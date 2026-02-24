import numpy as np
import math
from numba import jit
from skimage import measure
import zipfile
import os
import shutil
import json
import config
from collections import defaultdict

# --- File Loading ---
def load_npz(path):
    data = np.load(path)
    for key in data.files:
        if isinstance(data[key], np.ndarray) and data[key].ndim == 2:
            return data[key]
    raise ValueError(f"No valid 2D array found in {path}.")

# --- Function to Extract Label Map from ZIP ---
def extract_seg_label_map(zip_path, environment_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    target_file_in_zip = f"{environment_name}/seg_label_map.json"

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            if target_file_in_zip in zf.namelist():
                file_info = zf.getinfo(target_file_in_zip)
                new_filename = f"{environment_name}_seg_label_map.json"
                output_path = os.path.join(output_dir, new_filename)
                with zf.open(file_info) as source, open(output_path, 'wb') as target:
                    shutil.copyfileobj(source, target)
                return output_path
            else:
                print(f"   [Warning] '{target_file_in_zip}' not found in '{zip_path}'")
                return None
    except FileNotFoundError:
        print(f"   [Critical Error] ZIP file does not exist: {zip_path}")
        return None
    except Exception as e:
        print(f"   [Critical Error] Error processing ZIP file: {e}")
        return None

# --- Basic Geometry Calculations ---
def get_bbox(mask, padding=5, image_shape=None):
    y, x = np.where(mask)
    if y.size == 0 or x.size == 0: return None
    x1, y1 = max(0, int(x.min()) - padding), max(0, int(y.min()) - padding)
    x2, y2 = int(x.max()) + padding, int(y.max()) + padding
    if image_shape:
        h, w = image_shape
        x2, y2 = min(x2, w - 1), min(y2, h - 1)
    return [x1, y1, x2, y2]

def _bbox_area(b):
    return max(0, b[2]-b[0]) * max(0, b[3]-b[1])


# --- View Determination Functions ---

# --- [Core Fix] Added missing functions and dictionaries ---
_ADJ = {
    "front": {"left", "right", "top", "bottom"}, "back":  {"left", "right", "top", "bottom"},
    "left":  {"front", "back", "top", "bottom"}, "right": {"front", "back", "top", "bottom"},
    "top":   {"front", "back", "left", "right"}, "bottom":{"front", "back", "left", "right"},
}

def _are_adjacent_views(pa: str, pb: str) -> bool:
    if not pa or not pb: return False
    if pa == pb: return False
    return pb in _ADJ.get(pa, set())

@jit(nopython=True)
def _direction_to_face_index(dx, dy, dz):
    ax, ay, az = abs(dx), abs(dy), abs(dz)
    if ax >= ay and ax >= az: face_index = 1 if dx > 0 else 0
    elif ay >= az: face_index = 2 if dy > 0 else 3
    else: face_index = 4 if dz > 0 else 5
    return face_index

FACE_INDEX_TO_NAME = {0: "left", 1: "right", 2: "top", 3: "bottom", 4: "front", 5: "back"}

def _get_camera_name_for_point(x, y, erp_shape):
    erp_h, erp_w = erp_shape
    lat = (0.5 - y / erp_h) * np.pi
    lon = (x / erp_w - 0.5) * 2 * np.pi
    sy, cy = np.sin(lat), np.cos(lat)
    sx, cx = np.sin(lon), np.cos(lon)
    dx, dy, dz = cy * sx, sy, cy * cx
    face_index = _direction_to_face_index(dx, dy, dz)
    return FACE_INDEX_TO_NAME.get(face_index, "unknown")

def get_primary_camera_accurate(bbox, erp_shape):
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return _get_camera_name_for_point(center_x, center_y, erp_shape)

def analyze_camera_views_from_mask(mask_single, erp_shape):
    """
    Efficient unified function that iterates through all pixels in the mask once to calculate:
    1. primary_camera with the highest proportion (main view)
    2. Complete visible_cameras list (all visible views)
    """
    y_coords, x_coords = np.where(mask_single)
    pixel_count = len(y_coords)
    if pixel_count == 0:
        return {"primary_camera": "unknown", "visible_cameras": []}

    # Iterate through all pixels and count occurrences of each view
    camera_counts = defaultdict(int)
    for x, y in zip(x_coords, y_coords):
        cam_name = _get_camera_name_for_point(x, y, erp_shape)
        if cam_name != "unknown":
            camera_counts[cam_name] += 1
    
    if not camera_counts:
        return {"primary_camera": "unknown", "visible_cameras": []}
        
    # Derive both answers directly from the statistics
    primary_camera = max(camera_counts, key=camera_counts.get)
    visible_cameras = sorted(list(camera_counts.keys()))
    
    return {
        "primary_camera": primary_camera,
        "visible_cameras": visible_cameras
    }

@jit(nopython=True)
def _uvd_to_xyz_batch(y_coords, x_coords, depths, erp_h, erp_w):
    """
    Batch convert (y, x, depth) pixels to (X, Y, Z) 3D coordinates using Numba JIT.
    Coordinate system: camera center as origin, Y-axis up, Z-axis forward, X-axis left.
    """
    num_points = len(y_coords)
    xyz = np.empty((num_points, 3), dtype=np.float32)
    pi = np.pi
    
    for i in range(num_points):
        y, x, r = y_coords[i], x_coords[i], depths[i]
        
        lat = -(y / erp_h - 0.5) * pi
        lon = (x / erp_w - 0.5) * 2 * pi
        
        X = -r * np.cos(lat) * np.sin(lon)
        Y = r * np.sin(lat)
        Z = r * np.cos(lat) * np.cos(lon)
        xyz[i, 0] = X
        xyz[i, 1] = Y
        xyz[i, 2] = Z
        
    return xyz

def get_3d_representation(mask_single, depth_map, erp_shape, downsample_points=2000):
    """
    Generate 3D point cloud and axis-aligned bounding box (AABB) for a single object.
    """
    y_coords, x_coords = np.where(mask_single)
    pixel_count = len(y_coords)
    if pixel_count == 0:
        return None

    # Downsample if point cloud is too large to ensure performance
    if downsample_points > 0 and pixel_count > downsample_points:
        indices = np.random.choice(pixel_count, size=downsample_points, replace=False)
        y_coords, x_coords = y_coords[indices], x_coords[indices]

    depths = depth_map[y_coords, x_coords]
    valid_mask = np.isfinite(depths) & (depths > 0)
    
    if not np.any(valid_mask):
        return None

    y_coords, x_coords, depths = y_coords[valid_mask], x_coords[valid_mask], depths[valid_mask]

    point_cloud = _uvd_to_xyz_batch(y_coords, x_coords, depths, erp_shape[0], erp_shape[1])
    
    if point_cloud.shape[0] == 0:
        return None

    min_coords = np.min(point_cloud, axis=0)
    max_coords = np.max(point_cloud, axis=0)
    
    bbox_3d = {
        'min_x': min_coords[0], 'max_x': max_coords[0],
        'min_y': min_coords[1], 'max_y': max_coords[1],
        'min_z': min_coords[2], 'max_z': max_coords[2]
    }
    
    return bbox_3d
