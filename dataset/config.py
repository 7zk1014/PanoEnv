
# --- Path Configuration ---
BASE_DATA_DIR = "tartanair_output/"
OUTPUT_DIR = "final_output_qa"
SEG_ZIP_PATH = "segfiles.zip"
EXTRACTION_DIR = "extracted_maps"

# --- Environments to Exclude ---
EXCLUDED_ENVS = {
    'BrushifyMoon', 'ForestEnv', 'Gascola', 'GreatMarsh', 'OldScandinavia',
    'SeasonalForestAutumn', 'SeasonalForestSpring', 'SeasonalForestSummerNight',
    'SeasonalForestWinter', 'SeasonalForestWinterNight', 'ShoreCaves', 'Slaughter',
    'TerrainBlending'
}

# --- Quick Test Configuration (Optional) ---
TEST_MODE = False  # Large-scale processing mode: process all environments
TEST_ENVIRONMENTS = ['PolarSciFi', 'Downtown']  # Test environment list (disabled)
TEST_MAX_FRAMES_PER_ENV = 3  # Max frames per environment in test mode (disabled)

# --- Object Sampling and Filtering ---
TARGET_OBJECT_COUNT_PER_FRAME = 20
MIN_AREA = 900  # Lower minimum area, as objects may be smaller in ERP projection
MIN_WIDTH = 25   # Slightly lower minimum width
MIN_HEIGHT = 25  # Slightly lower minimum height
MAX_ASPECT_RATIO = 5
PADDING = 5

# Sampling Strategy Configuration
USE_CATEGORY_AWARE_SAMPLING = True  # True: use category diversity sampling, False: use traditional area sampling
CATEGORY_SAMPLING_MODE = "diversity_first"  # "diversity_first": prioritize category diversity, "quality_first": prioritize object quality

# ERP Special Region Sampling Configuration
USE_ERP_AWARE_SAMPLING = True  # Enable ERP-aware sampling (polar + seam priority)
ERP_POLAR_WEIGHT = 1.3  # Polar region object weight multiplier (1.0=no weighting, >1.0=priority sampling)

# === Performance Optimization Configuration ===
# Multi-processing Configuration
MAX_WORKERS = 8  # High-end hardware: utilize 12-core CPU (reserve 4 cores for system and GPU)
ENABLE_MULTIPROCESSING = True  # Enable multi-process processing

# GPU Acceleration Configuration (Reserved)
USE_GPU_ACCELERATION = True  # RTX 5060 available, but current task is CPU-focused
GPU_BATCH_SIZE = 32  # Used when GPU acceleration is enabled

# File I/O Optimization
SKIP_EXISTING_FILES = True  # Skip existing output files (important: avoid reprocessing)
BATCH_SIZE = 100  # High-performance system: increase batch size

# Memory Optimization
LAZY_LOAD_DEPTH_MAPS = True  # Lazy load depth maps
CLEAR_CACHE_FREQUENCY = 100  # High-memory system: reduce cleanup frequency for performance

# Visualization Configuration
GENERATE_VISUALIZATIONS = True  # Keep visualization enabled for large-scale processing

# Progress Saving
ENABLE_PROGRESS_SAVING = True  # Enable incremental processing and progress recovery
PROGRESS_SAVE_FREQUENCY = 5  # High-performance mode: save progress more frequently

# === High-Performance Hardware Optimization ===
HIGH_PERFORMANCE_MODE = True  # High-performance mode
PARALLEL_VISUALIZATION = True  # Parallel visualization processing
MEMORY_AGGRESSIVE_MODE = False  # Conservative memory management (high-memory systems can set True)
ERP_SEAM_WEIGHT = 1.2   # Seam object weight multiplier
ERP_MULTI_SPECIAL_BONUS = 1.5  # Extra bonus for objects with multiple special attributes (e.g., polar + seam)

# ERP Sampling Strategy Mode
ERP_SAMPLING_MODE = "balanced"  # "balanced": balance all dimensions, "special_first": special regions first, "category_first": category first

# ERP Region Representative Sampling Configuration
USE_ERP_REPRESENTATIVE_SAMPLING = True  # Enable region representative sampling
ERP_MIN_SEAM_REPRESENTATIVES = 3  # Minimum seam representatives
ERP_MIN_POLAR_REPRESENTATIVES = 2  # Minimum polar representatives (1 for top, 1 for bottom)

# Hierarchical Exclusion Strategy - Priority from high to low
# Layer 1: Always excluded base categories
ALWAYS_IGNORE_CLASSES = ['steps']

# Layer 2: Pure background large surfaces - always present, too large, very low QA discrimination
BACKGROUND_LARGE_CLASSES = [
    'sky', 'skysphere',           # Sky classes
    'ground', 'groundlandscape',  # Ground classes (but not specific terrain like mountains)
    'road', 'sidewalk',           # Road classes
    'wall', 'ceiling', 'floor', 'roof'  # Building surface classes (but keep details like walldetail)
]

# Layer 3: Cable/wire classes (usually only a few pixels thick in ERP, often misdetected)
CABLE_WIRE_CLASSES = [
    'powerline', 'tramlines', 'wire', 'cable'
]

# Layer 4: Miscellaneous debris/noise (unstable boundaries and semantics)
DEBRIS_NOISE_CLASSES = [
    'debris', 'rubble', 'trash'
]

# For backward compatibility, keep original IGNORE_CLASSES (Layer 1 + Layer 2)
IGNORE_CLASSES = ALWAYS_IGNORE_CLASSES + BACKGROUND_LARGE_CLASSES

# Minimum Sampling Requirement Configuration
MIN_OBJECTS_FOR_QA_GENERATION = 4  # Need at least 4 objects to generate complete question set

# --- Deduplication ---
DEDUP_TORUS_IOU_THRESH = 0.65
DEDUP_CONTAIN_RATIO  = 0.85
DEDUP_DEPTH_TAU_BASE = 0.40
DEDUP_KEEP_LARGER   = True
DEFAULT_ERP_SHAPE = (1280, 2560)  # Default ERP image dimensions

# --- QA General Configuration ---
QUESTIONS_PER_CATEGORY_PER_IMAGE = 5
SOFT_FAIL_ALLOW_PARTIAL = True
PER_FRAME_TOTAL_QUESTIONS = 25  # 5 categories * 5 questions

# --- Quality Control Strategy ---
STRICT_QA_MODE = True  # True: must generate complete 25 questions, otherwise skip image
                       # False: allow partial questions, dynamically adjust count
MIN_QUESTIONS_PER_CATEGORY = 5  # Minimum questions per category in strict mode

# --- View Question Configuration ---
VIEW_QA_MIN_AREA_RATIO = 0.003
VIEW_QA_TOP_QUANTILE  = 0.80


# --- Distance Question Configuration ---
DIST_SIMILARITY_BASE_M = 0.5
DIST_SIMILARITY_RATIO  = 0.10
DIST_IQR_BASE_M   = 0.6        # Fixed thickness threshold
DIST_IQR_RATIO    = 0.15       # Relative thickness threshold (relative to p50)
DIST_SIM_OVERLAP_JAC = 0.30    # Jaccard threshold for interval overlap "similarity"
DEPTH_NEAR_QUANTILE = 20       # "Near-end" quantile for comparison questions

# --- Relative Position Question Configuration ---
REL_POS_SIGNIFICANT_M = 0.5

# =========================
#  Environment Sets (Based on TartanAir Official In/Out & Type)
# =========================

# Indoor / Outdoor / Mixed
ENV_ATTRIBUTES = {
    "indoor": {
        "AbandonedFactory2","AmericanDiner","ArchVizTinyHouseDay","ArchVizTinyHouseNight",
        "CarWelding","CoalMine","CountryHouse","CyberPunkDowntown","Hospital","House",
        "Office","OldBrickHouseDay","OldBrickHouseNight","Restaurant","RetroOffice",
        "Sewerage","Supermarket"
    },
    "outdoor": {
        "AbandonedFactory","AmusementPark","AncientTowns","Antiquity3D","Apocalyptic",
        "BrushifyMoon","ConstructionSite","Cyberpunk","DesertGasStation","Downtown",
        "EndofTheWorld","FactoryWeather","Fantasy","ForestEnv","Gascola","GreatMarsh",
        "HongKong","IndustrialHangar","MiddleEast","ModularNeighborhood","NordicHarbor",
        "Ocean","OldIndustrialCity","OldScandinavia","OldTownFall","OldTownNight",
        "OldTownSummer","OldTownWinter","Rome","Ruins","SeasideTown",
        "SeasonalForestAutumn","SeasonalForestSpring","SeasonalForestSummerNight",
        "SeasonalForestWinter","SeasonalForestWinterNight","ShoreCaves","TerrainBlending",
        "UrbanConstruction","VictorianStreet","WaterMillDay","WaterMillNight"
    },
    # Mixed scenes with both indoor and outdoor areas (skip in env_attribute binary choice questions)
    "mix": {
        "AbandonedCable","AbandonedSchool","CastleFortress","GothicIsland","HQWesternSaloon",
        "JapaneseAlley","JapaneseCity","ModernCityDowntown","ModularNeighborhoodIntExt",
        "ModUrbanCity","PolarSciFi","Prison","Slaughter","SoulCity","WesternDesertTown"
    }
}

# Official 6 Category Types
ENV_CATEGORY_MAP = {
    "Infra": {
        "AbandonedCable","AbandonedFactory","AbandonedFactory2","AbandonedSchool",
        "CarWelding","CoalMine","ConstructionSite","FactoryWeather","Hospital",
        "IndustrialHangar","OldIndustrialCity","Prison","Sewerage"
    },
    "Domes": {
        "AmericanDiner","ArchVizTinyHouseDay","ArchVizTinyHouseNight","CountryHouse",
        "House","Office","OldBrickHouseDay","OldBrickHouseNight","Restaurant",
        "RetroOffice","Supermarket"
    },
    "Rural": {
        "AmusementPark","AncientTowns","CastleFortress","DesertGasStation","EndofTheWorld",
        "Fantasy","GothicIsland","GreatMarsh","HQWesternSaloon","SeasideTown",
        "WaterMillDay","WaterMillNight","WesternDesertTown"
    },
    "Them": {
        "Antiquity3D","Apocalyptic","Cyberpunk","CyberPunkDowntown","PolarSciFi","Rome","Slaughter"
    },
    "Nature": {
        "BrushifyMoon","ForestEnv","Gascola","Ocean","OldScandinavia","Ruins",
        "SeasonalForestAutumn","SeasonalForestSpring","SeasonalForestSummerNight",
        "SeasonalForestWinter","SeasonalForestWinterNight","ShoreCaves","TerrainBlending"
    },
    "Urban": {
        "Downtown","HongKong","JapaneseAlley","JapaneseCity","MiddleEast",
        "ModernCityDowntown","ModularNeighborhood","ModularNeighborhoodIntExt",
        "ModUrbanCity","NordicHarbor","OldTownFall","OldTownNight","OldTownSummer",
        "OldTownWinter","UrbanConstruction","VictorianStreet"
    }
}

# Convenience Sets
ENV_CATEGORIES   = list(ENV_CATEGORY_MAP.keys())
ALL_ENVIRONMENTS = sorted({e for s in ENV_CATEGORY_MAP.values() for e in s})