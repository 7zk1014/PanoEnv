import random
import numpy as np
from itertools import combinations
import math
import config
from utils import analyze_camera_views_from_mask, get_primary_camera_accurate, _are_adjacent_views
import re

def _smart_ref_question_answer(objects_in_question):
    labels = [obj['label'] for obj in objects_in_question]
    
    question_refs = {}
    answer_refs = {}
    
    if len(set(labels)) == len(labels):
        for obj in objects_in_question:
            question_refs[obj['object_id']] = f"the {obj['label']} (boundingbox: {str(obj['bbox'])})"
            answer_refs[obj['object_id']] = f"the {obj['label']}"
    else:
        label_counts = {}
        
        for obj in objects_in_question:
            label = obj['label']
            label_counts[label] = label_counts.get(label, 0) + 1
            
            if label_counts[label] == 1:
                if labels.count(label) > 1:
                    question_refs[obj['object_id']] = f"{label} A (boundingbox: {str(obj['bbox'])})"
                    answer_refs[obj['object_id']] = f"{label} A"
                else:
                    question_refs[obj['object_id']] = f"the {label} (boundingbox: {str(obj['bbox'])})"
                    answer_refs[obj['object_id']] = f"the {label}"
            else:
                letter = chr(ord('A') + label_counts[label] - 1)
                question_refs[obj['object_id']] = f"{label} {letter} (boundingbox: {str(obj['bbox'])})"
                answer_refs[obj['object_id']] = f"{label} {letter}"
    
    return question_refs, answer_refs

def _ref_question(o, question_refs):
    if question_refs and o['object_id'] in question_refs:
        return question_refs[o['object_id']]
    else:
        return f"object {o['object_id']} (the {o['label']} at boundingbox: {str(o['bbox'])})"

def _ref_answer(o, answer_refs):
    if answer_refs and o['object_id'] in answer_refs:
        return answer_refs[o['object_id']]
    else:
        return f"the {o['label']}"

def _smart_ref(objects_in_question):
    _, answer_refs = _smart_ref_question_answer(objects_in_question)
    return answer_refs

def _capitalize_smart(text):
    if not text:
        return text
    return text[0].upper() + text[1:]

def _ref(o, ref_mapping=None, is_question=True):
    if ref_mapping and o['object_id'] in ref_mapping:
        return ref_mapping[o['object_id']]
    else:
        if is_question:
            return f"object {o['object_id']} (the {o['label']} at boundingbox: {str(o['bbox'])})"
        else:
            return f"the {o['label']}"

def _calculate_bbox_overlap_ratio(bbox1, bbox2):
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    if x1_inter >= x2_inter or y1_inter >= y2_inter:
        return 0.0
    
    area_inter = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    area_1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    
    if area_1 == 0:
        return 0.0
        
    return area_inter / area_1

def _are_objects_suitable_for_comparison(objects, containment_threshold=0.9):
    if len(objects) < 2:
        return True
        
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            obj1, obj2 = objects[i], objects[j]
            
            overlap_ratio_1_in_2 = _calculate_bbox_overlap_ratio(obj1['bbox'], obj2['bbox'])
            if overlap_ratio_1_in_2 >= containment_threshold:
                return False
                
            overlap_ratio_2_in_1 = _calculate_bbox_overlap_ratio(obj2['bbox'], obj1['bbox'])
            if overlap_ratio_2_in_1 >= containment_threshold:
                return False
                
    return True

def _select_objects_for_comparison(candidates, num_objects, max_attempts=20, containment_threshold=0.9):
    if len(candidates) < num_objects:
        return None
        
    for attempt in range(max_attempts):
        selected = random.sample(candidates, num_objects)
        if _are_objects_suitable_for_comparison(selected, containment_threshold):
            return selected
            
    print(f"   [Warning] After {max_attempts} attempts, no suitable object combination found, using random selection as fallback")
    return random.sample(candidates, num_objects)

def _filter_unsuitable_objects(objects):
    unsuitable_classes = set([
        'sky', 'skysphere', 'cloud', 'clouds',
        'terrain', 'landscape'
    ])
    
    filtered_objects = []
    removed_count = 0
    
    for obj in objects:
        label = obj.get('label', '').lower()
        if label in unsuitable_classes:
            removed_count += 1
            continue
        if any(unsuitable in label for unsuitable in ['sky', 'ground', 'wall', 'ceiling', 'floor', 'road']):
            removed_count += 1
            continue
        filtered_objects.append(obj)
    
    if removed_count > 0:
        print(f"   [Filter] Removed {removed_count} unsuitable background objects")
    
    return filtered_objects

TEMPLATES = {
    "view_source": {
        "primary_view": {
            "question_type": "multiple_choice",
            "templates": [
                "From which view (front/back/left/right/top/bottom) does {a} primarily come?",
                "Which camera view (front/back/left/right/top/bottom) does {a} mainly originate from?",
                "Which view (front/back/left/right/top/bottom) is {a} most likely from?"
            ]
        },
        "multi_view_visibility": {
            "question_type": "open_ended",
            "templates": [
                "From how many different views (front/back/left/right/top/bottom) is {a} visible?",
                "In how many camera perspectives (front/back/left/right/top/bottom) does {a} appear?",
                "How many camera views (front/back/left/right/top/bottom) can observe {a}?"
            ]
        },
        "seam_attribution": {
            "question_type": "multiple_choice", 
            "templates": [
                "When {a} spans a boundary, which is its primary view (front/back/left/right/top/bottom)?",
                "At the view boundary, {a} belongs to which primary view (front/back/left/right/top/bottom)?",
                "Given {a} crosses two views, which view (front/back/left/right/top/bottom) should it be attributed to primarily?"
            ]
        },
        "shared_visibility": {
            "question_type": "true_false",
            "templates": [
                "Is {b} also visible from the primary camera view of {a}?",
                "Does the primary view for {a} also capture any part of {b}?",
                "Is it true that {b} is visible within the perspective of {a}'s main camera?"
            ]
        },
        "multi_object_relationship": {
            "question_type": "true_false",
            "templates": [
                "Do {a} and {b} come from the same primary view?",
                "Do {a} and {b} originate from adjacent views?",
                "Are {a} and {b} visible in the same camera view?"
            ]
        }
    },
    "distance": {
        "depth_compare": {
            "question_type": "multiple_choice",
            "templates": [
                "Which object is closer to the viewer, {a} or {b}?",
                "Between {a} and {b}, which one is nearer to the camera?",
                "Based on depth, which is closer to the viewer: {a} or {b}?"
            ]
        },
        "depth_binary": {
            "question_type": "true_false",
            "templates": [
                "Is {a} closer to the viewer than {b}?",
                "From the camera's perspective, is {a} positioned at a shorter distance than {b}?",
                "Is {a} located nearer to the viewer than {b}?"
            ]
        },
        "depth_similarity": {
            "question_type": "true_false",
            "templates": [
                "Are {a} and {b} at a similar distance from the viewer?",
                "Do {a} and {b} appear at the same depth level relative to the camera?",
                "Are {a} and {b} roughly at the same distance from the viewer?"
            ]
        },
        "distance_description": {
            "question_type": "open_ended",
            "templates": [
                "How far is {a} from the viewer?",
                "Estimate the distance to {a} from the camera.",
                "What is the distance between the camera and {a}?"
            ]
        },
        "depth_triplet_farthest": {
            "question_type": "multiple_choice", 
            "templates": [
                "Which of the three objects is farthest from the viewer: {a}, {b}, or {c}?",
                "Among {a}, {b}, and {c}, which is most distant from the camera?",
                "Who is located at the greatest distance from the viewer: {a}, {b}, or {c}?"
            ]
        }
    },
    "environment": {
        "env_mcq": {
            "question_type": "multiple_choice",
            "templates": [
                "Given that {a} is in the scene, which environment is it: {options}?",
                "Based on the image containing {a} and {b}, which of these environments fits the scene best: {options}?",
                "Observing {a}, which of the following is the correct environment: {options}?"
            ]
        },
        "env_binary_judgement": {
            "question_type": "true_false",
            "templates": [
                "Considering the object {a}, is the environment of this scene {env}?",
                "Is the environment shown with {a} the {env}?",
                "Is it plausible that the scene containing {a} is {env}?"
            ]
        },
        "env_attribute": {
            "question_type": "multiple_choice",
            "templates": [
                "Considering the setting around {a}, is this scene located indoors or outdoors?",
                "Based on the context of {a}, would you classify this environment as indoors or outdoors?",
                "From the visual cues in the image, is the setting of this scene indoors or outdoors?"
            ]
        },
        "env_confusable_pair": {
            "question_type": "multiple_choice",
            "templates": [
                "Given the presence of {a}, which is a more likely environment for this scene: {env1} or {env2}?",
                "Which environment is a better fit for the image: {env1} or {env2}?",
                "If you had to choose between {env1} and {env2}, which is the correct environment for this scene?"
            ]
        },
        "env_category_identification": {
            "question_type": "multiple_choice",
            "templates": [
                "Which general category does the environment containing {a} belong to: {options}?",
                "To which general classification does this scene belong: {options}?",
                "The environment in the image can be best described as which of the following categories: {options}?"
            ]
        },
        "env_scene_judgement": {
            "question_type": "true_false",
            "templates": [
                "Based on the overall visual characteristics of this scene, is this environment {env}?",
                "Does this scene appear to be set in {env}?",
                "Would you classify the environment depicted in this image as {env}?"
            ]
        }
    },
    "relative_position": {
        "relpos_binary": {
            "question_type": "true_false",
            "templates": [
                "In the real world, is {a} to the left of {b}?",
                "Considering their 3D positions, is {a} located to the right of {b}?",
                "Is {a} positioned above {b} in 3D space?"
            ]
        },
        "relpos_cardinal": {
            "question_type": "open_ended",
            "templates": [
                "What is the spatial relationship of {a} relative to {b} in the 3D world?",
                "Describe the 3D position of {a} with respect to {b}.",
                "In 3D space, how is {a} positioned relative to {b}?"
            ]
        },
        "relpos_distance_straightline": {
            "question_type": "open_ended",
            "templates": [
                "What is the straight-line distance between the centers of {a} and {b} in meters?",
                "How far apart are {a} and {b} in the 3D scene?",
                "What is the distance between the center of {a} and the center of {b}?"
            ]
        },
        "relpos_triplet_extreme": {
            "question_type": "multiple_choice",
            "templates": [
                "Among {a}, {b}, and {c}, which one is the highest?",
                "Out of {a}, {b}, and {c}, which is located furthest to the left in the 3D scene?",
                "Which of the three objects, {a}, {b}, or {c}, is positioned furthest forward in the scene?"
            ]
        },
        "relpos_distance_components": {
            "question_type": "open_ended",
            "templates": [
                "What is the horizontal distance between {a} and {b}?",
                "Ignoring height differences, how far apart are {a} and {b} on the ground plane?",
                "What is the vertical distance (height difference) between {a} and {b}?"
            ]
        }
    },
    "attribute_comparison": {
        "volume_comparison": {
            "question_type": "multiple_choice",
            "templates": [
                "In the real world, which object has a larger volume, {a} or {b}?",
                "Between {a} and {b}, which one is physically bigger?",
                "Which of these two objects, {a} or {b}, occupies less space in 3D?"
            ]
        },
        "volume_binary": {
            "question_type": "true_false",
            "templates": [
                "In terms of real-world size, is {a} smaller than {b}?",
                "Is the physical volume of {a} greater than that of {b}?",
                "Considering their 3D dimensions, is it correct that {a} is bigger than {b}?"
            ]
        },
        "shape_flatness": {
            "question_type": "multiple_choice",
            "templates": [
                "Which object looks flatter, {a} or {b}?",
                "Between {a} and {b}, which one is more like a sheet or a plate?",
                "Which of these two seems much thinner than it is wide, {a} or {b}?"
            ]
        },
        "shape_elongation": {
            "question_type": "multiple_choice",
            "templates": [
                "Which object has a more elongated or stick-like shape, {a} or {b}?",
                "Between {a} and {b}, which one is much longer than it is wide?",
                "Which of these two is more stretched out, {a} or {b}?"
            ]
        },
        "size_triplet_extreme": {
            "question_type": "multiple_choice",
            "templates": [
                "Among {a}, {b}, and {c}, which one has the largest volume?",
                "Out of the three objects {a}, {b}, and {c}, which is the smallest in the real world?",
                "Please identify the biggest object among {a}, {b}, and {c}."
            ]
        }
    }
}

def build_view_questions(objects, erp_shape, per_cat):
    if not objects: return []
    final_qas = []
    used_ids = set()

    def add_qa(sub_category, obj_actors):
        template_config = TEMPLATES["view_source"][sub_category]
        question_type = template_config["question_type"]
        
        t = random.choice(template_config["templates"])
        
        ans, q_text = "", ""

        if len(obj_actors) == 1:
            o = obj_actors[0]
            question_refs, answer_refs = _smart_ref_question_answer(obj_actors)
            q_text = t.format(a=_ref_question(o, question_refs))
            if sub_category == "primary_view":
                ans = _capitalize_smart(o['primary_camera'])
            elif sub_category == "multi_view_visibility":
                count = len(set([c for c in o["visible_cameras"] if c != "unknown"]))
                if count == 1:
                    ans = "1 different view"
                else:
                    ans = f"{count} different views"
            elif sub_category == "seam_attribution":
                ans = _capitalize_smart(o['primary_camera'])

        elif len(obj_actors) == 2:
            a, b = obj_actors[0], obj_actors[1]
            tt = t.lower()
            question_refs, answer_refs = _smart_ref_question_answer(obj_actors)
            q_text = t.format(a=_ref_question(a, question_refs), b=_ref_question(b, question_refs))

            if sub_category == "multi_object_relationship":
                if "adjacent" in tt:
                    ans = "Yes" if _are_adjacent_views(a["primary_camera"], b["primary_camera"]) else "No"
                elif "same camera view" in tt:
                    a_cameras = set([c for c in a.get("visible_cameras", []) if c != "unknown"])
                    b_cameras = set([c for c in b.get("visible_cameras", []) if c != "unknown"])
                    common_cameras = a_cameras & b_cameras
                    ans = "Yes" if len(common_cameras) > 0 else "No"
                else:
                    ans = "Yes" if a["primary_camera"] == b["primary_camera"] else "No"

            elif sub_category == "shared_visibility":
                ans = "Yes" if a["primary_camera"] in set(b["visible_cameras"]) else "No"

        final_qas.append({
            "major_category": "view_source",
            "sub_category": sub_category,
            "question_type": question_type,
            "question": q_text,
            "answer": ans,
            "related_object_ids": [o["object_id"] for o in obj_actors],
        })
        for o in obj_actors: 
            used_ids.add(o["object_id"])
        return True

    target_subcategories = [
        "primary_view", "multi_view_visibility", "seam_attribution",
        "shared_visibility", "multi_object_relationship"
    ]

    for sub in target_subcategories:
        unused_objects = [o for o in objects if o["object_id"] not in used_ids]
        
        if sub == "seam_attribution":
            pool = [o for o in unused_objects if o.get("is_seam", False)]
            if not pool:
                pool = [o for o in objects if o.get("is_seam", False)]
            if pool: 
                add_qa(sub, [random.choice(pool)])
                
        elif sub in ["shared_visibility", "multi_object_relationship"]:
            if len(unused_objects) >= 2:
                selected_objects = _select_objects_for_comparison(unused_objects, 2)
                if selected_objects:
                    add_qa(sub, selected_objects)
            elif len(objects) >= 2:
                selected_objects = _select_objects_for_comparison(objects, 2)
                if selected_objects:
                    add_qa(sub, selected_objects)
                
        else:
            if unused_objects:
                add_qa(sub, [random.choice(unused_objects)])
            elif objects:
                add_qa(sub, [random.choice(objects)])
                

    def _is_yesno(s):
        return s in ("Yes", "No", "True", "False")
    for qa in final_qas:
        if qa["major_category"] == "view_source" and qa["sub_category"] == "multi_object_relationship":
            qlower = qa["question"].lower()
            ans = qa["answer"]
            if "which views" in qlower and _is_yesno(ans):
                try:
                    a_id, b_id = qa["related_object_ids"]
                    oa = next(o for o in objects if o["object_id"] == a_id)
                    ob = next(o for o in objects if o["object_id"] == b_id)
                    qa["answer"] = f"{oa['primary_camera']} and {ob['primary_camera']}"
                except Exception:
                    pass
            if ("same view" in qlower or "adjacent" in qlower) and not _is_yesno(ans):
                try:
                    a_id, b_id = qa["related_object_ids"]
                    oa = next(o for o in objects if o["object_id"] == a_id)
                    ob = next(o for o in objects if o["object_id"] == b_id)
                    qa["answer"] = "Yes" if oa["primary_camera"] == ob["primary_camera"] else "No"
                except Exception:
                    pass

    return final_qas[:per_cat]

def _eff_depth(o, for_compare=False):
    ds = o.get("depth_stats")
    if not ds:
        return float(o["depth"])
    p50 = float(ds.get("p50", o["depth"]))
    iqr = float(ds.get("iqr", 0.0))
    thick = iqr > max(config.DIST_IQR_BASE_M, config.DIST_IQR_RATIO * p50)
    if for_compare and thick:
        q = max(1, min(49, int(config.DEPTH_NEAR_QUANTILE)))
        return float(ds.get(f"p{q}", ds.get("p20", ds.get("p25", p50))))
    return p50

def _similar_depth(a, b):
    ds_a, ds_b = a.get("depth_stats"), b.get("depth_stats")
    if ds_a and ds_b:
        A = (float(ds_a["p25"]), float(ds_a["p75"]))
        B = (float(ds_b["p25"]), float(ds_b["p75"]))
        inter = max(0.0, min(A[1], B[1]) - max(A[0], B[0]))
        union = max(A[1], B[1]) - min(A[0], B[0]) + 1e-6
        jac = inter / union
        med_diff_ok = abs(float(ds_a["p50"]) - float(ds_b["p50"])) < max(
            config.DIST_SIMILARITY_BASE_M,
            config.DIST_SIMILARITY_RATIO * min(float(ds_a["p50"]), float(ds_b["p50"]))
        )
        if jac > config.DIST_SIM_OVERLAP_JAC or med_diff_ok:
            return True
        return False
    return abs(a["depth"] - b["depth"]) < max(
        config.DIST_SIMILARITY_BASE_M,
        config.DIST_SIMILARITY_RATIO * min(a["depth"], b["depth"])
    )

def build_distance_questions(objects, erp_shape, per_cat):
    if not objects: return []
    final_qas = []
    valid_objects = sorted(
        [o for o in objects if "depth" in o and o["depth"] > 0],
        key=lambda x: _eff_depth(x, for_compare=False)
    )
    if not valid_objects: return []
    used_ids = set()
    
    def add_qa(sub_category, obj_actors):
        template_config = TEMPLATES["distance"][sub_category]
        question_type = template_config["question_type"]
        t = random.choice(template_config["templates"])
        ans, q_text = "", ""

        if sub_category == "depth_compare":
            a, b = obj_actors
            question_refs, answer_refs = _smart_ref_question_answer(obj_actors)
            da = _eff_depth(a, for_compare=True)
            db = _eff_depth(b, for_compare=True)
            closer_obj = a if da < db else b
            closer_ref = answer_refs[closer_obj['object_id']]
            ans = _capitalize_smart(closer_ref)
            q_text = t.format(a=_ref_question(a, question_refs), b=_ref_question(b, question_refs))
        elif sub_category == "depth_binary":
            a, b = obj_actors
            question_refs, answer_refs = _smart_ref_question_answer(obj_actors)
            da = _eff_depth(a, for_compare=True)
            db = _eff_depth(b, for_compare=True)
            ans = "Yes" if da < db else "No"
            q_text = t.format(a=_ref_question(a, question_refs), b=_ref_question(b, question_refs))
        elif sub_category == "depth_similarity":
            a, b = obj_actors
            question_refs, answer_refs = _smart_ref_question_answer(obj_actors)
            ans = "Yes" if _similar_depth(a, b) else "No"
            q_text = t.format(a=_ref_question(a, question_refs), b=_ref_question(b, question_refs))
        elif sub_category == "distance_description":
            a = obj_actors[0]
            question_refs, answer_refs = _smart_ref_question_answer(obj_actors)
            ds = a.get("depth_stats")
            p50 = float(ds["p50"]) if ds else float(a["depth"])
            ans = f"About {round(p50, 1)} meters"
            q_text = t.format(a=_ref_question(a, question_refs))
        elif sub_category == "depth_triplet_farthest":
            a, b, c = obj_actors
            question_refs, answer_refs = _smart_ref_question_answer(obj_actors)
            def far_metric(o):
                ds = o.get("depth_stats")
                return float(ds["p80"]) if ds else float(o["depth"])
            farthest_obj = max([a, b, c], key=far_metric)
            farthest_ref = answer_refs[farthest_obj['object_id']]
            ans = _capitalize_smart(farthest_ref)
            q_text = t.format(a=_ref_question(a, question_refs), b=_ref_question(b, question_refs), c=_ref_question(c, question_refs))
            
        final_qas.append({
            "major_category": "distance",
            "sub_category": sub_category,
            "question_type": question_type,
            "question": q_text, "answer": ans,
            "related_object_ids": [o["object_id"] for o in obj_actors],
        })
        for o in obj_actors: 
            used_ids.add(o["object_id"])
        return True

    target_subcategories = ["distance_description", "depth_compare", "depth_binary", "depth_similarity", "depth_triplet_farthest"]
    
    for sub in target_subcategories:
        unused_objects = [o for o in valid_objects if o["object_id"] not in used_ids]
        
        if sub == "depth_similarity":
            if len(unused_objects) >= 2:
                selected_objects = _select_objects_for_comparison(unused_objects, 2)
                if selected_objects:
                    add_qa(sub, selected_objects)
            elif len(valid_objects) >= 2:
                selected_objects = _select_objects_for_comparison(valid_objects, 2)
                if selected_objects:
                    add_qa(sub, selected_objects)
                
        elif sub == "depth_triplet_farthest":
            if len(unused_objects) >= 3:
                selected_objects = _select_objects_for_comparison(unused_objects, 3)
                if selected_objects:
                    add_qa(sub, selected_objects)
            elif len(valid_objects) >= 3:
                selected_objects = _select_objects_for_comparison(valid_objects, 3)
                if selected_objects:
                    add_qa(sub, selected_objects)
                
        elif sub in ["depth_compare", "depth_binary"]:
            if len(unused_objects) >= 2:
                selected_objects = _select_objects_for_comparison(unused_objects, 2)
                if selected_objects:
                    add_qa(sub, selected_objects)
            elif len(valid_objects) >= 2:
                selected_objects = _select_objects_for_comparison(valid_objects, 2)
                if selected_objects:
                    add_qa(sub, selected_objects)
                
        elif sub == "distance_description":
            if unused_objects:
                add_qa(sub, [random.choice(unused_objects)])
            elif valid_objects:
                add_qa(sub, [random.choice(valid_objects)])
    return final_qas[:per_cat]

# -------------------------
# -------------------------
def build_environment_questions(objects, erp_shape, per_cat, environment):
    ALL_ENVIRONMENTS = config.ALL_ENVIRONMENTS
    ENV_ATTRIBUTES   = config.ENV_ATTRIBUTES
    ENV_CATEGORY_MAP = config.ENV_CATEGORY_MAP
    ENV_CATEGORIES   = config.ENV_CATEGORIES
    
    def template_needs_objects(template):
        import re
        return bool(re.search(r'{[abc]}', template))
    def get_confusable_envs(env, num_wrong=3):
        
        MEANINGLESS_GROUPS = [
            {"AbandonedFactory", "AbandonedFactory2"},
        ]
        
        EDUCATIONAL_GROUPS = [
            {"ArchVizTinyHouseDay", "ArchVizTinyHouseNight"},
            {"OldBrickHouseDay", "OldBrickHouseNight"},
            {"WaterMillDay", "WaterMillNight"},
            {"SeasonalForestAutumn", "SeasonalForestSpring", "SeasonalForestSummerNight", 
             "SeasonalForestWinter", "SeasonalForestWinterNight"},
            {"ModularNeighborhood", "ModularNeighborhoodIntExt"},
            {"OldTownFall", "OldTownNight", "OldTownSummer", "OldTownWinter"}
        ]
        
        meaningless_group = None
        for group in MEANINGLESS_GROUPS:
            if env in group:
                meaningless_group = group
                break
        
        educational_group = None
        for group in EDUCATIONAL_GROUPS:
            if env in group:
                educational_group = group
                break
        
        excluded = {env}
        if meaningless_group:
            excluded.update(meaningless_group)
            
        basic_candidates = [e for e in ALL_ENVIRONMENTS if e not in excluded]
        
        if len(basic_candidates) < num_wrong:
            random.shuffle(basic_candidates)
            return basic_candidates[:num_wrong]
        
        selected = []
        
        if educational_group and len(selected) < num_wrong:
            educational_candidates = [e for e in educational_group if e not in excluded]
            if educational_candidates:
                educational_pick_count = min(
                    max(1, num_wrong // 2),
                    len(educational_candidates),
                    num_wrong - len(selected)
                )
                selected.extend(random.sample(educational_candidates, educational_pick_count))
        
        if len(selected) < num_wrong:
            current_category = None
            for category, envs in ENV_CATEGORY_MAP.items():
                if env in envs:
                    current_category = category
                    break
            
            if current_category:
                same_category_candidates = [e for e in basic_candidates 
                                          if e in ENV_CATEGORY_MAP[current_category] and e not in selected]
                if same_category_candidates:
                    remaining_slots = num_wrong - len(selected)
                    same_cat_pick = min(max(1, remaining_slots // 2), len(same_category_candidates))
                    selected.extend(random.sample(same_category_candidates, same_cat_pick))
        
        if len(selected) < num_wrong:
            remaining_candidates = [c for c in basic_candidates if c not in selected]
            if remaining_candidates:
                remaining_needed = num_wrong - len(selected)
                selected.extend(random.sample(remaining_candidates, min(remaining_needed, len(remaining_candidates))))
        
        random.shuffle(selected)
        return selected[:num_wrong]
    def format_options_inline(options):
        return ", ".join(options)
    if environment in config.EXCLUDED_ENVS or len(ALL_ENVIRONMENTS) < 4: return []
    # if not objects: return []
    
    final_qas = []
    used_ids = set()
    is_mix = environment in ENV_ATTRIBUTES.get("mix", set())
    
    def add_qa(sub_category, obj_actors=None, extra_info=None):
        template_config = TEMPLATES["environment"][sub_category]
        question_type = template_config["question_type"]
        available_templates = template_config["templates"]
        
        if not obj_actors or len(obj_actors) == 0:
            no_object_templates = [t for t in available_templates if not template_needs_objects(t)]
            if no_object_templates:
                t = random.choice(no_object_templates)
            else:
                return False
        else:
            import re
            suitable_templates = []
            for template in available_templates:
                needed_keys = set(re.findall(r"{([abc])}", template))
                if len(needed_keys) <= len(obj_actors):
                    suitable_templates.append(template)
            
            if suitable_templates:
                t = random.choice(suitable_templates)
            else:
                template_obj_counts = []
                for template in available_templates:
                    needed_keys = set(re.findall(r"{([abc])}", template))
                    template_obj_counts.append((template, len(needed_keys)))
                template_obj_counts.sort(key=lambda x: x[1])
                t = template_obj_counts[0][0]
        
        ans, q_text, options = "", "", []
        import re
        needed_keys = sorted(set(re.findall(r"{([abc])}", t)))
        
        if obj_actors and len(obj_actors) > 0:
            question_refs, answer_refs = _smart_ref_question_answer(obj_actors)
            if len(obj_actors) >= len(needed_keys):
                fmt = {k: _ref_question(obj_actors[i], question_refs) for i, k in enumerate(needed_keys)}
            else:
                fmt = {k: _ref_question(obj_actors[i % len(obj_actors)], question_refs) for i, k in enumerate(needed_keys)}
        else:
            fmt = {k: "(missing)" for k in needed_keys}
        if sub_category == "env_mcq":
            wrong_envs = get_confusable_envs(environment, num_wrong=3)
            options = [environment] + wrong_envs
            random.shuffle(options)
            ans = _capitalize_smart(environment)
            q_text = t.format(**fmt, options=format_options_inline(options))
        elif sub_category == "env_binary_judgement":
            if random.random() > 0.5:
                ans = "Yes"; q_text = t.format(**fmt, env=environment)
            else:
                wrong_envs = get_confusable_envs(environment, num_wrong=1)
                if not wrong_envs: 
                    ans = "Yes"; q_text = t.format(**fmt, env=environment)
                else:
                    ans = "No"; q_text = t.format(**fmt, env=wrong_envs[0])
        elif sub_category == "env_attribute":
            if is_mix:
                return False
            if environment in ENV_ATTRIBUTES["indoor"]:
                ans = "Indoors"; q_text = t.format(**fmt)
            elif environment in ENV_ATTRIBUTES["outdoor"]:
                ans = "Outdoors"; q_text = t.format(**fmt)
            else:
                return False
        elif sub_category == "env_confusable_pair":
            if not extra_info: return False
            env1, env2 = extra_info['env1'], extra_info['env2']
            ans = _capitalize_smart(environment); q_text = t.format(**fmt, env1=env1, env2=env2)
        elif sub_category == "env_category_identification":
            correct_cat = next((cat for cat, envs in ENV_CATEGORY_MAP.items() if environment in envs), None)
            if not correct_cat: return False
            wrong_cats = [c for c in ENV_CATEGORIES if c != correct_cat]
            options = [correct_cat] + random.sample(wrong_cats, min(len(wrong_cats), 2))
            random.shuffle(options)
            ans = f"{correct_cat}"
            q_text = t.format(**fmt, options=format_options_inline(options))
        elif sub_category == "env_scene_judgement":
            if random.random() > 0.5:
                ans = "Yes"; q_text = t.format(**fmt, env=environment)
            else:
                wrong_envs = get_confusable_envs(environment, num_wrong=1)
                if not wrong_envs:
                    ans = "Yes"; q_text = t.format(**fmt, env=environment)
                else:
                    ans = "No"; q_text = t.format(**fmt, env=wrong_envs[0])

        qa_item = {
            "major_category": "environment",
            "sub_category": sub_category,
            "question_type": question_type,
            "question": q_text,
            "answer": ans,
            "related_object_ids": [o["object_id"] for o in obj_actors] if obj_actors else [],
        }
        if options: qa_item["options"] = options
        final_qas.append(qa_item)
        if obj_actors:
            for o in obj_actors: 
                used_ids.add(o["object_id"])
        return True
    
    if is_mix:
        target_subcategories = ["env_mcq", "env_binary_judgement", "env_confusable_pair", "env_scene_judgement", "env_category_identification"]
    else:
        target_subcategories = ["env_mcq", "env_binary_judgement", "env_attribute", "env_confusable_pair", "env_category_identification"]
    
    for i, sub in enumerate(target_subcategories):
        template_config = TEMPLATES["environment"][sub]
        available_templates = template_config["templates"]
        has_no_object_templates = any(not template_needs_objects(t) for t in available_templates)
        
        if has_no_object_templates and (not objects or len(objects) == 0):
            success = add_qa(sub, obj_actors=None)
            if success:
                continue
        
        unused_objects = [o for o in objects if o["object_id"] not in used_ids]
        
        import re
        max_objects_needed = 0
        min_objects_needed = 999
        for template in available_templates:
            needed_keys = set(re.findall(r"{([abc])}", template))
            template_obj_count = len(needed_keys)
            max_objects_needed = max(max_objects_needed, template_obj_count)
            min_objects_needed = min(min_objects_needed, template_obj_count)
        
        actors = None
        
        if len(unused_objects) >= max_objects_needed and max_objects_needed > 0:
            if max_objects_needed > 1:
                actors = _select_objects_for_comparison(unused_objects, max_objects_needed)
            else:
                actors = [random.choice(unused_objects)]
        elif len(objects) >= max_objects_needed and max_objects_needed > 0:
            if max_objects_needed > 1:
                actors = _select_objects_for_comparison(objects, max_objects_needed)
            else:
                actors = [random.choice(objects)]
        elif len(unused_objects) >= min_objects_needed and min_objects_needed > 0:
            if min_objects_needed > 1:
                actors = _select_objects_for_comparison(unused_objects, min_objects_needed)
            else:
                actors = [random.choice(unused_objects)]
        elif len(objects) >= min_objects_needed and min_objects_needed > 0:
            if min_objects_needed > 1:
                actors = _select_objects_for_comparison(objects, min_objects_needed)
            else:
                actors = [random.choice(objects)]
        
        if not actors and not has_no_object_templates:
            continue
            
        if sub == "env_confusable_pair":
            wrong = get_confusable_envs(environment, num_wrong=1)
            if wrong:
                pair = [environment, wrong[0]]; random.shuffle(pair)
                add_qa(sub, actors, extra_info={'env1': pair[0], 'env2': pair[1]})
        else:
            add_qa(sub, actors)
            
    return final_qas[:per_cat]

# -------------------------
# -------------------------
def _get_object_3d_centroid(obj, depth_map, erp_shape):
    H, W = erp_shape
    x1, y1, x2, y2 = obj['bbox']
    center_px, center_py = (x1 + x2) / 2, (y1 + y2) / 2
    center_px_int, center_py_int = int(np.clip(center_px, 0, W - 1)), int(np.clip(center_py, 0, H - 1))
    depth = obj.get('depth')
    if depth is None or depth <= 0:
        depth = float(depth_map[center_py_int, center_px_int])
        if not (np.isfinite(depth) and depth > 0): return None
    lon, lat = (center_px / W - 0.5) * 2 * math.pi, -(center_py / H - 0.5) * math.pi
    r = depth
    x = -r * math.cos(lat) * math.sin(lon)
    y = r * math.sin(lat)
    z = r * math.cos(lat) * math.cos(lon)
    return np.array([x, y, z])

def _vector_to_cardinal_direction(vec, threshold):
    dx, dy, dz = vec
    desc = []
    if dz > threshold: desc.append("in front of")
    elif dz < -threshold: desc.append("behind")
    if dx > threshold: desc.append("to the left of")
    elif dx < -threshold: desc.append("to the right of")
    if dy > threshold: desc.append("above")
    elif dy < -threshold: desc.append("below")
    if not desc: return "at a similar position to"
    return " and ".join(desc)

def build_relative_position_questions(objects, erp_shape, per_cat, depth_map):
    if not objects: return []
    for o in objects:
        o['centroid_3d'] = _get_object_3d_centroid(o, depth_map, erp_shape)
    valid_objects = [o for o in objects if o['centroid_3d'] is not None]
    if not valid_objects: return []
    
    final_qas = []
    used_ids = set()
    
    def add_qa(sub_category, obj_actors):
        template_config = TEMPLATES["relative_position"][sub_category]
        question_type = template_config["question_type"]
        t = random.choice(template_config["templates"])
        ans, q_text = "", ""
        if sub_category == "relpos_binary":
            a, b = obj_actors
            vec = a['centroid_3d'] - b['centroid_3d']

            opposites = {
                'left': 'right', 'right': 'left',
                'above': 'below', 'below': 'above',
                'front': 'behind', 'behind': 'front'
            }
            
            true_relations = []
            if vec[0] > config.REL_POS_SIGNIFICANT_M: true_relations.append('left')
            if vec[0] < -config.REL_POS_SIGNIFICANT_M: true_relations.append('right')
            if vec[1] > config.REL_POS_SIGNIFICANT_M: true_relations.append('above')
            if vec[1] < -config.REL_POS_SIGNIFICANT_M: true_relations.append('below')
            if vec[2] > config.REL_POS_SIGNIFICANT_M: true_relations.append('front')
            if vec[2] < -config.REL_POS_SIGNIFICANT_M: true_relations.append('behind')

            if not true_relations:
                return False

            if random.random() < 0.5:
                direction_to_ask = random.choice(true_relations)
                ans = "Yes"
            else:
                true_direction = random.choice(true_relations)
                direction_to_ask = opposites[true_direction]
                ans = "No"
            
            q_text = f"In the real world, is object {{a}} to the {direction_to_ask} of object {{b}}?"
            question_refs, answer_refs = _smart_ref_question_answer([a, b])
            q_text = q_text.format(a=_ref_question(a, question_refs), b=_ref_question(b, question_refs))
        elif sub_category == "relpos_cardinal":
            a, b = obj_actors
            question_refs, answer_refs = _smart_ref_question_answer(obj_actors)
            vec = a['centroid_3d'] - b['centroid_3d']
            relationship = _vector_to_cardinal_direction(vec, config.REL_POS_SIGNIFICANT_M)
            a_ref = answer_refs[a['object_id']]
            b_ref = answer_refs[b['object_id']]
            ans = f"{_capitalize_smart(a_ref)} is {relationship} {b_ref}."
            q_text = t.format(a=_ref_question(a, question_refs), b=_ref_question(b, question_refs))
        elif sub_category == "relpos_distance_straightline":
            a, b = obj_actors
            question_refs, answer_refs = _smart_ref_question_answer(obj_actors)
            dist = np.linalg.norm(a['centroid_3d'] - b['centroid_3d'])
            ans = f"About {round(dist, 1)} meters"
            q_text = t.format(a=_ref_question(a, question_refs), b=_ref_question(b, question_refs))
        elif sub_category == "relpos_triplet_extreme":
            a,b,c = obj_actors
            dim, dim_name, op = random.choice([
                (0, "leftmost", np.argmax), (0, "rightmost", np.argmin),
                (1, "highest", np.argmax), (1, "lowest", np.argmin),
                (2, "frontmost", np.argmax), (2, "rearmost", np.argmin)
            ])
            coords = np.array([o['centroid_3d'][dim] for o in [a,b,c]])
            extreme_obj = [a,b,c][op(coords)]
            question_refs, answer_refs = _smart_ref_question_answer([a, b, c])
            q_text = f"Among {{a}}, {{b}}, and {{c}}, which one is the {dim_name}?"
            q_text = q_text.format(a=_ref_question(a, question_refs), b=_ref_question(b, question_refs), c=_ref_question(c, question_refs))
            extreme_ref = answer_refs[extreme_obj['object_id']]
            ans = _capitalize_smart(extreme_ref)
        elif sub_category == "relpos_distance_components":
            a, b = obj_actors
            question_refs, answer_refs = _smart_ref_question_answer(obj_actors)
            vec = np.abs(a['centroid_3d'] - b['centroid_3d'])
            templates = TEMPLATES["relative_position"]["relpos_distance_components"]["templates"]
            if random.random() > 0.5:
                dist = np.linalg.norm(vec[[0,2]])
                q_text = random.choice(templates[:2]).format(a=_ref_question(a, question_refs), b=_ref_question(b, question_refs))
                ans = f"About {round(dist, 1)} meters"
            else:
                dist = vec[1]
                q_text = templates[2].format(a=_ref_question(a, question_refs), b=_ref_question(b, question_refs))
                ans = f"About {round(dist, 1)} meters"
        final_qas.append({
            "major_category": "relative_position",
            "sub_category": sub_category,
            "question_type": question_type,
            "question": q_text,
            "answer": ans,
            "related_object_ids": [o["object_id"] for o in obj_actors],
        })
        for o in obj_actors: 
            used_ids.add(o["object_id"])
        return True

    target_subcategories = ["relpos_binary", "relpos_cardinal", "relpos_distance_straightline", "relpos_triplet_extreme", "relpos_distance_components"]
    
    for sub in target_subcategories:
        unused_objects = [o for o in valid_objects if o["object_id"] not in used_ids]
        
        if sub == "relpos_triplet_extreme":
            if len(unused_objects) >= 3:
                selected_objects = _select_objects_for_comparison(unused_objects, 3)
                if selected_objects:
                    add_qa(sub, selected_objects)
            elif len(valid_objects) >= 3:
                selected_objects = _select_objects_for_comparison(valid_objects, 3)
                if selected_objects:
                    add_qa(sub, selected_objects)
        else:
            if len(unused_objects) >= 2:
                selected_objects = _select_objects_for_comparison(unused_objects, 2)
                if selected_objects:
                    add_qa(sub, selected_objects)
            elif len(valid_objects) >= 2:
                selected_objects = _select_objects_for_comparison(valid_objects, 2)
                if selected_objects:
                    add_qa(sub, selected_objects)
    
    return final_qas[:per_cat]

# -------------------------
# -------------------------
def _get_3d_dims(obj):
    bbox_3d = obj.get("bbox_3d")
    if not bbox_3d: return None
    return [
        bbox_3d['max_x'] - bbox_3d['min_x'],
        bbox_3d['max_y'] - bbox_3d['min_y'],
        bbox_3d['max_z'] - bbox_3d['min_z']
    ]

def _get_flatness_score(dims):
    if not dims or any(d <= 0 for d in dims): return 1.0
    dims.sort()
    return dims[0] / dims[2]

def build_attribute_questions(objects, erp_shape, per_cat):
    final_qas = []
    valid_objects = [o for o in objects if o.get("volume") and o.get("bbox_area")]
    if len(valid_objects) < 2: return []
    used_ids = set()

    def add_qa(sub_category, obj_actors):
        template_config = TEMPLATES["attribute_comparison"][sub_category]
        question_type = template_config["question_type"]
        t = random.choice(template_config["templates"])
        ans, q_text = "", ""

        if sub_category in ["volume_comparison", "volume_binary"]:
            a, b = obj_actors
            question_refs, answer_refs = _smart_ref_question_answer(obj_actors)
            
            t = random.choice(TEMPLATES["attribute_comparison"][sub_category]["templates"])
            is_asking_smaller = "smaller" in t.lower() or "less" in t.lower()

            if sub_category == "volume_comparison":
                if is_asking_smaller:
                    smaller_obj = a if a['volume'] < b['volume'] else b
                    smaller_ref = answer_refs[smaller_obj['object_id']]
                    ans = _capitalize_smart(smaller_ref)
                else:
                    bigger_obj = a if a['volume'] > b['volume'] else b
                    bigger_ref = answer_refs[bigger_obj['object_id']]
                    ans = _capitalize_smart(bigger_ref)
            
            else: # volume_binary ("Is a larger/smaller than b?")
                if is_asking_smaller:
                    ans = "Yes" if a['volume'] < b['volume'] else "No"
                else:
                    ans = "Yes" if a['volume'] > b['volume'] else "No"

            q_text = t.format(a=_ref_question(a, question_refs), b=_ref_question(b, question_refs))

        elif sub_category in ["shape_flatness", "shape_elongation"]:
            a, b = obj_actors
            question_refs, answer_refs = _smart_ref_question_answer(obj_actors)
            dims_a, dims_b = _get_3d_dims(a), _get_3d_dims(b)
            if not dims_a or not dims_b: return False
            
            score_a = _get_flatness_score(dims_a)
            score_b = _get_flatness_score(dims_b)
            
            target_obj = a if score_a < score_b else b
            target_ref = answer_refs[target_obj['object_id']]
            shape_desc = "flatter" if "flatness" in sub_category else "more elongated"
            ans = _capitalize_smart(target_ref)
            q_text = t.format(a=_ref_question(a, question_refs), b=_ref_question(b, question_refs))

        elif sub_category == "size_triplet_extreme":
            a, b, c = obj_actors
            question_refs, answer_refs = _smart_ref_question_answer(obj_actors)
            if "smallest" in t.lower():
                extreme_obj = min([a, b, c], key=lambda o: o['volume'])
                extreme_ref = answer_refs[extreme_obj['object_id']]
                ans = _capitalize_smart(extreme_ref)
            else: # largest
                extreme_obj = max([a, b, c], key=lambda o: o['volume'])
                extreme_ref = answer_refs[extreme_obj['object_id']]
                ans = _capitalize_smart(extreme_ref)
            q_text = t.format(a=_ref_question(a, question_refs), b=_ref_question(b, question_refs), c=_ref_question(c, question_refs))

        final_qas.append({
            "major_category": "attribute_comparison", "sub_category": sub_category,
            "question_type": question_type,
            "question": q_text, "answer": ans,
            "related_object_ids": [o["object_id"] for o in obj_actors],
        })
        for o in obj_actors: 
            used_ids.add(o["object_id"])
        return True

    target_subcategories = ["volume_comparison", "volume_binary", "shape_flatness", "shape_elongation", "size_triplet_extreme"]
    
    for sub in target_subcategories:
        unused_objects = [o for o in valid_objects if o["object_id"] not in used_ids]
        
        if sub == "size_triplet_extreme":
            if len(unused_objects) >= 3:
                selected_objects = _select_objects_for_comparison(unused_objects, 3)
                if selected_objects:
                    add_qa(sub, selected_objects)
            elif len(valid_objects) >= 3:
                selected_objects = _select_objects_for_comparison(valid_objects, 3)
                if selected_objects:
                    add_qa(sub, selected_objects)
        else:
            if len(unused_objects) >= 2:
                selected_objects = _select_objects_for_comparison(unused_objects, 2)
                if selected_objects:
                    add_qa(sub, selected_objects)
            elif len(valid_objects) >= 2:
                selected_objects = _select_objects_for_comparison(valid_objects, 2)
                if selected_objects:
                    add_qa(sub, selected_objects)
            
    return final_qas[:per_cat]

# -------------------------
# -------------------------
def generate_questions_for_frame(data_package):
    raw_objs = data_package["sampled_objects"]
    depth_map = data_package["depth_map"]
    erp_shape = data_package["erp_shape"]
    environment = data_package.get("environment", None)
    
    objs = _filter_unsuitable_objects(raw_objs)
    print(f"   [Filter] Raw objects: {len(raw_objs)} -> QA-eligible: {len(objs)}")
    
    num_objects = len(objs)
    if num_objects < config.MIN_OBJECTS_FOR_QA_GENERATION:
        print(f"   [Reject] Not enough objects ({num_objects}<{config.MIN_OBJECTS_FOR_QA_GENERATION}), skip this sample")
        return []
    
    questions_per_category = 5
    print(f"   [Info] Objects available: {num_objects}, generating {questions_per_category} questions per category (target {questions_per_category * 5})")

    view_qs = build_view_questions(objs, erp_shape, questions_per_category)
    dist_qs = build_distance_questions(objs, erp_shape, questions_per_category)
    env_qs = build_environment_questions(objs, erp_shape, questions_per_category, environment)
    relpos_qs = build_relative_position_questions(objs, erp_shape, questions_per_category, depth_map)
    attr_qs = build_attribute_questions(objs, erp_shape, questions_per_category)
    
    picked = view_qs + dist_qs + env_qs + relpos_qs + attr_qs

    category_counts = {
        "view_source": len(view_qs), "distance": len(dist_qs), "environment": len(env_qs),
        "relative_position": len(relpos_qs), "attribute_comparison": len(attr_qs)
    }
    total_questions = len(picked)
    
    failed_categories = [cat for cat, count in category_counts.items() if count < questions_per_category]
    if failed_categories:
        print(f"   [Warning] Insufficient questions for categories: {failed_categories}")
        print(f"   [Stats] Produced counts: {category_counts}, target per category: {questions_per_category}")
        
        min_total_questions = 10
        if total_questions < min_total_questions:
            print(f"   [Reject] Total questions too low ({total_questions}<{min_total_questions}), skip this sample")
            return []
        else:
            print(f"   [Accept] Some categories missing, but total questions ({total_questions}) meet the minimum")
    else:
        print(f"   [Success] Generated questions across all categories: {category_counts}, total: {total_questions}")
    
    print(f"   [Strategy] Balanced plan: {questions_per_category} subcategories per class, avoid reusing objects within a class")

    random.shuffle(picked)
    for i, q in enumerate(picked):
        q["question_id"] = i + 1
    return picked

# -------------------------
# -------------------------

def normalize_text(text):
    import re
    if not text:
        return ""
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    text = re.sub(r'\s+', ' ', text.strip())
    return text

def extract_all_choices_from_question(question_text):
    import re
    
    # "which environment is it: Hospital, School, Park, Office?"
    # "which is correct: A, B, C?"
    
    patterns = [
        r':\s*([^?]+)\?',
        r'\(([^)]+)\)',
        r'between\s+(.+?)\s+and\s+(.+?)(?:\s*[,?]|$)',
        r'(\w+)\s+or\s+(\w+)',
    ]
    
    choices = []
    
    for pattern in patterns:
        matches = re.findall(pattern, question_text, re.IGNORECASE)
        if matches:
            for match in matches:
                if isinstance(match, tuple):
                    for item in match:
                        if item.strip():
                            choices.extend([c.strip() for c in item.split(',') if c.strip()])
                else:
                    if ',' in match:
                        choices.extend([c.strip() for c in match.split(',') if c.strip()])
                    elif '/' in match:
                        choices.extend([c.strip() for c in match.split('/') if c.strip()])
                    else:
                        choices.append(match.strip())
    
    cleaned_choices = []
    for choice in choices:
        choice = re.sub(r'[^\w\s]', '', choice).strip()
        if len(choice) >= 3 and not choice.lower() in ['the', 'and', 'or', 'is', 'it', 'are']:
            cleaned_choices.append(choice)
    
    seen = set()
    unique_choices = []
    for choice in cleaned_choices:
        if choice not in seen:
            seen.add(choice)
            unique_choices.append(choice)
    
    return unique_choices

def extract_yes_no_simple(text):
    normalized = normalize_text(text)
    
    yes_keywords = [
        'yes', 'yeah', 'yep', 'yup', 'true', 'correct', 'right',
        'absolutely', 'definitely', 'certainly', 'indeed', 'agree',
        'positive', 'affirm', 'confirm', 'it is'
    ]
    
    no_keywords = [
        'no', 'nope', 'nah', 'false', 'incorrect', 'wrong',
        'negative', 'disagree', 'deny', 'refuse', 'it is not'
    ]
    
    negative_patterns = [
        'definitely not', 'absolutely not', 'certainly not',
        'not at all', 'not really', 'not correct', 'not true',
        'not right', 'it is not'
    ]
    
    for pattern in negative_patterns:
        if pattern in normalized:
            return 'no'
    
    for keyword in no_keywords:
        if keyword in normalized:
            return 'no'
    
    for keyword in yes_keywords:
        if keyword in normalized:
            return 'yes'
            
    return None

def extract_yes_no(model_output):
    return extract_yes_no_simple(model_output)

def extract_choice_answer(model_output, choices):
    normalized_output = normalize_text(model_output)
    normalized_choices = [normalize_text(choice) for choice in choices]
    
    for i, choice in enumerate(normalized_choices):
        if choice in normalized_output:
            return choices[i]
    
    import re
    for i, choice in enumerate(choices):
        keywords = re.findall(r'\b(?!the|a|an|and|or|in|on|at|to|for|of|with|by)\w{3,}\b', 
                             normalize_text(choice))
        
        matches = sum(1 for keyword in keywords if keyword in normalized_output)
        if matches > 0 and matches >= len(keywords) * 0.5:
            return choices[i]
    
    option_labels = ['a', 'b', 'c', 'd', 'e', 'f']
    for i, label in enumerate(option_labels[:len(choices)]):
        patterns = [
            f'\\b{label}\\b',
            f'\\({label}\\)',
            f'{label}\\.',
            f'option\\s+{label}',
            f'choice\\s+{label}',
        ]
        
        for pattern in patterns:
            if re.search(pattern, normalized_output):
                return choices[i]
    
    return None

def extract_judgment_keywords(answer, question_category, sub_category=None):
    import re
    
    if question_category == "distance":
        numbers = re.findall(r'\d+\.?\d*', answer)
        return {'type': 'numbers', 'keywords': numbers}
    
    elif question_category == "relative_position":
        position_keywords = [
            'left', 'right', 'above', 'below', 'front', 'behind',
            'up', 'down', 'top', 'bottom', 'forward', 'backward',
            'leftmost', 'rightmost', 'highest', 'lowest', 
            'frontmost', 'rearmost', 'closer', 'farther'
        ]
        
        normalized = normalize_text(answer)
        found_keywords = [kw for kw in position_keywords if kw in normalized]
        return {'type': 'positions', 'keywords': found_keywords}
    
    else:
        return {'type': 'general', 'keywords': []}

def get_position_synonyms():
    return {
        'left': ['left', 'leftward', 'leftmost'],
        'right': ['right', 'rightward', 'rightmost'],
        
        'above': ['above', 'up', 'upward', 'over', 'top', 'upper', 'highest'],
        'below': ['below', 'down', 'downward', 'under', 'bottom', 'lower', 'lowest'],
        'up': ['above', 'up', 'upward', 'over', 'top', 'upper', 'highest'],
        'down': ['below', 'down', 'downward', 'under', 'bottom', 'lower', 'lowest'],
        'top': ['above', 'up', 'upward', 'over', 'top', 'upper', 'highest'],
        'bottom': ['below', 'down', 'downward', 'under', 'bottom', 'lower', 'lowest'],
        'highest': ['above', 'up', 'upward', 'over', 'top', 'upper', 'highest'],
        'lowest': ['below', 'down', 'downward', 'under', 'bottom', 'lower', 'lowest'],
        
        'front': ['front', 'forward', 'ahead', 'frontmost'],
        'behind': ['behind', 'back', 'backward', 'rear', 'rearmost'],
        'forward': ['front', 'forward', 'ahead', 'frontmost'],
        'backward': ['behind', 'back', 'backward', 'rear', 'rearmost'],
        'frontmost': ['front', 'forward', 'ahead', 'frontmost'],
        'rearmost': ['behind', 'back', 'backward', 'rear', 'rearmost'],
        
        'closer': ['closer', 'nearer', 'near'],
        'farther': ['farther', 'further', 'distant', 'far'],
        'farthest': ['farthest', 'furthest', 'most distant']
    }

def normalize_position_words(position_words):
    synonyms = get_position_synonyms()
    base_words = set()
    
    for word in position_words:
        base_word = None
        for base, synonym_list in synonyms.items():
            if word in synonym_list:
                base_word = base
                break
        
        if base_word:
            base_words.add(base_word)
        else:
            base_words.add(word)
    
    return base_words
