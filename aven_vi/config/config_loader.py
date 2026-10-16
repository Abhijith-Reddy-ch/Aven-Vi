import os
import yaml
from typing import Dict, Any, List

DEFAULT_CONFIG: Dict[str, Any] = {
    "detection": {
        "model": "yolov8n.pt",
        "confidence": 0.35,
        "classes": [
            "person", "bicycle", "car", "motorcycle", "bus", "truck",
            "dog", "chair", "bench", "traffic light", "stop sign"
        ]
    },
    "motion": {
        "ema_alpha": 0.20,
        "max_history": 30,
        "max_missing_frames": 15
    },
    "visualization": {
        "draw_trajectory": True,
        "draw_velocity": True,
        "draw_centroid": True,
        "trajectory_color": [0, 255, 255],
        "box_thickness": 2,
        "font_scale": 0.50
    }
}

def load_config(config_path: str = "config/av_config.yaml") -> Dict[str, Any]:
    """
    Loads YAML configuration for AVEN-VI perception pipeline.
    Falls back to DEFAULT_CONFIG if file not found or invalid.
    """
    if not os.path.exists(config_path):
        print(f"[WARN] Config file '{config_path}' not found. Using default config.")
        return DEFAULT_CONFIG.copy()

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[ERROR] Failed to parse config file '{config_path}': {e}. Using default config.")
        return DEFAULT_CONFIG.copy()

    # Merge user config on top of defaults
    merged = DEFAULT_CONFIG.copy()
    for section, sec_vals in user_config.items():
        if isinstance(sec_vals, dict) and section in merged:
            merged[section].update(sec_vals)
        else:
            merged[section] = sec_vals

    return merged

def resolve_target_class_ids(configured_class_names: List[str], model_names: Dict[int, str]) -> List[int]:
    """
    Dynamically maps configured human-readable class names to model class IDs.
    Eliminates hard-coded integer class lists.
    """
    # Create case-insensitive lookup
    name_to_id = {v.lower().strip(): k for k, v in model_names.items()}
    resolved_ids = []
    for name in configured_class_names:
        norm_name = name.lower().strip()
        if norm_name in name_to_id:
            resolved_ids.append(name_to_id[norm_name])
        else:
            print(f"[WARN] Configured class '{name}' not found in model classes.")
    return sorted(list(set(resolved_ids)))
