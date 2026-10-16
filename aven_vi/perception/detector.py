from typing import List, Dict, Any, Optional
from ultralytics import YOLO
from aven_vi.config.config_loader import resolve_target_class_ids

class ObstacleDetector:
    """
    YOLOv8-based generic obstacle detector for AVEN-VI.
    Supports pedestrian-relevant COCO classes via external configuration.
    """
    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.35, target_class_names: Optional[List[str]] = None):
        print(f"[INFO] Initializing ObstacleDetector ({model_name})...")
        self.model = YOLO(model_name)
        self.confidence = float(confidence)
        
        # Resolve target classes dynamically from model vocabulary
        if target_class_names is not None:
            self.target_class_ids = resolve_target_class_ids(target_class_names, self.model.names)
        else:
            self.target_class_ids = list(self.model.names.keys())
            
        print(f"[INFO] Target Classes ({len(self.target_class_ids)}): {[self.model.names[i] for i in self.target_class_ids]}")

    def get_class_name(self, class_id: int) -> str:
        """Returns the human-readable class name for a given class ID."""
        return self.model.names.get(class_id, f"class_{class_id}")
