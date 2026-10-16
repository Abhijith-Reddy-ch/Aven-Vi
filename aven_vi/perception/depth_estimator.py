import cv2
import torch
import numpy as np
from typing import Optional, Tuple

class MonocularDepthEstimator:
    """
    Intel MiDaS Small monocular depth estimation wrapper.
    Generates dense disparity maps and extracts regional median disparity for tracked objects.
    """
    def __init__(self, model_type: str = "MiDaS_small", device: Optional[str] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[INFO] Initializing DepthEstimator ({model_type}) on {self.device}...")
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True).to(self.device)
        self.midas.eval()

        self.transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).small_transform
        self.latest_disparity_map: Optional[np.ndarray] = None

    def estimate_disparity(self, frame: np.ndarray) -> np.ndarray:
        """
        Computes dense 2D inverse depth (disparity map) matching frame dimensions.
        """
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transforms(img_rgb).to(self.device)

        with torch.no_grad():
            pred = self.midas(input_batch)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False
            ).squeeze()

        self.latest_disparity_map = pred.cpu().numpy()
        return self.latest_disparity_map

    def sample_roi_disparity(self, bbox: Tuple[int, int, int, int]) -> Optional[float]:
        """
        Samples the median disparity value within a detected object bounding box.
        """
        if self.latest_disparity_map is None:
            return None

        h, w = self.latest_disparity_map.shape[:2]
        x1, y1, x2, y2 = bbox
        x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
        y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))

        if x2 <= x1 or y2 <= y1:
            return None

        region = self.latest_disparity_map[y1:y2, x1:x2]
        if region.size == 0:
            return None

        return float(np.median(region))
