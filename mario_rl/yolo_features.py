from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


@dataclass
class DetectionSummary:
    object_count: float = 0.0
    nearest_x: float = -1.0
    nearest_y: float = -1.0
    largest_area: float = 0.0
    mean_x: float = -1.0
    mean_y: float = -1.0

    def to_vector(self, frame_width: int, frame_height: int) -> np.ndarray:
        fw = max(float(frame_width), 1.0)
        fh = max(float(frame_height), 1.0)
        return np.array([
            self.object_count / 20.0,
            self.nearest_x / fw if self.nearest_x >= 0 else -1.0,
            self.nearest_y / fh if self.nearest_y >= 0 else -1.0,
            self.largest_area / (fw * fh),
            self.mean_x / fw if self.mean_x >= 0 else -1.0,
            self.mean_y / fh if self.mean_y >= 0 else -1.0,
        ], dtype=np.float32)


class MarioYOLOFeatureExtractor:
    def __init__(self, weights: str, conf: float = 0.25, device: str = 'cpu'):
        if YOLO is None:
            raise ImportError('ultralytics is not installed correctly.')
        self.model = YOLO(weights)
        self.conf = conf
        self.device = device

    def detect(self, frame: np.ndarray) -> DetectionSummary:
        results = self.model.predict(source=frame, conf=self.conf, device=self.device, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return DetectionSummary()

        xyxy = boxes.xyxy.detach().cpu().numpy()
        object_count = float(len(xyxy))

        centers_x = []
        centers_y = []
        nearest_x = -1.0
        nearest_y = -1.0
        largest_area = 0.0

        for box in xyxy:
            x1, y1, x2, y2 = box[:4]
            cx = float((x1 + x2) / 2.0)
            cy = float((y1 + y2) / 2.0)
            area = float(max(0.0, x2 - x1) * max(0.0, y2 - y1))
            centers_x.append(cx)
            centers_y.append(cy)
            if nearest_x < 0 or cx < nearest_x:
                nearest_x = cx
                nearest_y = cy
            if area > largest_area:
                largest_area = area

        return DetectionSummary(
            object_count=object_count,
            nearest_x=nearest_x,
            nearest_y=nearest_y,
            largest_area=largest_area,
            mean_x=float(np.mean(centers_x)) if centers_x else -1.0,
            mean_y=float(np.mean(centers_y)) if centers_y else -1.0,
        )
