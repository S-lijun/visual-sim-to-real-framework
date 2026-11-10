"""YOLO detections, matching the existing notebook's model and threshold.

Ultralytics treats numpy images as BGR, while these tensors are RGB. Frames
are channel-flipped before ``predict`` so the network sees the same RGB order
as the notebook path, which reads files with OpenCV.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import ExtractConfig


class YoloEncoder:
    def __init__(self, config: ExtractConfig, device: str):
        from ultralytics import YOLO

        self.model = YOLO(str(config.yolo_weights))
        self.conf = config.yolo_conf
        self.batch = config.yolo_batch
        self.device = device
        self.names = {int(k): str(v) for k, v in self.model.names.items()}

    def detect(self, rgb: np.ndarray) -> dict[str, np.ndarray]:
        height, width = rgb.shape[1:3]
        counts = np.zeros(rgb.shape[0], dtype=np.int16)
        frame_ids: list[int] = []
        confs: list[float] = []
        cls_ids: list[int] = []
        boxes: list[list[float]] = []
        norm_wh: list[list[float]] = []
        norm_area: list[float] = []
        norm_center: list[list[float]] = []

        for start in range(0, rgb.shape[0], self.batch):
            chunk = rgb[start : start + self.batch]
            bgr = np.ascontiguousarray(chunk[..., ::-1])
            results = self.model.predict(
                source=[frame for frame in bgr],
                conf=self.conf,
                verbose=False,
                device=self.device,
                batch=self.batch,
            )
            for offset, result in enumerate(results):
                frame_index = start + offset
                found = result.boxes
                if found is None or len(found) == 0:
                    counts[frame_index] = 0
                    continue
                xyxy = found.xyxy.detach().cpu().numpy().astype(np.float32)
                conf = found.conf.detach().cpu().numpy().astype(np.float32)
                cls = found.cls.detach().cpu().numpy().astype(np.int16)
                counts[frame_index] = len(conf)
                for i in range(len(conf)):
                    x1, y1, x2, y2 = (float(v) for v in xyxy[i])
                    box_w = max(0.0, x2 - x1) / width
                    box_h = max(0.0, y2 - y1) / height
                    frame_ids.append(frame_index)
                    confs.append(float(conf[i]))
                    cls_ids.append(int(cls[i]))
                    boxes.append([x1, y1, x2, y2])
                    norm_wh.append([box_w, box_h])
                    norm_area.append(box_w * box_h)
                    norm_center.append([((x1 + x2) * 0.5) / width, ((y1 + y2) * 0.5) / height])

        return {
            "count": counts,
            "det_frame": np.asarray(frame_ids, dtype=np.int32),
            "det_conf": np.asarray(confs, dtype=np.float32),
            "det_cls": np.asarray(cls_ids, dtype=np.int16),
            "det_xyxy": np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            "det_norm_wh": np.asarray(norm_wh, dtype=np.float32).reshape(-1, 2),
            "det_norm_area": np.asarray(norm_area, dtype=np.float32),
            "det_norm_center": np.asarray(norm_center, dtype=np.float32).reshape(-1, 2),
        }


def class_table(names: dict[int, str], path: Path) -> None:
    path.write_text(
        "\n".join(f"{idx}\t{name}" for idx, name in sorted(names.items())),
        encoding="utf-8",
    )
