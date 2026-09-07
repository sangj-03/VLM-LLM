import math

import torch
from ultralytics import YOLO


def bbox_iou(a, b):
    if a is None or b is None:
        return 0.0

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


class DriverSelector:
    def __init__(
        self,
        model_path="yolo26n.pt",
        tracker="bytetrack.yaml",
        anchor_x=1260,
        anchor_y=550,
        fps=38.0,
    ):
        self.model = YOLO(model_path)

        self.tracker = tracker
        self.anchor = (anchor_x, anchor_y)

        self.device = 0 if torch.cuda.is_available() else "cpu"

        self.prev_box = None
        self.prev_id = None

        self.missed = 0

        # 최대 약 0.5초 동안 마지막 bbox 유지
        self.max_hold = max(1, int(round(fps * 0.5)))

    def choose_driver(self, candidates, frame_shape):
        if not candidates:
            return None

        h, w = frame_shape[:2]

        ax, ay = self.anchor

        # 이 영상에서는 운전자가 오른쪽
        min_driver_x = 0.56 * w

        best = None
        best_score = -1e9

        for c in candidates:
            x1, y1, x2, y2 = c["box"]

            bw = x2 - x1

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            # 너무 왼쪽이면 조수석 후보
            if cx < min_driver_x:
                continue

            # 두 사람이 합쳐진 거대한 bbox 제거
            if bw > 0.55 * w:
                continue

            dx = (cx - ax) / w
            dy = (cy - ay) / h

            seat_dist = math.sqrt(dx * dx + dy * dy)

            overlap = bbox_iou(c["box"], self.prev_box)

            same_id = (
                self.prev_id is not None
                and c["id"] is not None
                and c["id"] == self.prev_id
            )

            score = 0.0

            # 운전석 위치
            score -= 5.0 * seat_dist

            # 이전 운전자 bbox와 연속성
            score += 2.0 * overlap

            # 동일 tracker ID
            if same_id:
                score += 1.0

            # 오른쪽 사람을 약간 선호
            score += 0.5 * (cx / w)

            if score > best_score:
                best_score = score
                best = c

        return best

    def update(self, frame):
        results = self.model.track(
            frame,
            persist=True,
            classes=[0],       # COCO person
            conf=0.25,
            iou=0.5,
            tracker=self.tracker,
            verbose=False,
            device=self.device,
        )

        candidates = []

        boxes = results[0].boxes

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()

            if boxes.id is not None:
                ids = boxes.id.int().cpu().tolist()
            else:
                ids = [None] * len(xyxy)

            confs = boxes.conf.cpu().tolist()

            for box, track_id, conf in zip(xyxy, ids, confs):
                candidates.append(
                    {
                        "box": tuple(box.tolist()),
                        "id": track_id,
                        "conf": float(conf),
                    }
                )

        driver = self.choose_driver(
            candidates,
            frame.shape,
        )

        if driver is not None:
            self.prev_box = driver["box"]

            if driver["id"] is not None:
                self.prev_id = driver["id"]

            self.missed = 0

            driver["held"] = False
            return driver

        # 잠깐 detection이 끊겨도 마지막 driver bbox를 유지
        self.missed += 1

        if (
            self.prev_box is not None
            and self.missed <= self.max_hold
        ):
            return {
                "box": self.prev_box,
                "id": self.prev_id,
                "conf": 0.0,
                "held": True,
            }

        return None


def expand_driver_box(box, frame_shape):
    """
    MediaPipe가 머리/어깨/손을 볼 수 있도록
    YOLO bbox에 약간의 여백을 추가.
    """
    h, w = frame_shape[:2]

    x1, y1, x2, y2 = box

    bw = x2 - x1
    bh = y2 - y1

    # passenger가 다시 crop에 많이 들어오는 것을 피하기 위해
    # 너무 크게 확장하지 않는다.
    x1 -= 0.02 * bw
    x2 += 0.10 * bw

    y1 -= 0.20 * bh
    y2 += 0.10 * bh

    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))
    x2 = min(w, int(round(x2)))
    y2 = min(h, int(round(y2)))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2
