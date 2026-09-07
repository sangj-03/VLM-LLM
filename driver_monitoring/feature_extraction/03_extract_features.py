import argparse
from pathlib import Path
import math
import time

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

from common import FEATURE_NAMES, load_config, ensure_dir, clip_finite
from driver_selector import DriverSelector, expand_driver_box

# MediaPipe Face Mesh landmark indices used by the common EAR formula.
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12

HEAD_IDS = {
    "nose": 1,
    "chin": 152,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "left_mouth": 61,
    "right_mouth": 291,
}

MODEL_POINTS = np.array([
    [0.0, 0.0, 0.0],          # nose
    [0.0, -63.6, -12.5],      # chin
    [-43.3, 32.7, -26.0],     # eye outer
    [43.3, 32.7, -26.0],
    [-28.9, -28.9, -24.1],    # mouth corner
    [28.9, -28.9, -24.1],
], dtype=np.float64)


def pxy(lm):
    return np.array([float(lm.x), float(lm.y)], dtype=np.float64)


def dist(a, b):
    return float(np.linalg.norm(a-b))


def angle_delta(curr, prev):
    """Return the shortest signed angular difference in degrees."""
    return (curr - prev + 180.0) % 360.0 - 180.0


def ear(landmarks, ids):
    p1,p2,p3,p4,p5,p6 = [pxy(landmarks[i]) for i in ids]
    denom = 2.0 * dist(p1,p4)
    if denom < 1e-8:
        return np.nan
    return (dist(p2,p6) + dist(p3,p5)) / denom


def in_frame_ratio(landmarks, margin=0.05):
    if not landmarks:
        return 0.0
    ok = 0
    for lm in landmarks:
        if -margin <= lm.x <= 1+margin and -margin <= lm.y <= 1+margin:
            ok += 1
    return ok / len(landmarks)


def visibility_value(lm):
    # Legacy MediaPipe Holistic pose landmarks provide visibility.
    # Do not use protobuf presence=0 as a hard failure.
    if getattr(lm, "visibility", None) is not None:
        return float(lm.visibility)
    return 1.0


def pose_reliability(pose):
    # Neck proxy uses:
    # nose = 0
    # left shoulder = 11
    # right shoulder = 12
    if not pose or len(pose) <= 12:
        return 0.0

    ids = [0, 11, 12]

    vis = np.mean([
        visibility_value(pose[i])
        for i in ids
    ])

    frame = np.mean([
        1.0
        if -0.05 <= pose[i].x <= 1.05
        and -0.05 <= pose[i].y <= 1.05
        else 0.0
        for i in ids
    ])

    return float(np.clip(vis * frame, 0, 1))


def hand_rel(hand):
    if not hand or len(hand) < 21:
        return 0.0
    return float(np.clip(in_frame_ratio(hand), 0, 1))


def mean_motion(curr, prev, ids=None, dt=0.1):
    if not curr or not prev:
        return 0.0
    if ids is None:
        n = min(len(curr), len(prev))
        ids = range(n)
    ds = []
    for i in ids:
        if i < len(curr) and i < len(prev):
            ds.append(dist(pxy(curr[i]), pxy(prev[i])))
    return float(np.mean(ds)/dt) if ds else 0.0


def head_pose(face, width, height):
    if not face or max(HEAD_IDS.values()) >= len(face):
        return None
    idx = [HEAD_IDS[k] for k in ["nose","chin","left_eye_outer","right_eye_outer","left_mouth","right_mouth"]]
    image_points = np.array([[face[i].x*width, face[i].y*height] for i in idx], dtype=np.float64)
    focal = float(width)
    camera = np.array([[focal,0,width/2],[0,focal,height/2],[0,0,1]], dtype=np.float64)
    dist_coeffs = np.zeros((4,1), dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(MODEL_POINTS, image_points, camera, dist_coeffs,
                               flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    rmat, _ = cv2.Rodrigues(rvec)
    angles, *_ = cv2.RQDecomp3x3(rmat)
    pitch, yaw, roll = [float(a) for a in angles]
    return pitch, yaw, roll


def neck_values(pose):
    """
    Visual head-to-shoulder movement proxies.

    Uses MediaPipe Pose:
      nose           = 0
      left shoulder  = 11
      right shoulder = 12

    Values are normalized by shoulder width, so they are less
    sensitive to changes in crop size.

    These are NOT anatomical cervical joint angles.
    """

    if not pose or len(pose) <= 12:
        return None

    nose = pxy(pose[0])
    ls = pxy(pose[11])
    rs = pxy(pose[12])

    # Reject unreliable/out-of-frame nose.
    if visibility_value(pose[0]) < 0.2:
        return None

    if not (
        -0.10 <= pose[0].x <= 1.10
        and -0.10 <= pose[0].y <= 1.10
    ):
        return None

    shoulder_mid = (ls + rs) / 2.0
    shoulder_width = dist(ls, rs)

    if shoulder_width < 1e-6:
        return None

    rel = (nose - shoulder_mid) / shoulder_width

    # x: left/right displacement relative to shoulders
    neck_lateral_proxy = float(rel[0])

    # y grows downward in image coordinates.
    # Larger value means the head moved downward toward shoulders.
    neck_flexion_proxy = float(rel[1])

    return (
        neck_flexion_proxy,
        neck_lateral_proxy,
        rel.astype(np.float64),
    )


def main():
    p = argparse.ArgumentParser(description="영상 -> 10 FPS 22-D driver head-neck feature CSV")
    p.add_argument("--video", required=True)
    p.add_argument("--model", required=True, help="holistic_landmarker.task")
    p.add_argument("--config", default="config.json")
    p.add_argument("--out", default=None)
    p.add_argument("--preview", action="store_true")
    p.add_argument("--max-sec", type=float, default=None, help="처리할 길이(초)")
    p.add_argument("--start-sec", type=float, default=0.0, help="처리 시작 시간(초)")
    p.add_argument("--yolo-model", default="yolo26n.pt")
    p.add_argument("--tracker", default="bytetrack.yaml")
    p.add_argument("--anchor-x", type=int, default=1260)
    p.add_argument("--anchor-y", type=int, default=550)
    args = p.parse_args()

    cfg = load_config(args.config)
    target_fps = float(cfg["target_fps"])
    dt = 1.0/target_fps
    r = cfg["roi"]

    video_id = Path(args.video).stem
    out = args.out or f"data/features/{video_id}.csv"
    ensure_dir(Path(out).parent)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"영상 열기 실패: {args.video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS)

    driver_selector = DriverSelector(
        model_path=args.yolo_model,
        tracker=args.tracker,
        anchor_x=args.anchor_x,
        anchor_y=args.anchor_y,
        fps=src_fps,
    )
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = n_frames/src_fps if src_fps > 0 else 0

    start_sec = max(0.0, float(args.start_sec))
    end_sec = video_duration
    if args.max_sec is not None:
        end_sec = min(video_duration, start_sec + float(args.max_sec))

    if start_sec >= video_duration:
        raise SystemExit(f"start-sec가 영상 길이를 넘음: {start_sec:.3f} >= {video_duration:.3f}")

    cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)
    duration = end_sec - start_sec

    # DGX Spark / MediaPipe 0.10.18:
    # Tasks HolisticLandmarker can abort on an empty output packet.
    # Use legacy Holistic while keeping the same 10 Hz sampling pipeline.
    mp_holistic = mp.solutions.holistic

    prev_face = prev_left = prev_right = None
    prev_angles = None
    prev_neck_rel = None
    rows = []
    latencies = []

    n_steps = int(math.floor(duration * target_fps))
    progress = tqdm(total=n_steps, desc=video_id)
    next_sample_t = start_sec
    frame_idx = int(round(start_sec * src_fps))
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_t = frame_idx / src_fps
            frame_idx += 1

            # Track the driver on EVERY source frame.
            # MediaPipe processing below still runs only at target_fps.
            driver_info = driver_selector.update(frame)
            if frame_t + 1e-9 < next_sample_t:
                continue
            if frame_t > end_sec + 1e-9:
                break
            # 가장 가까운 원본 프레임을 10 Hz timeline에 대응시킴.
            t = next_sample_t
            next_sample_t += dt
            progress.update(1)
            H, W = frame.shape[:2]

            if driver_info is not None:
                driver_box = expand_driver_box(
                    driver_info["box"],
                    frame.shape
                )
            else:
                driver_box = None

            if driver_box is not None:
                x1, y1, x2, y2 = driver_box
                crop = frame[y1:y2, x1:x2]
                driver_found = 1
                driver_id = driver_info["id"]
            else:
                # No trustworthy driver candidate:
                # never substitute the passenger.
                crop = np.zeros((480, 480, 3), dtype=np.uint8)
                driver_found = 0
                driver_id = None
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            ts_ms = int(round(t*1000))
            tic = time.perf_counter()
            result = landmarker.process(rgb)
            latencies.append((time.perf_counter()-tic)*1000.0)

            face = result.face_landmarks.landmark if result.face_landmarks else None
            pose = result.pose_landmarks.landmark if result.pose_landmarks else None
            left = result.left_hand_landmarks.landmark if result.left_hand_landmarks else None
            right = result.right_hand_landmarks.landmark if result.right_hand_landmarks else None

            face_valid = int(face is not None and len(face) > 291)
            eye_valid = int(face_valid and max(LEFT_EYE+RIGHT_EYE) < len(face))
            face_rel = in_frame_ratio(face) if face_valid else 0.0
            pose_rel = pose_reliability(pose)
            lrel = hand_rel(left); rrel = hand_rel(right)
            hand_reliability = (lrel+rrel)/2.0

            if eye_valid:
                ear_l = ear(face, LEFT_EYE); ear_r = ear(face, RIGHT_EYE)
                ear_m = np.nanmean([ear_l, ear_r])
            else:
                ear_l = ear_r = ear_m = 0.0

            angles = head_pose(face, crop.shape[1], crop.shape[0]) if face_valid else None
            if angles is None:
                pitch=yaw=roll=0.0
                pvel=yvel=rvel=0.0
                prev_angles = None
            else:
                pitch,yaw,roll = angles
                if prev_angles is None:
                    pvel=yvel=rvel=0.0
                else:
                    pvel = angle_delta(pitch, prev_angles[0]) / dt
                    yvel = angle_delta(yaw, prev_angles[1]) / dt
                    rvel = angle_delta(roll, prev_angles[2]) / dt
                prev_angles = angles
            pvel = clip_finite(pvel,-720,720); yvel=clip_finite(yvel,-720,720); rvel=clip_finite(rvel,-720,720)

            head_motion = mean_motion(face, prev_face, ids=[1,33,263,61,291], dt=dt) if face_valid and prev_face else 0.0

            nv = neck_values(pose) if pose_rel > 0 else None

            if nv is None:
                neck_flexion_proxy = 0.0
                neck_lateral_proxy = 0.0
                neck_velocity = 0.0
                prev_neck_rel = None
            else:
                (
                    neck_flexion_proxy,
                    neck_lateral_proxy,
                    neck_rel,
                ) = nv

                if prev_neck_rel is None:
                    neck_velocity = 0.0
                else:
                    neck_velocity = float(
                        np.linalg.norm(
                            neck_rel - prev_neck_rel
                        ) / dt
                    )

                prev_neck_rel = neck_rel

            left_valid = int(left is not None and len(left)>=21)
            right_valid = int(right is not None and len(right)>=21)
            left_motion = mean_motion(left, prev_left, dt=dt) if left_valid and prev_left else 0.0
            right_motion = mean_motion(right, prev_right, dt=dt) if right_valid and prev_right else 0.0

            values = [
                ear_l, ear_r, ear_m, eye_valid,
                pitch, yaw, roll,
                pvel, yvel, rvel,
                head_motion,
                neck_flexion_proxy,
                neck_lateral_proxy,
                neck_velocity,
                left_motion, right_motion,
                left_valid, right_valid,
                face_valid,
                face_rel, pose_rel, hand_reliability,
            ]
            row = {"video_id":video_id,"timestamp":round(t,3)}
            row.update({k:clip_finite(v) for k,v in zip(FEATURE_NAMES, values)})
            rows.append(row)

            prev_face = face if face_valid else None
            prev_left = left if left_valid else None
            prev_right = right if right_valid else None

            if args.preview:
                ch,cw=crop.shape[:2]
                if face_valid:
                    for i in sorted(set(LEFT_EYE+RIGHT_EYE+[1,152,61,291])):
                        if i < len(face):
                            cv2.circle(crop,(int(face[i].x*cw),int(face[i].y*ch)),2,(0,255,0),-1)
                if pose:
                    for i in [0,11,12,15,16]:
                        if i < len(pose):
                            cv2.circle(crop,(int(pose[i].x*cw),int(pose[i].y*ch)),5,(255,255,0),-1)
                for hand in [left,right]:
                    if hand:
                        for lm in hand:
                            cv2.circle(crop,(int(lm.x*cw),int(lm.y*ch)),2,(255,0,255),-1)
                cv2.putText(crop, f"DRIVER id={driver_id} found={driver_found}",
                            (15,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                cv2.putText(crop, f"t={t:.1f}s EAR={ear_m:.3f} pitch={pitch:.1f} neckF={neck_flexion_proxy:.2f} neckL={neck_lateral_proxy:.2f}",
                            (15,35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                cv2.putText(crop, f"rel face={face_rel:.2f} pose={pose_rel:.2f} hand={hand_reliability:.2f}",
                            (15,68), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                scale=min(1.0,900/crop.shape[1])
                show = cv2.resize(crop, None, fx=scale, fy=scale)
                cv2.imshow("driver feature preview - q to stop", show)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    progress.close()
    cap.release(); cv2.destroyAllWindows()
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    print(f"rows={len(df)}, features={len(FEATURE_NAMES)}")
    if latencies:
        a=np.asarray(latencies)
        print(f"MediaPipe latency ms: mean={a.mean():.1f}, p50={np.percentile(a,50):.1f}, p95={np.percentile(a,95):.1f}")
        print(f"10 FPS synchronous criterion: p95 {'OK' if np.percentile(a,95) <= 100 else 'CHECK'} (100 ms)")

if __name__ == "__main__":
    main()
