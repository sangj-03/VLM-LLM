"""Causal motion evidence for an early-warning score, not a consciousness probability."""
import math


def motion_evidence(window, feature_names, fps=10.0):
    """Require sustained downward motion in reliable samples over at most 0.5 s.

    Seat-relative motion resists crop translation; head-to-shoulder motion
    provides a second signal. No padding, future samples, or missing-data jumps.
    Thresholds are provisional shoulder-width units and need held-out validation.
    """
    rows = list(window)[-max(3, int(round(0.5 * fps)) + 1):]
    keys = ('pose_reliability', 'seat_head_dy', 'head_to_shoulder_dy')
    idx = [feature_names.index(k) for k in keys]
    if len(rows) < 3:
        return False, 0.0, 0.0
    values = [[float(r[i]) for i in idx] for r in rows]
    if any(not all(math.isfinite(v) for v in r) or r[0] < 0.6 for r in values):
        return False, 0.0, 0.0
    # Derive velocity from positions to avoid a stale extractor derivative
    # after tracking loss. Two consecutive falling samples reject isolated spikes.
    velocities = [(b[1] - a[1]) * fps for a, b in zip(values, values[1:])]
    speed = min(velocities[-2:])
    acceleration = (velocities[-1] - velocities[-2]) * fps
    drop = values[-1][1] - min(r[1] for r in values[:-1])
    neck_drop = values[-1][2] - min(r[2] for r in values[:-1])
    active = speed >= 0.35 and drop >= 0.12 and neck_drop >= 0.06
    return active, speed, acceleration


def fuse_onset_risk(base_risk, onset_probability, onset_threshold, motion, floor=0.72):
    """Raise an operational warning score only with two independent gates.

    Keep the original calibrated TCN probability separate. An onset model
    predicts a transition, so its probability must not replace P(non_active).
    """
    active = bool(motion and onset_probability is not None
                  and onset_threshold is not None
                  and onset_probability >= onset_threshold)
    return max(base_risk, floor) if active else base_risk, active
