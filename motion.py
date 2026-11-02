import math
from collections import deque

import numpy as np
from vision import iou


def robust_line(times, values):
    # reject isolated measurement spikes
    t = np.asarray(times, dtype=float)
    y = np.asarray(values, dtype=float)
    t = t - t[-1]
    keep = np.ones(len(t), dtype=bool)
    for _ in range(3):
        a = np.column_stack((t[keep], np.ones(keep.sum())))
        slope, intercept = np.linalg.lstsq(a, y[keep], rcond=None)[0]
        residual = y - (slope * t + intercept)
        center = np.median(residual[keep])
        mad = 1.4826 * np.median(np.abs(residual[keep] - center))
        new = np.abs(residual - center) <= max(3 * mad, 1e-8)
        if new.sum() < 4 or np.array_equal(keep, new):
            break
        keep = new
    a = np.column_stack((t[keep], np.ones(keep.sum())))
    slope, intercept = np.linalg.lstsq(a, y[keep], rcond=None)[0]
    residual = y[keep] - (slope * t[keep] + intercept)
    error = math.sqrt(float(np.sum(residual**2)) / max(1, keep.sum() - 2))
    spread = float(np.sum((t[keep] - np.mean(t[keep])) ** 2))
    slope_error = error / math.sqrt(max(spread, 1e-12))
    return float(slope), float(intercept), slope_error, float(np.ptp(t[keep]))


class RiskFilter:
    def __init__(self, tau=0.3, release=0.6):
        self.tau = tau
        self.release = release
        self.value = None
        self.state = None
        self.previous = None
        self.candidate = None
        self.since = None

    def update(self, raw, now):
        if self.value is None:
            self.value = raw
        else:
            dt = max(0.0, now - self.previous)
            alpha = 1 - math.exp(-dt / self.tau)
            self.value += alpha * (raw - self.value)
        self.previous = now
        # escalate immediately, release slowly
        level = max(raw, self.value)
        if self.state is None:
            self.state = 2 if level >= 0.70 else 1 if level >= 0.40 else 0
        target = 2 if level >= 0.70 else 1 if level >= 0.40 else 0
        if target > self.state:
            self.state = target
            self.candidate = self.since = None
        else:
            lower = self.value < (0.60 if self.state == 2 else 0.30)
            if target < self.state and lower:
                if self.candidate != target:
                    self.candidate, self.since = target, now
                elif now - self.since >= self.release:
                    self.state = target
                    self.candidate = self.since = None
            else:
                self.candidate = self.since = None
        return self.state


class Track:
    def __init__(self, track_id, detection, now, window):
        self.id = track_id
        self.window = window
        self.samples = deque(maxlen=120)
        self.boxes = deque(maxlen=12)
        self.filter = RiskFilter()
        self.update(detection, now)

    def update(self, detection, now):
        self.box = np.asarray(detection[:4], dtype=float)
        self.confidence, self.cls = detection[4:]
        self.last_seen = now
        self.boxes.append((now, self.box.copy()))
        while self.boxes and now - self.boxes[0][0] > self.window:
            self.boxes.popleft()
        while self.samples and now - self.samples[0][0] > self.window:
            self.samples.popleft()

    def observe(self, width, height):
        x1, y1, x2, y2 = self.box
        h = y2 - y1
        # skip clipped or tiny boxes
        if h < 12 or min(x1, y1) <= 2 or x2 >= width - 3 or y2 >= height - 3:
            return
        if self.samples and self.samples[-1][0] == self.last_seen:
            return
        self.samples.append((self.last_seen, h, (x1 + x2) / 2))

    def predict(self, now):
        box = self.box.copy()
        if len(self.boxes) >= 3:
            times = [s[0] for s in self.boxes]
            dt = min(max(0.0, now - self.last_seen), 0.3)
            for i in range(4):
                slope, _, _, _ = robust_line(times, [s[1][i] for s in self.boxes])
                box[i] += slope * dt
        return box

    def motion(self, fx, fy, cx, car_height):
        if len(self.samples) < 4:
            return None
        t, heights, centers = np.asarray(self.samples).T
        if t[-1] - t[0] < 0.25:
            return None
        # inverse height is proportional to depth
        inv_slope, inv_h, slope_error, span = robust_line(t, 1 / heights)
        if inv_h <= 0 or span < 0.25 or slope_error * span > 0.15 * inv_h:
            return None
        z = fy * car_height * inv_h
        vz = fy * car_height * inv_slope
        lateral = (centers - cx) * (fy * car_height / heights) / fx
        vx, _, _, _ = robust_line(t, lateral)
        # ignore expansion within measurement noise
        approaching = inv_slope < -max(2 * slope_error, 0.00002)
        ttc = -inv_h / inv_slope if approaching else None
        return {
            "distance": z,
            "speed": math.hypot(vz, vx) * 3.6,
            "closing": max(0.0, -vz) if approaching else 0.0,
            "ttc": ttc,
            "approaching": approaching,
        }


class Tracker:
    def __init__(self, window=0.8, max_age=0.5):
        self.window, self.max_age = window, max_age
        self.tracks = []
        self.next_id = 1

    def expire(self, now):
        self.tracks = [t for t in self.tracks if now - t.last_seen <= self.max_age]

    def update(self, detections, now, width, height):
        self.expire(now)
        pairs = []
        for ti, track in enumerate(self.tracks):
            predicted = track.predict(now)
            pw, ph = np.maximum(predicted[2:] - predicted[:2], 1)
            pc = (predicted[:2] + predicted[2:]) / 2
            for di, detection in enumerate(detections):
                b = np.asarray(detection[:4], dtype=float)
                _bw, bh = np.maximum(b[2:] - b[:2], 1)
                ratio = bh / ph
                distance = np.linalg.norm(((b[:2] + b[2:]) / 2 - pc) / [pw, ph])
                overlap = iou(predicted, b)
                if 0.55 < ratio < 1.8 and (overlap >= 0.1 or distance < 0.5):
                    cost = 1 - overlap + 0.25 * distance + 0.1 * abs(math.log(ratio))
                    pairs.append((cost, ti, di))
        used_tracks, used_detections = set(), set()
        # gated, one-to-one association
        for _, ti, di in sorted(pairs):
            if ti in used_tracks or di in used_detections:
                continue
            track = self.tracks[ti]
            track.update(detections[di], now)
            track.observe(width, height)
            used_tracks.add(ti)
            used_detections.add(di)
        for di, detection in enumerate(detections):
            if di not in used_detections:
                track = Track(self.next_id, detection, now, self.window)
                track.observe(width, height)
                self.tracks.append(track)
                self.next_id += 1


def raw_risk(motion, args):
    if motion["ttc"] is None:
        return 0.0
    ttc_risk = np.clip((args.ttc_yellow - motion["ttc"]) / (args.ttc_yellow - args.ttc_red), 0, 1)
    # speed alone cannot establish a collision course
    near_factor = np.clip((2 * args.ttc_yellow - motion["ttc"]) / args.ttc_yellow, 0, 1)
    speed_risk = (
        np.clip((motion["closing"] * 3.6 - args.v_yellow) / (args.v_red - args.v_yellow), 0, 1)
        * near_factor
    )
    return float(max(ttc_risk, speed_risk))
