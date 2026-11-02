import cv2
import numpy as np
from vision import CLASS_NAMES

COLORS = {0: (0, 190, 0), 1: (0, 210, 255), 2: (0, 0, 230), None: (150, 150, 150)}


STATE_NAMES = {0: "LOW ESTIMATED RISK", 1: "CAUTION", 2: "HIGH RISK", None: "UNKNOWN"}


def wrap_text(text, scale, max_width):
    lines = []
    current = ""
    chunks = text.split(" | ") if " | " in text else text.split()
    separator = " | " if " | " in text else " "
    for word in chunks:
        trial = f"{current}{separator}{word}".strip() if current else word
        if current and cv2.getTextSize(trial, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > max_width:
            lines.append(current.rstrip(" |"))
            current = word.lstrip("|").strip()
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def make_panel(lines, color, max_width, header_scale, detail_scale):
    # supersample text
    pad = 9
    rows = []
    for i, text in enumerate(lines):
        scale = header_scale if i == 0 else detail_scale
        for part in wrap_text(text, scale, max_width - 2 * pad):
            (width, height), baseline = cv2.getTextSize(part, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            rows.append((part, scale, width, height, baseline, i == 0))
    panel_width = min(max_width, max(row[2] for row in rows) + 2 * pad)
    panel_height = sum(row[3] + row[4] + 9 for row in rows) + pad
    panel = np.full((panel_height * 2, panel_width * 2, 3), (23, 23, 23), dtype=np.uint8)
    y = pad
    for text, scale, width, height, baseline, header in rows:
        if header:
            cv2.rectangle(
                panel,
                (0, (y - 5) * 2),
                (panel_width * 2 - 1, (y + height + baseline + 4) * 2),
                color,
                -1,
            )
        # light text on red; dark text on yellow/green
        ink = (255, 255, 255) if not header or color[2] > color[1] * 1.5 else (10, 10, 10)
        cv2.putText(
            panel,
            text,
            (pad * 2, (y + height) * 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale * 2,
            ink,
            2,
            cv2.LINE_AA,
        )
        y += height + baseline + 9
    return cv2.resize(panel, (panel_width, panel_height), interpolation=cv2.INTER_AREA)


def draw_status_banner(frame, text, subtext, color_bgr):
    height, width = frame.shape[:2]
    panel = make_panel(
        [text, subtext],
        color_bgr,
        width - 16,
        max(0.58, min(0.9, width / 1200)),
        max(0.46, min(0.72, width / 1600)),
    )
    ph, pw = panel.shape[:2]
    if ph <= height - 16:
        frame[height - ph - 8 : height - 8, 8 : 8 + pw] = panel


def draw_track(frame, track, box, state, predicted):
    height, width = frame.shape[:2]
    b = np.clip(box, [0, 0, 0, 0], [width - 1, height - 1, width - 1, height - 1]).astype(int)
    x1, y1, x2, y2 = b
    if x2 <= x1 or y2 <= y1:
        return
    color = COLORS[state]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
    label = f"{CLASS_NAMES[track.cls]} #{track.id} | conf {track.confidence:.2f}"
    if predicted:
        label += " | predicted"
    scale = max(0.43, min(0.65, width / 1600))
    panel = make_panel([label], color, width - 8, scale, scale)
    ph, pw = panel.shape[:2]
    left = max(4, min(x1, width - pw - 4))
    top = y1 - ph - 3 if y1 >= ph + 3 else y1 + 3
    top = max(0, min(top, height - ph))
    if ph <= height:
        frame[top : top + ph, left : left + pw] = panel


def format_metrics(motion):
    if motion is None:
        return "Speed: -- | Dist: -- | TTC: --"
    ttc = f"{motion['ttc']:.1f}s" if motion["ttc"] is not None else "--"
    return f"Speed: ~{motion['speed']:.1f} km/h | Dist: ~{motion['distance']:.1f} m | TTC: {ttc}"
