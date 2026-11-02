import argparse
import math
import tempfile
from pathlib import Path

import cv2
from audio import BeepWriter, add_audio
from motion import Tracker, raw_risk
from overlay import COLORS, STATE_NAMES, draw_status_banner, draw_track, format_metrics
from tqdm import tqdm
from vision import CAR_CLASS_IDS, Detector


def parse_args():
    ap = argparse.ArgumentParser(
        description="Third Eye video prototype with tracking and beep audio"
    )
    ap.add_argument("video_path", type=Path)
    ap.add_argument("-o", "--output", type=Path)
    ap.add_argument(
        "--tflite", type=Path, default=Path(__file__).resolve().parent / "yolov5l.tflite"
    )
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--detect-stride", type=int, default=3)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--fov", type=float, default=70)
    ap.add_argument("--car-h", type=float, default=1.45)
    ap.add_argument("--fx", type=float, help="calibrated horizontal focal length in pixels")
    ap.add_argument("--fy", type=float, help="calibrated vertical focal length in pixels")
    ap.add_argument("--cx", type=float, help="calibrated horizontal principal point")
    ap.add_argument("--window", type=float, default=0.8)
    ap.add_argument("--max-age", type=float, default=0.5)
    ap.add_argument("--ttc-yellow", type=float, default=6)
    ap.add_argument("--ttc-red", type=float, default=3)
    ap.add_argument("--v-yellow", type=float, default=20)
    ap.add_argument("--v-red", type=float, default=35)
    ap.add_argument("--no-audio", action="store_true")
    args = ap.parse_args()
    if not 1 < args.fov < 179 or args.car_h <= 0:
        ap.error("invalid field of view or car height")
    if args.detect_stride < 1 or args.threads < 1 or args.window < 0.3 or args.max_age <= 0:
        ap.error("stride/threads must be positive; window >= 0.3; max-age > 0")
    if not 0 < args.ttc_red < args.ttc_yellow or not 0 <= args.v_yellow < args.v_red:
        ap.error("red speed must exceed yellow; yellow TTC must exceed positive red TTC")
    if not 0 < args.conf <= 1 or not 0 < args.iou <= 1:
        ap.error("conf and iou must be in (0, 1]")
    if (args.fx is None) != (args.fy is None):
        ap.error("supply both --fx and --fy")
    if args.fx is not None and min(args.fx, args.fy) <= 0:
        ap.error("focal lengths must be positive")
    args.video_path = args.video_path.expanduser()
    args.tflite = args.tflite.expanduser()
    args.output = (
        args.output.expanduser()
        if args.output
        else args.video_path.with_name(f"output_{args.video_path.stem}.mp4")
    )
    if not args.video_path.is_file() or not args.tflite.is_file():
        ap.error("input video or model not found")
    if args.output.exists() or args.output.resolve() == args.video_path.resolve():
        ap.error("output already exists or equals input; choose another -o path")
    if args.output.suffix.lower() != ".mp4":
        ap.error("use an .mp4 output path")
    return args


def annotate(frame, tracker, now, args, fx, fy, cx, fps):
    readings = []
    uncertain = False
    fresh_limit = max(0.18, 1.5 * args.detect_stride / fps)
    for track in tracker.tracks:
        age = now - track.last_seen
        recent = age <= fresh_limit
        motion = track.motion(fx, fy, cx, args.car_h)
        state = None
        fresh_motion = bool(track.samples) and now - track.samples[-1][0] <= fresh_limit
        if motion is not None and recent and fresh_motion:
            state = track.filter.update(raw_risk(motion, args), now)
        else:
            uncertain = True
        if state is not None:
            readings.append((state, raw_risk(motion, args), motion, track.id))
        draw_track(frame, track, track.predict(now), state, age > 1e-6)
    # highest warning controls the beep
    state = max((r[0] for r in readings), default=None)
    if uncertain and state != 2:
        state = None
    if state is None:
        note = "Collecting observations" if tracker.tracks else "No tracked car"
    else:
        chosen = max(readings, key=lambda item: (item[0], item[1]))
        note = format_metrics(chosen[2])
        if len(tracker.tracks) > 1:
            note += f" | car #{chosen[3]}"
    draw_status_banner(frame, STATE_NAMES[state], note, COLORS[state])
    return state


def process_video(args, silent_path, wav_path):
    model = Detector(args.tflite, args.threads)
    cap = cv2.VideoCapture(str(args.video_path))
    if not cap.isOpened():
        raise RuntimeError("cannot open input video")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0
    width, height = int(cap.get(3)), int(cap.get(4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("invalid video dimensions")
    if args.detect_stride / fps > min(args.max_age, args.window / 3):
        cap.release()
        raise RuntimeError("detection stride too large for motion window/track timeout")
    fx = args.fx or (width / 2) / math.tan(math.radians(args.fov / 2))
    fy = args.fy or fx
    cx = args.cx if args.cx is not None else width / 2
    writer = cv2.VideoWriter(
        str(silent_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("cannot create output video")
    beeps = BeepWriter(wav_path, fps) if wav_path is not None else None
    tracker = Tracker(args.window, args.max_age)
    index = 0
    try:
        with tqdm(total=total if total > 0 else None, unit="frame", desc="Processing") as bar:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                # shared video and audio clock
                now = index / fps
                tracker.expire(now)
                if index % args.detect_stride == 0:
                    detections = model.predict(frame, args.conf, args.iou)
                    detections = [
                        d
                        for d in detections
                        if d[5] in CAR_CLASS_IDS and d[2] - d[0] >= 4 and d[3] - d[1] >= 6
                    ]
                    tracker.update(detections, now, width, height)
                state = annotate(frame, tracker, now, args, fx, fy, cx, fps)
                writer.write(frame)
                if beeps is not None:
                    beeps.write(index, state)
                index += 1
                bar.update(1)
    finally:
        cap.release()
        writer.release()
        if beeps is not None:
            beeps.close()
    if index == 0:
        raise RuntimeError("input video contains no decodable frames")


def main():
    args = parse_args()
    ffmpeg = None
    if not args.no_audio:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="third-eye-", dir=args.output.parent) as tmp:
        tmp = Path(tmp)
        silent, wav = tmp / "video.mp4", tmp / "beeps.wav"
        process_video(args, silent, None if args.no_audio else wav)
        if args.no_audio:
            silent.replace(args.output)
        else:
            muxed = tmp / "annotated.mp4"
            add_audio(ffmpeg, silent, wav, muxed)
            muxed.replace(args.output)
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
