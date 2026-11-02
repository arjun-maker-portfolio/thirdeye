# Third Eye

A camera-based prototype for assessing approaching cars, motivated by the difficulty of crossing roads in unpredictable traffic in India. It estimates vehicle motion and adds coloured warnings and pitch-coded beeps to a video.

[Input video from YouTube](videodemo.mp4)

**[Annotated demo with sound](output.mp4)**



https://github.com/user-attachments/assets/035e129b-3d32-4c01-8308-1665633ae574


## Run
Distance is estimated as $`Z = \frac{f_y H}{h}`$.
Use **Python 3.11 or 3.12**. From this folder, in a virtual environment:

```bash
python -m pip install -r req.txt
python main.py videodemo.mp4 -o annotated.mp4 --threads 4
```

Open `annotated.mp4` after processing. Add `--no-audio` for silent output.
Beeps are embedded in the output. **220 Hz for red, 440 Hz for yellow, 880 Hz for green**.

## How it works

- **Detection and tracking:** A YOLOv5 TFLite model detects vehicles; this version uses the front, side, and rear car classes. Detections are matched to existing tracks using bounding-box overlap, normalized centre displacement, and height ratio. Each track maintains its own measurement history.

- **Distance and motion:** Distance is estimated as `Z = fy × H / h`, where `H` is assumed vehicle height, `h` is bounding-box height, and `fy` is vertical focal length in pixels. A short regression fits inverse height against time, rejecting isolated outliers. Its slope estimates longitudinal velocity; lateral velocity comes from changes in `X = (u − cx) × Z / fx`, where `u` is the box centre and `cx` is the camera’s horizontal principal point.

- **Time to contact:** For a reliably approaching vehicle, `TTC = −q / q′`, where `q = 1/h` and `q′` is its fitted rate of change. Under constant-velocity and fixed-height assumptions, focal length and assumed vehicle height cancel out. Expansion must exceed an estimated noise threshold before producing a TTC.

- **Warnings and missing detections:** Closing speed and TTC feed a bounded risk score. Time-based exponential smoothing, separate entry/exit thresholds, and delayed warning release reduce flicker while allowing immediate escalation. Predicted boxes bridge skipped frames without becoming new motion measurements; stale tracks expire.

`main.py` runs the pipeline. `vision.py` handles detection, `motion.py` tracking and risk, `overlay.py` drawing, and `audio.py` pitch-coded warnings.

## Assumptions

Defaults assume a 70° horizontal field of view and 1.45 m car height. Set `--fx`, `--fy`, and optionally `--cx` for calibrated camera intrinsics at the video's resolution.
Estimates are relative to the camera. Camera motion, occlusion, vehicle dimensions, and changing viewpoints affect them.
