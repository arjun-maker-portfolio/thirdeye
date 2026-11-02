# Third Eye

A camera-based prototype for assessing approaching cars, motivated by the difficulty of crossing roads in unpredictable traffic in India. It estimates vehicle motion and adds coloured warnings and pitch-coded beeps to a video.

[Input video from YouTube](videodemo.mp4)

**[Annotated demo with sound](output.mp4)**



https://github.com/user-attachments/assets/035e129b-3d32-4c01-8308-1665633ae574


## Run
Use **Python 3.11 or 3.12**. From this folder, in a virtual environment:

```bash
python -m pip install -r req.txt
python main.py videodemo.mp4 -o annotated.mp4 --threads 4
```

Open `annotated.mp4` after processing. Add `--no-audio` for silent output.
Beeps are embedded in the output.

## How it works

- **Detection and tracking:** A YOLOv5 TFLite model detects vehicles. This version uses the front, side, and rear car classes. Detections are matched to existing tracks using bounding-box overlap, normalized centre displacement, and height ratio.
- **Distance and motion:** Assuming vehicle height $H$, distance is estimated as $Z=f_yH/h$, where $h$ is bounding-box height and $f_y$ is vertical focal length in pixels. A short regression fits inverse height against time to reject any isolated outliers. Its slope estimates longitudinal velocity; lateral velocity comes from changes in $X=(u-c_x)Z/f_x$.
- **Time to contact:** For a reliably approaching vehicle, $\mathrm{TTC}=-(1/h)/\frac{d(1/h)}{dt}$. Under the constant-velocity and fixed-height assumptions, focal length and assumed vehicle height cancel out. Expansion must exceed an estimated noise threshold before producing a TTC.
- **Warnings and missing detections:** Closing speed and TTC feed a bounded risk score. Time-based exponential smoothing, separate entry/exit thresholds, and delayed warning release reduce flicker while allowing immediate escalation. Predicted boxes bridge skipped frames without becoming new motion measurements; stale tracks expire.

`main.py` runs the entire process. `vision.py` handles detection, `motion.py` tracking and risk, `overlay.py` drawing, and `audio.py` pitch-coded warnings.

## Assumptions

Defaults assume a 70° horizontal field of view and 1.45 m car height. Set `--fx`, `--fy`, and optionally `--cx` for calibrated camera intrinsics at the video's resolution.
Estimates are relative to the camera. Camera motion, occlusion, vehicle dimensions, and changing viewpoints affect them.
