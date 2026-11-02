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
Beeps are embedded in the output. **220 Hz for red, 440 Hz for yellow, 880 Hz for green**.

## How it works

- A YOLOv5 TFLite model detects vehicles. This version uses its three car classes and matches boxes across detections by overlap, position, and size.
- Each car has its own history. A short regression over real detections estimates motion and rejects isolated outliers. Predicted boxes fill skipped frames but do not count as measurements. Stale tracks also expire.
- Apparent height and camera geometry estimate distance. Inverse-height changes estimate approach time. Closing speed and time to contact feed a risk score, with smoothing, hysteresis, and delayed warning release.

`main.py` runs the entire process. `vision.py` handles detection, `motion.py` tracking and risk, `overlay.py` drawing, and `audio.py` sound.

## Assumptions

Defaults assume a 70° horizontal field of view and 1.45 m car height. Set `--fx`, `--fy`, and optionally `--cx` for calibrated camera intrinsics at the video's resolution.
Estimates are relative to the camera. Camera motion, occlusion, vehicle dimensions, and changing viewpoints affect them.
