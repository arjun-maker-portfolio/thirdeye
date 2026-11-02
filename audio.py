import subprocess
import wave

import numpy as np

PITCHES = {0: 880, 1: 440, 2: 220}


class BeepWriter:
    def __init__(self, path, fps, rate=24000):
        self.stream = wave.open(str(path), "wb")
        self.stream.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        self.fps, self.rate = fps, rate
        self.state = None
        self.start = 0.0

    def write(self, frame_index, state):
        now = frame_index / self.fps
        if state != self.state:
            self.start, self.state = now, state
        first = round(frame_index * self.rate / self.fps)
        last = round((frame_index + 1) * self.rate / self.fps)
        t = np.arange(first, last) / self.rate - self.start
        if state in PITCHES:
            phase = np.mod(t, 0.7)
            envelope = np.minimum(
                np.clip(phase / 0.008, 0, 1), np.clip((0.14 - phase) / 0.008, 0, 1)
            )
            signal = 0.22 * envelope * np.sin(2 * np.pi * PITCHES[state] * t)
        else:
            signal = np.zeros(last - first)
        self.stream.writeframesraw((signal * 32767).astype("<i2").tobytes())

    def close(self):
        self.stream.close()


def add_audio(ffmpeg, video, wav, output):
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(video),
        "-i",
        str(wav),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        "-movflags",
        "+faststart",
        str(output),
    ]
    subprocess.run(command, check=True)
