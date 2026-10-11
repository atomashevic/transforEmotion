"""Build the two-person example picture and clip in inst/extdata.

Run from the package root with a Python that has OpenCV, and ffmpeg on the
PATH (it re-encodes the clip as H.264, about 75 KB instead of 425 KB):
    python data-raw/example-media.py

two-people.jpg  Boris Johnson (boris-1.png) and Donald Trump (trump1.jpg)
                side by side, 400 pixels high.
two-people.mp4  12 seconds at 25 frames per second: both men for 4 seconds,
                4 seconds of black (a cut), then Boris Johnson alone on the
                right (boris-2.png) for 4 seconds.
"""
import os
import subprocess
import tempfile

import cv2
import numpy as np

EXTDATA = os.path.join("inst", "extdata")
HEIGHT = 400


def read(name):
    """Read an example image and scale it to HEIGHT pixels high."""
    img = cv2.imread(os.path.join(EXTDATA, name))
    h, w = img.shape[:2]
    return cv2.resize(img, (round(w * HEIGHT / h), HEIGHT), interpolation=cv2.INTER_AREA)


two = cv2.hconcat([read("boris-1.png"), read("trump1.jpg")])
cv2.imwrite(os.path.join(EXTDATA, "two-people.jpg"), two, [cv2.IMWRITE_JPEG_QUALITY, 90])

h, w = two.shape[:2]
black = np.zeros_like(two)
alone = cv2.hconcat([np.zeros_like(read("trump1.jpg")), read("boris-2.png")])
alone = cv2.resize(alone, (w, h), interpolation=cv2.INTER_AREA)

with tempfile.TemporaryDirectory() as tmp:
    raw = os.path.join(tmp, "two-people.mp4")
    writer = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
    for i in range(300):
        writer.write(two if i < 100 else black if i < 200 else alone)
    writer.release()
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", raw, "-c:v", "libx264",
                    "-preset", "veryslow", "-crf", "23", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", os.path.join(EXTDATA, "two-people.mp4")],
                   check=True)
