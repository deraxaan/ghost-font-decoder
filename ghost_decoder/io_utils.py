"""Video loading and binary dot-mask extraction."""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt


def load_dot_frames(video_path: str, thresh: int = 127) -> npt.NDArray[np.float32]:
    """Load a video and convert every frame to a binary dot mask.

    Each frame is grayscaled and thresholded so that dark dots become
    ``1.0`` and the background becomes ``0.0``.

    Args:
        video_path: Path to the input video file.
        thresh: Grayscale threshold (0-255) below which a pixel is
            considered part of a dot. The default (127) works for
            high-contrast black-dots-on-light-background videos; lower
            it if dots are being missed, raise it if background noise
            is being picked up.

    Returns:
        A float32 array of shape ``(num_frames, height, width)`` with
        values in ``{0.0, 1.0}``.

    Raises:
        IOError: If the video file cannot be opened.
        ValueError: If no frames could be read from the video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
            frames.append(mask.astype(np.float32) / 255.0)
    finally:
        cap.release()

    if not frames:
        raise ValueError(f"No frames could be read from: {video_path}")

    return np.stack(frames, axis=0)
