"""Motion-based dot classification and frame alignment.

This is the core of the decoder. The message dots and the decoy
background dots move at the same speed in opposite vertical directions.
The functions here (1) figure out that speed, (2) classify each dot in
each frame as "message" or "background" by comparing it against the
previous frame, and (3) stack a short run of classified frames into a
single clean density map for OCR.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import numpy.typing as npt

FrameStack = npt.NDArray[np.float32]       # shape (T, H, W), values in {0, 1}
DotMask = npt.NDArray[np.uint8]            # shape (H, W), values in {0, 255}
DensityMap = npt.NDArray[np.float32]       # shape (H, W), values in [0, 1]


def auto_detect_speed(frames: FrameStack, speed_range: tuple[int, int] = (1, 10)) -> int:
    """Estimate the dots' vertical speed (in pixels/frame) by grid search.

    For each candidate speed, frames are shifted and averaged as if that
    speed were correct. The correct speed produces a result that
    deviates sharply from a flat/uniform density in a small region (the
    message), while incorrect speeds just blur everything into a uniform
    gray. We score each candidate by how strong that peak deviation is
    and keep the best one.

    This is intentionally a coarse, integer-only search -- fast enough
    to run on every frame of a several-second clip, and precise enough
    for the frame-by-frame classification step that follows.

    Args:
        frames: Dot-mask stack from :func:`load_dot_frames`.
        speed_range: Inclusive ``(min, max)`` pixel/frame range to search.

    Returns:
        The best-scoring integer speed.
    """
    T, H, W = frames.shape
    best_speed, best_score = speed_range[0], -1.0

    for speed in range(speed_range[0], speed_range[1] + 1):
        acc = np.zeros((H, W), dtype=np.float32)
        step = max(1, T // 60)  # subsample for speed during search
        n = 0
        for t in range(0, T, step):
            acc += np.roll(frames[t], shift=int(round(-speed * t)), axis=0)
            n += 1
        acc /= n

        deviation = np.abs(acc - acc.mean())
        score = float(np.percentile(deviation, 99.9))
        if score > best_score:
            best_score, best_speed = score, speed

    return best_speed


def classify_message_dots(frames: FrameStack, speed: float) -> List[DotMask]:
    """Classify each dot in each frame as "message" or "background".

    For every consecutive frame pair ``(t-1, t)``, the previous frame is
    shifted both up and down by ``speed`` pixels. A dot in the current
    frame is labeled "message" if it aligns better with the up-shifted
    hypothesis than the down-shifted one (i.e. it more plausibly moved
    up rather than down).

    Args:
        frames: Dot-mask stack from :func:`load_dot_frames`.
        speed: Dot speed in pixels/frame, as returned by
            :func:`auto_detect_speed` or supplied by the user.

    Returns:
        A list of ``T - 1`` masks (one per frame transition), each of
        shape ``(H, W)`` with values ``0`` or ``255``. Index ``i`` in
        this list corresponds to original frame ``t = i + 1``.
    """
    T, H, W = frames.shape
    shift = int(round(speed))
    masks = []
    for t in range(1, T):
        prev, cur = frames[t - 1], frames[t]
        shifted_up = np.roll(prev, shift=-shift, axis=0)
        shifted_down = np.roll(prev, shift=shift, axis=0)

        agrees_with_up = cur * shifted_up
        agrees_with_down = cur * shifted_down
        is_message = (agrees_with_up > agrees_with_down) & (cur > 0)

        masks.append(is_message.astype(np.uint8) * 255)
    return masks


def filter_noisy_frames(masks: Sequence[DotMask], tolerance: float = 1.15) -> List[int]:
    """Drop frame transitions where classification broke down.

    Occasionally the up/down classification described in
    :func:`classify_message_dots` produces a frame where almost every
    dot is (incorrectly) labeled "message" -- visually these look like
    a burst of solid noise. Such frames have a message-dot count far
    above the typical value, so we flag and drop them by comparing each
    frame's count to the median.

    Args:
        masks: Output of :func:`classify_message_dots`.
        tolerance: A frame is kept if its dot count is below
            ``median(counts) * tolerance``.

    Returns:
        Indices (into ``masks``) of the frames considered clean.
    """
    counts = np.array([int((m > 0).sum()) for m in masks])
    good_threshold = np.median(counts) * tolerance
    return [i for i, c in enumerate(counts) if c < good_threshold]


def build_density_for_window(
    masks: Sequence[DotMask],
    good_idx: Sequence[int],
    speed: float,
    start_i: int,
    window_size: int = 30,
) -> Optional[DensityMap]:
    """Stack one local window of good frames into a density map.

    Frames are aligned relative to the *window's own start* rather than
    to the very first frame of the video. This matters: the total
    vertical travel of the message dots over a full clip can exceed the
    frame height, so aligning everything back to frame 0 causes
    ``np.roll`` to wrap pixels around the top/bottom edge and corrupt
    the reconstruction. A short window keeps the cumulative shift small
    enough that wraparound never becomes visible.

    Args:
        masks: Output of :func:`classify_message_dots`.
        good_idx: Indices of clean frames, from :func:`filter_noisy_frames`.
        speed: Dot speed in pixels/frame.
        start_i: Index into ``good_idx`` marking the start of the window.
        window_size: Number of *original* frame slots the window spans
            (not all of them need to be "good" -- see ``min_fraction``
            behavior below).

    Returns:
        A float density map in ``[0, 1]`` (higher = more frames agreed a
        message dot was present at that pixel), or ``None`` if fewer
        than half the frames in the window were usable.
    """
    H, W = masks[0].shape
    window = [
        i for i in good_idx
        if good_idx[start_i] <= i < good_idx[start_i] + window_size
    ]
    if len(window) < window_size * 0.5:
        return None

    t0 = window[0] + 1  # +1 because masks[i] corresponds to original frame i+1
    acc = np.zeros((H, W), dtype=np.float32)
    for i in window:
        t = i + 1
        message_dots = (masks[i] > 0).astype(np.float32)
        dy = int(round(speed * (t - t0)))
        acc += np.roll(message_dots, shift=dy, axis=0)
    acc /= len(window)
    return acc


def reconstruct_best_window(
    masks: Sequence[DotMask],
    good_idx: Sequence[int],
    speed: float,
    window_size: int = 30,
) -> Optional[DensityMap]:
    """Build a density map from the single densest run of good frames.

    Scans all possible window start positions and picks the one
    containing the most usable (non-noisy) frames. Useful for a quick
    human-viewable preview; :func:`ghost_decoder.ocr.decode_message`
    additionally scans *multiple* windows and votes across them for
    higher OCR reliability.

    Returns:
        A density map (see :func:`build_density_for_window`), or
        ``None`` if ``good_idx`` is empty.
    """
    if not good_idx:
        return None

    best_start, best_count = 0, -1
    for start in range(0, max(1, len(good_idx) - 1)):
        count = sum(
            1 for i in good_idx
            if good_idx[start] <= i < good_idx[start] + window_size
        )
        if count > best_count:
            best_count, best_start = count, start
        if good_idx[start] > good_idx[-1] - window_size:
            break

    return build_density_for_window(masks, good_idx, speed, best_start, window_size)
