"""Locating text content within a reconstructed density map."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import numpy.typing as npt


def crop_to_content(
    img: npt.NDArray[np.uint8], pad: int = 15, dark_thresh: int = 20
) -> npt.NDArray[np.uint8]:
    """Crop an image to the bounding box of its dark (non-background) pixels.

    Args:
        img: Grayscale image where message content is dark.
        pad: Extra margin (pixels) to keep around the detected content.
        dark_thresh: Pixel values above this are considered "content".
            Note this expects dark content to have *higher* values in
            the ``img > dark_thresh`` sense -- callers pass in the
            inverted (message-dot-as-bright) representation.

    Returns:
        The cropped image, or the original image unchanged if no
        content pixels were found.
    """
    dark = img > dark_thresh
    ys, xs = np.where(dark)
    if len(xs) == 0:
        return img
    x0, x1 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad)
    y0, y1 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + pad)
    return img[y0:y1, x0:x1]


def find_text_line_bands(
    density: npt.NDArray[np.float32],
    smooth_k: int = 25,
    active_ratio: float = 0.08,
    valley_ratio: float = 0.4,
) -> List[Tuple[int, int]]:
    """Split a density map into horizontal text-line bands.

    The row-wise density profile is smoothed and then cut at "valleys"
    -- rows much sparser than the surrounding text lines, i.e. the gaps
    between lines of text.

    Smoothing is important: on the raw (unsmoothed) profile, a single
    sparse row in the middle of a letter can look like a line break and
    fragment a single line of text into unusable slivers.

    Args:
        density: Float density map from
            :func:`ghost_decoder.motion.build_density_for_window`.
        smooth_k: Moving-average window (rows) applied to the row-density
            profile before valley detection.
        active_ratio: Rows with smoothed density below
            ``active_ratio * max_density`` are considered outside any
            text content.
        valley_ratio: Within the active region, rows below
            ``valley_ratio * local_max_density`` are treated as a gap
            between lines.

    Returns:
        A list of ``(y_start, y_end)`` row ranges, one per detected text
        line. Falls back to a single band covering the whole image if no
        content is found.
    """
    row_sum = density.sum(axis=1)
    smooth = np.convolve(row_sum, np.ones(smooth_k) / smooth_k, mode="same")

    active = smooth > smooth.max() * active_ratio
    active_rows = np.where(active)[0]
    if len(active_rows) == 0:
        return [(0, density.shape[0])]
    lo, hi = active_rows.min(), active_rows.max()

    valley_thresh = smooth[lo:hi].max() * valley_ratio
    bands: List[Tuple[int, int]] = []
    start, in_gap = lo, False
    for row in range(lo, hi + 1):
        is_gap = smooth[row] < valley_thresh
        if is_gap and not in_gap:
            bands.append((start, row))
        if not is_gap and in_gap:
            start = row
        in_gap = is_gap
    if not in_gap:
        # Only close the final band if we ended inside text content --
        # otherwise we'd append a spurious trailing (gap_start, hi+1) band.
        bands.append((start, hi + 1))

    bands = [b for b in bands if b[1] - b[0] > 10]
    return bands or [(0, density.shape[0])]
