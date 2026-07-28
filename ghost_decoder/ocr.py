"""OCR extraction from reconstructed density maps.

The key trick here is applying Gaussian blur to the *continuous* density
map before thresholding, rather than dilating an already-binarized dot
mask. Blurring first lets nearby dots fuse into solid strokes with
naturally smooth edges; dilating a hard binary mask tends to produce
blobby, disconnected shapes that OCR engines don't recognize as letters.
"""

from __future__ import annotations

import os
import shutil
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import numpy.typing as npt

from .motion import DotMask, build_density_for_window
from .text_bands import find_text_line_bands

try:
    import pytesseract

    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

# On Windows, the Tesseract installer does not reliably add itself to
# PATH (and other tools on the system can silently break PATH resolution
# entirely). If `tesseract` isn't already resolvable, fall back to
# checking the installer's default locations directly, so a working
# install still works even when PATH doesn't cooperate.
if HAS_TESSERACT and sys.platform == "win32" and shutil.which("tesseract") is None:
    for _candidate in (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ):
        if os.path.isfile(_candidate):
            pytesseract.pytesseract.tesseract_cmd = _candidate
            break


def tesseract_engine_available() -> bool:
    """Check whether the actual Tesseract OCR engine binary is reachable.

    ``HAS_TESSERACT`` only confirms that the *Python wrapper* package
    (``pytesseract``) imported successfully -- ``pip install pytesseract``
    does not install the underlying OCR engine itself, which is a
    separate program that must be installed independently and be on
    PATH. Without this check, a missing engine binary silently causes
    every OCR attempt deep in :func:`ocr_density_region` to fail and
    return an empty string, which looks identical to "the image just
    wasn't readable" -- this function lets callers catch the real cause
    up front instead.

    Returns:
        ``True`` if OCR calls should actually be able to run.
    """
    if not HAS_TESSERACT:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


DEFAULT_SIGMAS = (7, 9, 11)
MIN_TEXT_LENGTH = 4


def ocr_density_region(
    density_slice: npt.NDArray[np.float32],
    pad: int = 20,
    sigmas: Sequence[int] = DEFAULT_SIGMAS,
) -> str:
    """Run OCR on one region (typically one text line) of a density map.

    For each candidate blur strength, the region is blurred, Otsu
    thresholded, cropped tightly to content, upscaled, and passed to
    Tesseract under a few page-segmentation modes. All non-empty,
    alphabetic-only results are tallied and the most frequent one wins
    -- this makes the result robust to any single blur/PSM combination
    producing a fluke misread.

    Args:
        density_slice: A ``(H, W)`` float density map, or a row-slice of
            one representing a single text line.
        pad: Blank rows added above/below before blurring, so the blur
            kernel doesn't get cut off at the region's edge.
        sigmas: Gaussian blur standard deviations to try.

    Returns:
        The most-voted OCR result (letters only, uppercased), or an
        empty string if nothing usable was recognized or Tesseract is
        unavailable.
    """
    if not HAS_TESSERACT:
        return ""

    padded = np.pad(density_slice, ((pad, pad), (0, 0)), mode="constant")
    votes: Dict[str, int] = {}

    for sigma in sigmas:
        kernel_size = sigma * 2 + 1
        blurred = cv2.GaussianBlur(padded, (kernel_size, kernel_size), sigma)
        blurred_u8 = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, thresholded = cv2.threshold(
            blurred_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        text_on_white = 255 - thresholded

        dark = text_on_white < 128
        ys, xs = np.where(dark)
        if len(xs) == 0:
            continue
        x0, x1 = max(0, xs.min() - pad), min(text_on_white.shape[1], xs.max() + pad)
        y0, y1 = max(0, ys.min() - pad), min(text_on_white.shape[0], ys.max() + pad)
        tight = text_on_white[y0:y1, x0:x1]
        upscaled = cv2.resize(tight, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        try:
            for psm in (7, 8):
                raw = pytesseract.image_to_string(upscaled, config=f"--psm {psm}").strip()
                letters_only = "".join(c for c in raw if c.isalpha())
                if letters_only:
                    key = letters_only.upper()
                    votes[key] = votes.get(key, 0) + 1
        except pytesseract.TesseractNotFoundError:
            return ""

    if not votes:
        return ""
    return max(votes.items(), key=lambda kv: kv[1])[0]


def decode_message(
    masks: Sequence[DotMask],
    good_idx: Sequence[int],
    speed: float,
    window_size: int = 30,
    stride: int = 8,
    debug: bool = False,
) -> Optional[str]:
    """Decode the full hidden message by scanning multiple alignment windows.

    Different alignment windows land on different "phases" of the dot
    pattern and vary in reconstruction quality, so no single window is
    guaranteed to OCR cleanly. This scans several windows across the
    video and votes across all attempts.

    Within each window, splitting the density map into per-line bands
    (see :func:`ghost_decoder.text_bands.find_text_line_bands`) and OCRing
    each line separately is significantly more reliable than OCRing the
    whole block at once -- so line-split results are weighted higher.
    Whole-block OCR is kept only as a fallback for messages with no
    detectable line gap.

    Args:
        masks: Output of :func:`ghost_decoder.motion.classify_message_dots`.
        good_idx: Output of :func:`ghost_decoder.motion.filter_noisy_frames`.
        speed: Dot speed in pixels/frame.
        window_size: Frame-count span of each alignment window.
        stride: Step size (in ``good_idx`` positions) between windows.
        debug: If ``True``, print each window's band count and every
            line/whole-block OCR attempt (including ones too short or
            empty to be registered) to help diagnose why decoding is
            failing on a particular video.

    Returns:
        The winning decoded string (e.g. ``"HELLO HUMAN"``), or ``None``
        if no window produced a usable OCR result.
    """
    votes: Dict[str, int] = {}
    display_text: Dict[str, str] = {}
    windows_tried = 0
    windows_with_density = 0

    def register(text: str, weight: int) -> None:
        if not text:
            return
        signature = text.replace(" ", "").upper()
        votes[signature] = votes.get(signature, 0) + weight
        prefer_this = signature not in display_text or (
            " " in text and " " not in display_text[signature]
        )
        if prefer_this:
            display_text[signature] = text

    for start in range(0, max(1, len(good_idx) - 1), stride):
        windows_tried += 1
        density = build_density_for_window(masks, good_idx, speed, start, window_size)
        if density is None:
            if debug:
                print(f"[debug] window start={start}: no density (too few good frames)")
            continue
        windows_with_density += 1

        bands = find_text_line_bands(density)
        if debug:
            print(f"[debug] window start={start}: {len(bands)} band(s) -> {bands}")

        line_split_succeeded = False
        if len(bands) >= 2:
            line_texts = [ocr_density_region(density[y0:y1, :]) for (y0, y1) in bands]
            if debug:
                print(f"[debug]   per-line OCR: {line_texts}")
            combined = " ".join(t for t in line_texts if t)
            if combined and len(combined.replace(" ", "")) >= MIN_TEXT_LENGTH:
                register(combined, weight=3)
                line_split_succeeded = True
            elif debug:
                print(f"[debug]   combined line text too short/empty: {combined!r}")

        # The whole-block fallback exists for messages with no detectable
        # line gap. When line-split already produced a usable result, it's
        # both lower-priority (weight 1 vs. 3) and, empirically, far more
        # likely to be garbage anyway -- skip the extra Tesseract calls.
        if not line_split_succeeded:
            whole_text = ocr_density_region(density, sigmas=(9, 13))
            if debug:
                print(f"[debug]   whole-block OCR: {whole_text!r}")
            if whole_text and len(whole_text) >= MIN_TEXT_LENGTH:
                register(whole_text, weight=1)

        # Once one candidate has a decisive lead (multiple independent
        # windows agreeing via line-split), further windows are very
        # unlikely to change the outcome -- stop early rather than
        # exhausting every remaining window for no benefit.
        if votes:
            top_signature, top_weight = max(votes.items(), key=lambda kv: kv[1])
            if top_weight >= 9:  # three independent line-split confirmations
                if debug:
                    print(
                        f"[debug] early stop: {top_signature!r} confirmed "
                        f"with weight {top_weight} after {windows_tried} window(s)"
                    )
                break

    if debug:
        print(f"[debug] {windows_with_density}/{windows_tried} windows had usable density")
        print(f"[debug] final votes: {votes}")

    if not votes:
        return None
    winning_signature = max(votes.items(), key=lambda kv: kv[1])[0]
    return display_text[winning_signature]
