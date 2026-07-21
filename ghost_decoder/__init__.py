"""
ghost_decoder
=============

A small computer-vision pipeline for recovering text hidden in
"Ghost Font" style videos, where a message is encoded as dots moving
in one direction against a background of decoy dots moving in the
opposite direction.

See the top-level README.md for background and usage.
"""

from .io_utils import load_dot_frames
from .motion import (
    auto_detect_speed,
    classify_message_dots,
    filter_noisy_frames,
    build_density_for_window,
    reconstruct_best_window,
)
from .text_bands import crop_to_content, find_text_line_bands
from .ocr import decode_message, HAS_TESSERACT

__all__ = [
    "load_dot_frames",
    "auto_detect_speed",
    "classify_message_dots",
    "filter_noisy_frames",
    "build_density_for_window",
    "reconstruct_best_window",
    "crop_to_content",
    "find_text_line_bands",
    "decode_message",
    "HAS_TESSERACT",
]

__version__ = "0.1.0"
