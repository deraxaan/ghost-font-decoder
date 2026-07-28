"""Command-line entry point for the Ghost Font decoder."""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

from .io_utils import load_dot_frames
from .motion import (
    auto_detect_speed,
    build_density_for_window,
    classify_message_dots,
    filter_noisy_frames,
    reconstruct_best_window,
)
from .ocr import HAS_TESSERACT, decode_message, tesseract_engine_available
from .text_bands import crop_to_content


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ghost-decode",
        description="Decode a Ghost Font video and print the hidden message.",
    )
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument(
        "--speed", type=float, default=None,
        help="Known dot speed in pixels/frame. If omitted, use --auto-speed.",
    )
    parser.add_argument(
        "--auto-speed", action="store_true",
        help="Automatically estimate dot speed (tries 1-10 px/frame).",
    )
    parser.add_argument(
        "--thresh", type=int, default=127,
        help="Grayscale threshold (0-255) for dot detection (default: 127).",
    )
    parser.add_argument(
        "--window", type=int, default=30,
        help="Local alignment window size in frames (default: 30).",
    )
    parser.add_argument(
        "--out", default="ghost_message_clean.png",
        help="Path to save the human-viewable reconstruction PNG.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Print per-window band detection and OCR attempts to diagnose "
             "why decoding might be failing on a specific video.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.speed is None and not args.auto_speed:
        print("error: provide --speed N or --auto-speed", file=sys.stderr)
        return 2

    if not HAS_TESSERACT:
        print(
            "error: the 'pytesseract' Python package is not installed.\n"
            "  pip install pytesseract",
            file=sys.stderr,
        )
        return 1

    if not tesseract_engine_available():
        print(
            "error: pytesseract is installed, but the Tesseract OCR *engine*\n"
            "itself could not be found. 'pip install pytesseract' only installs\n"
            "a Python wrapper -- the engine is a separate program.\n"
            "\n"
            "This was already checked against PATH and the default Windows\n"
            "install locations:\n"
            "  C:\\Program Files\\Tesseract-OCR\\tesseract.exe\n"
            "  C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe\n"
            "\n"
            "If you installed it somewhere else, find tesseract.exe and add\n"
            "these two lines near the top of ghost_decoder\\ocr.py (after the\n"
            "'import pytesseract' line):\n"
            "  pytesseract.pytesseract.tesseract_cmd = r'FULL_PATH_TO\\tesseract.exe'\n"
            "\n"
            "If it's not installed at all, get it from:\n"
            "  https://github.com/UB-Mannheim/tesseract/wiki",
            file=sys.stderr,
        )
        return 1

    print("Loading video...")
    frames = load_dot_frames(args.video, thresh=args.thresh)
    print(f"  {frames.shape[0]} frames, {frames.shape[2]}x{frames.shape[1]}")

    speed = args.speed if args.speed is not None else auto_detect_speed(frames)
    print(f"Speed: {speed} px/frame")

    print("Classifying message dots...")
    masks = classify_message_dots(frames, speed)

    print("Filtering noisy/glitched frames...")
    good_idx = filter_noisy_frames(masks)
    print(f"  keeping {len(good_idx)}/{len(masks)} frames")

    if len(good_idx) < args.window // 2:
        print(
            "warning: very few clean frames survived filtering -- results "
            "may be unreliable. Try a different --thresh or --speed.",
            file=sys.stderr,
        )

    print("Reconstructing a preview image...")
    density = reconstruct_best_window(masks, good_idx, speed, window_size=args.window)
    if density is not None:
        # Blur the continuous density map before thresholding -- the same
        # technique ocr.py already uses correctly for the actual OCR path.
        # Skipping this (as the previous version did) leaves raw per-pixel
        # density noise, which looks unreadable even when the underlying
        # reconstruction is fine.
        density_u8 = cv2.normalize(density, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        blurred = cv2.GaussianBlur(density_u8, (9, 9), 3)
        _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # After THRESH_OTSU, message pixels (higher density) end up as
        # 255 and background as 0. Invert so the saved PNG follows the
        # normal convention: dark text on a light background.
        preview = 255 - thresholded

        cv2.imwrite(args.out, crop_to_content(preview, dark_thresh=128))
        print(f"Saved preview image to: {args.out}")

    print("Scanning alignment windows and voting on the OCR result...")
    result = decode_message(masks, good_idx, speed, window_size=args.window, debug=args.debug)

    print("\n=== RESULT ===")
    if result:
        print("Decoded message:", result)
    else:
        print(f"OCR could not confidently read the message. Open {args.out} and read it yourself.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
