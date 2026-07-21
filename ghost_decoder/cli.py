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
from .ocr import HAS_TESSERACT, decode_message
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.speed is None and not args.auto_speed:
        print("error: provide --speed N or --auto-speed", file=sys.stderr)
        return 2

    if not HAS_TESSERACT:
        print(
            "error: pytesseract is not installed, or the Tesseract OCR engine "
            "binary is not on PATH.\n"
            "  pip install pytesseract\n"
            "  # plus the engine itself:\n"
            "  #   Ubuntu/Debian: sudo apt install tesseract-ocr\n"
            "  #   macOS:         brew install tesseract\n"
            "  #   Windows:       https://github.com/UB-Mannheim/tesseract/wiki",
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
        preview = 255 - (np.clip(density, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(args.out, crop_to_content(preview, dark_thresh=128))
        print(f"Saved preview image to: {args.out}")

    print("Scanning alignment windows and voting on the OCR result...")
    result = decode_message(masks, good_idx, speed, window_size=args.window)

    print("\n=== RESULT ===")
    if result:
        print("Decoded message:", result)
    else:
        print(f"OCR could not confidently read the message. Open {args.out} and read it yourself.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
