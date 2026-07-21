#!/usr/bin/env python3
"""
Convenience entry point: `python decode.py video.mp4 --speed 4`

Equivalent to `python -m ghost_decoder.cli video.mp4 --speed 4`.
"""

from ghost_decoder.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
