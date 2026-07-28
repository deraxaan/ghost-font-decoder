# Contributing

Thanks for considering contributing to Ghost Font Decoder. This is a
small, single-purpose project, so contribution guidelines are kept
light on process and heavy on context.

## Setup

```
git clone https://github.com/DeraXaan/ghost-font-decoder.git
cd ghost-font-decoder
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

You'll also need the Tesseract OCR engine installed system-wide (see the
README's Requirements section) to run the full pipeline end-to-end.

## Project status

There is currently no automated test suite, and every heuristic
threshold in `motion.py`, `text_bands.py`, and `ocr.py` (noisy-frame
tolerance, valley ratio, blur sigmas, minimum text length, etc.) has
only been validated against a single sample video. Contributions that
validate or generalize these are especially welcome.

## Good first contributions

- Add sample Ghost Font videos (or a script to generate synthetic ones)
  under `examples/`, along with expected decoded output, to make manual
  testing repeatable.
- Add a `pytest`-based test suite around the pure functions in
  `motion.py` and `text_bands.py` (these don't need a real video --
  synthetic `numpy` arrays are enough).
- Extend `auto_detect_speed` to support horizontal or diagonal dot
  motion, not just vertical.
- Try an alternative OCR backend (e.g. EasyOCR or another neural OCR
  engine) behind the same `decode_message` interface, and compare
  reliability against the current Tesseract-based path.

## Code style

- Type hints and docstrings on public functions (see any existing file
  in `ghost_decoder/` for the expected style).
- Comments should explain *why*, not just *what* -- especially for the
  non-obvious bits (local windowed alignment, blur-before-threshold,
  the up/down motion classification trick).
- Keep functions small and independently testable; the pipeline is
  intentionally built as a chain of plain functions rather than a
  monolithic class.

## Submitting changes

1. Fork the repo and create a branch for your change.
2. Keep pull requests focused -- one logical change per PR.
3. Describe what you tested it against (which video, what the output
   was) in the PR description, since there's no CI yet to verify this
   automatically.
