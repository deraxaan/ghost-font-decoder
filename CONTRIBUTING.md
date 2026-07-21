# Contributing to Ghost Font Decoder

Thanks for your interest — this started as a small personal experiment,
so contributions that keep it approachable and well-documented are very
welcome.

## Ways to contribute

- **Bug reports**: open an issue with the video (or a description of
  its properties: resolution, dot size, speed, single vs. multi-line
  text) and the command you ran.
- **Robustness improvements**: the noisy-frame filter, speed detection,
  and text-line splitting all use heuristic thresholds tuned on one
  sample video (see [Limitations](README.md#results--limitations) in
  the README). Testing against other Ghost Font samples and generalizing
  these is the most valuable kind of contribution right now.
- **New motion models**: the current pipeline assumes constant vertical
  velocity with two opposite directions. Support for horizontal motion,
  more than two dot populations, or non-constant speed would be a
  meaningful extension — happy to discuss design in an issue first.
- **Tests**: the project currently has no automated test suite. Unit
  tests around `ghost_decoder/motion.py` and `ghost_decoder/text_bands.py`
  (which are pure NumPy/OpenCV and don't require a real video) would be
  a great first contribution.

## Development setup

```bash
git clone https://github.com/<your-username>/ghost-font-decoder.git
cd ghost-font-decoder
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

You'll also need the Tesseract OCR engine installed system-wide (see
the README's [Requirements](README.md#requirements) section).

## Before submitting a pull request

- Keep functions focused and documented — this codebase favors small,
  well-named functions with docstrings over clever one-liners.
- If you change the alignment or classification logic, re-run against
  a real sample video and confirm the decoded output is still correct
  (there's no CI/test video checked in yet — see Tests above).
- Explain *why* in your PR description, not just *what*, especially for
  changes to the heuristic thresholds — they were tuned empirically and
  future readers will want to know the reasoning.

## Code style

- Python 3.10+, type hints on public functions.
- No hard dependency beyond `opencv-python`, `numpy`, and `pytesseract`
  — keep it that way unless there's a strong reason.

## Not sure where to start?

Open an issue describing what you'd like to work on before diving in,
especially for larger changes — happy to align on approach first.
