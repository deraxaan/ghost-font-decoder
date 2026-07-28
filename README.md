# Ghost Font Decoder

**Recovering motion-hidden text from "Ghost Font" videos using per-frame motion classification instead of single-frame image analysis.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

---

## Motivation

[Ghost Font](https://www.mixfont.com/ghost-font) renders text as a field of
dots: the "message" dots move in one direction, a much larger set of
decoy dots move the opposite way, same speed. Any single frame looks like
noise. Watch it play, and a human eye picks the message out almost
effortlessly.

That gap — trivial for a human watching motion, hard for anything that
reasons about one frame at a time — is what this project is about:
recover the hidden text by tracking motion over time instead of looking
for a static pattern.

It's a small, self-contained computer vision experiment, not a
production OCR system. See [Results & Limitations](#results--limitations)
for an honest account of what does and doesn't work.



https://github.com/user-attachments/assets/51731bcd-cd7a-4b2f-b20d-5ca110ecb36d



## Background: why this breaks frame-based approaches

Ghost Font encodes a message as a dense field of small dots:

- A subset of dots — shaped like the message text — move at a constant
  velocity in one direction (e.g. upward).
- The remaining dots (the majority) move at the same speed in the
  **opposite** direction, acting as camouflage.
- In any single frame, both populations look identical: same dot size,
  same density, same distribution. There is no static visual boundary
  between "message" and "background."

This defeats approaches that treat video as a bag of independent images:

- **Frame-based OCR / image classifiers** see only noise-like dot fields;
  there's no letterform for a single frame to contain.
- **Naive frame-stacking or averaging** (e.g. "take the max/mean over all
  frames") doesn't help either, because both dot populations are equally
  present across the whole clip — averaging blurs the message and the
  decoys into the same uniform gray.

What *does* carry the signal is **relative motion between consecutive
frames** — the cue this project's pipeline is built around.

## What this project does

Given a Ghost Font video, this tool:

1. Estimates the dots' common speed (or accepts a known speed).
2. Classifies every dot in every frame as "message" or "background" by
   comparing it to the previous frame.
3. Filters out frames where that classification broke down.
4. Reconstructs a clean, static density image of the message from a
   short run of good frames.
5. Locates individual text lines within that image and runs OCR on each,
   voting across several reconstruction attempts for reliability.
6. Prints the decoded text to the console.

On the one sample video used during development, it reliably and
deterministically decodes the hidden message: `HELLO HUMAN`.

The interesting part wasn't recovering that specific phrase — it was
building a pipeline that consistently turns a temporal motion signal
into something an off-the-shelf OCR engine can actually read.

## How the approach works

The core insight: **a single frame carries no signal, but the
transition between two frames does.**

Each stage below is independent and separately tunable — a design choice
that also means individual pieces (e.g. the motion classifier, or the
OCR backend) could be swapped out without touching the rest of the
pipeline.

### 1. Motion classification, not motion detection

For consecutive frames `t-1` and `t`, the previous frame is shifted both
up and down by the estimated speed. Each dot present in frame `t` is
labeled "message" if it lines up better with the *up*-shifted version of
frame `t-1` than the *down*-shifted version — i.e., it more plausibly
moved up than down. This sidesteps the much harder problem of explicit
dot-to-dot correspondence (tracking): we never ask *which* dot in the
previous frame this one came from, only *which direction of travel
better explains it*.

### 2. Local, wraparound-safe alignment

Once dots are classified per-frame, message-only frames still need to be
stacked into one static image. The naive approach — shift every frame
back to a single global reference (e.g. frame 0) and average — fails
here: over a multi-second clip, the cumulative shift can exceed the
frame height, causing pixel shifts to wrap around the top/bottom edge
and corrupt the reconstruction. The fix is to align and stack only a
short local window of frames (aligned relative to *that window's own
start*, not the whole video), keeping the cumulative shift small enough
that wraparound never becomes visible.

### 3. Noisy-frame filtering

Occasionally the per-frame classification in step 1 breaks down and
almost every dot gets (incorrectly) labeled "message" for a frame —
visually a burst of solid noise. These frames have a message-dot count
far above the typical value, so they're detected and excluded by
comparing each frame's count against the running median.

### 4. Blur-before-threshold, not dilate-after-threshold

To turn a sparse field of message dots into OCR-readable strokes, a
Gaussian blur is applied to the **continuous density map** (how often
each pixel was classified as "message" across the stacked frames)
*before* thresholding. This was the single biggest lever for OCR
accuracy: blurring the continuous signal produces smooth, letterform-like
regions, whereas dilating an already-binarized mask tends to produce
blobby, disconnected shapes that OCR engines don't recognize as text.

### 5. Line splitting + multi-window voting

The row-wise density profile is smoothed and cut at valleys to isolate
individual text lines — OCR-ing one line at a time is much more reliable
than OCR-ing a multi-line block at once. This whole process (alignment →
reconstruction → line split → OCR) is repeated across several different
alignment windows spanning the clip, since some windows land on a
cleaner "phase" of the dot pattern than others. Results are tallied by a
simple vote, and the most consistent reading wins — and once one answer
has been independently confirmed by several windows, remaining windows
are skipped rather than scanned for no benefit.

## Core ideas

- Motion-based dot classification (no explicit dot tracking)
- Automatic dot-speed estimation, or a known speed via `--speed`
- Wraparound-safe local frame alignment
- Noisy/corrupted frame detection and filtering
- Density-map reconstruction, blurred before thresholding
- Automatic text-line segmentation
- Multi-window OCR voting for reliability, with early stopping once confident
- Small, dependency-light, pure OpenCV/NumPy/Tesseract pipeline

## Architecture / Pipeline

```
                       ┌─────────────────────┐
                       │   Input video (.mp4) │
                       └──────────┬───────────┘
                                  │
                     load_dot_frames()   [io_utils.py]
                     grayscale + threshold → binary dot masks
                                  │
                                  ▼
              ┌───────────────────────────────────┐
              │ auto_detect_speed() or --speed N   │  [motion.py]
              └──────────────────┬──────────────────┘
                                  │
                     classify_message_dots()        [motion.py]
                 per-frame up-shift vs down-shift comparison
                                  │
                                  ▼
                     filter_noisy_frames()           [motion.py]
                   drop frames with abnormal dot counts
                                  │
                                  ▼
              build_density_for_window() × N windows [motion.py]
             local wraparound-safe alignment + stacking
                                  │
                                  ▼
                     find_text_line_bands()        [text_bands.py]
                  smoothed row-density valley detection
                                  │
                                  ▼
                     ocr_density_region()                [ocr.py]
              Gaussian blur (continuous) → Otsu → Tesseract
                                  │
                                  ▼
                     decode_message()                    [ocr.py]
                vote across windows/lines → final string
                                  │
                                  ▼
                       "HELLO HUMAN"  (stdout)
                    + ghost_message_clean.png (preview)
```

## Requirements

- Python 3.10+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) engine
  installed **system-wide**. This is easy to miss: `pytesseract` (the
  Python package installed via `pip`) is only a *wrapper* around the
  actual engine, which is a separate program you install independently:
  - Ubuntu/Debian: `sudo apt install tesseract-ocr`
  - macOS: `brew install tesseract`
  - Windows: [UB-Mannheim installer](https://github.com/UB-Mannheim/tesseract/wiki)
    — installs by default to `C:\Program Files\Tesseract-OCR\`. The CLI
    auto-detects this location even if it isn't on your PATH; see
    [Troubleshooting](#troubleshooting) if it still isn't found.

Python dependencies are listed in [`requirements.txt`](requirements.txt):

```
opencv-python>=4.8
numpy>=1.24
pytesseract>=0.3.10
```

## Installation

```bash
git clone https://github.com/DeraXaan/ghost-font-decoder.git
cd ghost-font-decoder
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

## Usage

```bash
# If you know the dot speed (pixels/frame):
python decode.py path/to/video.mp4 --speed 4

# If you don't:
python decode.py path/to/video.mp4 --auto-speed

# Custom output path / alignment window size:
python decode.py path/to/video.mp4 --speed 4 --out result.png --window 40

# Print per-window band detection and OCR attempts (useful if decoding
# fails on a video and you need to see exactly where it's breaking):
python decode.py path/to/video.mp4 --auto-speed --debug
```

### CLI options

| Flag           | Description                                                 | Default                    |
|----------------|--------------------------------------------------------------|-----------------------------|
| `video`        | Path to the input video (positional, required)               | —                           |
| `--speed`      | Known dot speed in px/frame                                   | `None`                      |
| `--auto-speed` | Estimate dot speed automatically (searches 1–10 px/frame)    | off                         |
| `--thresh`     | Grayscale threshold for dot detection (0–255)                 | `127`                       |
| `--window`     | Local alignment window size, in frames                        | `30`                        |
| `--out`        | Path to save the human-viewable preview PNG                   | `ghost_message_clean.png`   |
| `--debug`      | Print per-window band detection and OCR attempts              | off                         |

### Library usage

The pipeline is also usable as a library rather than only via the CLI:

```python
from ghost_decoder import (
    load_dot_frames, auto_detect_speed, classify_message_dots,
    filter_noisy_frames, decode_message,
)

frames = load_dot_frames("video.mp4")
speed = auto_detect_speed(frames)
masks = classify_message_dots(frames, speed)
good_idx = filter_noisy_frames(masks)

print(decode_message(masks, good_idx, speed))
```

## Input / Output example

**Input:** a ~6 second, 1280×720 video where every frame looks like
uniform dot noise (see `docs/assets/input-frame-example.png`).

**Console output:**

```
Loading video...
  119 frames, 1280x720
Speed: 4.0 px/frame
Classifying message dots...
Filtering noisy/glitched frames...
  keeping 101/118 frames
Reconstructing a preview image...
Saved preview image to: ghost_message_clean.png
Scanning alignment windows and voting on the OCR result...

=== RESULT ===
Decoded message: HELLO HUMAN
```

**Output image:** a two-line reconstruction reading `HELLO` / `HUMAN`
(`docs/assets/reconstructed-output.png`).

## Demo

A full end-to-end run is embedded at the top of this README. For a
static side-by-side of a raw noisy frame vs. the final reconstruction:

- `docs/assets/input-frame-example.png` — single raw frame (looks like noise)
- `docs/assets/reconstructed-output.png` — final decoded image

## Troubleshooting

A few issues that are easy to hit on first setup, especially on Windows:

- **`ModuleNotFoundError: No module named 'cv2'`, even after `pip install
  -r requirements.txt` reports it's already satisfied.** This means
  `pip` and `python` are resolving to two different Python installs.
  Force them to match:
  ```
  python -m pip install -r requirements.txt
  ```
  If that doesn't fix it, check which `python` actually runs
  (`Get-Command python` on Windows, `which python` elsewhere) — on
  Windows specifically, watch for it resolving to the Microsoft Store
  stub (a path containing `WindowsApps`) instead of your real install.

- **"OCR could not confidently read the message" with no obvious
  reason, or every per-line OCR attempt in `--debug` output comes back
  as an empty string (`''`) rather than garbage text.** This almost
  always means the Tesseract *engine* binary itself isn't reachable —
  `pip install pytesseract` only installs the Python wrapper. The CLI
  checks for this directly at startup and will tell you if this is the
  cause; if it slips through, confirm with:
  ```
  tesseract --version
  ```
  and install the engine itself (see [Requirements](#requirements)) if
  that fails.

- **Multiple tools (Python version managers, other CLI installers,
  etc.) fighting over your PATH**, causing `python`/`pip`/`tesseract` to
  silently resolve to unexpected locations depending on install order.
  Calling the intended executable by its full path
  (e.g. `& "C:\Program Files\...\python.exe" decode.py ...`) sidesteps
  this entirely and is a reliable way to confirm whether PATH is the
  culprit.

## Results & Limitations

**What's been verified:** the full pipeline reliably and repeatedly
decodes the one sample video used during development
(`HELLO HUMAN`, two lines, ~4 px/frame vertical motion, 1280×720, 119
frames). Re-running the CLI produces the same result every time.

**What hasn't been verified / likely limitations:**

- **Single sample size.** All thresholds — noisy-frame tolerance
  (`1.15×` median), line-split valley ratio (`0.4`), minimum text
  length, blur sigmas — were tuned empirically against one video. They
  are reasonable starting points, not validated general-purpose
  defaults.
- **Motion model assumptions.** The pipeline assumes exactly two dot
  populations moving at constant, purely vertical, opposite-direction
  velocity. Diagonal motion, non-constant speed, more than two
  populations, or horizontal scrolling are not currently handled.
- **Scope.** This targets one specific encoding scheme, using its known
  geometry (constant opposite-direction velocity) as a strong prior. It
  isn't a general claim about defeating arbitrary motion-based text
  encodings or about OCR robustness in general — only about recovering
  *this* specific one.
- **OCR is still the bottleneck.** Even after reconstruction, Tesseract
  needs a fairly clean, high-contrast letterform to succeed. Thinner
  fonts, smaller dot pitch, or lower-contrast source videos may need
  different blur/threshold parameters than the defaults.
- **Runtime is dominated by OCR subprocess overhead, not the CV
  pipeline.** Each OCR attempt launches a separate Tesseract process;
  the motion classification and reconstruction stages are fast by
  comparison. Multi-window voting stops early once a result is
  confirmed by several independent windows, which keeps this bounded in
  practice, but a from-scratch reimplementation using an in-process OCR
  library (rather than a CLI-wrapped one) would likely be
  meaningfully faster.

The implementation intentionally favors interpretability over model
complexity: every stage is a plain, inspectable function, which makes it
a reasonable baseline to compare future, more complex approaches against.

## Future improvements

- [ ] Support horizontal and diagonal dot motion, not just vertical
- [ ] Validate against additional Ghost-Font-style samples and generalize
      the current heuristic thresholds (or make them auto-tuning)
- [ ] Replace the fixed 1–10 px/frame search range in `auto_detect_speed`
      with a coarse-to-fine search for wider speed coverage
- [ ] Add an automated test suite (see [CONTRIBUTING.md](CONTRIBUTING.md))
- [ ] Compare the current heuristic (up/down shift comparison) against a
      small learned classifier, and benchmark robustness of each
- [ ] Package as a proper CLI (`pip install`-able, console entry point)
- [ ] Evaluate an in-process OCR backend (e.g. a neural OCR library)
      alongside Tesseract, both for potential accuracy gains on noisier
      input and to remove the per-call subprocess overhead

## Project Structure

```
ghost-font-decoder/
├── decode.py                  # CLI entry point
├── ghost_decoder/
│   ├── __init__.py            # public API exports
│   ├── io_utils.py            # video loading → binary dot masks
│   ├── motion.py              # speed estimation, classification, alignment
│   ├── text_bands.py          # text-line segmentation
│   ├── ocr.py                 # blur→threshold→OCR, multi-window voting
│   └── cli.py                 # argument parsing, pipeline orchestration
├── examples/
│   └── README.md              # how to get/use sample videos (not checked in)
├── docs/
│   └── assets/                # demo images/GIFs (placeholders)
├── requirements.txt
├── LICENSE
├── CONTRIBUTING.md
└── README.md
```

## What I learned

The hard part of this project was never OCR itself — Tesseract handles
clean letterforms fine. The hard part was producing a representation
that Tesseract could call clean in the first place. Getting from "motion
signal buried across 100+ noisy frames" to "one crisp static image" took
more iteration than the recognition step that followed it.

That's roughly the lesson: for problems like this, how you represent the
data usually matters more than which model you throw at it afterward. A
better OCR engine wouldn't have saved a bad reconstruction; a better
reconstruction made an ordinary OCR engine sufficient.

A second, smaller lesson from getting this running cross-platform:
`pip install pytesseract` looks like a complete install but isn't — it's
a wrapper around a separate system binary. Distinguishing "the Python
package failed to import" from "the underlying engine can't be found"
as two different failure modes (rather than one vague error) made
diagnosing real-world setup issues far faster.

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for
setup instructions and good first-contribution ideas (the project
currently has no automated tests, and the heuristic thresholds have
only been validated on one sample video).

## License

Released under the [MIT License](LICENSE).
