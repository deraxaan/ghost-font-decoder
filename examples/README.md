# Examples

Sample Ghost Font videos are not checked into this repository (to keep
it small). To get a sample video to test against:

1. Generate one at [mixfont.com/ghost-font](https://www.mixfont.com/ghost-font)
   using a short phrase (e.g. "HELLO HUMAN").
2. Download the generated clip and place it in this folder, e.g.
   `examples/sample.mp4`.
3. Run the decoder against it:
   ```
   python decode.py examples/sample.mp4 --auto-speed
   ```

If you contribute a real sample video and its expected decoded output,
please open a PR -- see [CONTRIBUTING.md](../CONTRIBUTING.md).
