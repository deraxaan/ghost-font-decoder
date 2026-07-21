# Examples

Sample Ghost Font videos are **not** checked into this repository —
they're excluded by `.gitignore` (see the note there) because:

1. Video files are large and don't belong in git history.
2. The sample used during development was sourced from
   [mixfont.com/ghost-font](https://www.mixfont.com/ghost-font); it is
   not this project's asset to redistribute.

To try the decoder yourself:

1. Generate or obtain your own Ghost Font video (e.g. from the tool
   linked above).
2. Place it anywhere on disk and point the CLI at it:
   ```bash
   python decode.py /path/to/your-video.mp4 --auto-speed
   ```
3. If you'd like to contribute a small, clearly-licensed sample video
   for CI/testing purposes, please open an issue first to discuss.
