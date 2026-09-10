# Project notes for Claude Code

This repo is a song-production pipeline (doom metal cover → rap track) built entirely in Claude Code sessions.

- Run everything with the venv Python and `PYTHONIOENCODING=utf-8` (Windows cp1251 console).
- The chain: `analyze.py` → `chords.py` → Demucs → `melody.py` → `live.py [--instrumental]` → `flow.py` → `studio.py` / `karaoke.py` → `mixvocal.py`.
- `live.py` renders in ~6 min; `mixvocal.py` in ~2 min. `plots2.py <wav> <png>` is the visual sanity check (no listening possible in-session).
- Sound must be "live band", never synth: real samples, neural amp captures, humanized timing (see `.claude/memory/`).
- Assets (`assets/`), the source song (`src/`), stems, WAVs and vocal takes (`vocals/`) are not committed.
