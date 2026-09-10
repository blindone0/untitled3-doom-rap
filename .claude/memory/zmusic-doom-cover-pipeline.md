---
name: zmusic-doom-cover-pipeline
description: "zmusic dir holds a \"song -> doom metal cover\" pipeline; v2 (live.py) uses real samples + neural amp captures after the user rejected the synth-sounding v1"
metadata: 
  node_type: memory
  type: project
  originSessionId: 0bbbf8df-472c-4eeb-bc39-0b004662565a
  modified: 2026-09-10T17:35:54.854Z
---

`C:\Users\igor\zmusic` started empty on 2026-09-10; the user asked for a doom metal cover of a YouTube song (TIK - "Олені").
Not a git repo; venv in `.venv` (torch-cpu, demucs, librosa, mido, tinysoundfont, gdown, matplotlib). Console is cp1251 -> run with `PYTHONIOENCODING=utf-8`.

Chain: `analyze.py` (tempo/beats/key/chords) -> `chords.py` (per-beat chords) -> Demucs vocal split -> `melody.py`
(CQT harmonic-summation melody tracker; pyin alone failed on layered vocals) -> render:
- v1 `cover.py` + `doomsynth.py`: pure numpy synth. User verdict: "sounds like Sega Mega Drive / 8-bit" — do not reuse for delivery.
- v2 `live.py`: sampled drums (`drumkit.py`, AVL Red Zeppelin SFZ in `assets/avl`), DI guitar samples (`gtrsampler.py`,
  Unreal Standard Guitar in `assets/stdguitar`; lowest real sample is E2, low roots are pitched down by resampling so riffs sit in
  drop-A register A1..G#2 as the user wanted; bass E1..D#2), quad-tracked with pedal captures (TS9 / Big Muff / Rat) in front of the amps,
  GuitarML Proteus LSTM amp captures (`ampsim.py`, `assets/proteus`) + cab IRs (`assets/ir`), soundfont bass (`sfsynth.py`,
  `assets/GeneralUser-GS.sf2`), Voxengo room IRs (`assets/voxengo`), mix tools in `mixfx.py`. Current output `out/doom_cover_heavy.{wav,mp3,mid}`
  (user asked for heavier + lower + louder bass on 2026-09-10; bass sits at -17 dB RMS target). Full render ~6 min.
  `plots2.py <wav> <png> [offset_s]` gives an envelope/spectrogram sanity check. Ask before multi-GB downloads.

Vocal stage (2026-09-10): `live.py --instrumental` renders `out/doom_cover_heavy_instrumental.*` (no lead melody in vocal sections) and
`out/bar_times.json`; `make_lyrics_doc.py` holds the original Ukrainian doom-rap lyrics («Олені біжать», themes: cruel love, dead faith)
and writes `out/lyrics_oleni_doom_rap.{md,txt}` with per-bar timestamps; `record.py` (sounddevice playrec, HATOR Signify mic is default input)
records takes into `vocals/`, `mixvocal.py` processes and mixes them -> `out/oleni_doom_rap.mp3`.
Later additions: `flow.py` (syllable grid -> `out/flow.json`, `out/flow_sheet.txt`, `out/guide_syllables.mp3` with a tick per syllable),
`karaoke.py` (Tk prompter with per-syllable highlighting + duplex recording), `studio.py` (Tk take manager: numbered takes per section,
`vocals/selection.json` chooses the used take, play bar, latency field, launches karaoke.py and mixvocal.py). Lyrics: explicit street version
(syrup/fights/sex), no religion, no family.

**Why:** the user wants "absolute live recordings" feel — groove and real post-production, not synth tones.
**How to apply:** for tweaks, edit `live.py` (BPM, patterns per `level`, TAKES for amp/IR choice, mix section); for a new song, replace
`src/original.wav` and rerun the chain. Related: [[user-igor-music]].
