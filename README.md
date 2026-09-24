# UNTITLED3' — doom-rap cover pipeline

A doom metal cover of TIK's «Олені» turned into a rap track with original Ukrainian lyrics, made end to end in
[Claude Code](https://claude.com/claude-code). **Produced by Claude Code, performed by Igor.**

Listen: **https://youtu.be/yEsHZU2ehVs** · `out/oleni_doom_rap_lovell.mp3` (final) · `out/doom_cover_heavy_instrumental.mp3` (beat only).

## How it was made

1. **Analysis** — `analyze.py`, `chords.py`: tempo, beat grid, key and chord chart of the original (136 BPM, A minor).
   Vocals were separated with Demucs and the sung melody traced by `melody.py` (CQT harmonic-summation tracker).
2. **Arrangement & rendering** — `live.py` (+ `drumkit.py`, `gtrsampler.py`, `sfsynth.py`, `ampsim.py`, `mixfx.py`):
   half-time 70 BPM doom arrangement in drop A. Sampled acoustic drums with humanized timing, DI guitar samples through
   neural amp captures (Tube Screamer / Big Muff / Rat into a Peavey 6505+ and a Mesa Rectifier) and real 4x12 cabinet
   impulse responses, sampled bass, tempo drift, room mics, bus compression, mastering.
   `cover.py` + `doomsynth.py` are the first, fully synthesized attempt (kept for history).
3. **Lyrics** — `make_lyrics_doc.py` holds the original text and stamps every bar with its time; `flow.py` places every
   syllable on the 16th-note grid (flow sheet + a guide audio with a tick per syllable).
4. **Recording** — `karaoke.py`: karaoke-style prompter with per-syllable highlighting that plays the beat and records the
   mic (sample-locked); `studio.py`: take manager (every retry kept, pick the best take, play bar, latency, export).
5. **Vocal mix & master** — `mixvocal.py`: declipping, gate, EQ, de-esser, compression, WORLD-vocoder auto-tune to A minor
   with a 2-semitone drop (`--voice lovell`), slap/dark echoes, room + hall, beat ducking, bus compression, limiter.

## Reproduce

```
python -m venv .venv && .venv\Scripts\pip install numpy scipy soundfile librosa mido tinysoundfont sounddevice pyworld matplotlib
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu   # for the amp captures (and demucs for stems)
```

Assets are not in the repo (size / licences). Put them under `assets/`:

- AVL Drumkits 1.0 (GPL) — http://www.bandshed.net/sounds/sfz/AVL_Drumkits_1.0.zip → `assets/avl/AVL_Drumkits_1.0/`
- Unreal Instruments Standard Guitar (free) — https://sfzinstruments.github.io/guitars/standard_guitar/ → `assets/stdguitar/UI_Standard_Guitar/`
- GuitarML ToneLibrary Proteus models — https://github.com/GuitarML/ToneLibrary → `assets/proteus/*.json`
- Cabinet IRs — https://github.com/fnpngn/IR → `assets/ir/`
- Voxengo free impulses — https://www.voxengo.com/impulses/ → `assets/voxengo/`
- GeneralUser GS soundfont — https://github.com/mrbumpy409/GeneralUser-GS → `assets/GeneralUser-GS.sf2`

Then: `python live.py --instrumental` → `python flow.py` → `python studio.py` → `python mixvocal.py`.
The console is cp1251 on this machine, so run scripts with `PYTHONIOENCODING=utf-8`.

## Track 2: STATIC (BONES-style, English)

An original lo-fi trap instrumental in the style of BONES — E minor, 68 BPM half-time, all-minor loop (Em–Am–Bm–Am).
The "sample" is a ghostly guitar swell pad, a sparse low pulse and a slow mournful top line (sampled Standard Guitar
through a Tube Screamer capture and a Fender-style cab, faint piano double), played 3 semitones up and slowed down like
a pitched-down record, then wow/flutter, band-limiting, tape saturation and vinyl crackle. Drums: real snare + hand clap
with transient shaping, swung hats, reversed crash swells, a kick-sample thump under a tuned 808 with a distorted
presence layer. Original English lyrics: numb, deadpan, a cold ex, fake friends, TV static.

- `static_beat.py` → `out/static_instrumental.{wav,mp3}` + `out/static_bar_times.json` (~1.5 min);
  `--variant b` → `out/static_instrumental_b.{wav,mp3}`: sparser 8th-note hats, darker loop, longer 808, more room — same bar grid.
- `static_lyrics.py` → `out/lyrics_static.{md,txt}` (lyrics with a timestamp per bar and delivery notes per section).
- `track.py static` switches every vocal tool (`flow.py`, `karaoke.py`, `studio.py`, `mixvocal.py`, `record.py`) to this
  track: its beat, bar grid, English syllable splitting, takes folder `vocals_static/`, the E minor key for auto-tune if it is
  used, final mix
  `out/static.mp3`. `track.py oleni` switches back; without `out/track.json` the original track is used.

```
python track.py static
python static_beat.py && python static_lyrics.py && python flow.py
python studio.py            # record intro / hook1 / verse1 ... then export (mixvocal.py)
```

The lyrics are written in a strict metre: **every line is exactly 16 syllables**, one per sixteenth note, for the whole
song, so the reading is one even pulse with no gaps (`static_lyrics.py` refuses to write a sheet that breaks the metre;
the syllable splitter lives in `syllables.py` and is shared with `flow.py`).

No singer available? `static_vocal.py` has a machine read it, with Windows' built-in "Microsoft Zira Desktop" (SAPI,
offline; the closest thing here to the original 2011 Siri voice; helper `sapi_batch.ps1`). Every
distinct WORD is synthesized on its own at a speaking rate chosen so it already comes out about the length of its
syllables, then placed so its first vowel lands on its sixteenth. Pitch is levelled by playback speed rather than by a
vocoder — that one choice is what makes the words survive (a WORLD round trip on this voice turns "static" into "sad":
83 % of words read back by speech recognition, against 98 % for resampling). `mixvocal.py` ducks the beat's own vocal
band under the voice (`--carve-db`), and the vocal is mixed dry, because echo smears a reading this tight.

```
python static_vocal.py
python mixvocal.py --voice clean --vocal-db 7 --carve-db 10 --duck-db 3 --carve-lo 300 --carve-hi 5000 --dry
python static_vocal.py --voice "Microsoft David Desktop" --f0 110    # male robot instead
```

## Credits

Composition of the source song: TIK — «Олені» (this is a cover / rearrangement; lyrics here are original).
Production, arrangement, code and lyrics: Claude Code (Claude Fable 5.1). Voice: Igor.
