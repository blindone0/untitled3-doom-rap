# -*- coding: utf-8 -*-
"""Generated vocal for the active track — v5: a machine reading, one syllable per sixteenth, nothing else.

Voice: Windows' built-in "Microsoft Zira Desktop" (SAPI, offline) — the closest thing on this machine to the original
2011 Siri voice (the same generation of concatenative synthesis; Apple's own voice is not available on Windows).

Every word is synthesized ON ITS OWN and then placed so that its vowels fall on the grid:
  * pass 1 speaks every distinct word at a base rate and measures it, pass 2 re-speaks it at a rate chosen so the word
    already comes out about the length of its syllables (both passes are one PowerShell call; the audio is cached).
    A word is only ever sped up, never slowed down: dragging "the" out to a whole sixteenth stops it being the word,
    and a word shorter than its slot simply leaves silence before the next one — a pulse, not a drone.
  * pitch is levelled by PLAYBACK SPEED, not by a vocoder. This is the single thing that decides whether the song is
    understandable: a WORLD analysis/resynthesis round trip on this voice turns "static" into "sad" (measured with
    speech recognition: 83 % of words read back, against 98 % for plain resampling), because it smears the stops.
    Resampling leaves every consonant exactly as the synthesizer made it, and the voice only varies about a semitone
    between words, so the result still sits on one note.
  * each word is placed so its first vowel lands exactly on its sixteenth, its consonants leading into the beat.
  * every word is level-matched and gets a ~12 ms articulation gap, so the reading is flat and words stay separate.
Takes go to the track's takes folder with latency 0, so mixvocal.py mixes them like recorded ones. One centred take
per section: a doubled robot voice combs against itself and the words stop being words.

Two more things that were measured rather than assumed: "the" spoken alone is heard as "they", and this voice accepts
SSML <phoneme> and then ignores it (identical duration and spectrum), so the weak form is coaxed out by spelling it
"thee"; and the vocal must be mixed DRY — echo and reverb smear a reading this tight.

  python static_vocal.py [--voice "Microsoft David Desktop"] [--f0 110] [--nuc 0.35]
  python mixvocal.py --voice clean --vocal-db 7 --carve-db 10 --duck-db 3 --carve-lo 300 --carve-hi 5000 --dry
"""
import argparse
import hashlib
import json
import os
import subprocess

import numpy as np
import pyworld as pw
import soundfile as sf
from scipy import signal
from scipy.ndimage import uniform_filter1d

from syllables import words_of_line
from track import T

SR = 44100
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--voice", default="Microsoft Zira Desktop")
ap.add_argument("--f0", type=float, default=164.81, help="the one note the whole vocal is spoken on, Hz (164.81 = E3, the key of the track and close to the voice's own pitch)")
ap.add_argument("--split", action="store_true", help="also cut multi-syllable words at the valley between their vowels and put every "
                "syllable on its own sixteenth (tighter to the grid; off by default because the plain version is the one that was approved)")
ap.add_argument("--nuc", type=float, default=0.35, help="where inside its sixteenth the vowel sits (0 = on the click)")
ap.add_argument("--base-rate", type=int, default=0, help="SAPI rate for the measuring pass")
ap.add_argument("--edge-ms", type=float, default=9.0, help="how gently each word is eased in, ms")
ap.add_argument("--gap-ms", type=float, default=22.0, help="how gently each word is eased out, ms — it rings into the next one")
ap.add_argument("--level", type=float, default=0.12, help="loudness every word is matched to (RMS), not its peak")
args = ap.parse_args()
VOICE = args.voice

FLOW = sorted(json.load(open(T["flow"], encoding="utf-8")), key=lambda l: l["t"])
BT = json.load(open(T["bars"]))
BARS = {int(k): v for k, v in BT["bars"].items()}
TAKES = T["takes"]
CACHE = os.path.join(TAKES, "words")
os.makedirs(CACHE, exist_ok=True)
EN = T.get("lang") == "en"


# ---------------- batch synthesis, cached per (voice, rate, word) ----------------
# Spoken alone, the synthesizer gives "the" a form that is heard as "they". SSML <phoneme> is accepted by this voice
# and then ignored (measured: identical duration and spectrum), so the weak form is coaxed out by spelling instead.
SPELL = {"the": "thee"}


def spoken(text):
    return SPELL.get(text.lower(), text)


def path_of(text, rate):
    key = hashlib.md5(f"v5|{VOICE}|{rate}|{spoken(text)}".encode("utf-8")).hexdigest()[:16]
    return os.path.join(CACHE, f"{key}.wav")


def synth(items):
    """items: [(text, rate)] — synthesize the ones that are not cached yet, in a single PowerShell call"""
    todo, seen = [], set()
    for text, rate in items:
        p = path_of(text, rate)
        if p in seen or os.path.exists(p):
            continue
        seen.add(p)
        todo.append({"text": spoken(text), "rate": int(rate), "out": p})
    if not todo:
        return 0
    spec = os.path.join(CACHE, "spec.json")
    json.dump(todo, open(spec, "w", encoding="utf-8"), ensure_ascii=False)
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", os.path.join(HERE, "sapi_batch.ps1"),
                    "-Spec", spec, "-Voice", VOICE], check=True, capture_output=True)
    os.remove(spec)
    missing = [t["text"] for t in todo if not os.path.exists(t["out"])]
    if missing:
        raise SystemExit(f"the synthesizer produced nothing for: {missing[:5]}")
    return len(todo)


def load(text, rate):
    x, sr = sf.read(path_of(text, rate))
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SR:
        x = signal.resample_poly(x, SR, sr)
    return x.astype(np.float64)


def speech_span(x):
    """(start, end) of the actual speech, cutting the silence the synthesizer leaves around a word"""
    env = uniform_filter1d(np.abs(x), int(0.004 * SR))
    if env.max() < 1e-6:
        return 0.0, len(x) / SR
    idx = np.where(env > env.max() * 0.02)[0]
    a = max(0, idx[0] - int(0.022 * SR))
    b = min(len(x), idx[-1] + int(0.02 * SR))
    return a / SR, b / SR


def nuclei_of(x, a, b, n):
    """time of each vowel peak inside the word (300-1000 Hz energy); evenly spread if they cannot be found"""
    even = [a + (b - a) * (k + 0.5) / n for k in range(n)]
    if n <= 1:
        return [(a + b) / 2]
    i0, i1 = int(a * SR), int(b * SR)
    if i1 - i0 < int(0.04 * SR) * n:
        return even
    sos = signal.butter(2, [300, 1000], "bandpass", fs=SR, output="sos")
    env = uniform_filter1d(np.abs(signal.sosfilt(sos, x[i0:i1])), int(0.012 * SR))
    if env.max() < 1e-6:
        return even
    pk, _ = signal.find_peaks(env, distance=int(0.04 * SR), height=env.max() * 0.12)
    if len(pk) < n:
        return even
    best = np.sort(pk[np.argsort(env[pk])[-n:]])
    return [(i0 + p) / SR for p in best]


# ---------------- every distinct word, spoken at the length of its syllables ----------------
step = (BARS[1] - BARS[0]) / 16.0          # one sixteenth
CALIB = os.path.join(CACHE, "calib.json")
vocab = {}                                  # spoken text -> syllable count
for l in FLOW:
    for w, c in words_of_line(l["text"], EN):
        vocab[w] = max(c, vocab.get(w, 0))
words = sorted(vocab)
n_new = synth([(w, args.base_rate) for w in words])
dur0 = {}
for w in words:
    a, b = speech_span(load(w, args.base_rate))
    dur0[w] = b - a
if os.path.exists(CALIB):
    B = json.load(open(CALIB))["b"]
else:                                       # how much one rate unit shortens the speech, measured on this voice
    probe = words[: min(24, len(words))]
    synth([(w, args.base_rate + 6) for w in probe])
    r = [np.log(dur0[w] / (lambda ab: ab[1] - ab[0])(speech_span(load(w, args.base_rate + 6)))) for w in probe]
    B = -float(np.median(r)) / 6.0
    json.dump({"b": B, "n": len(probe)}, open(CALIB, "w"))
rate = {}
for w in words:
    target = vocab[w] * step
    # only ever speak a word FASTER, never slower: dragging a short word like "the" out to a whole sixteenth is what
    # stops it sounding like the word. A word shorter than its slot simply leaves silence before the next one.
    rate[w] = int(np.clip(round(args.base_rate + np.log(target / dur0[w]) / B), args.base_rate, 10))
n_new += synth([(w, rate[w]) for w in words])
ratios = []
for w in words:
    a, b = speech_span(load(w, rate[w]))
    ratios.append(vocab[w] * step / (b - a))
print(f"{len(words)} distinct words, {n_new} newly synthesized; rates {min(rate.values())}..{max(rate.values())} "
      f"(1 unit = {(np.exp(B) - 1) * 100:+.0f}% length); residual stretch median {np.median(ratios):.2f}, "
      f"p10 {np.percentile(ratios, 10):.2f}, p90 {np.percentile(ratios, 90):.2f}")


# ---------------- render one word onto its steps ----------------
_cache = {}


def word_audio(w, cents=0.0):
    """(audio, head): the word on one constant note, its k-th vowel at (k + args.nuc) steps from the word's slot start;
    `head` is where the audio begins relative to that slot start (negative: the consonants lead into the beat)"""
    key = (w, round(cents, 1))
    if key in _cache:
        return _cache[key]
    c = vocab[w]
    x = load(w, rate[w])
    a, b = speech_span(x)
    y = x[int(a * SR):int(b * SR)].copy()
    # Pitch is levelled by PLAYBACK SPEED, not by a vocoder. A WORLD round trip on this voice turns "static" into
    # "sad" — it smears the stops — while resampling leaves every consonant exactly as the synthesizer made it.
    # The voice only varies ~1 semitone between words, so the shift is small and the words come out on one note.
    if args.f0 > 0 and len(y) > 400:
        f0, t = pw.dio(y, SR, frame_period=10.0, f0_floor=70.0, f0_ceil=500.0)
        f0 = pw.stonemask(y, f0, t, SR)
        v = f0[f0 > 0]
        med = float(np.median(v)) if len(v) else args.f0
        ratio = float(np.clip(args.f0 * 2 ** (cents / 1200.0) / med, 0.80, 1.35))
        if abs(ratio - 1.0) > 2e-3:
            n_out = max(2, int(round(len(y) / ratio)))
            y = np.interp(np.linspace(0, len(y) - 1, n_out), np.arange(len(y)), y)
    nuc = nuclei_of(y, 0.0, len(y) / SR, c)
    # A word of several syllables is CUT at the quiet point between its vowels and each piece is moved onto its own
    # sixteenth. Cutting at a valley keeps the consonants intact (unlike stretching them), and without this only the
    # first vowel of a word lands on the beat while the rest drift — measured: 50 ms median error against 22 ms.
    if c > 1 and args.split:
        sos = signal.butter(2, [300, 1000], "bandpass", fs=SR, output="sos")
        env = uniform_filter1d(np.abs(signal.sosfilt(sos, y)), int(0.012 * SR))
        cuts = []
        for k in range(c - 1):
            i0, i1 = int(nuc[k] * SR), int(nuc[k + 1] * SR)
            cuts.append(i0 + int(np.argmin(env[i0:i1])) if i1 - i0 > 4 else (i0 + i1) // 2)
        bounds = [0] + cuts + [len(y)]
        pieces = [(y[bounds[k]:bounds[k + 1]].copy(), nuc[k] * SR - bounds[k]) for k in range(c)]
    else:
        pieces = [(y, nuc[0] * SR)]
    starts = [(k + args.nuc) * step * SR - off for k, (_, off) in enumerate(pieces)]
    base = min(starts)
    head = base / SR                          # the word starts this far from its slot: its first vowel is on the beat
    fade = int(0.004 * SR)
    total = int(round(max(s - base + len(seg) for s, (seg, _) in zip(starts, pieces))))
    y = np.zeros(total)
    for j, (s, (seg, _)) in enumerate(zip(starts, pieces)):
        seg = seg.copy()
        if len(seg) > 2 * fade:               # soften both sides of a cut so it does not click
            if j > 0:
                seg[:fade] *= np.linspace(0, 1, fade)
            if j < len(pieces) - 1:
                seg[-fade:] *= np.linspace(1, 0, fade)
        i = int(round(s - base))
        y[i:i + len(seg)] += seg
    # Words are eased in and out with a raised cosine instead of a straight line, and the tail is left to ring into
    # the next word rather than being chopped: a hard edge on every word is what makes the reading sound spiky.
    e = int(args.edge_ms / 1000 * SR)
    if e > 1 and len(y) > 2 * e:
        y[:e] *= (1 - np.cos(np.linspace(0, np.pi, e))) / 2
    g = int(args.gap_ms / 1000 * SR)
    g = min(g, len(y) // 3)
    if g > 1:
        y[-g:] *= (1 + np.cos(np.linspace(0, np.pi, g))) / 2
    # Level by loudness, not by peak: matching peaks makes every burst of a consonant as loud as a whole vowel, which
    # is the other half of the sharpness. A safety ceiling keeps the loudest word from clipping the take.
    v = y[np.abs(y) > 1e-4]
    r = float(np.sqrt((v ** 2).mean())) if len(v) > 64 else 0.0
    if r > 1e-6:
        y = y * (args.level / r)
        pk = np.abs(y).max()
        if pk > 0.92:
            y *= 0.92 / pk
    _cache[key] = (y, head)
    return y, head


# ---------------- sections -> takes ----------------
sections = [tuple(s) for s in T["sections"]]
selection = {}
for name, a, b in sections:
    lines = [l for l in FLOW if BARS[a] - 0.06 <= l["t"] < BARS[b] - 0.06]
    if not lines:
        continue
    right = name.endswith("_R")
    cents = 25.0 if right else 0.0
    song_start = max(0.0, lines[0]["t"] - 0.5)
    length = int((lines[-1]["end"] + 1.5 - song_start) * SR)
    buf = np.zeros(length)
    for l in lines:
        pos = 0
        for w, c in words_of_line(l["text"], EN):
            # every VOWEL lands on its sixteenth; `head` is where the word's consonants start ahead of it
            y, head = word_audio(w, cents)
            s = int((l["t"] + (0.010 if right else 0.0) - song_start) * SR + round((pos * step + head) * SR))
            e = min(length, s + len(y))
            if e > s:
                buf[s:e] += y[: e - s]
            pos += c
    buf = buf / (np.abs(buf).max() + 1e-9) * 0.7
    take = f"{name}_take1"
    sf.write(os.path.join(TAKES, take + ".wav"), buf.astype(np.float32), SR, subtype="FLOAT")
    json.dump({"song_start": song_start, "part_start": lines[0]["t"], "beat": T["beat"], "generated": VOICE, "v": 4},
              open(os.path.join(TAKES, take + ".json"), "w"))
    selection[name] = take
    print(f"{name:8s} {len(lines):2d} lines  {song_start:6.1f}s  {length / SR:5.1f}s  -> {take}.wav")
json.dump(selection, open(os.path.join(TAKES, "selection.json"), "w"), indent=1)
json.dump({"latency_ms": 0.0}, open(os.path.join(TAKES, "latency.json"), "w"))   # generated takes are already on the grid
print(f"voice {VOICE}, one syllable every {step * 1000:.0f} ms, one note at {args.f0:.0f} Hz, vowels at {args.nuc:.2f} of a step")
print(f"wrote {len(selection)} takes into {TAKES}/ -> now: python mixvocal.py --voice clean --vocal-db 5")
