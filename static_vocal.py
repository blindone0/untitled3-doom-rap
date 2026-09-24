# -*- coding: utf-8 -*-
"""Generated vocal for the active track — v6: the line is spoken once, then bent onto the grid.

Voice: Windows' built-in "Microsoft Zira Desktop" (SAPI, offline) — the closest thing on this machine to the original
2011 Siri voice (the same generation of concatenative synthesis; Apple's own voice is not available on Windows).

  --mode line (default): the WHOLE LINE is spoken as ONE utterance at a rate fitted to its bar, and then time-warped
    so that every syllable's vowel lands on its sixteenth. The warp is WSOLA — overlap-add of real waveform frames,
    each taken from wherever it continues the previous one best. Nothing is resynthesized and nothing is resampled,
    so the pitch, the timbre and the coarticulation of that one sentence survive intact. This is what stops the song
    sounding like words lifted from different recordings, which is exactly what speaking each word alone produces.
    The syllables are found in the audio itself (the vowel-band energy peaks): the lyrics are written to a fixed 16
    syllables a line, so their number is known, and the synthesizer's own word marks are not needed — measured, they
    run on a different clock than the audio (a mark at 4.22 s in a 3.42 s file). An anchor that would demand a
    violent stretch is dropped rather than forced, and the line rides its natural timing across that one span.
  --mode word: every word is synthesized on its own instead — dead uniform in tone and tighter to the grid, but each
    word carries its own intonation. Kept because it is the version that was approved once.

Neither mode ever puts the voice through a vocoder: a WORLD round trip on this voice turns "static" into "sad"
(measured with speech recognition: 83 % of words read back, against 98-100 % without it) because it smears the stops.
--f0 levels a take onto one note by playback speed, which is safe; 0 leaves the voice alone.

"the" is spelled "thee" for the synthesizer: unstressed it comes out as something heard as "they", and this voice
accepts SSML <phoneme> and then ignores it (measured: identical duration and spectrum).

Takes go to the track's takes folder with latency 0, so mixvocal.py mixes them like recorded ones.

  python static_vocal.py [--mode line|word] [--voice "Microsoft David Desktop"] [--f0 0]
  python mixvocal.py --voice clean --vocal-db 4 --carve-db 5 --duck-db 1.5 --carve-lo 500 --carve-hi 3500 --dry --soft 1 --glue-db -15
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

from syllables import syllabify_word, words_of_line
from track import T

SR = 44100
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="line", choices=["line", "word"], help="speak whole lines and cut them up, or speak each word alone")
ap.add_argument("--voice", default="Microsoft Zira Desktop")
ap.add_argument("--f0", type=float, default=0.0, help="level every line onto this note by playback speed, Hz (0 = leave the voice alone)")
ap.add_argument("--nuc", type=float, default=0.0, help="extra offset of the vowel inside its sixteenth (0 = right on the beat; the consonants before it lead in on their own)")
ap.add_argument("--base-rate", type=int, default=0, help="SAPI rate for the measuring pass")
ap.add_argument("--edge-ms", type=float, default=9.0, help="how gently each take is eased in, ms")
ap.add_argument("--join-ms", type=float, default=22.0, help="crossfade where two words overlap, ms")
ap.add_argument("--level", type=float, default=0.12, help="word mode: loudness every word is matched to (RMS)")
ap.add_argument("--split", action="store_true", help="word mode: also put every syllable of a word on its own sixteenth")
args = ap.parse_args()
VOICE = args.voice
LINE_MODE = args.mode == "line"

FLOW = sorted(json.load(open(T["flow"], encoding="utf-8")), key=lambda l: l["t"])
BT = json.load(open(T["bars"]))
BARS = {int(k): v for k, v in BT["bars"].items()}
TAKES = T["takes"]
CACHE = os.path.join(TAKES, "words")
os.makedirs(CACHE, exist_ok=True)
EN = T.get("lang") == "en"
step = (BARS[1] - BARS[0]) / 16.0          # one sixteenth


# ---------------- synthesis, cached per (voice, rate, text) ----------------
SPELL = {"the": "thee"}


def spoken(text):
    """what the synthesizer is actually given"""
    out = []
    for w in text.split(" "):
        k = w.lower().strip(",.!?'\"")
        out.append(SPELL[k] + w[len(w.rstrip(",.!?'\"")):] if k in SPELL else w)
    return " ".join(out)


def path_of(text, rate):
    key = hashlib.md5(f"v6|{VOICE}|{rate}|{spoken(text)}".encode("utf-8")).hexdigest()[:16]
    return os.path.join(CACHE, key + ".wav")


def marks_of(text, rate):
    return path_of(text, rate)[:-4] + ".words.json"


def synth(items, want_marks=False):
    """items: [(text, rate)] — synthesize whatever is not cached yet, in a single PowerShell call"""
    todo, seen = [], set()
    for text, rate in items:
        p = path_of(text, rate)
        if p in seen or (os.path.exists(p) and (not want_marks or os.path.exists(marks_of(text, rate)))):
            continue
        seen.add(p)
        item = {"text": spoken(text), "rate": int(rate), "out": p}
        if want_marks:
            item["words"] = marks_of(text, rate)
        todo.append(item)
    if not todo:
        return 0
    spec = os.path.join(CACHE, "spec.json")
    json.dump(todo, open(spec, "w", encoding="utf-8"), ensure_ascii=False)
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", os.path.join(HERE, "sapi_batch.ps1"),
                    "-Spec", spec, "-Voice", VOICE], check=True, capture_output=True)
    os.remove(spec)
    missing = [t["text"] for t in todo if not os.path.exists(t["out"])]
    if missing:
        raise SystemExit(f"the synthesizer produced nothing for: {missing[:3]}")
    return len(todo)


def load(text, rate):
    x, sr = sf.read(path_of(text, rate))
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SR:
        x = signal.resample_poly(x, SR, sr)
    return x.astype(np.float64)


def speech_span(x):
    """(start, end) in seconds of the actual speech, without the silence the synthesizer pads around it"""
    env = uniform_filter1d(np.abs(x), int(0.004 * SR))
    if env.max() < 1e-6:
        return 0.0, len(x) / SR
    idx = np.where(env > env.max() * 0.02)[0]
    return max(0, idx[0] - int(0.022 * SR)) / SR, min(len(x), idx[-1] + int(0.02 * SR)) / SR


def vowel_env(x):
    sos = signal.butter(2, [300, 1000], "bandpass", fs=SR, output="sos")
    return uniform_filter1d(np.abs(signal.sosfilt(sos, x)), int(0.012 * SR))


def nuclei_of(x, a, b, n):
    """The n vowel peaks inside (a, b). The detector is loosened step by step until it finds at least n candidates,
    then the n strongest are kept — the lyrics are written to a fixed syllable count, so n is known exactly and does
    not have to be guessed from the synthesizer's word marks (which, measured, run on a different clock)."""
    even = [a + (b - a) * (k + 0.5) / n for k in range(n)]
    i0, i1 = max(0, min(len(x), int(a * SR))), max(0, min(len(x), int(b * SR)))
    if n < 1 or i1 - i0 < int(0.03 * SR) * n:
        return even
    env = vowel_env(x[i0:i1])
    if env.max() < 1e-6:
        return even
    for dist in (0.090, 0.070, 0.055, 0.045, 0.035):
        for h in (0.30, 0.22, 0.16, 0.11, 0.07, 0.04):
            pk, _ = signal.find_peaks(env, distance=int(dist * SR), height=env.max() * h)
            if len(pk) >= n:
                best = np.sort(pk[np.argsort(env[pk])[-n:]])
                return [(i0 + p) / SR for p in best]
    return even


def calibrate(sample):
    """how much one rate unit shortens this voice, so the synthesizer's word marks can be put on the real clock"""
    path = os.path.join(CACHE, f"calib_{args.mode}.json")
    if os.path.exists(path):
        return json.load(open(path))["b"]
    probe = sample[: min(16, len(sample))]
    synth([(t, args.base_rate) for t in probe])
    synth([(t, args.base_rate + 6) for t in probe])
    r = []
    for t in probe:
        a0, b0 = speech_span(load(t, args.base_rate))
        a1, b1 = speech_span(load(t, args.base_rate + 6))
        r.append(np.log((b0 - a0) / max(1e-6, b1 - a1)))
    b = -float(np.median(r)) / 6.0
    json.dump({"b": b, "n": len(probe)}, open(path, "w"))
    return b


def fit_rate(text, target, lo, hi):
    a, b = speech_span(load(text, args.base_rate))
    return int(np.clip(round(args.base_rate + np.log(target / max(1e-6, b - a)) / B), lo, hi))


def pitch_level(y):
    """shift the whole take onto args.f0 by playback speed (keeps the contour, keeps every consonant intact)"""
    if args.f0 <= 0 or len(y) < 1000:
        return y
    f0, t = pw.dio(y, SR, frame_period=10.0, f0_floor=70.0, f0_ceil=500.0)
    f0 = pw.stonemask(y, f0, t, SR)
    v = f0[f0 > 0]
    if not len(v):
        return y
    ratio = float(np.clip(args.f0 / float(np.median(v)), 0.85, 1.25))
    if abs(ratio - 1) < 2e-3:
        return y
    n_out = max(2, int(round(len(y) / ratio)))
    return np.interp(np.linspace(0, len(y) - 1, n_out), np.arange(len(y)), y)


def wsola(x, src_of_out, n_out, frame_ms=46.0, seek_ms=11.0):
    """Move a recording in time without touching its pitch or its consonants: overlap-add of real waveform frames,
    each taken from wherever it continues the previous one best (WSOLA). Unlike a vocoder nothing is resynthesized,
    so the line still sounds like the one sentence it was spoken as. src_of_out maps output sample -> input sample."""
    N = int(frame_ms / 1000 * SR)
    H = N // 2
    win = np.hanning(N + 1)[:-1]              # periodic Hann: two of them at 50 % overlap sum to one
    seek = int(seek_ms / 1000 * SR)
    out = np.zeros(int(n_out) + N)
    ref = None
    for pos in range(0, int(n_out), H):
        nom = int(src_of_out(pos))
        if ref is None:
            best = nom
        else:
            lo, hi = max(0, nom - seek), min(len(x) - N - H, nom + seek)
            if hi <= lo:
                best = nom
            else:
                c = np.correlate(x[lo:hi + len(ref)], ref, mode="valid")
                best = lo + int(np.argmax(c))
        best = int(np.clip(best, 0, max(0, len(x) - N - H)))
        out[pos:pos + N] += x[best:best + N] * win
        ref = x[best + H:best + 2 * H]        # what the next frame has to continue from
    return out[: int(n_out)]


def place(buf, start, seg, join):
    """add seg at `start` samples, crossfading into whatever already ends there"""
    s = max(0, start)
    seg = seg[max(0, -start):]
    e = min(len(buf), s + len(seg))
    if e <= s:
        return
    seg = seg[: e - s]
    tail = np.where(np.abs(buf[:s]) > 1e-5)[0]
    over = int(min(join, (tail[-1] + 1 - s) if len(tail) and tail[-1] >= s else 0))
    if over > 1:                               # equal power, so the join does not dip or bump
        t = np.linspace(0, np.pi / 2, over)
        buf[s:s + over] *= np.cos(t)
        seg = seg.copy()
        seg[:over] *= np.sin(t)
    buf[s:e] += seg


# ---------------- build the takes ----------------
sections = [tuple(s) for s in T["sections"]]
join = int(args.join_ms / 1000 * SR)

if LINE_MODE:
    texts = sorted({l["text"] for l in FLOW})
    B = calibrate(texts)
    synth([(t, args.base_rate) for t in texts])
    rate = {t: fit_rate(t, 16 * step, -6, 10) for t in texts}
    n_new = synth([(t, rate[t]) for t in texts], want_marks=True)
    print(f"{len(texts)} lines spoken whole, {n_new} newly synthesized; rates "
          f"{min(rate.values())}..{max(rate.values())} (1 unit = {(np.exp(B) - 1) * 100:+.0f}% length)")
else:
    vocab = {}
    for l in FLOW:
        for w, c in words_of_line(l["text"], EN):
            vocab[w] = max(c, vocab.get(w, 0))
    words = sorted(vocab)
    B = calibrate(words)
    synth([(w, args.base_rate) for w in words])
    rate = {w: fit_rate(w, vocab[w] * step, args.base_rate, 10) for w in words}
    n_new = synth([(w, rate[w]) for w in words])
    print(f"{len(words)} distinct words, {n_new} newly synthesized; rates {min(rate.values())}..{max(rate.values())}")

_cache = {}
stats = {"stretch": [], "pinned": 0, "syl": 0}
STRETCH_OK = (0.55, 1.85)


def line_audio(text):
    """(audio, head): the line spoken once, cut at its word boundaries, every first vowel on its sixteenth"""
    if text in _cache:
        return _cache[text]
    wordlist = words_of_line(text, EN)
    n = sum(c for _, c in wordlist)
    x = pitch_level(load(text, rate[text]))
    a, b = speech_span(x)
    # Every syllable of the line, found in the audio itself, pinned to its own sixteenth. Anchors that would demand a
    # violent stretch are dropped instead of forced: the line then rides its natural timing across that one span.
    nuc = nuclei_of(x, a, b, n)
    src, dst = [], []
    for k, v in enumerate(nuc):
        tgt = (k + args.nuc) * step
        if src:
            f = (tgt - dst[-1]) / max(1e-3, v - src[-1])
            if not (STRETCH_OK[0] <= f <= STRETCH_OK[1]):
                continue
        src.append(v)
        dst.append(tgt)
    if len(src) < 2:
        return None, 0.0
    lead = min(0.20, src[0] - a)
    src = [src[0] - lead] + src + [min(b, src[-1] + 0.32)]
    dst = [dst[0] - lead] + dst + [dst[-1] + 0.32]
    head = dst[0]
    n_out = int(round((dst[-1] - dst[0]) * SR))
    si, di = np.array(src) * SR, (np.array(dst) - dst[0]) * SR
    stats["stretch"].extend((np.diff(di) / np.maximum(1.0, np.diff(si))).tolist())
    stats["pinned"] += len(src) - 2
    stats["syl"] += n
    y = wsola(x, lambda p: np.interp(p, di, si), n_out)
    e = int(args.edge_ms / 1000 * SR)
    if e > 1 and len(y) > 2 * e:
        y[:e] *= (1 - np.cos(np.linspace(0, np.pi, e))) / 2
        y[-e:] *= (1 + np.cos(np.linspace(0, np.pi, e))) / 2
    _cache[text] = (y, head)
    return y, head


def word_audio(w):
    """(audio, head): one word spoken alone, its vowels on the grid — the --mode word path"""
    if w in _cache:
        return _cache[w]
    c = vocab[w]
    y = pitch_level(load(w, rate[w]))
    a, b = speech_span(y)
    y = y[int(a * SR):int(b * SR)]
    nuc = nuclei_of(y, 0.0, len(y) / SR, c)
    if args.split and c > 1:
        env = vowel_env(y)
        bounds = [0] + [int(nuc[k] * SR) + int(np.argmin(env[int(nuc[k] * SR):int(nuc[k + 1] * SR)]))
                        for k in range(c - 1)] + [len(y)]
        pieces = [(y[bounds[k]:bounds[k + 1]], nuc[k] - bounds[k] / SR) for k in range(c)]
    else:
        pieces = [(y, nuc[0])]
    starts = [(k + args.nuc) * step - off for k, (_, off) in enumerate(pieces)]
    head = min(starts)
    total = int(round(max(s - head + len(seg) / SR for s, (seg, _) in zip(starts, pieces)) * SR))
    out = np.zeros(total)
    for s, (seg, _) in zip(starts, pieces):
        place(out, int(round((s - head) * SR)), seg, join)
    e = int(args.edge_ms / 1000 * SR)
    if e > 1 and len(out) > 2 * e:
        out[:e] *= (1 - np.cos(np.linspace(0, np.pi, e))) / 2
        out[-e:] *= (1 + np.cos(np.linspace(0, np.pi, e))) / 2
    v = out[np.abs(out) > 1e-4]
    r = float(np.sqrt((v ** 2).mean())) if len(v) > 64 else 0.0
    if r > 1e-6:
        out = out * (args.level / r)
        pk = np.abs(out).max()
        if pk > 0.92:
            out *= 0.92 / pk
    _cache[w] = (out, head)
    return out, head


selection = {}
for name, a, b in sections:
    lines = [l for l in FLOW if BARS[a] - 0.06 <= l["t"] < BARS[b] - 0.06]
    if not lines:
        continue
    song_start = max(0.0, lines[0]["t"] - 0.6)
    length = int((lines[-1]["end"] + 1.5 - song_start) * SR)
    buf = np.zeros(length)
    for l in lines:
        if LINE_MODE:
            y, head = line_audio(l["text"])
            if y is None:
                continue
            place(buf, int(round((l["t"] + head - song_start) * SR)), y, join)
        else:
            pos = 0
            for w, c in words_of_line(l["text"], EN):
                y, head = word_audio(w)
                place(buf, int(round((l["t"] - song_start + pos * step + head) * SR)), y, join)
                pos += c
    buf = buf / (np.abs(buf).max() + 1e-9) * 0.7
    take = f"{name}_take1"
    sf.write(os.path.join(TAKES, take + ".wav"), buf.astype(np.float32), SR, subtype="FLOAT")
    json.dump({"song_start": song_start, "part_start": lines[0]["t"], "beat": T["beat"], "generated": VOICE, "mode": args.mode},
              open(os.path.join(TAKES, take + ".json"), "w"))
    selection[name] = take
    print(f"{name:8s} {len(lines):2d} lines  {song_start:6.1f}s  {length / SR:5.1f}s  -> {take}.wav")
json.dump(selection, open(os.path.join(TAKES, "selection.json"), "w"), indent=1)
json.dump({"latency_ms": 0.0}, open(os.path.join(TAKES, "latency.json"), "w"))   # generated takes are already on the grid
if LINE_MODE and stats["stretch"]:
    st = np.array(stats["stretch"])
    print(f"{stats['pinned']} of {stats['syl']} syllables pinned to the grid; time scale median {np.median(st):.2f}, "
          f"p10 {np.percentile(st, 10):.2f}, p90 {np.percentile(st, 90):.2f}; pitch and consonants untouched")
print(f"voice {VOICE}, mode {args.mode}, one syllable every {step * 1000:.0f} ms, vowels at {args.nuc:.2f} of a step")
print(f"wrote {len(selection)} takes into {TAKES}/")
