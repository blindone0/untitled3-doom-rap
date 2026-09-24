# -*- coding: utf-8 -*-
"""Generated vocal for the active track — v3: an old-school TTS voice rapping with every syllable on the grid.

Voice: Windows' built-in "Microsoft Zira Desktop" (SAPI, offline) — the closest thing on this machine to the original
2011 Siri voice (same generation of concatenative synthesis; Apple's own voice is not available on Windows).
Rhythm: the syllables of every line are placed at EQUAL spacing — 16th notes (8th notes when the line has 8 syllables
or fewer), starting on beat 1 of the line's bar, no gaps inside the line ("ta-ta-ta-ta-ta"), the rest of the bar rests.
How: each line is synthesized as one phrase at a speaking rate fitted so its natural length ≈ syllables × grid step,
word timings come from the synthesizer, syllable boundaries inside a word are found at energy valleys, and every
syllable is warped with the WORLD vocoder to exactly one grid step. "(whisper)" lines are resynthesized unvoiced.
Hook doubles (_L/_R) are two renders (pitch +3 % on the right). Takes go to the track's takes folder (latency 0).

  python static_vocal.py [--engine sapi|edge] [--voice NAME] [--grid 16] [--flatten 0..1] [--fill 1.0]
  python mixvocal.py --voice clean --vocal-db 3      -> out/static.mp3
"""
import argparse
import asyncio
import hashlib
import json
import os
import re
import subprocess

import numpy as np
import pyworld as pw
import soundfile as sf
from scipy import signal
from scipy.ndimage import uniform_filter1d

from track import T

SR = 44100
FP = 5.0  # ms per WORLD frame
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--engine", default="sapi", choices=["sapi", "edge"])
ap.add_argument("--voice", default=None, help="sapi: 'Microsoft Zira Desktop' / 'Microsoft David Desktop'; edge: e.g. en-US-BrianNeural")
ap.add_argument("--grid", type=int, default=16, help="syllable grid for normal lines (16 = sixteenth notes)")
ap.add_argument("--flatten", type=float, default=0.0, help="0 = the voice's own intonation, 1 = monotone")
ap.add_argument("--fill", type=float, default=1.0, help="part of each syllable slot the syllable fills (1 = legato)")
ap.add_argument("--seed", type=int, default=1)
args = ap.parse_args()
SAPI = args.engine == "sapi"
VOICE = args.voice or ("Microsoft Zira Desktop" if SAPI else "en-US-BrianNeural")
PITCH0 = "+0%" if SAPI else "-10Hz"
RMIN, RMAX = (-10, 10) if SAPI else (-40, 50)

FLOW = sorted(json.load(open(T["flow"], encoding="utf-8")), key=lambda l: l["t"])
BT = json.load(open(T["bars"]))
BARS = {int(k): v for k, v in BT["bars"].items()}
TAKES = T["takes"]
CACHE = os.path.join(TAKES, "tts")
os.makedirs(CACHE, exist_ok=True)
rng = np.random.default_rng(args.seed)


# ---------------- TTS (cached) ----------------
def _load(wav):
    x, sr = sf.read(wav)
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SR:
        x = signal.resample_poly(x, SR, sr)
    return x.astype(np.float64)


def tts(text, rate, pitch):
    """-> (mono float64 @ SR, words [{t, d, w}]). rate: sapi -10..10 | edge percent. pitch: sapi '+3%' | edge '-10Hz'"""
    key = hashlib.md5(f"v3|{args.engine}|{VOICE}|{rate}|{pitch}|{text}".encode("utf-8")).hexdigest()[:16]
    wav, meta = os.path.join(CACHE, key + ".wav"), os.path.join(CACHE, key + ".json")
    if not (os.path.exists(wav) and os.path.exists(meta)):
        if SAPI:
            txt = os.path.join(CACHE, key + ".txt")
            open(txt, "w", encoding="utf-8").write(text)
            raw = os.path.join(CACHE, key + ".words.json")
            subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", os.path.join(HERE, "sapi_tts.ps1"),
                            "-TextFile", txt, "-Out", wav, "-Words", raw, "-Voice", VOICE, "-Rate", str(int(rate)), "-Pitch", pitch],
                           check=True, capture_output=True)
            ev = json.load(open(raw, encoding="utf-8"))
            ev = sorted(ev, key=lambda e: e["t"])
            x = _load(wav)
            end = len(x) / SR
            words = [{"t": e["t"], "d": (ev[i + 1]["t"] if i + 1 < len(ev) else end) - e["t"], "w": e["w"]} for i, e in enumerate(ev)]
            os.remove(txt)
            os.remove(raw)
        else:
            import edge_tts

            async def go():
                com = edge_tts.Communicate(text, VOICE, rate=f"{int(rate):+d}%", pitch=pitch, boundary="WordBoundary")
                audio, words = bytearray(), []
                async for ch in com.stream():
                    if ch["type"] == "audio":
                        audio += ch["data"]
                    elif ch["type"] == "WordBoundary":
                        words.append({"t": ch["offset"] / 1e7, "d": ch["duration"] / 1e7, "w": ch["text"]})
                return bytes(audio), words

            audio, words = asyncio.run(go())
            mp3 = os.path.join(CACHE, key + ".mp3")
            open(mp3, "wb").write(audio)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp3, "-ar", str(SR), "-ac", "1", wav], check=True)
            os.remove(mp3)
        json.dump({"text": text, "rate": rate, "pitch": pitch, "words": words}, open(meta, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    return _load(wav), json.load(open(meta, encoding="utf-8"))["words"]


def speech_span(x, words):
    env = uniform_filter1d(np.abs(x), int(0.005 * SR))
    thr = env.max() * 0.03
    idx = np.where(env > thr)[0]
    if len(idx) == 0:
        return 0.0, len(x) / SR
    a, b = idx[0] / SR, idx[-1] / SR
    if words:
        a = min(a, words[0]["t"])
    return a, b


def calibrate():
    """slope of log(speech length) per rate unit for this engine/voice (cached)"""
    path = os.path.join(CACHE, f"calib_{args.engine}_{re.sub(r'[^A-Za-z]', '', VOICE)}.json")
    if os.path.exists(path):
        return json.load(open(path))["b"]
    sent = "Static in the speakers, static in my head. Cold house, no lights, blue screen on the bed."
    rates = [-6, -3, 0, 3, 6] if SAPI else [-30, -15, 0, 15, 30]
    durs = []
    for r in rates:
        x, w = tts(sent, r, PITCH0)
        a, b = speech_span(x, w)
        durs.append(b - a)
    b = float(np.polyfit(rates, np.log(durs), 1)[0])
    json.dump({"b": b, "rates": rates, "durs": durs}, open(path, "w"))
    return b


B = calibrate()


# ---------------- words -> syllables ----------------
def norm_word(w):
    return re.sub(r"[^a-z0-9']", "", w.lower())


def match_words(flow_words, tts_words):
    fw = [norm_word(w) for w in flow_words]
    tw = [norm_word(w["w"]) for w in tts_words]
    spans = [None] * len(fw)
    i = j = 0
    while i < len(fw) and j < len(tw):
        a, b = tts_words[j]["t"], tts_words[j]["t"] + tts_words[j]["d"]
        if fw[i] == tw[j]:
            spans[i] = (a, b)
            i += 1
            j += 1
        elif i + 1 < len(fw) and fw[i] + fw[i + 1] == tw[j]:
            mid = a + (b - a) * max(1, len(fw[i])) / max(2, len(fw[i]) + len(fw[i + 1]))
            spans[i], spans[i + 1] = (a, mid), (mid, b)
            i += 2
            j += 1
        elif j + 1 < len(tw) and fw[i] == tw[j] + tw[j + 1]:
            spans[i] = (a, tts_words[j + 1]["t"] + tts_words[j + 1]["d"])
            i += 1
            j += 2
        else:
            spans[i] = (a, b)
            i += 1
            j += 1
    if any(s is None for s in spans):
        return None
    return spans


def word_spans(x, tts_words, flow_words, rate):
    """(start, end) seconds of every flow word in the TTS audio: synthesizer word marks (SAPI reports them in
    nominal-rate time, so they are scaled by the calibrated speed factor), snapped to the nearest energy valley,
    trimmed to the word's own speech"""
    if not tts_words:
        return None
    scale = float(np.exp(B * rate)) if SAPI else 1.0
    tw = [{"t": w["t"] * scale, "d": w["d"] * scale, "w": w["w"]} for w in tts_words]
    sa, sb = speech_span(x, None)
    tw[-1]["d"] = max(0.05, sb - tw[-1]["t"])
    spans_w = match_words(flow_words, tw)
    if spans_w is None:
        return None
    env = uniform_filter1d(np.abs(x), int(0.012 * SR))
    n = len(env)

    def snap(t):                                            # nearest energy valley within +-60 ms
        lo, hi = max(0, int((t - 0.06) * SR)), min(n, int((t + 0.06) * SR))
        return (lo + int(np.argmin(env[lo:hi]))) / SR if hi - lo > 4 else t

    starts = [max(sa, snap(a)) for (a, _) in spans_w]
    starts[0] = sa
    out = []
    for i, a in enumerate(starts):
        b = starts[i + 1] if i + 1 < len(starts) else sb
        seg = env[int(a * SR):int(b * SR)]
        if len(seg) > 10:                                   # drop the pause after the word, keep its consonants
            thr = seg.max() * 0.05
            idx = np.where(seg > thr)[0]
            if len(idx):
                a, b = a + max(0, idx[0] - int(0.01 * SR)) / SR, a + min(len(seg), idx[-1] + 1 + int(0.02 * SR)) / SR
        out.append((a, max(b, a + 0.04)))
    return out


# ---------------- WORLD warp onto the grid ----------------
def analyze(x):
    f0, t = pw.dio(x, SR, frame_period=FP, f0_floor=60.0, f0_ceil=500.0)
    f0 = pw.stonemask(x, f0, t, SR)
    return f0, pw.cheaptrick(x, f0, t, SR), pw.d4c(x, f0, t, SR)


def render_line(line, mode, pitch, rate_bias=0):
    """-> audio starting at the line's bar start: every syllable exactly one grid step long"""
    text = re.sub(r"\([^)]*\)", "", line["text"]).strip()
    syl = line["syl"]
    n = len(syl)
    if not text or n == 0:
        return None, 0
    flow_words = re.findall(r"[^\s-]+", text)
    order = sorted(set(s["w"] for s in syl))
    counts = [sum(1 for s in syl if s["w"] == w) for w in order]
    D = line["end"] - line["t"]
    step = D / 8 if n <= 8 else D / args.grid
    x0, w0 = tts(text, 0, pitch)
    a0, b0 = speech_span(x0, w0)
    r = int(np.clip(round(np.log(n * step / max(0.2, b0 - a0)) / B) + rate_bias, RMIN, RMAX))
    x, words = (x0, w0) if r == 0 else tts(text, r, pitch)
    # WORDS are warped uniformly (kept intact, so they stay intelligible) onto exactly as many grid steps as they have
    # syllables; the pulse at every word boundary is on the grid, inside a word the natural syllable proportions remain
    spans = word_spans(x, words, flow_words, r)
    units = list(zip(spans, counts)) if spans is not None and len(spans) == len(counts) else None
    if units is None:                                    # could not match the words: even split of the whole phrase
        a, b = speech_span(x, words)
        units = [((a + (b - a) * k / n, a + (b - a) * (k + 1) / n), 1) for k in range(n)]
    else:                                                # a word much longer than its syllables' slots gets one extra step
        steps = [cnt for (_, cnt) in units]
        for i, ((sa, sb), cnt) in enumerate(units):
            if (sb - sa) / (cnt * step) > 1.35 and sum(steps) < int(D / step) - 1:
                steps[i] += 1
        units = [(span, st) for (span, _), st in zip(units, steps)]
    f0, sp, apr = analyze(x)
    nfr = len(f0)
    if mode == "whisper":
        apr = np.ones_like(apr)
    if args.flatten > 0:
        v = f0 > 0
        if v.any():
            med = np.median(f0[v])
            f0[v] = med * (f0[v] / med) ** (1.0 - args.flatten)
    total = int(round((sum(st for _, st in units) * step + 0.25) * 1000 / FP))
    f0o = np.zeros(total)
    spo = np.full((total, sp.shape[1]), 1e-9)
    apo = np.ones((total, apr.shape[1]))
    k = 0
    for (sa, sb), cnt in units:
        fs = int(sa * 1000 / FP)
        fe = min(nfr, max(fs + 2, int(sb * 1000 / FP)))
        o0 = int(round(k * step * 1000 / FP))
        o1 = min(total, int(round((k + cnt - 1 + args.fill) * step * 1000 / FP)))
        k += cnt
        if o1 - o0 < 2 or fe - fs < 2:
            continue
        idx = np.clip(np.round(np.linspace(fs, fe - 1, o1 - o0)).astype(int), 0, nfr - 1)
        f0o[o0:o1] = f0[idx]
        spo[o0:o1] = sp[idx]
        apo[o0:o1] = apr[idx]
    y = pw.synthesize(np.ascontiguousarray(f0o), np.ascontiguousarray(spo), np.ascontiguousarray(np.clip(apo, 0, 1)), SR, frame_period=FP)
    if mode == "whisper":
        y = signal.sosfilt(signal.butter(2, 300, "high", fs=SR, output="sos"), y) * 0.5
    return y, r


def mode_of(text):
    if "(whisper)" in text:
        return "whisper"
    if "(spoken)" in text:
        return "spoken"
    return "rap"


# ---------------- sections -> takes ----------------
sections = [tuple(s) for s in T["sections"]]
selection, rates_used = {}, []
for name, a, b in sections:
    lines = [l for l in FLOW if BARS[a] - 0.06 <= l["t"] < BARS[b] - 0.06]
    if not lines:
        continue
    right = name.endswith("_R")
    pitch = ("+3%" if SAPI else "-8Hz") if right else PITCH0
    song_start = max(0.0, lines[0]["t"] - 0.5)
    length = int((lines[-1]["end"] + 1.5 - song_start) * SR)
    buf = np.zeros(length)
    for l in lines:
        y, r = render_line(l, mode_of(l["text"]), pitch, rate_bias=(1 if right else 0))
        if y is None:
            continue
        rates_used.append(r)
        # 35 ms early: a word starts with its consonants, the vowel (the beat you hear) lands on the grid
        s = int((l["t"] - 0.035 + (0.012 if right else 0.0) - song_start) * SR)
        e = min(length, s + len(y))
        if e > s:
            buf[s:e] += y[: e - s]
    buf = buf / (np.abs(buf).max() + 1e-9) * 0.6
    take = f"{name}_take1"
    sf.write(os.path.join(TAKES, take + ".wav"), buf.astype(np.float32), SR, subtype="FLOAT")
    json.dump({"song_start": song_start, "part_start": lines[0]["t"], "beat": T["beat"], "generated": f"{args.engine}:{VOICE}", "v": 3},
              open(os.path.join(TAKES, take + ".json"), "w"))
    selection[name] = take
    print(f"{name:8s} {len(lines):2d} lines  {song_start:6.1f}s  {length / SR:5.1f}s  -> {take}.wav")
json.dump(selection, open(os.path.join(TAKES, "selection.json"), "w"), indent=1)
json.dump({"latency_ms": 0.0}, open(os.path.join(TAKES, "latency.json"), "w"))   # generated takes are already on the grid
print(f"voice {VOICE} ({args.engine}), grid 1/{args.grid} (1/8 for lines of <= 8 syllables), rates used {min(rates_used)}..{max(rates_used)}, "
      f"calibration slope {B:.3f}/unit")
print(f"wrote {len(selection)} takes into {TAKES}/ -> now: python mixvocal.py --voice clean --vocal-db 3")
