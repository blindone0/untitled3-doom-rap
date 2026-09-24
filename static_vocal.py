# -*- coding: utf-8 -*-
"""Generated vocal for the active track (for when nobody can record) — v2: whole natural phrases.

v1 cut the TTS into words and warped every word onto the 16th grid with a vocoder: intelligible (whisper heard 90 % of
the words) but staccato, a robot reading a list. v2 keeps each line as one continuous natural phrase:
  * every lyric line is spoken by a neural TTS voice (edge-tts, en-US-BrianNeural, low and flat),
  * it is rendered once at a base speaking rate, its natural length is measured, and it is re-rendered at the rate that
    makes the phrase fill ~80 % of the time until the next line (no time-stretching, no vocoder — the TTS just talks
    faster or slower, rate is rounded to 5 %),
  * the phrase starts exactly on its first flow syllable; the pitch is lowered by the TTS itself,
  * hook doubles (_L/_R) are two different renders (slightly different rate and pitch) — a real double-track,
  * "(whisper)" lines are the only ones resynthesized (WORLD, unvoiced); "(spoken)" lines are plain speech.
Takes go to the track's takes folder like recorded ones (latency 0). Then: python mixvocal.py --voice clean --vocal-db 3

  python static_vocal.py [--voice en-US-BrianNeural] [--pitch -10Hz] [--fill 0.8]     (TTS renders are cached in <takes>/tts/)
Needs internet for the TTS (edge-tts, free, no key). Run with the venv python from the project folder.
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

from track import T

SR = 44100
ap = argparse.ArgumentParser()
ap.add_argument("--voice", default="en-US-BrianNeural")
ap.add_argument("--pitch", default="-10Hz", help="TTS base pitch shift")
ap.add_argument("--fill", type=float, default=0.80, help="fraction of the time until the next line a phrase should take")
ap.add_argument("--base-rate", type=int, default=-5, help="first render rate in %% (used to measure the natural length)")
ap.add_argument("--seed", type=int, default=1)
args = ap.parse_args()

FLOW = sorted(json.load(open(T["flow"], encoding="utf-8")), key=lambda l: l["t"])
BT = json.load(open(T["bars"]))
BARS = {int(k): v for k, v in BT["bars"].items()}
TAKES = T["takes"]
CACHE = os.path.join(TAKES, "tts")
os.makedirs(CACHE, exist_ok=True)
rng = np.random.default_rng(args.seed)


# ---------------- TTS (cached) ----------------
def tts(text, rate_pct, pitch):
    """-> (mono float64 @ SR, words [{t, d, w}]) for the phrase at the given speaking rate"""
    rate = f"{rate_pct:+d}%"
    key = hashlib.md5(f"v2|{args.voice}|{rate}|{pitch}|{text}".encode("utf-8")).hexdigest()[:16]
    wav, meta = os.path.join(CACHE, key + ".wav"), os.path.join(CACHE, key + ".json")
    if not (os.path.exists(wav) and os.path.exists(meta)):
        import edge_tts

        async def go():
            com = edge_tts.Communicate(text, args.voice, rate=rate, pitch=pitch, boundary="WordBoundary")
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
    x, sr = sf.read(wav)
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SR:
        x = signal.resample_poly(x, SR, sr)
    return x.astype(np.float64), json.load(open(meta, encoding="utf-8"))["words"]


def speech_span(x, words):
    """(start, end) seconds of the actual speech in a TTS clip (word boundaries, checked against the energy)"""
    env = np.abs(x)
    thr = env.max() * 0.03
    idx = np.where(env > thr)[0]
    if len(idx) == 0:
        return 0.0, len(x) / SR
    a, b = idx[0] / SR, idx[-1] / SR
    if words:
        a = min(a, words[0]["t"])
        b = max(b, words[-1]["t"] + words[-1]["d"])
    return a, b


def fit_rate(text, slot, pitch, rate0):
    """render at rate0, measure, re-render at the rate that fills args.fill of the slot"""
    x0, w0 = tts(text, rate0, pitch)
    a, b = speech_span(x0, w0)
    natural = max(0.2, b - a)
    target = slot * args.fill
    factor = natural / target                     # >1 = must talk faster
    pct = int(round(((1 + rate0 / 100) * factor - 1) * 100 / 5.0)) * 5
    pct = int(np.clip(pct, -30, 45))
    if pct == rate0:
        return x0, w0, pct
    x, w = tts(text, pct, pitch)
    return x, w, pct


def whisper_it(x):
    """unvoiced WORLD resynthesis = whispered"""
    f0, t = pw.dio(x, SR, frame_period=5.0)
    f0 = pw.stonemask(x, f0, t, SR)
    sp = pw.cheaptrick(x, f0, t, SR)
    apr = np.ones_like(pw.d4c(x, f0, t, SR))
    y = pw.synthesize(np.ascontiguousarray(f0), np.ascontiguousarray(sp), np.ascontiguousarray(apr), SR, frame_period=5.0)
    y = signal.sosfilt(signal.butter(2, 300, "high", fs=SR, output="sos"), y)
    return y / (np.abs(y).max() + 1e-9) * 0.5


def mode_of(text):
    if "(whisper)" in text:
        return "whisper"
    if "(spoken)" in text:
        return "spoken"
    return "rap"


# ---------------- lines -> phrases -> takes ----------------
sections = [tuple(s) for s in T["sections"]]
selection = {}
report = []
for name, a, b in sections:
    lines = [l for l in FLOW if BARS[a] - 0.06 <= l["t"] < BARS[b] - 0.06]
    if not lines:
        continue
    double = name.endswith(("_L", "_R"))
    right = name.endswith("_R")
    # per-take variation for doubles: a different render (rate +3 %, pitch +2 Hz) and ~15 ms later on the right side
    d_rate = 3 if right else 0
    pitch = args.pitch if not right else f"{int(args.pitch[:-2]) + 2:+d}Hz"
    song_start = max(0.0, lines[0]["t"] - 0.5)
    length = int((lines[-1]["end"] + 1.5 - song_start) * SR)
    buf = np.zeros(length)
    for i, l in enumerate(lines):
        text = re.sub(r"\([^)]*\)", "", l["text"]).strip()
        if not text:
            continue
        m = mode_of(l["text"])
        start = l["syl"][0]["t"] if l["syl"] else l["t"]
        nxt = lines[i + 1]["syl"][0]["t"] if i + 1 < len(lines) and lines[i + 1]["syl"] else l["end"]
        slot = max(0.8, nxt - start)
        if m == "spoken":
            x, words, pct = tts(text, args.base_rate - 5 + d_rate, pitch) + (args.base_rate - 5,)
        else:
            x, words, pct = fit_rate(text, slot, pitch, args.base_rate + d_rate)
        sa, sb = speech_span(x, words)
        y = x[int(sa * SR):int(sb * SR) + int(0.15 * SR)]
        if m == "whisper":
            y = whisper_it(y)
        # never run into the next line: fade out 60 ms before it starts
        limit = int((slot - 0.06) * SR)
        if len(y) > limit > 1000:
            y = y[:limit].copy()
            y[-int(0.05 * SR):] *= np.linspace(1, 0, int(0.05 * SR))
        off = 0.015 if right else 0.0
        s = int((start + off - song_start) * SR) + (int(rng.normal(0, 0.006) * SR) if double else 0)
        e = min(length, s + len(y))
        if e > s:
            buf[s:e] += y[: e - s]
        report.append((name, round(start, 1), pct, round(sb - sa, 2), round(slot, 2), text))
    buf = buf / (np.abs(buf).max() + 1e-9) * 0.6
    take = f"{name}_take1"
    sf.write(os.path.join(TAKES, take + ".wav"), buf.astype(np.float32), SR, subtype="FLOAT")
    json.dump({"song_start": song_start, "part_start": lines[0]["t"], "beat": T["beat"], "generated": args.voice, "v": 2},
              open(os.path.join(TAKES, take + ".json"), "w"))
    selection[name] = take
    print(f"{name:8s} {len(lines):2d} lines  {song_start:6.1f}s  {length / SR:5.1f}s  -> {take}.wav")
json.dump(selection, open(os.path.join(TAKES, "selection.json"), "w"), indent=1)
json.dump({"latency_ms": 0.0}, open(os.path.join(TAKES, "latency.json"), "w"))   # generated takes are already on the grid
rates = [r[2] for r in report if r[0] in ("verse1", "verse2", "hook1_L")]
print(f"speaking rates used: min {min(rates):+d}% median {int(np.median(rates)):+d}% max {max(rates):+d}%  (phrase / slot, verse 1):")
for r in report:
    if r[0] == "verse1":
        print(f"   [{r[1]:6.1f}s] rate {r[2]:+4d}%  {r[3]:4.2f}s of {r[4]:4.2f}s  {r[5]}")
print(f"wrote {len(selection)} takes into {TAKES}/ -> now: python mixvocal.py --voice clean --vocal-db 3")
