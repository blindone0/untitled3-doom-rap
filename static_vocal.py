# -*- coding: utf-8 -*-
"""Generated vocal for the active track (for when nobody can record): every lyric line is spoken by a neural TTS voice
(edge-tts, en-US-BrianNeural — low and flat), then each WORD is time-warped with the WORLD vocoder onto the flow grid
(the same syllable slots the karaoke prompter shows, from the track's flow json), the pitch contour is flattened
(deadpan) and lowered, "(whisper)" lines are resynthesized without voicing, "(spoken)" lines keep their natural prosody.
Hooks are rendered twice with small random differences (the _L / _R doubles). The results are written as takes into the
track's takes folder (<section>_take1.wav + .json, selection.json, latency 0), so mixvocal.py mixes them like real takes.

  python static_vocal.py                 (TTS audio is cached in <takes>/tts/, so re-runs are fast)
  python mixvocal.py --voice lovell --tune 0.5   -> out/static.mp3
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
FP = 5.0  # ms per WORLD frame
ap = argparse.ArgumentParser()
ap.add_argument("--voice", default="en-US-BrianNeural")
ap.add_argument("--rate", default="-12%", help="TTS speaking rate")
ap.add_argument("--pitch", default="-6Hz", help="TTS base pitch")
ap.add_argument("--flatten", type=float, default=0.55, help="0 = natural intonation, 1 = fully monotone (deadpan)")
ap.add_argument("--drop", type=float, default=1.0, help="semitones to lower the voice here (mixvocal --drop adds more)")
ap.add_argument("--seed", type=int, default=1)
args = ap.parse_args()

FLOW = sorted(json.load(open(T["flow"], encoding="utf-8")), key=lambda l: l["t"])
BT = json.load(open(T["bars"]))
BARS = {int(k): v for k, v in BT["bars"].items()}
TAKES = T["takes"]
CACHE = os.path.join(TAKES, "tts")
os.makedirs(CACHE, exist_ok=True)
rng = np.random.default_rng(args.seed)


# ---------------- TTS with word boundaries (cached) ----------------
def tts(text):
    key = hashlib.md5(f"{args.voice}|{args.rate}|{args.pitch}|{text}".encode("utf-8")).hexdigest()[:16]
    wav, meta = os.path.join(CACHE, key + ".wav"), os.path.join(CACHE, key + ".json")
    if not (os.path.exists(wav) and os.path.exists(meta)):
        import edge_tts

        async def go():
            com = edge_tts.Communicate(text, args.voice, rate=args.rate, pitch=args.pitch, boundary="WordBoundary")
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
        json.dump({"text": text, "words": words}, open(meta, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    x, sr = sf.read(wav)
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SR:
        x = signal.resample_poly(x, SR, sr)
    return x.astype(np.float64), json.load(open(meta, encoding="utf-8"))["words"]


def norm_word(w):
    return re.sub(r"[^a-z0-9']", "", w.lower())


def match_words(flow_words, tts_words):
    """greedy alignment flow word index -> (tts start, tts end) in seconds; handles hyphen/contraction splits"""
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
        elif i + 1 < len(fw) and fw[i] + fw[i + 1] == tw[j]:      # "Thrift-store" is one TTS word, two flow words
            mid = a + (b - a) * max(1, len(fw[i])) / max(2, len(fw[i]) + len(fw[i + 1]))
            spans[i], spans[i + 1] = (a, mid), (mid, b)
            i += 2
            j += 1
        elif j + 1 < len(tw) and fw[i] == tw[j] + tw[j + 1]:      # one flow word, two TTS tokens
            spans[i] = (a, tts_words[j + 1]["t"] + tts_words[j + 1]["d"])
            i += 1
            j += 2
        else:                                                    # give up on this pair, keep going 1:1
            spans[i] = (a, b)
            i += 1
            j += 1
    # anything unmatched: spread over the remaining audio
    for k in range(len(spans)):
        if spans[k] is None:
            prev_end = spans[k - 1][1] if k > 0 and spans[k - 1] else 0.0
            spans[k] = (prev_end, prev_end + 0.25)
    return spans


# ---------------- WORLD warping ----------------
def analyze(x):
    f0, t = pw.dio(x, SR, frame_period=FP, f0_floor=60.0, f0_ceil=400.0)
    f0 = pw.stonemask(x, f0, t, SR)
    sp = pw.cheaptrick(x, f0, t, SR)
    apr = pw.d4c(x, f0, t, SR)
    return f0, sp, apr


def flatten_f0(f0, amount, drop_st):
    f0 = f0.copy()
    v = f0 > 0
    if v.any():
        med = np.median(f0[v])
        f0[v] = med * (f0[v] / med) ** (1.0 - amount)
        f0[v] *= 2 ** (-drop_st / 12.0)
    return f0


def render_line(line, mode, jitter_ms=0.0, cents=0.0, warp_jit=0.0):
    """-> (audio starting at line['t'], seconds) : words warped onto their flow slots"""
    text = re.sub(r"\([^)]*\)", "", line["text"]).strip()
    words_flow = re.findall(r"[^\s-]+", text)
    x, tts_words = tts(text)
    if not tts_words or len(x) < SR // 10:
        return np.zeros(int(0.1 * SR))
    f0, sp, apr = analyze(x)
    nfr = len(f0)
    if mode == "whisper":
        apr = np.ones_like(apr)                      # noise excitation only = whisper
    elif mode == "rap":
        f0 = flatten_f0(f0, args.flatten, args.drop)
    else:                                            # spoken: natural, just a little lower
        f0 = flatten_f0(f0, 0.15, args.drop)
    if cents:
        f0 = f0 * 2 ** (cents / 1200.0)
    # target word slots from the flow (first syllable of each word)
    syl = line["syl"]
    starts = {}
    for s in syl:
        starts.setdefault(s["w"], s["t"])
    order = sorted(starts)
    D = line["end"] - line["t"]
    tgt = []
    for n, w in enumerate(order):
        a = starts[w]
        b = starts[order[n + 1]] if n + 1 < len(order) else min(line["end"], syl[-1]["t"] + D / 8)
        tgt.append((a - line["t"], b - line["t"]))
    spans = match_words(words_flow, tts_words)
    if len(spans) != len(tgt):                       # word count mismatch: warp the whole line to the whole slot
        spans = [(tts_words[0]["t"], tts_words[-1]["t"] + tts_words[-1]["d"])]
        tgt = [(tgt[0][0], tgt[-1][1])]
    total = int(round((tgt[-1][1] + 0.35) * 1000 / FP))
    f0o = np.zeros(total)
    spo = np.full((total, sp.shape[1]), 1e-8)
    apo = np.ones((total, apr.shape[1]))
    for (sa, sb), (ta, tb) in zip(spans, tgt):
        fs, fe = int(sa * 1000 / FP), max(int(sa * 1000 / FP) + 2, int(sb * 1000 / FP) + 2)   # + a little tail
        fe = min(fe, nfr)
        if fe - fs < 2:
            continue
        nat = (fe - fs) * FP / 1000
        slot = tb - ta
        r = slot * 0.92 / nat                          # natural -> slot; leave a breath before the next word
        r = float(np.clip(r * (1 + warp_jit * rng.normal()), 0.5, 1.12))   # never drag a word out much: gaps become pauses
        n_out = max(2, int(round(nat * r * 1000 / FP)))
        o0 = int(round((ta + jitter_ms / 1000 * rng.normal()) * 1000 / FP))
        o0 = max(0, o0)
        o1 = min(total, o0 + n_out)
        if o1 <= o0:
            continue
        idx = np.clip(np.round(np.linspace(fs, fe - 1, o1 - o0)).astype(int), 0, nfr - 1)
        f0o[o0:o1] = f0[idx]
        spo[o0:o1] = sp[idx]
        apo[o0:o1] = apr[idx]
    y = pw.synthesize(np.ascontiguousarray(f0o), np.ascontiguousarray(spo), np.ascontiguousarray(np.clip(apo, 0, 1)), SR, frame_period=FP)
    if mode == "whisper":
        y = signal.sosfilt(signal.butter(2, 300, "high", fs=SR, output="sos"), y) * 0.5
    return y


# ---------------- sections -> takes ----------------
def mode_of(text):
    if "(whisper)" in text:
        return "whisper"
    if "(spoken)" in text:
        return "spoken"
    return "rap"


sections = [tuple(s) for s in T["sections"]]
selection = {}
for name, a, b in sections:
    lines = [l for l in FLOW if BARS[a] - 0.06 <= l["t"] < BARS[b] - 0.06]
    if not lines:
        continue
    double = name.endswith(("_L", "_R"))
    side = name.endswith("_R")
    song_start = max(0.0, lines[0]["t"] - 0.5)
    length = int((lines[-1]["end"] + 1.5 - song_start) * SR)
    buf = np.zeros(length)
    for l in lines:
        m = mode_of(l["text"])
        if double:
            y = render_line(l, m, jitter_ms=12.0, cents=(+7 if side else -7) + rng.normal(0, 3), warp_jit=0.04)
        else:
            y = render_line(l, m)
        s = int((l["t"] - song_start) * SR)
        e = min(length, s + len(y))
        buf[s:e] += y[: e - s]
    buf = buf / (np.abs(buf).max() + 1e-9) * 0.6
    take = f"{name}_take1"
    sf.write(os.path.join(TAKES, take + ".wav"), buf.astype(np.float32), SR, subtype="FLOAT")
    json.dump({"song_start": song_start, "part_start": lines[0]["t"], "beat": T["beat"], "generated": args.voice},
              open(os.path.join(TAKES, take + ".json"), "w"))
    selection[name] = take
    print(f"{name:8s} {len(lines):2d} lines  {song_start:6.1f}s  {length / SR:5.1f}s  -> {take}.wav")
json.dump(selection, open(os.path.join(TAKES, "selection.json"), "w"), indent=1)
json.dump({"latency_ms": 0.0}, open(os.path.join(TAKES, "latency.json"), "w"))   # generated takes are already on the grid
print(f"wrote {len(selection)} takes into {TAKES}/ (selection.json, latency 0) -> now: python mixvocal.py --voice lovell --tune 0.5")
