# -*- coding: utf-8 -*-
"""STATIC — original BONES-style lo-fi trap instrumental (own composition, real samples, nothing borrowed). v3.

E minor, 68 BPM half-time (snare on 3, swung hats, tuned 808s). 4-bar loop Em - Am - Bm - Am, all minor.
The "sample": a ghostly guitar swell pad (volume-pedal swells into a long hall), a sparse low pulse and a slow mournful
top line, all played on the sampled Standard Guitar through a Tube Screamer capture and a Fender-style cab, plus a faint
piano double. The whole loop is played 3 semitones up and faster, then slowed down like a record — the classic
"pitched-down sample" trick (thicker, darker, sounds sampled). Then wow/flutter, band-limiting, tape saturation, crackle.
Drums = AVL Red Zeppelin samples: real snare + hand clap with transient shaping, swung hats, reversed crash swells,
a kick sample thump under a tuned 808 (sub + distorted presence layer so it is audible on small speakers).

  python static_beat.py                 -> out/static_instrumental.{wav,mp3}   (variant A: full trap drums)
  python static_beat.py --variant b     -> out/static_instrumental_b.{wav,mp3} (variant B: sparser, darker, more ambient)
Both use the same bar grid (out/static_bar_times.json), so the lyrics / flow / takes fit either.
Run with the venv python from the project folder: PYTHONIOENCODING=utf-8 .venv/Scripts/python static_beat.py
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np
import soundfile as sf
from scipy import signal

import mixfx as fx
from ampsim import Proteus, cab, load_ir as load_cab
from drumkit import RZ, SFZKit, place_stereo
from gtrsampler import StandardGuitar
from sfsynth import render as sf_render

ap = argparse.ArgumentParser()
ap.add_argument("--variant", default="a", choices=["a", "b"])
args = ap.parse_args()
VB = args.variant == "b"

SR = 44100
BPM = 68.0
spb = 60.0 / BPM * SR          # samples per beat
s16 = spb / 4                  # samples per 16th
UP = 3                         # record trick: melodic layers are played UP semitones higher / faster, then slowed down
R = 2 ** (UP / 12)
spb_hi = spb / R
s16_hi = spb_hi / 4
SWING = 0.10                   # off-16ths late by 10 % of a 16th (~22 ms): lazy trap hats
rng = np.random.default_rng(809 if VB else 808)
T0 = time.time()
AVL = "assets/avl/AVL_Drumkits_1.0"
OUT = "out/static_instrumental" + ("_b" if VB else "")


def log(m):
    print(f"[{time.time() - T0:5.0f}s] {m}", flush=True)


# ---------------- arrangement ----------------
# name, pad voicing (whole-bar swells), low pulse [(beat, midi, beats)], top line [(beat, midi, beats)], 808 midi
CHORDS = [
    ("Em", [52, 59, 67, 71], [(0, 40, 2.5), (3.5, 47, 0.5)], [(0, 71, 2.0), (2.5, 72, 0.5), (3, 71, 1.0)], 40),
    ("Am", [57, 64, 67, 72], [(0, 45, 2.5), (3.5, 52, 0.5)], [(0, 69, 1.5), (1.5, 72, 1.5), (3, 71, 1.0)], 45),
    ("Bm", [59, 66, 69, 74], [(0, 47, 2.5), (3.5, 54, 0.5)], [(0, 74, 1.0), (1, 72, 1.0), (2, 71, 1.0), (3, 69, 1.0)], 47),
    ("Am", [57, 64, 67, 72], [(0, 45, 2.5), (3.5, 52, 0.5)], [(0, 67, 1.5), (1.5, 64, 1.0), (2.5, 66, 0.5), (3, 64, 1.0)], 45),
]
# (section, bars, level) level 0 = loop only, 1 = verse, 2 = hook, 3 = last hook (busier)
STRUCT = [("intro", 4, 0), ("hook", 8, 2), ("verse", 12, 1), ("hook", 8, 2), ("verse", 12, 1), ("hook", 8, 3), ("outro", 4, 0)]
bars = []
for sec, n, lvl in STRUCT:
    for i in range(n):
        bars.append((i % 4, lvl, sec, i, n))
NB = len(bars)
TAIL = 6.0
N = int(NB * 4 * spb + TAIL * SR)
N_hi = int(np.ceil(N / R)) + SR


def bar_start(k):
    return int(round(k * 4 * spb))


def t_slot(k, slot, swing=0.0):
    """sample index of 16th-note slot (float ok) in bar k; odd 16ths can be swung"""
    off = swing * s16 if (swing and abs(slot - round(slot)) < 1e-6 and int(round(slot)) % 2 == 1) else 0.0
    return int(round(bar_start(k) + slot * s16 + off))


def t_beat_hi(k, beat):
    return int(round(k * 4 * spb_hi + beat * spb_hi))


def top_line_on(sec, i, n):
    """the mournful top line: hooks and outro always, verses only in their last 4 bars (build into the hook)"""
    if sec in ("hook", "outro"):
        return True
    if sec == "verse":
        return i >= n - 4
    return sec == "intro" and i >= 2


# ---------------- the loop (guitar swells + pulse + top line + faint piano), played UP and slowed down ----------------
guitar = StandardGuitar("assets/stdguitar/UI_Standard_Guitar/Samples")


def add(buf, s, y):
    s = max(0, s)
    e = min(len(buf), s + len(y))
    if e > s:
        buf[s:e] += y[: e - s]


def swell_env(n, attack_s, release_s):
    t = np.arange(n) / SR
    env = np.minimum(1.0, (t / attack_s) ** 1.6)
    r = min(n, int(release_s * SR))
    if r > 0:
        env[-r:] *= np.linspace(1, 0, r) ** 0.8
    return env


def render_pad(chan, drift, seed):
    """volume-pedal swells of the chord tones, one bar each — the ghost pad"""
    trng = np.random.default_rng(seed)
    di = np.zeros(N_hi)
    for k, (ci, lvl, sec, i, n) in enumerate(bars):
        name, pad, pulse, top, _ = CHORDS[ci]
        if sec == "outro" and i >= 3:
            continue
        for j, m in enumerate(pad):
            s = t_beat_hi(k, 0) + int(trng.normal(0.03 * j, 0.02) * SR)
            dur = int(spb_hi * trng.uniform(4.0, 4.4))
            y = guitar.note("long", m + UP, dur, 0.5, trng, chan=chan, drift_cents=drift + trng.normal(0, 4.0))
            y *= swell_env(len(y), trng.uniform(0.7, 1.1), 0.5)
            add(di, s, y)
    return di


def render_pulse_top(chan, drift, seed):
    """sparse low pulse + the top line (fingerpicked, drier than the pad)"""
    trng = np.random.default_rng(seed)
    di = np.zeros(N_hi)
    for k, (ci, lvl, sec, i, n) in enumerate(bars):
        name, pad, pulse, top, _ = CHORDS[ci]
        if sec == "outro" and i >= 2:
            continue
        for (beat, m, d) in pulse:
            s = t_beat_hi(k, beat) + int(trng.normal(0.004, 0.006) * SR)
            y = guitar.note("sus_down", m + UP, int(spb_hi * d * 1.1), 0.55, trng, chan=chan, drift_cents=drift)
            add(di, s, y * 0.9)
        if top_line_on(sec, i, n):
            oct_up = 12 if (lvl == 3 and i >= 4) else 0
            for (beat, m, d) in top:
                s = t_beat_hi(k, beat) + int(trng.normal(0.006, 0.008) * SR)
                vel = float(np.clip(trng.normal(0.62, 0.05), 0.4, 0.8))
                y = guitar.note("sus_down", m + UP + oct_up, int(spb_hi * d * 1.25), vel, trng, chan=chan, drift_cents=drift + trng.normal(0, 3.0))
                add(di, s, y)
    return di


ts9 = Proteus("assets/proteus/TS9_HighDrive.json")
ir_clean = load_cab("assets/ir/Allure Pack/Allure_64_USDeluxe_P12N.wav", max_ms=120)


def amp(di, drive_in):
    """DI -> Tube Screamer capture (driven gently) -> Fender-style cab: a real, slightly hairy clean tone"""
    x = di * drive_in / (np.percentile(np.abs(di), 99.9) + 1e-9)
    y = ts9.process(np.clip(x, -1, 1))
    y = cab(y, ir_clean)
    y = fx.hp(y, 100)
    y = fx.peak(y, 3000, -2.0, 1.2)
    return y


pad = amp(render_pad(0, -6.0, 21) + render_pad(1, +6.0, 22), 0.28)
lead = amp(render_pulse_top(0, -3.0, 23), 0.35)
log(f"guitars done: pad rms {fx.rms_db(pad):.1f}, lead rms {fx.rms_db(lead):.1f} dB")

# faint piano doubling the top line an octave down (it gets pitched down with the loop -> dusty sample piano)
pev = []
for k, (ci, lvl, sec, i, n) in enumerate(bars):
    name, pad_v, pulse, top, _ = CHORDS[ci]
    if not top_line_on(sec, i, n) or (sec == "outro" and i >= 2):
        continue
    for (beat, m, d) in top:
        pev.append({"s": max(0, t_beat_hi(k, beat) + int(rng.normal(0.012, 0.01) * SR)), "n": int(spb_hi * d * 1.6), "m": m - 12 + UP,
                    "v": float(np.clip(rng.normal(0.45, 0.05), 0.3, 0.6))})
piano = sf_render("assets/GeneralUser-GS.sf2", 0, pev, N_hi, gain_db=-6.0)
piano = fx.lp(piano, 2500)
piano = fx.hp(piano, 120)


def level_to(x, tgt, core=None):
    ref = x if core is None else x[core]
    return x * 10 ** ((tgt - fx.rms_db(ref)) / 20)


CORE_HI = slice(int(4 * 4 * spb_hi), int(12 * 4 * spb_hi))
loop_hi = np.zeros((N_hi, 2))
p = level_to(pad, -24.0, CORE_HI)
l = level_to(lead, -21.0, CORE_HI)
pn = level_to(piano, -31.0, CORE_HI)
# pad wide (Haas-free: two takes were rendered from different channels + drift, keep both sides), lead slightly left
loop_hi[:, 0] += p * 0.9 + l * 0.8 + pn * 0.7
loop_hi[:, 1] += p * 0.9 + l * 0.6 + pn * 0.7
hall = fx.load_ir("assets/voxengo/Large Long Echo Hall.wav", max_s=4.0)
pad_st = np.stack([p, p], 1)
loop_hi += fx.convolve(pad_st, hall) * (0.9 if VB else 0.7) + fx.convolve(np.stack([l, l], 1), hall) * 0.35
log("loop assembled (hi)")

# ---- the record trick: slow the whole loop down by R (pitch -UP semitones, tempo -> 68) ----
idx = np.arange(N) / R
loop = np.empty((N, 2))
src = np.arange(N_hi)
for ch in range(2):
    loop[:, ch] = np.interp(idx, src, loop_hi[:, ch])
del loop_hi


def wow_flutter(x, wow_hz=0.4, wow_depth=0.003, flutter_hz=5.7, flutter_depth=0.0006):
    n = len(x)
    t = np.arange(n) / SR
    rate = 1 + wow_depth * np.sin(2 * np.pi * wow_hz * t + 0.7) + flutter_depth * np.sin(2 * np.pi * flutter_hz * t)
    pos = np.cumsum(rate)
    pos -= pos[0]
    pos = np.clip(pos, 0, n - 1)
    out = np.empty_like(x)
    for ch in range(x.shape[1]):
        out[:, ch] = np.interp(pos, np.arange(n), x[:, ch])
    return out


def vinyl_noise(n, rng):
    crackle = np.zeros((n, 2))
    k = int(22.0 * n / SR)
    pos = rng.integers(0, n - 4, k)
    amp_ = rng.exponential(0.5, k) * rng.choice([-1, 1], k)
    for chn in range(2):
        sel = rng.random(k) < 0.6
        crackle[pos[sel], chn] += amp_[sel]
        crackle[pos[sel] + 1, chn] -= amp_[sel] * 0.5
    crackle = fx.hp(crackle, 700)
    crackle = fx.lp(crackle, 6500)
    crackle *= 10 ** (-33 / 20) / (np.abs(crackle).max() + 1e-9)
    hiss = rng.standard_normal((n, 2))
    hiss = fx.hp(fx.lp(hiss, 4200), 250)
    hiss *= 10 ** (-58 / 20) / (np.sqrt((hiss ** 2).mean()) + 1e-9)
    return crackle + hiss


CORE = slice(bar_start(4), bar_start(12))  # first hook = reference for levels
loop = wow_flutter(loop)
loop = fx.hp(loop, 85)
loop = fx.lp(loop, 4200 if VB else 9000, 1)
loop = fx.saturate(loop, drive=1.8, mix=0.5)
loop = fx.peak(loop, 900, 1.0, 0.8)
loop = level_to(loop, -21.5 if VB else -20.0, CORE)   # set AFTER the saturation stage (tanh drive adds linear gain)
# intro: the record fades in with a filter opening over 4 bars; outro: it dulls and dies
b4 = bar_start(4)
dark = fx.lp(loop[:b4], 500)
ramp = (np.arange(b4) / b4)[:, None] ** 1.4
loop[:b4] = dark * (1 - ramp) + loop[:b4] * ramp
o0 = bar_start(NB - 4)
dull = fx.lp(loop[o0:], 900)
ramp = np.linspace(0, 1, N - o0)[:, None]
loop[o0:] = loop[o0:] * (1 - ramp) + dull * ramp
fade0 = bar_start(NB - 2)
loop[fade0:] *= np.linspace(1, 0, N - fade0)[:, None] ** 0.7
noise = vinyl_noise(N, rng)
noise[int(N - TAIL * SR):] *= np.linspace(1, 0, N - int(N - TAIL * SR))[:, None]
loop += noise
log(f"loop bus rms {fx.rms_db(loop[CORE]):.1f} dB")

# ---------------- drums ----------------
kit = SFZKit(f"{AVL}/Red_Zeppelin_5pc.sfz")
_cache = {}


def wav(name):
    if name not in _cache:
        x, sr = sf.read(f"{AVL}/{name}")
        if x.ndim > 1:
            x = x.mean(1)
        if sr != SR:
            x = signal.resample_poly(x, SR, sr)
        _cache[name] = x.astype(np.float64)
    return _cache[name]


def repitch(x, semis):
    ratio = 2 ** (semis / 12)
    idx_ = np.arange(0, len(x) - 1, ratio)
    return np.interp(idx_, np.arange(len(x)), x)


def clap(vel, rng):
    i = int(np.clip(round(vel * 5), 1, 5))
    x = wav(f"39-HandClap-{i}.wav")
    return repitch(x, rng.normal(0, 0.3)) * (0.4 + 0.6 * vel)


DR = []  # (sample_index, kind, vel)
E8 = []  # 808 events: (sample_index, midi, vel)
REV = []  # reversed crash targets (sample index of the downbeat)


def hat_roll(k, a, b, n, v0, v1):
    for j in range(n):
        DR.append((t_slot(k, a + (b - a) * j / n), "hhc", v0 + (v1 - v0) * j / max(1, n - 1)))


for k, (ci, lvl, sec, i, n) in enumerate(bars):
    root808 = CHORDS[ci][4]
    nxt = CHORDS[bars[k + 1][0]][4] if k + 1 < NB else root808
    last = (i == n - 1)
    if sec == "hook" and i == 0:
        REV.append(bar_start(k))
    if lvl == 0:
        if sec == "intro" and i == 3:
            E8.append((t_slot(k, 0), root808, 0.8))
            E8.append((t_slot(k, 12), nxt, 0.7))
        if sec == "outro":
            E8.append((t_slot(k, 0), root808, 0.9 - 0.2 * i))
        continue
    # 808 pattern in a 2-bar phrase; the pickup on slot 13 already plays the next bar's root
    kicks = [0, 7, 10] if k % 2 == 0 else [0, 6, 10, 13]
    if VB and k % 2 == 1:
        kicks = [0, 6, 11]
    if last and sec == "verse":
        kicks = [0]
    if lvl == 3 and i == n - 1:
        kicks = kicks + [14, 15]
    for sl in kicks:
        E8.append((t_slot(k, sl, SWING), nxt if sl >= 12 else root808, 1.0 if sl == 0 else 0.85))
    # snare + clap on 3; fill in the last bar of a verse
    if last and sec == "verse":
        for j, sl in enumerate((8, 12, 13, 14, 15)):
            DR.append((t_slot(k, sl, SWING), "snare", 0.95 if sl == 8 else 0.5 + 0.1 * j))
    else:
        DR.append((t_slot(k, 8), "snare", 1.0))
        if lvl >= 2 and i % 4 == 3 and not VB:
            DR.append((t_slot(k, 15, SWING), "snare", 0.3))
    # hats
    if lvl == 1 or VB:
        for sl in range(0, 16, 2):
            if last and sec == "verse" and sl >= 8:
                break
            DR.append((t_slot(k, sl, SWING), "hhc", 0.66 if sl % 4 == 0 else 0.46))
        if i % 2 == 1 and not last:
            for sl in (13, 15):
                DR.append((t_slot(k, sl, SWING), "hhc", 0.4))
        if i % 4 == 3 and not last:
            hat_roll(k, 6, 8, 6, 0.3, 0.55)
        if lvl >= 2 and i % 2 == 0:
            DR.append((t_slot(k, 14, SWING), "hho", 0.45))
    else:
        for sl in range(16):
            v = 0.7 if sl % 4 == 0 else (0.55 if sl % 2 == 0 else 0.36)
            DR.append((t_slot(k, sl, SWING), "hhc", v))
        if i % 2 == 1:
            hat_roll(k, 14, 16, 6, 0.32, 0.6)
        if i % 4 == 3:
            hat_roll(k, 6, 8, 8, 0.28, 0.6)
        if i % 2 == 0:
            DR.append((t_slot(k, 14, SWING), "hho", 0.5))
        if lvl == 3:
            DR.append((t_slot(k, 4), "side", 0.45))
            DR.append((t_slot(k, 12), "side", 0.4))

log(f"arrangement: {NB} bars, {len(DR)} drum hits, {len(E8)} 808s, {len(REV)} reversed crashes, {N / SR:.0f}s")

drng = np.random.default_rng(3)
snare_bus = np.zeros((N, 2))
hat_bus = np.zeros((N, 2))
for (s, name, vel) in DR:
    v = float(np.clip(vel * drng.normal(1.0, 0.05), 0.1, 1.0))
    if name == "snare":
        s = s + int(drng.normal(0, 0.003) * SR)
        x, _ = kit.hit(RZ["snare"], v, drng)
        x = repitch(x, (-3.0 if VB else -2.0) + drng.normal(0, 0.15))
        x = x[: int(0.4 * SR)]
        t = np.arange(len(x)) / SR
        x = x * np.exp(-t / 0.13) * (1 + 1.2 * np.exp(-t / 0.008))   # tight + transient shaper
        cl = clap(v, drng)
        place_stereo(snare_bus, s, x, 0.0, 1.0)
        place_stereo(snare_bus, s + int(0.005 * SR), cl, 0.0, 0.9)
    else:
        s = s + int(drng.normal(0, 0.003) * SR)
        x, pan = kit.hit(RZ[name], v, drng)
        if name == "hhc":
            x = x[: int(0.1 * SR)]
            x = x * np.exp(-np.arange(len(x)) / (0.035 * SR))
        elif name == "hho":
            x = x[: int(0.32 * SR)]
            x = x * np.exp(-np.arange(len(x)) / (0.12 * SR))
        pan = 0.25 if name in ("hhc", "hho") else -0.35
        place_stereo(hat_bus, s, x, pan, 1.0 if name != "side" else 0.6)
# reversed crash swells into every hook
for target in REV:
    x, _ = kit.hit(RZ["crash1"], 0.75, drng)
    x = x[: int(1.7 * SR)]
    x = x * np.linspace(0, 1, len(x)) ** 2
    x = x[::-1]
    x = fx.hp(x, 500)
    place_stereo(hat_bus, target - len(x), x, -0.2, 0.55)

snare_bus = fx.hp(snare_bus, 130)
snare_bus = fx.peak(snare_bus, 190, 2.5, 1.0)
snare_bus = fx.peak(snare_bus, 3800, 3.0, 1.0)
snare_bus = fx.lp(snare_bus, 12000)
snare_bus = fx.saturate(snare_bus, drive=2.0, mix=0.6)
snare_bus = snare_bus + fx.convolve(snare_bus, fx.load_ir("assets/voxengo/Small Drum Room.wav")) * (0.5 if VB else 0.3)
hat_bus = fx.hp(hat_bus, 900)
hat_bus = fx.saturate(hat_bus, drive=1.3, mix=0.3)
log(f"drums: snare rms {fx.rms_db(snare_bus[CORE]):.1f}, hats rms {fx.rms_db(hat_bus[CORE]):.1f} dB")

# ---------------- 808 (sub + presence) + kick thump ----------------
kick = wav("36-Ludwig26Kick-4.wav")
kick_click = fx.hp(kick, 80)[: int(0.06 * SR)]
kick_click *= np.exp(-np.arange(len(kick_click)) / (0.02 * SR))
kick_thump = fx.lp(kick, 160)[: int(0.35 * SR)]
kick_thump *= np.exp(-np.arange(len(kick_thump)) / (0.09 * SR))
E8.sort()
sub = np.zeros(N)
TAU = 1.1 if VB else 0.85
for j, (s, midi, vel) in enumerate(E8):
    f0 = 440.0 * 2 ** ((midi - 69) / 12)
    nxt = E8[j + 1][0] if j + 1 < len(E8) else N
    n = int(min(1.8 * SR, max(0.2 * SR, nxt - s - int(0.008 * SR))))
    t = np.arange(n) / SR
    f = f0 * (1 + 2.0 * np.exp(-t / 0.02))
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.tanh(1.5 * np.sin(ph)) / np.tanh(1.5)
    env = (1 - np.exp(-t / 0.002)) * np.exp(-t / TAU)
    env[-int(0.008 * SR):] *= np.linspace(1, 0, int(0.008 * SR))
    add(sub, s, body * env * vel)
    add(sub, s, kick_click * (0.5 + 0.5 * vel) * 0.8)
    add(sub, s, kick_thump * vel * 0.6)
grit = fx.lp(fx.hp(np.tanh(6.0 * sub), 90), 2800)          # distorted layer: the 808 you hear on a phone
e808 = fx.lp(sub, 1000) + grit * 10 ** ((-13.0 if VB else -11.0) / 20) * (np.sqrt((sub ** 2).mean()) / (np.sqrt((grit ** 2).mean()) + 1e-9))
e808, _ = fx.compressor(e808[:, None], thr_db=-12, ratio=3, attack_ms=15, release_ms=150, makeup_db=2)
e808 = e808[:, 0]
log(f"808 rms {fx.rms_db(e808[CORE]):.1f} dB")

# ---------------- mix + master ----------------
mix = np.zeros((N, 2))
mix += loop
mix += level_to(snare_bus, -19.0 if VB else -18.5, CORE)
mix += level_to(hat_bus, -25.0 if VB else -22.5, CORE)
b = level_to(e808, -14.0, CORE)
mix[:, 0] += b
mix[:, 1] += b
mix, _ = fx.compressor(mix, thr_db=-9, ratio=1.8, attack_ms=25, release_ms=220, makeup_db=1.0)  # light glue: keep the drop
mix = fx.saturate(mix, drive=1.2, mix=0.25)
mix = fx.hp(mix, 27)
mix = fx.shelf(mix, 10000, -1.5, "high")
mix *= 10 ** (((-12.5 if VB else -11.5) - fx.rms_db(mix[CORE])) / 20)
mix = np.tanh(mix * 0.6) / np.tanh(0.6)
mix = fx.limiter(mix, thr=0.97, lookahead_ms=3, release_ms=120)
mix = mix / (np.abs(mix).max() + 1e-9) * 0.97

os.makedirs("out", exist_ok=True)
sf.write(OUT + ".wav", mix.astype(np.float32), SR, subtype="PCM_16")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", OUT + ".wav", "-codec:a", "libmp3lame", "-b:a", "320k",
                "-metadata", f"title=STATIC (instrumental{' B' if VB else ''})", "-metadata", "artist=Igor / Claude Code", OUT + ".mp3"], check=True)

# bar grid + structure for flow.py / sections.py / studio.py (identical for both variants)
sections, k0 = [], 0
for sec, n, lvl in STRUCT:
    sections.append([sec, k0, k0 + n])
    k0 += n
json.dump({"bpm": BPM, "levels": [b_[1] for b_ in bars], "bars": {str(k): round(bar_start(k) / SR, 3) for k in range(NB + 1)},
           "structure": sections}, open("out/static_bar_times.json", "w"), indent=1)
log(f"MASTER rms {fx.rms_db(mix):.1f} dBFS (core {fx.rms_db(mix[CORE]):.1f}) peak {np.abs(mix).max():.2f} {N / SR:.0f}s -> {OUT}.mp3")
