# -*- coding: utf-8 -*-
"""STATIC — original BONES-style lo-fi trap instrumental (own composition, real samples, nothing borrowed).

E minor, 68 BPM half-time (snare on 3, hats in 8ths/16ths, tuned 808s). 4-bar loop Em - C - Am - B.
Sound: the "dusty record" loop = clean sampled guitar (Unreal Standard Guitar DI -> Fender-style cab IR), sampled piano
and choir (GeneralUser GS) through wow/flutter, band-limiting, tape saturation, vinyl crackle/hiss and a long hall.
Drums = AVL Red Zeppelin samples (snare pitched down + hand clap, hats) lo-fi'd; 808 = tuned sub with a real kick sample
for the click (that is how an 808 is made — there is no other way to get one from a drum kit).
Outputs: out/static_instrumental.wav / .mp3 and out/static_bar_times.json (bar grid + sections for the vocal tools).
Run with the venv python from the project folder: PYTHONIOENCODING=utf-8 .venv/Scripts/python static_beat.py
"""
import json
import os
import subprocess
import time

import numpy as np
import soundfile as sf
from scipy import signal

import mixfx as fx
from ampsim import cab, load_ir as load_cab
from drumkit import RZ, SFZKit, place_stereo
from gtrsampler import StandardGuitar
from sfsynth import render as sf_render

SR = 44100
BPM = 68.0
spb = 60.0 / BPM * SR          # samples per beat
s16 = spb / 4                  # samples per 16th
rng = np.random.default_rng(808)
T0 = time.time()
AVL = "assets/avl/AVL_Drumkits_1.0"
OUT = "out/static_instrumental"


def log(m):
    print(f"[{time.time() - T0:5.0f}s] {m}", flush=True)


# ---------------- arrangement ----------------
# chord: (name, arpeggio midi per 8th slot (None = rest), low root midi, 808 midi, piano chord, choir notes)
CHORDS = [
    ("Em", [71, 67, 64, 67, 71, 67, 64, None], 52, 40, [52, 55, 59, 64], [40 + 12, 47 + 12]),
    ("C",  [72, 67, 64, 67, 72, 67, 64, None], 48, 48, [48, 52, 55, 64], [48, 55]),
    ("Am", [69, 64, 60, 64, 69, 64, 60, None], 45, 45, [45, 52, 57, 60], [45, 52]),
    ("B",  [71, 66, 63, 66, 71, 66, 63, None], 47, 47, [47, 51, 54, 59], [47, 54]),
]
# (section, bars, level) level 0 = loop only, 1 = verse, 2 = hook, 3 = last hook (busier hats)
STRUCT = [("intro", 4, 0), ("hook", 8, 2), ("verse", 12, 1), ("hook", 8, 2), ("verse", 12, 1), ("hook", 8, 3), ("outro", 4, 0)]
bars = []
for sec, n, lvl in STRUCT:
    for i in range(n):
        bars.append((i % 4, lvl, sec, i, n))
NB = len(bars)
TAIL = 6.0
N = int(NB * 4 * spb + TAIL * SR)


def bar_start(k):
    return int(round(k * 4 * spb))


def t_slot(k, slot):
    """sample index of 16th-note slot (float ok) in bar k"""
    return int(round(bar_start(k) + slot * s16))


# ---------------- the loop (guitar + piano + choir) ----------------
guitar = StandardGuitar("assets/stdguitar/UI_Standard_Guitar/Samples")


def render_guitar(chan, drift_cents, seed):
    trng = np.random.default_rng(seed)
    di = np.zeros(N)
    for k, (ci, lvl, sec, i, n) in enumerate(bars):
        name, arp, low, _, _, _ = CHORDS[ci]
        # arpeggio, 8th notes, let each note ring into the next (legato picking)
        for j, m in enumerate(arp):
            if m is None:
                continue
            if sec == "outro" and j >= 4 and i >= 2:  # outro thins out
                continue
            s = t_slot(k, 2 * j) + int(trng.normal(0.004, 0.006) * SR)
            dur = int(spb * 0.5 * trng.uniform(1.6, 2.2))
            vel = float(np.clip(trng.normal(0.62 if j % 4 == 0 else 0.5, 0.06), 0.3, 0.8))
            y = guitar.note("sus_down", m, dur, vel, trng, chan=chan, drift_cents=drift_cents + trng.normal(0, 3.0))
            s = max(0, s)
            e = min(N, s + len(y))
            di[s:e] += y[: e - s]
        # low root, one long note per bar, soft
        s = t_slot(k, 0) + int(trng.normal(0.0, 0.005) * SR)
        dur = int(spb * 3.9)
        y = guitar.note("long", low, dur, 0.42, trng, chan=chan, drift_cents=drift_cents) * 0.7
        s = max(0, s)
        e = min(N, s + len(y))
        di[s:e] += y[: e - s]
    return di


gtr_L = render_guitar(0, -5.0, 11)
gtr_R = render_guitar(1, +5.0, 12)
ir_clean = load_cab("assets/ir/Allure Pack/Allure_64_USDeluxe_P12N.wav", max_ms=120)


def clean_amp(x):
    x = x * 0.6 / (np.percentile(np.abs(x), 99.9) + 1e-9)
    y = np.tanh(1.6 * x) / np.tanh(1.6)          # a warm clean channel pushed a little
    y = cab(y, ir_clean)
    y = fx.hp(y, 110)
    y = fx.peak(y, 2400, -2.5, 1.2)
    y = fx.peak(y, 380, 1.5, 1.0)
    return y


gtr_L, gtr_R = clean_amp(gtr_L), clean_amp(gtr_R)
log(f"guitar done, rms {fx.rms_db(gtr_L):.1f} dB")

# piano: dark chord on beat 1, single high answer on the 'and' of 3 in half of the bars
pev, cev = [], []
for k, (ci, lvl, sec, i, n) in enumerate(bars):
    name, arp, low, _, pch, choir = CHORDS[ci]
    if sec == "outro" and i >= 2:
        continue
    s = t_slot(k, 0) + int(rng.normal(0.012, 0.008) * SR)
    for j, m in enumerate(pch):
        pev.append({"s": max(0, s + int(j * rng.uniform(0.008, 0.02) * SR)), "n": int(spb * 3.6), "m": m, "v": float(np.clip(rng.normal(0.5, 0.05), 0.3, 0.7))})
    if k % 2 == 1:
        pev.append({"s": t_slot(k, 10) + int(rng.normal(0.01, 0.01) * SR), "n": int(spb * 1.4), "m": arp[0] + 12, "v": 0.42})
    for m in choir:
        cev.append({"s": t_slot(k, 0), "n": int(spb * 4.0), "m": m, "v": 0.5})
piano = sf_render("assets/GeneralUser-GS.sf2", 0, pev, N, gain_db=-4.0)
piano = fx.lp(piano, 3200)
piano = fx.hp(piano, 90)
choir = sf_render("assets/GeneralUser-GS.sf2", 52, cev, N, gain_db=-8.0)
choir = fx.lp(choir, 1800)
choir = fx.hp(choir, 120)
log(f"piano rms {fx.rms_db(piano):.1f} dB, choir rms {fx.rms_db(choir):.1f} dB")


def level_to(x, tgt, core=None):
    ref = x if core is None else x[core]
    return x * 10 ** ((tgt - fx.rms_db(ref)) / 20)


CORE = slice(bar_start(4), bar_start(12))  # first hook = reference for levels
loop = np.zeros((N, 2))
th = (-0.55 + 1) * np.pi / 4
gl = level_to(gtr_L, -26.5, CORE)
gr = level_to(gtr_R, -26.5, CORE)
loop[:, 0] += gl * np.cos(th) + gr * np.sin(th)
loop[:, 1] += gl * np.sin(th) + gr * np.cos(th)
p = level_to(piano, -31.0, CORE)
c = level_to(choir, -37.0, CORE)
loop[:, 0] += p + c
loop[:, 1] += p + c

# long hall on the loop, then the whole thing goes through the "old record" chain
hall = fx.load_ir("assets/voxengo/Large Long Echo Hall.wav", max_s=3.5)
loop = loop + fx.convolve(loop, hall) * 0.32


def wow_flutter(x, wow_hz=0.45, wow_depth=0.0035, flutter_hz=6.1, flutter_depth=0.0007):
    """variable-speed playback (tape / warped record): pitch and time wobble together"""
    n = len(x)
    t = np.arange(n) / SR
    rate = 1 + wow_depth * np.sin(2 * np.pi * wow_hz * t + 0.7) + flutter_depth * np.sin(2 * np.pi * flutter_hz * t)
    pos = np.cumsum(rate)
    pos -= pos[0]
    pos = np.clip(pos, 0, n - 1)
    idx = np.arange(n)
    out = np.empty_like(x)
    for ch in range(x.shape[1]):
        out[:, ch] = np.interp(pos, idx, x[:, ch])
    return out


def vinyl_noise(n, rng):
    """crackle (sparse random clicks) + hiss (band-limited noise), stereo"""
    crackle = np.zeros((n, 2))
    per_s = 28.0
    k = int(per_s * n / SR)
    pos = rng.integers(0, n - 4, k)
    amp = rng.exponential(0.5, k) * rng.choice([-1, 1], k)
    for chn in range(2):
        sel = rng.random(k) < 0.6
        crackle[pos[sel], chn] += amp[sel]
        crackle[pos[sel] + 1, chn] -= amp[sel] * 0.5
    crackle = fx.hp(crackle, 700)
    crackle = fx.lp(crackle, 6500)
    crackle *= 10 ** (-30 / 20) / (np.abs(crackle).max() + 1e-9)
    hiss = rng.standard_normal((n, 2))
    hiss = fx.hp(fx.lp(hiss, 4200), 250)
    hiss *= 10 ** (-56 / 20) / (np.sqrt((hiss ** 2).mean()) + 1e-9)
    return crackle + hiss


loop = wow_flutter(loop)
loop = fx.hp(loop, 95)
loop = fx.lp(loop, 6800, 2)
loop = fx.saturate(loop, drive=2.2, mix=0.6)
loop = fx.peak(loop, 1100, 1.5, 0.8)          # boxy mid, like a cheap sample
loop = level_to(loop, -19.5, CORE)             # set AFTER the saturation stage (tanh drive adds linear gain)
# outro: the record gets duller and fades
o0, o1 = bar_start(NB - 4), N
dull = fx.lp(loop[o0:o1], 1400)
ramp = np.linspace(0, 1, o1 - o0)[:, None]
loop[o0:o1] = loop[o0:o1] * (1 - ramp) + dull * ramp
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
    idx = np.arange(0, len(x) - 1, ratio)
    return np.interp(idx, np.arange(len(x)), x)


def clap(vel, rng):
    i = int(np.clip(round(vel * 5), 1, 5))
    x = wav(f"39-HandClap-{i}.wav")
    return repitch(x, rng.normal(0, 0.3)) * (0.4 + 0.6 * vel)


DR = []  # (sample_index, kind, vel)
E8 = []  # 808 events: (sample_index, midi, vel)


def hat_roll(k, a, b, n, v0, v1):
    for j in range(n):
        DR.append((t_slot(k, a + (b - a) * j / n), "hhc", v0 + (v1 - v0) * j / max(1, n - 1)))


for k, (ci, lvl, sec, i, n) in enumerate(bars):
    root808 = CHORDS[ci][3]
    nxt = CHORDS[bars[k + 1][0]][3] if k + 1 < NB else root808
    last = (i == n - 1)
    if lvl == 0:
        if sec == "intro" and i == 3:
            E8.append((t_slot(k, 0), root808, 0.8))
            E8.append((t_slot(k, 12), nxt, 0.7))
        if sec == "outro":
            E8.append((t_slot(k, 0), root808, 0.9 - 0.2 * i))
        continue
    # 808 / kick pattern in a 2-bar phrase, the pickup on slot 13 already plays the next bar's root
    kicks = [0, 7, 10] if k % 2 == 0 else [0, 6, 10, 13]
    if last and sec == "verse":
        kicks = [0]
    for sl in kicks:
        E8.append((t_slot(k, sl), nxt if sl >= 12 else root808, 1.0 if sl == 0 else 0.85))
    # snare + clap on 3; fill in the last bar of a verse
    if last and sec == "verse":
        for j, sl in enumerate((8, 12, 13, 14, 15)):
            DR.append((t_slot(k, sl), "snare", 0.95 if sl == 8 else 0.55 + 0.1 * j))
    else:
        DR.append((t_slot(k, 8), "snare", 1.0))
        if lvl >= 2 and i % 4 == 3:
            DR.append((t_slot(k, 15), "snare", 0.35))
    # hats
    if lvl == 1:
        for sl in range(0, 16, 2):
            if last and sl >= 8:
                break
            DR.append((t_slot(k, sl), "hhc", 0.62 if sl % 4 == 0 else 0.42))
        if i % 2 == 1 and not last:
            for sl in (13, 15):
                DR.append((t_slot(k, sl), "hhc", 0.38))
        if i % 4 == 3 and not last:
            hat_roll(k, 6, 8, 6, 0.3, 0.55)
    else:
        for sl in range(16):
            v = 0.66 if sl % 4 == 0 else (0.5 if sl % 2 == 0 else 0.34)
            DR.append((t_slot(k, sl), "hhc", v))
        if i % 2 == 1:
            hat_roll(k, 14, 16, 6, 0.32, 0.6)
        if i % 4 == 3:
            hat_roll(k, 6, 8, 8, 0.28, 0.6)
        if lvl == 3 and i % 2 == 0:
            DR.append((t_slot(k, 14), "hho", 0.55))
        if lvl == 3:
            DR.append((t_slot(k, 4), "side", 0.5))
            DR.append((t_slot(k, 12), "side", 0.45))
    if sec == "hook" and i == 0:
        DR.append((t_slot(k, 0), "crash1", 0.55))

log(f"arrangement: {NB} bars, {len(DR)} drum hits, {len(E8)} 808s, {N / SR:.0f}s")

drng = np.random.default_rng(3)
snare_bus = np.zeros((N, 2))
hat_bus = np.zeros((N, 2))
for (s, name, vel) in DR:
    s = s + int(drng.normal(0, 0.004) * SR)
    v = float(np.clip(vel * drng.normal(1.0, 0.05), 0.1, 1.0))
    if name == "snare":
        x, _ = kit.hit(RZ["snare"], v, drng)
        x = repitch(x, -4.0 + drng.normal(0, 0.2))
        x = x[: int(0.5 * SR)]
        x = x * np.exp(-np.arange(len(x)) / (0.16 * SR))
        cl = clap(v, drng)
        place_stereo(snare_bus, s, x, 0.0, 1.0)
        place_stereo(snare_bus, s + int(0.006 * SR), cl, 0.0, 0.85)
    elif name == "crash1":
        x, pan = kit.hit(RZ["crash1"], v, drng)
        place_stereo(hat_bus, s, x, -0.3, 0.6)
    else:
        x, pan = kit.hit(RZ[name], v, drng)
        if name == "hhc":
            x = x[: int(0.12 * SR)]
            x = x * np.exp(-np.arange(len(x)) / (0.04 * SR))
        pan = 0.28 if name in ("hhc", "hho") else -0.35
        place_stereo(hat_bus, s, x, pan, 1.0 if name != "side" else 0.7)

snare_bus = fx.hp(snare_bus, 130)
snare_bus = fx.lp(snare_bus, 7000)
snare_bus = fx.saturate(snare_bus, drive=2.5, mix=0.7)
snare_bus = fx.peak(snare_bus, 200, 3.0, 1.0)
snare_bus = snare_bus + fx.convolve(snare_bus, fx.load_ir("assets/voxengo/Small Drum Room.wav")) * 0.35
hat_bus = fx.hp(hat_bus, 600)
hat_bus = fx.lp(hat_bus, 9500)
hat_bus = fx.saturate(hat_bus, drive=1.5, mix=0.4)
log(f"drums: snare rms {fx.rms_db(snare_bus[CORE]):.1f}, hats rms {fx.rms_db(hat_bus[CORE]):.1f} dB")

# ---------------- 808 ----------------
kick_click = fx.hp(wav("36-Ludwig26Kick-4.wav"), 80)[: int(0.06 * SR)]
kick_click *= np.exp(-np.arange(len(kick_click)) / (0.02 * SR))
E8.sort()
e808 = np.zeros(N)
for idx, (s, midi, vel) in enumerate(E8):
    f0 = 440.0 * 2 ** ((midi - 69) / 12)
    nxt = E8[idx + 1][0] if idx + 1 < len(E8) else N
    n = int(min(1.5 * SR, max(0.2 * SR, nxt - s - int(0.008 * SR))))
    t = np.arange(n) / SR
    f = f0 * (1 + 2.6 * np.exp(-t / 0.022))
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.tanh(2.4 * np.sin(ph)) / np.tanh(2.4)
    env = (1 - np.exp(-t / 0.003)) * np.exp(-t / 0.7)
    env[-int(0.008 * SR):] *= np.linspace(1, 0, int(0.008 * SR))
    body = body * env * vel
    e = min(N, s + n)
    e808[s:e] += body[: e - s]
    ck = kick_click * (0.5 + 0.5 * vel)
    e = min(N, s + len(ck))
    e808[s:e] += ck[: e - s] * 0.9
e808 = fx.lp(e808, 2200)
e808 = fx.saturate(e808, drive=1.8, mix=0.35)
e808, _ = fx.compressor(e808[:, None], thr_db=-12, ratio=3, attack_ms=15, release_ms=150, makeup_db=2)
e808 = e808[:, 0]
log(f"808 rms {fx.rms_db(e808[CORE]):.1f} dB")

# ---------------- mix + master ----------------
mix = np.zeros((N, 2))
mix += loop
mix += level_to(snare_bus, -19.5, CORE)
mix += level_to(hat_bus, -24.0, CORE)
b = level_to(e808, -14.0, CORE)
mix[:, 0] += b
mix[:, 1] += b
mix, _ = fx.compressor(mix, thr_db=-9, ratio=1.8, attack_ms=25, release_ms=220, makeup_db=1.0)  # light glue: keep the drop
mix = fx.saturate(mix, drive=1.25, mix=0.3)
mix = fx.hp(mix, 27)
mix = fx.shelf(mix, 9000, -2.0, "high")   # BONES mixes are dark on top
core = mix[CORE]
mix *= 10 ** ((-11.0 - fx.rms_db(core)) / 20)
mix = np.tanh(mix * 0.9) / np.tanh(0.9)
mix = fx.limiter(mix, thr=0.97, lookahead_ms=3, release_ms=120)
mix = mix / (np.abs(mix).max() + 1e-9) * 0.97

os.makedirs("out", exist_ok=True)
sf.write(OUT + ".wav", mix.astype(np.float32), SR, subtype="PCM_16")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", OUT + ".wav", "-codec:a", "libmp3lame", "-b:a", "320k",
                "-metadata", "title=STATIC (instrumental)", "-metadata", "artist=Igor / Claude Code", OUT + ".mp3"], check=True)

# bar grid + recording sections for flow.py / sections.py / studio.py
sections, k0 = [], 0
for sec, n, lvl in STRUCT:
    sections.append([sec, k0, k0 + n])
    k0 += n
json.dump({"bpm": BPM, "levels": [b[1] for b in bars], "bars": {str(k): round(bar_start(k) / SR, 3) for k in range(NB + 1)},
           "structure": sections}, open("out/static_bar_times.json", "w"), indent=1)
log(f"MASTER rms {fx.rms_db(mix):.1f} dBFS (core {fx.rms_db(mix[CORE]):.1f}) peak {np.abs(mix).max():.2f} {N / SR:.0f}s -> {OUT}.mp3, out/static_bar_times.json")
