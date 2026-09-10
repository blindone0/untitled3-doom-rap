"""Process the recorded vocal takes (vocals/*.wav), mix them onto the beat and master the result.

  python mixvocal.py [--offset-ms 0] [--vocal-db 0] [--voice crypto|clean] [--beat ...] [--out out/oleni_doom_rap]

Takes: vocals/<section>_takeN.wav (+ .json with song_start); vocals/selection.json picks the take per section.
Section names ending in _L / _R are panned (doubled hook); 'adlib' / 'whisper' in the name = more space, -5 dB.
--voice crypto: disguised dark voice (pitch -2, octave-down monster layer, megaphone grit, ring-mod buzz, dark echoes).
"""
import argparse
import glob
import json
import os
import re
import subprocess

import librosa
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import maximum_filter1d, uniform_filter1d

import mixfx as fx

SR = 44100
ap = argparse.ArgumentParser()
ap.add_argument("--beat", default="out/doom_cover_heavy_instrumental.wav")
ap.add_argument("--takes", default="vocals")
ap.add_argument("--offset-ms", type=float, default=None, help="shift all takes earlier by this (default: vocals/latency.json or 0)")
ap.add_argument("--vocal-db", type=float, default=0.0, help="vocal level trim in dB")
ap.add_argument("--voice", default="lovell", choices=["lovell", "crypto", "clean"])
ap.add_argument("--drop", type=float, default=2.0, help="lovell: semitones to lower the voice")
ap.add_argument("--retune-ms", type=float, default=40.0, help="lovell: auto-tune retune speed (smaller = harder tune)")
ap.add_argument("--formant", type=float, default=0.97, help="lovell: formant ratio (<1 = deeper / darker timbre)")
ap.add_argument("--scale", default="Am", help="lovell: key for pitch correction, e.g. Am, Em, C")
ap.add_argument("--tune", type=float, default=1.0, help="lovell: correction strength 0..1")
ap.add_argument("--out", default="out/oleni_doom_rap")
ap.add_argument("--dry", action="store_true", help="no delay/reverb on the vocal")
ap.add_argument("--loud", type=float, default=-9.5, help="master RMS target dBFS")
args = ap.parse_args()

beat, bsr = sf.read(args.beat)
if beat.ndim == 1:
    beat = np.stack([beat, beat], 1)
N = len(beat)
lat_ms = args.offset_ms
if lat_ms is None:
    p = os.path.join(args.takes, "latency.json")
    lat_ms = json.load(open(p))["latency_ms"] if os.path.exists(p) else 0.0
print(f"beat {N / SR:.0f}s, latency compensation {lat_ms:.0f} ms, voice={args.voice}")


# ---------------- repair ----------------
def declip(x, thr=0.985):
    """rebuild clipped peaks with a cubic fit through the neighbours (mic was too hot)"""
    clipped = np.abs(x) >= thr
    if not clipped.any():
        return x, 0
    idx = np.where(clipped)[0]
    brk = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[brk + 1]]
    ends = np.r_[idx[brk], idx[-1]]
    y = x.copy()
    n = len(x)
    for a, b in zip(starts, ends):
        lo, hi = max(0, a - 6), min(n - 1, b + 6)
        xs = np.r_[np.arange(lo, a), np.arange(b + 1, hi + 1)]
        if len(xs) < 5:
            continue
        pcoef = np.polyfit(xs - a, x[xs], 3)
        rec = np.polyval(pcoef, np.arange(a, b + 1) - a)
        sgn = np.sign(x[a]) if x[a] != 0 else 1.0
        y[a:b + 1] = sgn * np.clip(np.abs(rec), thr, 2.5)
    return y, len(starts)


def gate(x, thr_db=-48.0, hold_ms=120, release_ms=150):
    env = maximum_filter1d(np.abs(x), size=int(hold_ms / 1000 * SR))
    env = uniform_filter1d(env, size=int(release_ms / 1000 * SR))
    g = np.clip((20 * np.log10(env + 1e-9) - thr_db) / 12.0, 0, 1)
    return x * g


def deess(x, lo=5000, hi=9000, thr_db=-24, ratio=4.0):
    band = fx.hp(fx.lp(x, hi), lo)
    env = fx.env_follow(band, 1.0, 40.0)
    over = 20 * np.log10(env + 1e-9) - thr_db
    gr = np.where(over > 0, over * (1 - 1 / ratio), 0.0)
    return x - band + band * 10 ** (-gr / 20)


def norm(v, target=0.5):
    a = np.abs(v)
    return v / (np.percentile(a[a > 1e-4], 99.5) + 1e-9) * target if np.any(a > 1e-4) else v


def comp(v, **kw):
    y, _ = fx.compressor(v[:, None], **kw)
    return y[:, 0]


# ---------------- voice chains ----------------
def voice_clean(v):
    v = fx.peak(v, 300, -3.0, 1.2)
    v = fx.peak(v, 3000, 3.0, 1.0)
    v = fx.shelf(v, 9000, 2.0, "high")
    v = norm(v)
    v = deess(v)
    v = comp(v, thr_db=-20, ratio=4.0, attack_ms=5, release_ms=80, makeup_db=6)
    v = comp(v, thr_db=-14, ratio=2.0, attack_ms=30, release_ms=300, makeup_db=3)
    v = fx.saturate(v[:, None], drive=1.3, mix=0.3)[:, 0]
    return np.stack([v, v], 1)


def voice_crypto(v):
    """disguised, dark: main pitched down 2 semitones, octave-down monster layer, megaphone grit, ring-mod buzz"""
    v = norm(v)
    main = librosa.effects.pitch_shift(v.astype(np.float32), sr=SR, n_steps=-2.0).astype(np.float64)
    sub = librosa.effects.pitch_shift(v.astype(np.float32), sr=SR, n_steps=-12.0).astype(np.float64)
    sub = fx.lp(sub, 1200, 2)
    grit = fx.hp(fx.lp(v, 3500, 2), 400, 2)
    grit = np.tanh(4.0 * grit) / np.tanh(4.0)
    grit = fx.peak(grit, 1800, 4.0, 1.0)
    t = np.arange(len(v)) / SR
    buzz = main * (0.12 * np.sin(2 * np.pi * 33.0 * t))  # ring-mod growl, subtle
    main = fx.peak(main, 250, -2.5, 1.2)
    main = fx.peak(main, 2600, 3.0, 1.0)
    main = fx.shelf(main, 8000, 1.5, "high")
    main = deess(main)
    main = comp(main, thr_db=-20, ratio=4.0, attack_ms=5, release_ms=80, makeup_db=6)
    main = comp(main, thr_db=-14, ratio=2.0, attack_ms=30, release_ms=300, makeup_db=3)
    main = fx.saturate(main[:, None], drive=1.5, mix=0.4)[:, 0]
    sub = comp(norm(sub, 0.4), thr_db=-18, ratio=4.0, attack_ms=10, release_ms=120, makeup_db=4)
    grit = comp(norm(grit, 0.4), thr_db=-18, ratio=3.0, attack_ms=5, release_ms=100, makeup_db=3)
    d = int(0.009 * SR)  # Haas width on the grit layer
    gl, gr_ = np.zeros_like(grit), np.zeros_like(grit)
    gl[:len(grit) - d] = grit[d:]
    gr_[d:] = grit[:len(grit) - d]
    L = main + 0.35 * sub + 0.30 * gl + buzz
    R = main + 0.35 * sub + 0.30 * gr_ + buzz
    return np.stack([L, R], 1)


def scale_pcs(name):
    roots = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}
    minor = name.endswith("m")
    root = roots[name[:-1] if minor else name]
    steps = [0, 2, 3, 5, 7, 8, 10] if minor else [0, 2, 4, 5, 7, 9, 11]
    return sorted((root + s) % 12 for s in steps)


def voice_lovell(v):
    """auto-tuned, lowered, darkened voice via the WORLD vocoder (analysis -> f0 edit -> resynthesis)"""
    import pyworld as pw
    from scipy.ndimage import map_coordinates
    x = norm(v, 0.6).astype(np.float64)
    fp = 5.0
    f0, tt = pw.harvest(x, SR, f0_floor=55.0, f0_ceil=600.0, frame_period=fp)
    sp = pw.cheaptrick(x, f0, tt, SR)
    apr = pw.d4c(x, f0, tt, SR)
    pcs = scale_pcs(args.scale)
    voiced = f0 > 0
    midi = np.zeros_like(f0)
    midi[voiced] = 69 + 12 * np.log2(f0[voiced] / 440.0)
    # nearest scale note
    target = midi.copy()
    for i in np.where(voiced)[0]:
        m = midi[i]
        cands = [m2 for k in range(int(m) - 2, int(m) + 3) for m2 in [k] if k % 12 in pcs]
        target[i] = min(cands, key=lambda c: abs(c - m))
    corr = np.zeros_like(f0)
    corr[voiced] = (target[voiced] - midi[voiced]) * args.tune
    # retune speed: one-pole smoothing restarted at every voiced run
    a = np.exp(-fp / max(1.0, args.retune_ms))
    sm = np.zeros_like(corr)
    prev, prev_v = 0.0, False
    for i in range(len(corr)):
        if voiced[i]:
            prev = corr[i] if not prev_v else a * prev + (1 - a) * corr[i]
            sm[i] = prev
        else:
            prev = 0.0
        prev_v = bool(voiced[i])
    new_f0 = f0 * 2 ** (sm / 12.0) * 2 ** (-args.drop / 12.0)
    # formant shift: read the envelope at scaled frequency
    nb = sp.shape[1]
    if abs(args.formant - 1.0) > 1e-3:
        rows = np.repeat(np.arange(sp.shape[0])[:, None], nb, axis=1)
        cols = np.repeat((np.arange(nb) / args.formant)[None, :], sp.shape[0], axis=0)
        sp = np.exp(map_coordinates(np.log(sp + 1e-12), [rows, cols], order=1, mode="nearest"))
    y = pw.synthesize(new_f0, sp, apr, SR, frame_period=fp)
    y2 = pw.synthesize(new_f0 * 2 ** (8 / 1200.0), sp, apr, SR, frame_period=fp)  # slightly detuned double
    n = len(x)
    y = np.pad(y, (0, max(0, n - len(y))))[:n]
    y2 = np.pad(y2, (0, max(0, n - len(y2))))[:n]
    main = norm(y, 0.5)
    main = fx.hp(main, 80, 2)
    main = fx.peak(main, 220, -2.0, 1.2)
    main = fx.peak(main, 3000, 2.0, 1.0)
    main = fx.shelf(main, 9000, 1.5, "high")
    main = deess(main)
    main = comp(main, thr_db=-20, ratio=4.0, attack_ms=5, release_ms=80, makeup_db=6)
    main = comp(main, thr_db=-14, ratio=2.0, attack_ms=30, release_ms=300, makeup_db=3)
    main = fx.saturate(main[:, None], drive=1.15, mix=0.2)[:, 0]
    dbl = comp(norm(fx.hp(y2, 120, 2), 0.5), thr_db=-18, ratio=4.0, attack_ms=5, release_ms=100, makeup_db=4)
    d = int(0.016 * SR)
    dl, dr = np.zeros(n), np.zeros(n)
    dl[d:] = dbl[:n - d]
    dr[:n - d] = dbl[d:]
    L = main + 0.22 * dl
    R = main + 0.22 * dr
    print(f"    lovell: voiced {voiced.mean() * 100:.0f}%, median correction {np.median(np.abs(corr[voiced])) * 100 if voiced.any() else 0:.0f} cents, drop {args.drop} st, formant {args.formant}")
    return np.stack([L, R], 1)


# ---------------- takes ----------------
bus = np.zeros((N, 2))
fxsend = np.zeros((N, 2))
takes = sorted(glob.glob(os.path.join(args.takes, "*.wav")))
sel_path = os.path.join(args.takes, "selection.json")
selection = json.load(open(sel_path)) if os.path.exists(sel_path) else {}
chosen = set(selection.values())
if not takes:
    raise SystemExit("no takes found in vocals/ - record some with studio.py first")
try:
    from sections import window_of
except Exception:
    window_of = lambda name: None
for path in takes:
    name = os.path.splitext(os.path.basename(path))[0]
    section = re.sub(r"_take\d+$", "", name)
    if section in selection and name not in chosen:
        continue
    meta_path = os.path.splitext(path)[0] + ".json"
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {"song_start": 0.0}
    v, vsr = sf.read(path)
    if v.ndim > 1:
        v = v.mean(1)
    if vsr != SR:
        v = signal.resample_poly(v, SR, vsr)
    v = v - v.mean()
    v, nclip = declip(v)
    v = v / (np.abs(v).max() + 1e-9) * 0.9
    v = fx.hp(v, 90, 2)
    v = gate(v)
    kind = "adlib" if "adlib" in section.lower() else ("whisper" if "whisper" in section.lower() else "main")
    st = {"crypto": voice_crypto, "clean": voice_clean, "lovell": voice_lovell}[args.voice](v)
    if kind != "main":
        st *= 10 ** (-5 / 20)
    r = min(len(st), int(0.02 * SR))
    st[:r] *= np.linspace(0, 1, r)[:, None]
    st[-r:] *= np.linspace(1, 0, r)[:, None]
    w = window_of(section)
    if w:  # keep only this section's phrases
        w0, w1, _ = w
        a = int((w0 - meta["song_start"]) * SR)
        b = int((w1 - meta["song_start"]) * SR)
        f = int(0.05 * SR)
        if a > 0:
            st[:max(0, a - f)] = 0
            st[max(0, a - f):a] *= np.linspace(0, 1, a - max(0, a - f))[:, None]
        if 0 <= b < len(st):
            st[b + f:] = 0
            seg = st[b:b + f]
            seg *= np.linspace(1, 0, len(seg))[:, None]
    pan = -0.6 if section.endswith("_L") else (0.6 if section.endswith("_R") else 0.0)
    gl, gr_ = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
    st = np.stack([st[:, 0] * gl * 1.3, st[:, 1] * gr_ * 1.3], 1)
    start = int((meta["song_start"] - lat_ms / 1000) * SR)
    if start < 0:
        st = st[-start:]
        start = 0
    e = min(N, start + len(st))
    bus[start:e] += st[: e - start]
    fxsend[start:e] += st[: e - start] * (1.0 if kind == "main" else 1.8)
    print(f"  take {name:16s} ({section:8s}) at {meta['song_start']:6.1f}s len {len(v) / SR:5.1f}s pan {pan:+.1f} declipped {nclip} peaks")

# ---------------- vocal space ----------------
vox = bus.copy()
if not args.dry:
    D = int(60 / 70 / 2 * SR)  # 1/8
    slap = np.zeros((N, 2))
    slap[D:] = fx.lp(fxsend[: N - D], 3500) * 0.16
    slap[2 * D:] += fx.lp(fxsend[: N - 2 * D], 2500) * 0.06
    vox += slap
    if args.voice in ("crypto", "lovell"):  # dark quarter-note echoes with feedback
        Q = int(60 / 70 * SR)
        tap = fxsend.copy()
        g = 0.16 if args.voice == "crypto" else 0.07
        for k in range(1, 5):
            tap = fx.lp(tap, 2200)
            shifted = np.zeros((N, 2))
            shifted[k * Q:] = tap[: N - k * Q]
            vox += shifted[:, ::-1] * g if k % 2 else shifted * g
            g *= 0.5
    vox += fx.convolve(fxsend, fx.load_ir("assets/voxengo/Small Drum Room.wav")) * 0.14
    vox += fx.convolve(fxsend, fx.load_ir("assets/voxengo/Large Wide Echo Hall.wav", max_s=2.5)) * 0.08

# level: voiced parts ~1.5 dB under the beat, then trim
act = uniform_filter1d(np.abs(bus).max(1), int(0.3 * SR)) > 0.01
beat_rms = fx.rms_db(beat[act]) if act.any() else fx.rms_db(beat)
vox_rms = fx.rms_db(bus[act]) if act.any() else fx.rms_db(bus)
g = 10 ** ((beat_rms - 1.5 + args.vocal_db - vox_rms) / 20)
vox *= g
env = uniform_filter1d(np.abs(vox).max(1), int(0.05 * SR))
duck = 1 - 0.18 * np.clip(env / (np.percentile(env, 99) + 1e-9), 0, 1)
duck = uniform_filter1d(duck, int(0.08 * SR))
mix = beat * duck[:, None] + vox

# ---------------- master ----------------
mix, gr1 = fx.compressor(mix, thr_db=-14, ratio=2.0, attack_ms=30, release_ms=250, makeup_db=1.0)
mix = fx.saturate(mix, drive=1.2, mix=0.3)
mix = fx.hp(mix, 25)
mix = fx.shelf(mix, 5000, 1.0, "high")
core = mix[int(30 * SR):int(120 * SR)]
mix *= 10 ** ((args.loud - fx.rms_db(core)) / 20)
mix = np.tanh(mix * 0.85) / np.tanh(0.85)
mix = fx.limiter(mix, thr=0.97, lookahead_ms=4, release_ms=120)
mix = mix / (np.abs(mix).max() + 1e-9) * 0.97
sf.write(args.out + ".wav", mix.astype(np.float32), SR, subtype="PCM_16")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", args.out + ".wav", "-codec:a", "libmp3lame", "-b:a", "320k",
                "-metadata", "title=Олені біжать", "-metadata", "artist=Igor", args.out + ".mp3"], check=True)
print(f"vocal gain {20 * np.log10(g):+.1f} dB, beat rms {beat_rms:.1f} dB, master GR {gr1:.1f} dB, final rms {fx.rms_db(mix):.1f} dBFS, peak {np.abs(mix).max():.2f}")
print(f"wrote {args.out}.wav and {args.out}.mp3")
