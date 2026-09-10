"""Live-sounding doom cover: sampled drums (AVL Red Zeppelin), DI guitar samples (Unreal Standard Guitar) through
neural amp captures (GuitarML Proteus) and real cabinet IRs, soundfont bass, humanized timing / velocity / tempo,
and live-style mixing (room mics, bus compression, saturation)."""
import json
import subprocess
import sys
import time

import mido
import numpy as np
import soundfile as sf

import mixfx as fx
from ampsim import Proteus, cab, load_ir as load_cab
from drumkit import RZ, SFZKit, place_stereo
from gtrsampler import StandardGuitar
from sfsynth import BLOCK, render as sf_render

SR = 44100
BPM = 70.0
A = json.load(open("out/analysis.json"))
PB = json.load(open("out/perbeat.json"))["per_beat"]
M = json.load(open("out/melody.json"))
NOTES, BARS = M["notes"], M["bars"]
PHASE = 1
nbars = len(BARS)
INTRO_BARS, OUTRO_BARS = 2, 2
TOTAL_BEATS = (INTRO_BARS + nbars + OUTRO_BARS) * 4
rng = np.random.default_rng(2024)
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:5.0f}s] {msg}", flush=True)


# ---------------- chord chart & intensity (same logic as cover.py) ----------------
ROOTPC = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}


def root_of(ch):
    return ROOTPC[ch.replace("dim", "").rstrip("m")]


def gtr_root(pc):
    return 33 + ((pc - 9) % 12)  # A1 .. G#2 (drop-A register; low roots are pitched-down E2 samples)


def bass_root(pc):
    return 28 + ((pc - 4) % 12)  # E1 .. D#2 (standard 4-string range: an octave under the guitar for E..G#)


chart = []
for k in range(nbars):
    c = PB[PHASE + 4 * k:PHASE + 4 * k + 4]
    h1 = c[0] if c[0] == c[1] else (c[1] if c[1] == c[2] else c[0])
    h2 = c[2] if c[2] == c[3] else h1
    chart.append([h1, h2])
flat = [h for pair in chart for h in pair]
for i in range(1, len(flat) - 1):
    if flat[i] != flat[i - 1] and flat[i - 1] == flat[i + 1]:
        flat[i] = flat[i - 1]
chart = [[flat[2 * k], flat[2 * k + 1]] for k in range(nbars)]
mx = max(b["mix_db"] for b in BARS)
raw_lvl = [2 if (b["mix_db"] - mx) > -2.0 else (1 if (b["mix_db"] - mx) > -5.5 else 0) for b in BARS]
level = []
for k in range(nbars):
    blk = raw_lvl[(k // 4) * 4:(k // 4) * 4 + 4]
    level.append(0 if raw_lvl[k] == 0 else int(round(np.mean(blk))))
level[0:4] = [1, 1, 1, 1]
level[-2:] = [0, 0]

# ---------------- tempo map with human drift ----------------
# per-beat duration multipliers: slow random walk + section feel + final ritardando
nb_total = TOTAL_BEATS + 8
walk = np.zeros(nb_total)
v = 0.0
for i in range(nb_total):
    v = 0.92 * v + rng.normal(0, 0.004)
    walk[i] = v
walk = np.clip(walk, -0.02, 0.02)
feel = np.zeros(nb_total)
for k in range(nbars):
    q0 = (INTRO_BARS + k) * 4
    feel[q0:q0 + 4] = {0: +0.025, 1: 0.0, 2: -0.008}[level[k]]
q_out = (INTRO_BARS + nbars) * 4
feel[q_out:q_out + 8] = np.linspace(0.02, 0.35, 8)  # ritardando in the outro
beat_dur = 60.0 / BPM * (1 + walk + feel)
beat_start = np.concatenate([[0.0], np.cumsum(beat_dur)])  # seconds at each new-timeline beat


def t_of(b_orig):
    """original beat index (float) -> sample in the new timeline"""
    q = INTRO_BARS * 4 + (b_orig - PHASE)
    q = float(np.clip(q, 0, nb_total - 1e-6))
    i = int(q)
    return int((beat_start[i] + (q - i) * beat_dur[i]) * SR)


def dur_samples(b_orig, d):
    return t_of(b_orig + d) - t_of(b_orig)


N = int(beat_start[TOTAL_BEATS] * SR) + 8 * SR
INSTRUMENTAL = "--instrumental" in sys.argv  # no lead melody inside the song: room for a vocalist
json.dump({"bpm": BPM, "levels": level,
           "bars": {str(k): round(float(beat_start[(INTRO_BARS + k) * 4]), 3) for k in range(-INTRO_BARS, nbars + OUTRO_BARS + 1)}},
          open("out/bar_times.json", "w"), indent=1)

# ---------------- events ----------------
G = []      # (beat, dur, root, mute, up, vel, soft)
BASS = []   # (beat, dur, midi, vel)
DR = []     # (beat, name, vel)
LEAD = []   # (beat, dur, midi, vel)
F, MU = False, True


def gtr(beat, dur, pc_root, mute=False, up=False, vel=1.0, soft=False, bass=True, approach_midi=None):
    root = gtr_root(pc_root) if approach_midi is None else approach_midi
    G.append((beat, dur, root, mute, up, vel, soft))
    if bass:
        broot = bass_root(pc_root) if approach_midi is None else bass_root(approach_midi % 12)
        BASS.append((beat, dur if not mute else min(dur, 0.5), broot, vel * (0.7 if soft else 1.0)))


def drum(beat, name, vel):
    DR.append((beat, name, vel))


def flam(beat, name, vel):
    drum(beat - 0.03 * BPM / 60, name, vel * 0.6)
    drum(beat, name, vel)


def approach_pc(nxt_pc, cur_pc):
    """chromatic approach note below the next root (as a guitar midi), or None if the chord does not change"""
    if nxt_pc == cur_pc:
        return None
    m = gtr_root(nxt_pc) - 1
    return m if m >= 33 else gtr_root(nxt_pc) + 1


for k in range(nbars):
    b0 = PHASE + 4 * k
    p1, p2 = [root_of(c) for c in chart[k]]
    pn = root_of(chart[k + 1][0]) if k + 1 < nbars else 9
    lvl = level[k]
    last = (k % 8 == 7) and k + 1 < nbars
    first = (k % 8 == 0)
    ap = approach_pc(pn, p2)
    if lvl == 0:
        if p1 != p2:
            gtr(b0, 2, p1, vel=0.8, soft=True); gtr(b0 + 2, 2, p2, vel=0.75, soft=True)
        else:
            gtr(b0, 4, p1, vel=0.8, soft=True)
        drum(b0, "kick", 0.75); drum(b0 + 2, "ftom2", 0.5)
        for o in range(4):
            drum(b0 + o, "ride", 0.45 if o % 2 else 0.55)
        if first:
            drum(b0, "crash1", 0.5)
    elif lvl == 1:
        gtr(b0, 2, p1, vel=1.0)
        if k % 4 == 3:
            gtr(b0 + 2, 1.5, p2, vel=0.92)
            gtr(b0 + 3.5, 0.25, p2, mute=True, vel=0.8)
            gtr(b0 + 3.75, 0.25, p2, mute=True, up=True, vel=0.7)
        else:
            gtr(b0 + 2, 1.5, p2, vel=0.92)
            if ap is not None and k % 2 == 1:
                gtr(b0 + 3.5, 0.5, pn, mute=True, vel=0.85, approach_midi=ap)
            else:
                gtr(b0 + 3.5, 0.5, p2, mute=True, vel=0.85)
        drum(b0, "kick", 1.0); drum(b0 + 2, "kick", 0.95)
        drum(b0 + 1, "snare", 0.95); drum(b0 + 3, "snare", 1.0)
        if rng.random() < 0.5:
            drum(b0 + 3.5, "kick", 0.75)
        elif rng.random() < 0.5:
            drum(b0 + 2.5, "kick", 0.7)
        if rng.random() < 0.4:
            drum(b0 + 2.75, "snare", 0.22)
        if rng.random() < 0.3:
            drum(b0 + 0.75, "snare", 0.2)
        for o in range(8):
            drum(b0 + o / 2, "ride", 0.85 if o % 2 == 0 else 0.55)
        drum(b0 + 1, "hhp", 0.4); drum(b0 + 3, "hhp", 0.4)
        if first:
            drum(b0, "crash1", 0.9)
    else:
        gtr(b0, 1.5, p1, vel=1.0)
        gtr(b0 + 1.5, 1.0, p1, vel=0.95, up=(rng.random() < 0.3))
        if last or (k % 4 == 3):
            gtr(b0 + 2.5, 0.5, p2, vel=0.95)
            gtr(b0 + 3, 0.25, p2, mute=True, vel=0.8); gtr(b0 + 3.25, 0.25, p2, mute=True, up=True, vel=0.7)
            gtr(b0 + 3.5, 0.25, p2, mute=True, vel=0.85); gtr(b0 + 3.75, 0.25, pn if ap is not None else p2, mute=True, up=True, vel=0.8, approach_midi=ap)
        else:
            gtr(b0 + 2.5, 1.0, p2, vel=0.97)
            if ap is not None:
                gtr(b0 + 3.5, 0.5, pn, mute=True, vel=0.9, approach_midi=ap)
            else:
                gtr(b0 + 3.5, 0.5, p2, mute=True, vel=0.9)
        drum(b0, "kick", 1.0); drum(b0 + 1.5, "kick", 0.9); drum(b0 + 2.5, "kick", 0.95); drum(b0 + 3.5, "kick", 0.85)
        if rng.random() < 0.35:
            drum(b0 + 3.75, "kick", 0.7)
        drum(b0 + 1, "rim", 1.0)
        if last:
            flam(b0 + 3, "rim", 1.0)
        else:
            drum(b0 + 3, "rim", 1.0)
        if rng.random() < 0.5:
            drum(b0 + 1.75, "snare", 0.25)
        if rng.random() < 0.5:
            drum(b0 + 3.75, "snare", 0.28)
        bell = (k % 8) >= 4
        for o in range(8):
            if bell and o % 4 == 0:
                drum(b0 + o / 2, "bell", 0.9)
            else:
                drum(b0 + o / 2, "ride", 0.9 if o % 2 == 0 else 0.6)
        drum(b0 + 1, "hhp", 0.45); drum(b0 + 3, "hhp", 0.45)
        if k % 2 == 0:
            drum(b0, "crash1" if k % 4 == 0 else "crash2", 0.95)
        if k % 2 == 1 and rng.random() < 0.5:
            drum(b0 + 3.5, "hhs", 0.6)
    if last and lvl > 0:
        DR = [e for e in DR if not (b0 + 2.9 <= e[0] < b0 + 4 and e[1] in ("ride", "bell", "snare", "rim", "kick", "hhp"))]
        var = rng.integers(4)
        if var == 0:
            for i, (o, nm) in enumerate([(3, "tom"), (3.25, "tom"), (3.5, "ftom1"), (3.75, "ftom1"), (4 - 0.001, "ftom2")]):
                drum(b0 + o, nm, 0.7 + 0.075 * i)
            drum(b0 + 3, "kick", 0.9); drum(b0 + 3.5, "kick", 0.9)
        elif var == 1:
            drum(b0 + 3, "snare", 0.9); drum(b0 + 3.5, "snare", 0.95)
            for i, (o, nm) in enumerate([(3.75, "tom"), (4 - 0.25, "ftom1")]):
                drum(b0 + o, nm, 0.9)
            drum(b0 + 3, "kick", 0.9); drum(b0 + 3.5, "kick", 0.9)
        elif var == 2:
            flam(b0 + 3, "snare", 1.0); drum(b0 + 3.5, "ftom2", 1.0); drum(b0 + 3.5, "kick", 1.0); drum(b0 + 3.5, "crash2", 0.8)
        else:
            for i, o in enumerate([3, 3.25, 3.5, 3.75]):
                drum(b0 + o, "snare", 0.55 + 0.15 * i)
            drum(b0 + 3, "kick", 0.9); drum(b0 + 3.5, "kick", 0.95); drum(b0 + 3.75, "kick", 0.9)
        if lvl == 2 and rng.random() < 0.5:
            drum(b0 + 2, "china", 0.8)

# lead = vocal melody
for n in NOTES:
    if PHASE <= n["b"] < PHASE + 4 * nbars and not INSTRUMENTAL:
        LEAD.append((n["b"], n["d"], n["m"], float(np.clip(rng.normal(0.9, 0.06), 0.6, 1.0))))

# intro: soft chord + feedback swell + fill into bar 0
ib = PHASE - 8
gtr(ib, 4, 9, vel=0.85, soft=True); gtr(ib + 4, 4, 9, vel=1.0)
drum(ib, "crash1", 0.6); drum(ib, "kick", 0.8); drum(ib + 2, "ftom2", 0.5); drum(ib + 4, "kick", 1.0); drum(ib + 6, "kick", 0.9)
for o in range(6):
    drum(ib + o, "ride", 0.45)
for i, (o, nm) in enumerate([(6, "snare"), (6.5, "snare"), (7, "tom"), (7.25, "tom"), (7.5, "ftom1"), (7.75, "ftom2")]):
    drum(ib + o, nm, 0.7 + 0.05 * i)
drum(PHASE, "crash1", 1.0)
drum(PHASE, "china", 0.7)
# outro: two big hits, ritardando, ring out
ob = PHASE + 4 * nbars
gtr(ob, 4, 9, vel=1.0); gtr(ob + 4, 4.5, 9, vel=1.0)
drum(ob, "crash1", 1.0); drum(ob, "china", 0.9); drum(ob, "kick", 1.0); drum(ob + 3, "snare", 1.0); drum(ob + 3.5, "ftom2", 1.0)
drum(ob + 4, "crash2", 1.0); drum(ob + 4, "crash1", 0.9); drum(ob + 4, "kick", 1.0)
LEAD.append((ob, 8.0, 69, 0.95))

log(f"events: gtr {len(G)} bass {len(BASS)} drums {len(DR)} lead {len(LEAD)}; length {N / SR:.0f}s")
print("chart / level:")
for k in range(0, nbars, 8):
    print(f"bar {k:2d}: " + "  ".join(f"{chart[j][0]:>3}{'|' + chart[j][1] if chart[j][1] != chart[j][0] else '':<4}L{level[j]}" for j in range(k, min(k + 8, nbars))))

# ---------------- render: guitars ----------------
guitar = StandardGuitar("assets/stdguitar/UI_Standard_Guitar/Samples")
P = "assets/proteus/"
TAKES = [  # (channel, timing bias s, detune cents, pedal model, amp model, cab IR, pan, level dB)
    (0, +0.003, -3.0, P + "TS9_HighDrive.json", P + "6505Plus_Red_DirectOut.json", "assets/ir/0Miscellaneous/5150 IC V1.wav", -0.9, -20.0),
    (1, -0.002, +3.0, P + "TS9_HighDrive.json", P + "MesaMiniRec_HighGain_DirectOut.json", "assets/ir/0Miscellaneous/SMG_UK_V30 57_dc.wav", +0.9, -20.0),
    (1, +0.006, -6.0, P + "LittleBigMuff_HighGainPedal.json", P + "6505Plus_Red_DirectOut.json", "assets/ir/0Miscellaneous/Mesa Modern IC.wav", -0.6, -26.0),
    (0, -0.005, +6.0, P + "ProcoRatPedal_HighGain.json", P + "MesaMiniRec_HighGain_DirectOut.json", "assets/ir/1960a/1960a Au i5 misc/jr4x12-i5-1in-2c.wav", +0.6, -26.0),
]
gtr_tracks = []
for ti, (chan, bias, cents, pedal_path, amp_path, ir_path, pan, lvl_db) in enumerate(TAKES):
    trng = np.random.default_rng(100 + ti)
    di = np.zeros(N)
    for (beat, dur, root, mute, up, vel, soft) in G:
        s = t_of(beat) + int((bias + trng.normal(0, 0.009)) * SR)
        n = dur_samples(beat, dur) - int(0.006 * SR)
        if n < 400:
            continue
        v = float(np.clip(vel * trng.normal(1.0, 0.07), 0.3, 1.0))
        art_n = min(n, int(1.0 * SR)) if mute else n
        sig = guitar.power_chord(root, art_n, v, trng, mute=mute, up=up, chan=chan, drift_cents=cents, long=(n > 3.8 * SR))
        if soft:
            sig *= 0.3
        s = max(0, s)
        e = min(N, s + len(sig))
        di[s:e] += sig[: e - s]
    di *= 0.55 / (np.percentile(np.abs(di), 99.9) + 1e-9)
    log(f"guitar take {ti} DI done, peak {np.abs(di).max():.2f}")
    if pedal_path:
        y = Proteus(pedal_path).process(np.clip(di, -1, 1))
        y *= 0.7 / (np.percentile(np.abs(y), 99.9) + 1e-9)  # pedal into the front of the amp, hot
    else:
        y = di
    y = Proteus(amp_path).process(np.clip(y, -1, 1))
    y = cab(y, load_cab(ir_path))
    y = fx.hp(y, 55)
    y = fx.lp(y, 9500)
    y = fx.peak(y, 300, -2.0, 1.0)
    y = fx.peak(y, 3200, 1.5, 1.0)
    gtr_tracks.append((y, pan, lvl_db))
    log(f"guitar take {ti} pedal+amp+cab done, rms {fx.rms_db(y):.1f} dB")

# ---------------- render: bass ----------------
bev = []
for (beat, dur, midi, vel) in BASS:
    s = t_of(beat) + int((0.006 + rng.normal(0, 0.008)) * SR)
    n = dur_samples(beat, dur) - int(0.01 * SR)
    if n > 300:
        bev.append({"s": max(0, s), "n": n, "m": midi, "v": float(np.clip(vel * rng.normal(0.95, 0.06), 0.3, 1.0))})
bass_di = sf_render("assets/GeneralUser-GS.sf2", 34, bev, N, gain_db=0.0)
bass_di *= 0.5 / (np.percentile(np.abs(bass_di), 99.9) + 1e-9)
bass_lo = fx.lp(bass_di, 180, 2)
bass_hi = fx.hp(bass_di, 180, 2)
rat = Proteus("assets/proteus/ProcoRatPedal_HighGain.json")
bass_hi = rat.process(np.clip(bass_hi * 1.0, -1, 1))
bass_hi = cab(bass_hi, load_cab("assets/ir/Bass IR Shiftline/07_Ampeg_SVT-810E_by_Shift_Line.wav", max_ms=300))
bass_hi = fx.hp(bass_hi, 150)
bass_lo = np.tanh(2.6 * bass_lo) / np.tanh(2.6)
bass = bass_lo * 1.0 + bass_hi * (1.3 * np.sqrt((bass_lo ** 2).mean()) / (np.sqrt((bass_hi ** 2).mean()) + 1e-9))
bass, _ = fx.compressor(bass[:, None], thr_db=-16, ratio=4, attack_ms=10, release_ms=120, makeup_db=4)
bass = bass[:, 0]
bass = fx.peak(bass, 800, 2.0, 1.0)  # growl so the bass reads on small speakers too
log(f"bass done, rms {fx.rms_db(bass):.1f} dB")

# ---------------- render: drums ----------------
kit = SFZKit("assets/avl/AVL_Drumkits_1.0/Red_Zeppelin_5pc.sfz")
close = np.zeros((N, 2))
room_in = np.zeros((N, 2))
OFF = {"kick": -0.003, "snare": 0.010, "rim": 0.010, "ride": 0.0, "bell": 0.0, "crash1": 0.004, "crash2": 0.004, "china": 0.004,
       "tom": 0.003, "ftom1": 0.003, "ftom2": 0.003, "hhp": 0.0, "hhs": 0.0, "hhc": 0.0, "hho": 0.0}
GAIN = {"kick": 1.0, "snare": 0.9, "rim": 0.85, "ride": 0.7, "bell": 0.75, "crash1": 0.75, "crash2": 0.75, "china": 0.7,
        "tom": 0.85, "ftom1": 0.85, "ftom2": 0.9, "hhp": 0.5, "hhs": 0.6, "hhc": 0.5, "hho": 0.6}
PANFIX = {"kick": 0.0, "snare": 0.0, "rim": 0.0}
drng = np.random.default_rng(3)
for (beat, name, vel) in DR:
    s = t_of(beat) + int((OFF[name] + drng.normal(0, 0.006)) * SR)
    v = float(np.clip(vel * drng.normal(1.0, 0.06), 0.1, 1.0))
    x, pan = kit.hit(RZ[name], v, drng)
    pan = PANFIX.get(name, pan)
    place_stereo(close, s, x, pan, GAIN[name])
    place_stereo(room_in, s, x, pan * 0.6, GAIN[name])
close = fx.peak(close, 55, 3.5, 0.9)
close = fx.peak(close, 3800, 2.5, 1.0)
close = fx.peak(close, 350, -2.0, 1.2)
room = fx.convolve(room_in, fx.load_ir("assets/voxengo/Nice Drum Room.wav"))
room_sm, gr = fx.compressor(room, thr_db=-34, ratio=10, attack_ms=2, release_ms=180, makeup_db=16)
drums = close + room_sm * 0.75 + room * 0.3
drums, gr2 = fx.compressor(drums, thr_db=-16, ratio=4, attack_ms=20, release_ms=160, makeup_db=3)
drums = fx.saturate(drums, drive=1.6, mix=0.6)
drums = fx.shelf(drums, 7000, 3.0, "high")
del close, room_in, room, room_sm
log(f"drums done, rms {fx.rms_db(drums):.1f} dB (room GR {gr:.0f} dB, bus GR {gr2:.0f} dB)")

# ---------------- render: lead ----------------
lead_di = np.zeros(N)
lrng = np.random.default_rng(11)
LEAD.sort(key=lambda e: e[0])
prev_end, prev_m = None, None
for (beat, dur, m, vel) in LEAD:
    s = t_of(beat) + int((0.015 + lrng.normal(0, 0.012)) * SR)
    n = dur_samples(beat, dur) - int(0.004 * SR)
    if n < 300:
        continue
    t = np.arange(n) / SR
    long_note = dur >= 0.5
    depth = (0.35 if dur >= 1.0 else 0.22) if long_note else 0.0
    ramp = np.clip((t - 0.22) / 0.4, 0, 1)
    fr = 4.7 + lrng.normal(0, 0.3)
    bend = depth * ramp * 0.5 * (1 - np.cos(2 * np.pi * fr * t + lrng.uniform(0, 6.28))) + lrng.normal(0, 0.03)
    legato = prev_end is not None and abs(beat - prev_end) < 0.01 and prev_m is not None and abs(m - prev_m) <= 4
    art = "long" if dur >= 3.0 else "sus_down"
    sig = guitar.note(art, m, n, vel, lrng, chan=None, bend=bend, glide_from=(prev_m if legato else None), glide_ms=70)
    s = max(0, s)
    e = min(N, s + len(sig))
    lead_di[s:e] += sig[: e - s]
    prev_end, prev_m = beat + dur, m
# feedback swell during the intro
n = dur_samples(ib, 8.0)
t = np.arange(n) / SR
bend = 0.45 * np.clip((t - 1.0) / 2.0, 0, 1) * 0.5 * (1 - np.cos(2 * np.pi * 4.2 * t))
sw = guitar.note("long", 64, n, 1.0, lrng, bend=bend) * (np.linspace(0, 1, n) ** 2.2)
s = t_of(ib)
lead_di[s:s + n] += sw
lead_di *= 0.45 / (np.percentile(np.abs(lead_di), 99.9) + 1e-9)
lead_amp = Proteus("assets/proteus/MesaIICplus_Drive8_5EQoff.json")
lead = lead_amp.process(np.clip(lead_di, -1, 1))
lead = cab(lead, load_cab("assets/ir/0Miscellaneous/Mesa c414 XL2.wav"))
lead = fx.hp(lead, 140)
lead = fx.peak(lead, 2500, 2.0, 1.0)
lead = fx.shelf(lead, 5000, 1.5, "high")
lead, _ = fx.compressor(lead[:, None], thr_db=-18, ratio=3, attack_ms=15, release_ms=150, makeup_db=3)
lead = lead[:, 0]
# delay (3/16 note, ping-pong) + hall
D = int(beat_dur[TOTAL_BEATS // 2] * 0.75 * SR)
lead_st = np.stack([lead, lead], 1)
tap = lead.copy()
g = 0.30
for r in range(1, 6):
    tap = fx.lp(tap, 3200)
    shifted = np.zeros(N)
    shifted[r * D:] = tap[: N - r * D]
    lead_st[:, r % 2] += shifted * g
    g *= 0.42
hall = fx.convolve(lead_st, fx.load_ir("assets/voxengo/Large Wide Echo Hall.wav", max_s=3.0))
lead_st = lead_st + hall * 0.22
log(f"lead done, rms {fx.rms_db(lead_st):.1f} dB")

# ---------------- mix ----------------
def level_to(x, target_db):
    return x * 10 ** ((target_db - fx.rms_db(x)) / 20)


mix = np.zeros((N, 2))
for (y, pan, lvl_db) in gtr_tracks:
    th = (pan + 1) * np.pi / 4
    y = level_to(y, lvl_db)
    mix[:, 0] += y * np.cos(th)
    mix[:, 1] += y * np.sin(th)
bass_l = level_to(bass, -17.0)
mix[:, 0] += bass_l
mix[:, 1] += bass_l
mix += level_to(drums, -18.5)
mix += lead_st * 10 ** (-11.1 / 20)  # fixed gain (matches the full version) so a sparse lead is not normalized up
# everyone in the same room (a little live-room glue) + bus processing
mix = mix + fx.convolve(mix, fx.load_ir("assets/voxengo/Small Drum Room.wav")) * 0.08
del gtr_tracks, drums, lead_st, hall, bass
# section dynamics: the band plays softer in quiet parts (gain rides before the bus compressor)
auto_db = np.zeros(N)
for k in range(nbars):
    auto_db[t_of(PHASE + 4 * k):t_of(PHASE + 4 * k + 4)] = {0: -5.0, 1: -1.5, 2: 0.0}[level[k]]
auto_db[t_of(ib):t_of(ib + 4)] = -6.0
auto_db[t_of(ib + 4):t_of(ib + 8)] = -2.5
from scipy.ndimage import uniform_filter1d
auto = 10 ** (uniform_filter1d(auto_db, int(0.12 * SR)) / 20)
mix *= auto[:, None]
mix, gr3 = fx.compressor(mix, thr_db=-16, ratio=2.5, attack_ms=30, release_ms=250, makeup_db=2.0)
mix = fx.saturate(mix, drive=1.4, mix=0.45)
mix = fx.hp(mix, 26)
mix = fx.shelf(mix, 6000, 2.5, "high")
# fade out after the last outro hit rings
fade_start = t_of(ob + 6.0)
mix[fade_start:] *= (np.linspace(1, 0, N - fade_start) ** 1.4)[:, None]
core = mix[t_of(PHASE + 32):t_of(PHASE + 64)]
mix *= 10 ** ((-11.5 - fx.rms_db(core)) / 20)
mix = np.tanh(mix * 0.9) / np.tanh(0.9)
mix = fx.limiter(mix, thr=0.97, lookahead_ms=3, release_ms=120)
mix = mix / (np.abs(mix).max() + 1e-9) * 0.97
log(f"mix done: rms {fx.rms_db(mix):.1f} dBFS, glue GR {gr3:.1f} dB, duration {N / SR:.0f}s")
OUT = "out/doom_cover_heavy" + ("_instrumental" if INSTRUMENTAL else "")
sf.write(OUT + ".wav", mix.astype(np.float32), SR, subtype="PCM_16")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", OUT + ".wav", "-codec:a", "libmp3lame", "-b:a", "320k",
                "-metadata", "title=Oleni (doom metal cover, heavy drop-A version)", "-metadata", "artist=Claude x TIK", OUT + ".mp3"], check=True)

# ---------------- MIDI (nominal, unhumanized) ----------------
mid = mido.MidiFile(ticks_per_beat=480)
tt = mido.MidiTrack()
tt.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(BPM), time=0))
tt.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
mid.tracks.append(tt)


def track(name, program, ch):
    t = mido.MidiTrack()
    t.append(mido.MetaMessage("track_name", name=name, time=0))
    if program is not None:
        t.append(mido.Message("program_change", program=program, channel=ch, time=0))
    return t


def write_notes(t, evs, ch):
    msgs = []
    for (b, d, n, v) in evs:
        msgs.append((b, 1, mido.Message("note_on", note=int(n), velocity=int(np.clip(v * 110, 1, 127)), channel=ch, time=0)))
        msgs.append((b + max(d, 0.1) - 0.02, 0, mido.Message("note_off", note=int(n), velocity=0, channel=ch, time=0)))
    msgs.sort(key=lambda x: (x[0], x[1]))
    last = 0.0
    for (b, _, m) in msgs:
        m.time = int(round((b - last) * 480))
        t.append(m)
        last = b


def nb(beat):
    return INTRO_BARS * 4 + (beat - PHASE)


tg = track("Rhythm guitar", 30, 0)
write_notes(tg, [(nb(b), d, r + iv, v * (0.7 if mu else 1.0)) for (b, d, r, mu, up, v, so) in G for iv in (0, 7, 12)], 0)
mid.tracks.append(tg)
tb = track("Bass", 34, 1)
write_notes(tb, [(nb(b), d, m, v) for (b, d, m, v) in BASS], 1)
mid.tracks.append(tb)
tl = track("Lead guitar (vocal melody)", 30, 2)
write_notes(tl, [(nb(b), d, m, v) for (b, d, m, v) in LEAD], 2)
mid.tracks.append(tl)
GM = {"kick": 36, "snare": 38, "rim": 40, "tom": 48, "ftom1": 45, "ftom2": 41, "crash1": 49, "crash2": 57, "china": 52,
      "ride": 51, "bell": 53, "hhp": 44, "hhs": 46, "hhc": 42, "hho": 46}
td = track("Drums", None, 9)
write_notes(td, [(nb(b), 0.25, GM[n], v) for (b, n, v) in DR], 9)
mid.tracks.append(td)
mid.save(OUT + ".mid")
log(f"wrote {OUT}.wav, .mp3, .mid")
