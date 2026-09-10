"""Arrange and render the doom-metal cover from the analysis (chords, structure) and the extracted melody."""
import json
import subprocess
import sys
import time

import mido
import numpy as np
import soundfile as sf

from doomsynth import (SR, add, amp_bass, amp_guitar, amp_lead, bass_note, convolve_stereo, delay_fx, drum, hp,
                       lead_note, limiter, midi2f, power_chord, reverb_ir, shelf)

BPM = 70.0
spb = 60.0 / BPM * SR  # samples per beat
A = json.load(open("out/analysis.json"))
PB = json.load(open("out/perbeat.json"))["per_beat"]
M = json.load(open("out/melody.json"))
NOTES, BARS = M["notes"], M["bars"]
PHASE = 1
nbars = len(BARS)
INTRO_BARS, OUTRO_BARS = 2, 2
TOTAL_BEATS = (INTRO_BARS + nbars + OUTRO_BARS) * 4
N = int(TOTAL_BEATS * spb) + 7 * SR


def b2s(b):
    """original beat index (float) -> sample position in the new timeline"""
    return int(round((INTRO_BARS * 4 + (b - PHASE)) * spb))


def add1(buf, s, sig):
    """add a mono signal into a mono buffer at sample s, clipping both ends"""
    if s < 0:
        sig = sig[-s:]
        s = 0
    e = min(len(buf), s + len(sig))
    if e > s:
        buf[s:e] += sig[: e - s]


ROOTPC = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}


def root_of(ch):
    return ROOTPC[ch.replace("dim", "").rstrip("m")]


def root_midi(pc):
    return 33 + ((pc - 9) % 12)  # A1 .. G#2 (drop-A register)


# ---------- chord chart: two half-bar roots per bar ----------
chart = []
for k in range(nbars):
    i = PHASE + 4 * k
    c = PB[i:i + 4]
    h1 = c[0] if c[0] == c[1] else (c[1] if c[1] == c[2] else c[0])
    h2 = c[2] if c[2] == c[3] else h1
    chart.append([h1, h2])
flat = [h for pair in chart for h in pair]
for i in range(1, len(flat) - 1):  # isolated half-bar chord between identical chords -> noise
    if flat[i] != flat[i - 1] and flat[i - 1] == flat[i + 1]:
        flat[i] = flat[i - 1]
chart = [[flat[2 * k], flat[2 * k + 1]] for k in range(nbars)]

# ---------- intensity levels per bar ----------
mx = max(b["mix_db"] for b in BARS)
raw_lvl = []
for b in BARS:
    d = b["mix_db"] - mx
    raw_lvl.append(2 if d > -2.0 else (1 if d > -5.5 else 0))
level = []
for k in range(nbars):
    blk = raw_lvl[(k // 4) * 4:(k // 4) * 4 + 4]
    l = int(round(np.mean(blk)))
    if raw_lvl[k] == 0:
        l = 0
    level.append(l)
level[0:4] = [1, 1, 1, 1]  # riff intro before the first vocal
level[-2:] = [0, 0]

print("chart / level:")
for k in range(0, nbars, 8):
    print(f"bar {k:2d}: " + "  ".join(f"{chart[j][0]:>3}{'|' + chart[j][1] if chart[j][1] != chart[j][0] else '':<4}L{level[j]}" for j in range(k, min(k + 8, nbars))))

# ---------- events ----------
rng = np.random.default_rng(42)
G = [[], []]  # per take: (beat, dur, root_midi, mute, vel)
BASS = []     # (beat, dur, root_midi, vel)
DR = []       # (beat, name, vel)
LEAD = []     # (beat, dur, midi, vel)
F, MU = False, True


def gtr(beat, dur, root, mute, vel, soft=False):
    for tk in range(2):
        G[tk].append((beat, dur, root, mute, vel, soft))
    BASS.append((beat, dur if not mute else min(dur, 0.5), root, vel * (0.7 if soft else 1.0)))


def approach(nxt, cur):
    if nxt == cur:
        return cur
    a = nxt - 1
    return a if a >= 33 else nxt + 1


song_start = PHASE
for k in range(nbars):
    b0 = PHASE + 4 * k
    r1, r2 = [root_midi(root_of(c)) for c in chart[k]]
    nxt = root_midi(root_of(chart[k + 1][0])) if k + 1 < nbars else 33
    lvl = level[k]
    last = (k % 8 == 7) and k + 1 < nbars

    def R(o):
        return r1 if o < 2 else r2

    if lvl == 0:
        if r1 != r2:
            gtr(b0, 2, r1, F, 0.9, soft=True); gtr(b0 + 2, 2, r2, F, 0.85, soft=True)
        else:
            gtr(b0, 4, r1, F, 0.9, soft=True)
        DR += [(b0, "kick", 0.7), (b0 + 2, "kick", 0.55)]
        DR += [(b0 + o, "ride", 0.45) for o in range(4)]
        if k % 8 == 0:
            DR.append((b0, "crash", 0.5))
    elif lvl == 1:
        gtr(b0, 2, r1, F, 1.0); gtr(b0 + 2, 1.5, r2, F, 0.9)
        gtr(b0 + 3.5, 0.5, approach(nxt, r2) if k % 2 == 1 else r2, MU, 0.85)
        DR += [(b0, "kick", 1.0), (b0 + 2, "kick", 0.95), (b0 + 3.5, "kick", 0.7), (b0 + 1, "snare", 0.95), (b0 + 3, "snare", 1.0)]
        DR += [(b0 + o / 2, "ride", 0.75 if o % 2 == 0 else 0.5) for o in range(8)]
        if k % 8 == 0:
            DR.append((b0, "crash", 0.85))
    else:
        gtr(b0, 1.5, r1, F, 1.0); gtr(b0 + 1.5, 1.0, r1, F, 0.9); gtr(b0 + 2.5, 1.0, r2, F, 0.95)
        gtr(b0 + 3.5, 0.5, approach(nxt, r2), MU, 0.9)
        DR += [(b0, "kick", 1.0), (b0 + 1.5, "kick", 0.9), (b0 + 2, "kick", 0.9), (b0 + 3.5, "kick", 0.8),
               (b0 + 1, "snare", 1.0), (b0 + 3, "snare", 1.0)]
        DR += [(b0 + o / 2, "ride", 0.8 if o % 2 == 0 else 0.55) for o in range(8)]
        if k % 2 == 0:
            DR.append((b0, "crash", 0.9))
    if last and lvl > 0:
        DR = [e for e in DR if not (b0 + 2 <= e[0] < b0 + 4 and e[1] in ("ride", "snare", "kick"))]
        DR += [(b0 + 2, "kick", 1.0), (b0 + 2, "snare", 1.0), (b0 + 2.5, "snare", 0.8), (b0 + 3, "tomhi", 0.9),
               (b0 + 3.25, "tomhi", 0.8), (b0 + 3.5, "tommid", 0.95), (b0 + 3.75, "tomlo", 1.0), (b0 + 3, "kick", 0.9)]
        if lvl == 2:
            DR += [(b0 + 2.25, "kick", 0.8), (b0 + 2.75, "kick", 0.8)]

# lead from melody (only inside the song)
for n in NOTES:
    if PHASE <= n["b"] < PHASE + 4 * nbars:
        LEAD.append((n["b"], n["d"], n["m"], 1.0))

# intro: two bars of the tonic drone + feedback swell + fill into bar 0
ib = PHASE - 8
gtr(ib, 4, 33, F, 0.9, soft=True); gtr(ib + 4, 4, 33, F, 1.0, soft=True)
DR += [(ib, "crash", 0.6), (ib, "kick", 0.8), (ib + 2, "kick", 0.6), (ib + 4, "kick", 0.9), (ib + 6, "kick", 0.9), (ib + 6, "snare", 0.9),
       (ib + 6.5, "snare", 0.7), (ib + 7, "tomhi", 0.9), (ib + 7.25, "tomhi", 0.8), (ib + 7.5, "tommid", 0.95), (ib + 7.75, "tomlo", 1.0)]
DR += [(ib + o, "ride", 0.4) for o in range(6)]
DR.append((PHASE, "crash", 1.0))
# outro: two final hits, everything rings out
ob = PHASE + 4 * nbars
gtr(ob, 4, 33, F, 1.0); gtr(ob + 4, 4.5, 33, F, 1.0)
DR += [(ob, "crash", 1.0), (ob, "kick", 1.0), (ob + 4, "crash", 1.0), (ob + 4, "kick", 1.0), (ob + 3, "snare", 1.0), (ob + 3.5, "snare", 0.9)]
LEAD.append((ob, 8.5, 69, 0.9))  # long A4 with vibrato
LEAD_SWELL = (ib, 8.0, 64)        # feedback E4 rising during the intro

# ---------- render ----------
t0 = time.time()
print("rendering guitars...", flush=True)
gtr_out = []
for tk in range(2):
    buf = np.zeros(N)
    soft_buf = np.zeros(N)
    trng = np.random.default_rng(100 + tk)
    for (beat, dur, root, mute, vel, soft) in G[tk]:
        s = b2s(beat) + int(trng.normal(0, 0.006) * SR)
        n = int(dur * spb) - int(0.004 * SR)
        add1(soft_buf if soft else buf, s, power_chord(root, n, trng, vel, mute))
    out = amp_guitar(buf, gain=22.0, cab_lp=4600 + 500 * tk, presence=4.0 + tk)
    # quiet sections: crunchy low-gain chain, mixed lower -> dynamic contrast
    out += 0.55 * amp_guitar(soft_buf, gain=4.0, cab_lp=4200.0, presence=2.0, low_hp=60.0)
    gtr_out.append(out)
    del buf, soft_buf
print(f"  guitars done {time.time() - t0:.0f}s", flush=True)

print("rendering bass...", flush=True)
buf = np.zeros(N)
brng = np.random.default_rng(7)
for (beat, dur, root, vel) in BASS:
    s = b2s(beat)
    n = int(dur * spb) - int(0.004 * SR)
    add1(buf, s, bass_note(midi2f(root), n, brng, vel))
bass_out = amp_bass(buf)
del buf

print("rendering drums...", flush=True)
drums = np.zeros((N, 2))
drums_verb = np.zeros((N, 2))
PAN = {"kick": 0.0, "snare": 0.0, "tomhi": -0.35, "tommid": 0.0, "tomlo": 0.4, "crash": 0.3, "ride": -0.3, "hat": -0.4}
GAIN = {"kick": 1.0, "snare": 0.85, "tomhi": 0.7, "tommid": 0.7, "tomlo": 0.75, "crash": 0.5, "ride": 0.36, "hat": 0.2}
VERB = {"kick": 0.05, "snare": 0.35, "tomhi": 0.3, "tommid": 0.3, "tomlo": 0.3, "crash": 0.25, "ride": 0.12, "hat": 0.1}
drng = np.random.default_rng(3)
for (beat, name, vel) in DR:
    s = b2s(beat) + int(drng.normal(0, 0.003) * SR)
    sig = drum(name) * vel * GAIN[name] * drng.uniform(0.92, 1.0)
    p = PAN[name]
    st = np.stack([sig * min(1, 1 - p), sig * min(1, 1 + p)], 1)
    add(drums, s, st)
    add(drums_verb, s, st, VERB[name])

print("rendering lead...", flush=True)
lead = np.zeros(N)
lead_sub = np.zeros(N)
lrng = np.random.default_rng(11)
LEAD.sort(key=lambda e: e[0])
prev_end, prev_f = -1, None
for (beat, dur, m, vel) in LEAD:
    s = b2s(beat)
    n = int(dur * spb) - int(0.005 * SR)
    f = midi2f(m)
    f_from = prev_f if (prev_end is not None and abs(beat - prev_end) < 0.01) else None
    add1(lead, s, lead_note(f, n, lrng, vel, f_from=f_from))
    add1(lead_sub, s, lead_note(f / 2, n, lrng, vel * 0.8, f_from=(f_from / 2 if f_from else None), vib_depth=10))
    prev_end, prev_f = beat + dur, f
# feedback swell in the intro
sb, sd, sm_ = LEAD_SWELL
n = int(sd * spb)
sw = lead_note(midi2f(sm_), n, lrng, 1.0, vib_depth=35, vib_rate=4.5)
ramp = (np.linspace(0, 1, n) ** 2.5) * 0.9
add1(lead, b2s(sb), sw * ramp)
lead_amp = amp_lead(lead)
sub_amp = amp_guitar(lead_sub, gain=18.0, cab_lp=3500.0, presence=2.0, low_hp=80.0)
del lead, lead_sub
lead_st = delay_fx(lead_amp, 60.0 / BPM * 0.75, fb=0.35, mix=0.28, damp=3000)[:N]
print(f"  lead done {time.time() - t0:.0f}s", flush=True)

print("mixing...", flush=True)
mix = np.zeros((N, 2))
gL, gR = gtr_out
mix[:, 0] += 0.33 * gL + 0.06 * gR
mix[:, 1] += 0.33 * gR + 0.06 * gL
mix[:, 0] += 0.30 * bass_out
mix[:, 1] += 0.30 * bass_out
mix += drums * 0.95
mix += lead_st * 0.46
mix[:, 0] += 0.16 * sub_amp
mix[:, 1] += 0.16 * sub_amp
# reverb bus
verb_in = drums_verb.copy()
verb_in[:, 0] += 0.09 * gL + 0.06 * bass_out * 0.3
verb_in[:, 1] += 0.09 * gR + 0.06 * bass_out * 0.3
verb_in += lead_st * 0.18
ir = reverb_ir(rt60=3.0)
wet = convolve_stereo(verb_in, ir)[:N]
mix += wet * 0.9
del wet, verb_in, drums, drums_verb, gtr_out, gL, gR
# final fade (outro ring-out)
fade_start = b2s(ob + 5.5)
fl = N - fade_start
mix[fade_start:] *= np.linspace(1, 0, fl)[:, None] ** 1.5
# master EQ: clean the sub, tame the boom, a little air
mix = hp(mix, 28)
mix = shelf(mix, 75, -3.0, "low")
mix = shelf(mix, 3500, 2.0, "high")
# loudness: aim ~ -14 dBFS RMS over the loud part, gentle soft clip, then limit
core = mix[b2s(PHASE + 32):b2s(PHASE + 64)]
rms_db = 20 * np.log10(np.sqrt((core ** 2).mean()) + 1e-9)
gain = 10 ** ((-14.0 - rms_db) / 20)
print(f"  pre-limiter rms {rms_db:.1f} dB, gain {20 * np.log10(gain):.1f} dB, peak {np.abs(mix).max() * gain:.2f}")
mix = np.tanh(mix * gain * 0.7) / np.tanh(0.7)
mix = limiter(mix, thr=0.96)
mix = mix / (np.abs(mix).max() + 1e-9) * 0.97
print(f"  final rms {20 * np.log10(np.sqrt((mix ** 2).mean())):.1f} dBFS, duration {N / SR:.1f}s, total {time.time() - t0:.0f}s")
sf.write("out/doom_cover.wav", mix.astype(np.float32), SR, subtype="PCM_16")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "out/doom_cover.wav", "-codec:a", "libmp3lame", "-b:a", "320k",
                "-metadata", "title=Oleni (doom metal cover)", "-metadata", "artist=Claude x TIK", "out/doom_cover.mp3"], check=True)

# ---------- MIDI export ----------
mid = mido.MidiFile(ticks_per_beat=480)
def track(name, program, ch):
    t = mido.MidiTrack(); t.append(mido.MetaMessage("track_name", name=name, time=0))
    if program is not None:
        t.append(mido.Message("program_change", program=program, channel=ch, time=0))
    return t
def write_notes(t, evs, ch):
    # evs: list of (start_beat_song, dur, note, vel)  in new-timeline beats
    msgs = []
    for (b, d, n, v) in evs:
        msgs.append((b, 1, mido.Message("note_on", note=n, velocity=int(min(127, v * 110)), channel=ch, time=0)))
        msgs.append((b + max(d, 0.1) - 0.02, 0, mido.Message("note_off", note=n, velocity=0, channel=ch, time=0)))
    msgs.sort(key=lambda x: (x[0], x[1]))
    last = 0.0
    for (b, _, m) in msgs:
        m.time = int(round((b - last) * 480)); t.append(m); last = b
def nb(beat):  # original beat idx -> new timeline beats
    return INTRO_BARS * 4 + (beat - PHASE)
tempo_tr = mido.MidiTrack(); tempo_tr.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(BPM), time=0))
tempo_tr.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0)); mid.tracks.append(tempo_tr)
tg = track("Rhythm guitar (drop A power chords)", 30, 0)
ev = []
for (beat, dur, root, mute, vel, soft) in G[0]:
    for iv in (0, 7, 12):
        ev.append((nb(beat), dur, root + iv + 12, vel * (0.7 if mute else 1.0)))  # +12: GM guitar sounds an octave low
write_notes(tg, ev, 0); mid.tracks.append(tg)
tb = track("Bass", 33, 1); write_notes(tb, [(nb(b), d, r, v) for (b, d, r, v) in BASS], 1); mid.tracks.append(tb)
tl = track("Lead guitar (vocal melody)", 29, 2); write_notes(tl, [(nb(b), d, m, v) for (b, d, m, v) in LEAD], 2); mid.tracks.append(tl)
GM = {"kick": 36, "snare": 38, "tomlo": 45, "tommid": 47, "tomhi": 50, "crash": 49, "ride": 51, "hat": 42}
td = track("Drums", None, 9); write_notes(td, [(nb(b), 0.25, GM[n], v) for (b, n, v) in DR], 9); mid.tracks.append(td)
mid.save("out/doom_cover.mid")
print("wrote out/doom_cover.wav, out/doom_cover.mp3, out/doom_cover.mid")
