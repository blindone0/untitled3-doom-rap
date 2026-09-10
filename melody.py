"""Extract the vocal melody from the Demucs vocal stem (harmonic-summation salience on a CQT,
with pyin used where it is confident), plus per-bar energy stats."""
import json

import librosa
import numpy as np
from scipy.ndimage import median_filter

sr = 22050
A = json.load(open("out/analysis.json"))
beats = np.array(A["beats"])
voc, _ = librosa.load("out/stems/htdemucs/original/vocals.wav", sr=sr, mono=True)
mix, _ = librosa.load("src/original.wav", sr=sr, mono=True)
hop = 256
BPO = 36  # bins per octave (3 per semitone)
FMIN = librosa.note_to_hz("E2")
NB = BPO * 5  # E2 .. E7
inst, _ = librosa.load("out/stems/htdemucs/original/no_vocals.wav", sr=sr, mono=True)
TUNE = float(librosa.estimate_tuning(y=inst, sr=sr))  # instruments give a steadier tuning estimate than vibrato vocals
del inst
print(f"tuning offset {TUNE:+.2f} semitones")
C = np.abs(librosa.cqt(voc, sr=sr, hop_length=hop, fmin=FMIN, n_bins=NB, bins_per_octave=BPO, tuning=TUNE * 3))
C = C / (C.max() + 1e-9)
nfr = C.shape[1]
times = librosa.frames_to_time(np.arange(nfr), sr=sr, hop_length=hop)
# harmonic summation salience for candidate bins in E2..G5 (flat weights scored best against pyin)
cand = int(BPO * np.log2(librosa.note_to_hz("G5") / FMIN))
sal = np.zeros((cand, nfr))
for h, w in [(1, 1.0), (2, 0.9), (3, 0.8), (4, 0.7), (5, 0.6), (6, 0.5), (7, 0.4), (8, 0.3)]:
    off = int(round(BPO * np.log2(h)))
    hi = min(cand, NB - off)
    sal[:hi] += w * C[off:off + hi]
rms = librosa.feature.rms(y=voc, frame_length=2048, hop_length=hop)[0][:nfr]
gate = rms > 0.10 * np.percentile(rms, 97)
peak_bin = sal.argmax(0)
peak_val = sal.max(0)
prom = peak_val / (sal.mean(0) + 1e-9)
sal_ok = gate & (prom > 3.0)
midi_sal = 40.0 + peak_bin / 3.0  # bin 0 = E2 in tuning-compensated coordinates
# pyin where confident
f0, vflag, vprob = librosa.pyin(voc, fmin=FMIN, fmax=librosa.note_to_hz("G5"), sr=sr,
                                frame_length=2048, hop_length=hop, fill_na=np.nan)
f0 = f0[:nfr]; vprob = vprob[:nfr]
pyin_ok = (~np.isnan(f0)) & (vprob > 0.6) & gate
midi = np.full(nfr, np.nan)
midi[sal_ok] = midi_sal[sal_ok]
midi[pyin_ok] = librosa.hz_to_midi(f0[pyin_ok]) - TUNE
print(f"frames: {nfr}, salience-voiced {sal_ok.mean():.2f}, pyin-voiced {pyin_ok.mean():.2f}, any {(~np.isnan(midi)).mean():.2f}")

valid = ~np.isnan(midi)
sm = midi.copy()
sm_f = median_filter(np.where(valid, midi, 0.0), size=9)
sm[valid] = sm_f[valid]
# fill tiny holes (< 3 frames) so notes do not fragment
holes = np.where(~valid)[0]
for i in holes:
    if 1 <= i < nfr - 2 and valid[i - 1] and (valid[i + 1] or valid[i + 2]):
        sm[i] = sm[i - 1]; valid[i] = True

# segment into notes
notes = []
cur = None
for i in range(nfr):
    if not valid[i]:
        if cur:
            notes.append(cur)
            cur = None
        continue
    p = sm[i]
    if cur is None:
        cur = {"s": times[i], "e": times[i] + hop / sr, "p": [p]}
    elif abs(p - np.median(cur["p"][-6:])) < 0.8:
        cur["e"] = times[i] + hop / sr
        cur["p"].append(p)
    else:
        notes.append(cur)
        cur = {"s": times[i], "e": times[i] + hop / sr, "p": [p]}
if cur:
    notes.append(cur)
notes = [{"s": n["s"], "e": n["e"], "m": int(round(np.median(n["p"])))} for n in notes if n["e"] - n["s"] >= 0.08]

merged = []
for n in notes:
    if merged and n["m"] == merged[-1]["m"] and n["s"] - merged[-1]["e"] < 0.06:
        merged[-1]["e"] = n["e"]
    else:
        merged.append(dict(n))
notes = merged

bidx = np.arange(len(beats))


def to_beat(t):
    return float(np.interp(t, beats, bidx,
                           left=(t - beats[0]) / (beats[1] - beats[0]),
                           right=len(beats) - 1 + (t - beats[-1]) / (beats[-1] - beats[-2])))


q = []
for n in notes:
    b = round(to_beat(n["s"]) * 4) / 4
    e = round(to_beat(n["e"]) * 4) / 4
    if e <= b:
        e = b + 0.25
    q.append({"b": b, "d": e - b, "m": n["m"]})
q.sort(key=lambda x: x["b"])
mono = []
for n in q:
    if mono and n["b"] < mono[-1]["b"] + mono[-1]["d"]:
        if n["b"] == mono[-1]["b"]:
            continue
        mono[-1]["d"] = max(0.25, n["b"] - mono[-1]["b"])
    mono.append(n)

ms = np.array([n["m"] for n in mono], float)
loc = median_filter(ms, size=11, mode="nearest")
for n, l in zip(mono, loc):
    while n["m"] - l > 8:
        n["m"] -= 12
    while l - n["m"] > 8:
        n["m"] += 12

# short out-of-scale slivers are usually slides between notes: snap them to A harmonic/natural minor
SCALE = {9, 11, 0, 2, 4, 5, 7, 8}
snapped = 0
for i, n in enumerate(mono):
    if n["d"] <= 0.25 and n["m"] % 12 not in SCALE:
        nxt = mono[i + 1]["m"] if i + 1 < len(mono) else n["m"]
        up, dn = n["m"] + 1, n["m"] - 1
        n["m"] = up if abs(up - nxt) <= abs(dn - nxt) else dn
        snapped += 1
merged = []
for n in mono:
    if merged and n["m"] == merged[-1]["m"] and abs(merged[-1]["b"] + merged[-1]["d"] - n["b"]) < 1e-6:
        merged[-1]["d"] += n["d"]
    else:
        merged.append(n)
mono = merged
pcs = np.bincount([n["m"] % 12 for n in mono], minlength=12)
print(f"snapped {snapped} slivers; in-scale note fraction {pcs[list(SCALE)].sum() / pcs.sum():.2f}")
print("notes:", len(mono), "pitch range", min(n["m"] for n in mono), "-", max(n["m"] for n in mono),
      "median", int(np.median([n["m"] for n in mono])))
durs = sorted(set(n["d"] for n in mono))[:8]
print("dur histogram (beats):", {d: sum(1 for n in mono if n["d"] == d) for d in durs})

PHASE = 1
nbars = (len(beats) - PHASE) // 4
mix_rms = librosa.feature.rms(y=mix, frame_length=2048, hop_length=hop)[0]
mt = librosa.frames_to_time(np.arange(len(mix_rms)), sr=sr, hop_length=hop)
stats = []
for k in range(nbars):
    t0 = beats[PHASE + 4 * k]
    t1 = beats[PHASE + 4 * k + 4] if PHASE + 4 * k + 4 < len(beats) else t0 + 4 * 60 / A["tempo"]
    sel = (times >= t0) & (times < t1)
    msel = (mt >= t0) & (mt < t1)
    stats.append({"bar": k, "mix_db": float(20 * np.log10(mix_rms[msel].mean() + 1e-9)),
                  "voc_db": float(20 * np.log10(rms[sel].mean() + 1e-9)),
                  "nnotes": sum(1 for n in mono if PHASE + 4 * k <= n["b"] < PHASE + 4 * k + 4)})
mx = max(s["mix_db"] for s in stats)
vx = max(s["voc_db"] for s in stats)
line = []
for s in stats:
    line.append(f"{s['bar']:3d}:{s['mix_db'] - mx:5.1f}/{s['voc_db'] - vx:5.1f}/{s['nnotes']:2d}")
    if len(line) == 6:
        print("  ".join(line)); line = []
if line:
    print("  ".join(line))
# compact melody dump per bar
for k in range(nbars):
    ns = [n for n in mono if PHASE + 4 * k <= n["b"] < PHASE + 4 * k + 4]
    if ns:
        print(f"bar {k:2d}: " + " ".join(f"{librosa.midi_to_note(n['m'], unicode=False)}@{n['b'] - PHASE - 4 * k:.2f}x{n['d']}" for n in ns))
json.dump({"notes": mono, "bars": stats}, open("out/melody.json", "w"), indent=0)
