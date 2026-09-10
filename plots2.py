import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import signal

path = sys.argv[1]
out = sys.argv[2]
y, sr = sf.read(path)
mono = y.mean(1) if y.ndim > 1 else y
print("dur", round(len(mono) / sr, 1), "peak", np.abs(y).max().round(3), "nan", bool(np.isnan(y).any()))
win = sr
nsec = len(mono) // win
rms = np.array([np.sqrt((mono[i * win:(i + 1) * win] ** 2).mean()) for i in range(nsec)])
rms_db = 20 * np.log10(rms + 1e-9)
print("rms dB per 10 s:", " ".join(f"{v:.0f}" for v in rms_db[::10]))
# crest factor per 10 s (peak/rms) - transient life
crest = []
for i in range(0, nsec - 10, 10):
    seg = mono[i * win:(i + 10) * win]
    crest.append(20 * np.log10(np.abs(seg).max() / (np.sqrt((seg ** 2).mean()) + 1e-9)))
print("crest dB per 10 s:", " ".join(f"{v:.0f}" for v in crest))
fig, ax = plt.subplots(3, 1, figsize=(16, 12))
ax[0].plot(np.arange(nsec), rms_db); ax[0].set_title("RMS per second (dBFS)"); ax[0].grid(True)


def spec(axis, seg, title):
    f, t, S = signal.spectrogram(seg, sr, nperseg=4096, noverlap=3072)
    axis.pcolormesh(t, f, 10 * np.log10(S + 1e-12), shading="auto", vmin=-110, vmax=-30, cmap="magma")
    axis.set_ylim(0, 8000); axis.set_yscale("symlog", linthresh=200); axis.set_title(title)


spec(ax[1], mono[0:int(25 * sr)], "intro (0-25 s)")
c0 = int(float(sys.argv[3]) * sr) if len(sys.argv) > 3 else int(150 * sr)
spec(ax[2], mono[c0:c0 + int(25 * sr)], f"excerpt ({c0 // sr}-{c0 // sr + 25} s)")
plt.tight_layout(); plt.savefig(out, dpi=80)
for name, lo, hi in [("sub 30-90", 30, 90), ("low 90-250", 90, 250), ("mid 250-2k", 250, 2000), ("high 2k-8k", 2000, 8000), ("air 8k+", 8000, 20000)]:
    sos = signal.butter(4, [lo, min(hi, sr / 2 - 1)], btype="band", fs=sr, output="sos")
    b = signal.sosfilt(sos, mono[c0:c0 + int(25 * sr)])
    print(f"{name}: {20 * np.log10(np.sqrt((b ** 2).mean()) + 1e-9):.1f} dB")
