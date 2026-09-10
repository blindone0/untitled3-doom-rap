import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import signal

y, sr = sf.read("out/doom_cover.wav")
mono = y.mean(1)
print("samples", len(y), "sr", sr, "peak", np.abs(y).max().round(3), "nan", bool(np.isnan(y).any()))
# 1) RMS envelope per second + stereo correlation
win = sr
nsec = len(mono) // win
rms = np.array([np.sqrt((mono[i * win:(i + 1) * win] ** 2).mean()) for i in range(nsec)])
rms_db = 20 * np.log10(rms + 1e-9)
print("rms dB per 10 s:", " ".join(f"{v:.0f}" for v in rms_db[::10]))
print("silent seconds (< -40 dB):", int((rms_db < -40).sum()), "of", nsec)
fig, ax = plt.subplots(3, 1, figsize=(16, 12))
ax[0].plot(np.arange(nsec), rms_db)
ax[0].set_title("RMS per second (dBFS)"); ax[0].set_xlabel("s"); ax[0].grid(True)
# 2) spectrogram of the intro + first bars (0-25 s)
def spec(axis, seg, title):
    f, t, S = signal.spectrogram(seg, sr, nperseg=4096, noverlap=3072)
    axis.pcolormesh(t, f, 10 * np.log10(S + 1e-12), shading="auto", vmin=-110, vmax=-30, cmap="magma")
    axis.set_ylim(0, 6000); axis.set_yscale("symlog", linthresh=200); axis.set_title(title)
spec(ax[1], mono[0:int(25 * sr)], "intro + first riffs (0-25 s)")
c0 = int(150 * sr)
spec(ax[2], mono[c0:c0 + int(25 * sr)], "chorus excerpt (150-175 s)")
plt.tight_layout(); plt.savefig("out/check.png", dpi=80)
# band energy split to see that low end, mids and highs are all present
for name, lo, hi in [("sub 30-90", 30, 90), ("low 90-250", 90, 250), ("mid 250-2k", 250, 2000), ("high 2k-8k", 2000, 8000), ("air 8k+", 8000, 20000)]:
    sos = signal.butter(4, [lo, min(hi, sr / 2 - 1)], btype="band", fs=sr, output="sos")
    b = signal.sosfilt(sos, mono[c0:c0 + int(25 * sr)])
    print(f"{name}: {20 * np.log10(np.sqrt((b ** 2).mean()) + 1e-9):.1f} dB")
