"""Mixing/mastering helpers: compressor, saturation, EQ, convolution reverb, limiter."""
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import maximum_filter1d, uniform_filter1d

SR = 44100


def _butter(x, fc, kind, order=2):
    fc = float(np.clip(fc, 10, SR * 0.49))
    sos = signal.butter(order, fc, btype=kind, fs=SR, output="sos")
    return signal.sosfilt(sos, x, axis=0)


def lp(x, fc, order=2): return _butter(x, fc, "low", order)
def hp(x, fc, order=2): return _butter(x, fc, "high", order)


def peak(x, fc, gain_db, q=1.0):
    A = 10 ** (gain_db / 40.0); w0 = 2 * np.pi * fc / SR; al = np.sin(w0) / (2 * q)
    b = np.array([1 + al * A, -2 * np.cos(w0), 1 - al * A]); a = np.array([1 + al / A, -2 * np.cos(w0), 1 - al / A])
    return signal.lfilter(b / a[0], a / a[0], x, axis=0)


def shelf(x, fc, gain_db, kind="low", s=0.9):
    A = 10 ** (gain_db / 40.0); w0 = 2 * np.pi * fc / SR
    al = np.sin(w0) / 2 * np.sqrt((A + 1 / A) * (1 / s - 1) + 2); c = np.cos(w0); sq = 2 * np.sqrt(A) * al
    if kind == "low":
        b = [A * ((A + 1) - (A - 1) * c + sq), 2 * A * ((A - 1) - (A + 1) * c), A * ((A + 1) - (A - 1) * c - sq)]
        a = [(A + 1) + (A - 1) * c + sq, -2 * ((A - 1) + (A + 1) * c), (A + 1) + (A - 1) * c - sq]
    else:
        b = [A * ((A + 1) + (A - 1) * c + sq), -2 * A * ((A - 1) + (A + 1) * c), A * ((A + 1) + (A - 1) * c - sq)]
        a = [(A + 1) - (A - 1) * c + sq, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - sq]
    b, a = np.array(b), np.array(a)
    return signal.lfilter(b / a[0], a / a[0], x, axis=0)


def env_follow(x, attack_ms, release_ms):
    """peak envelope with attack (via max-filter lookahead-free hold) and one-pole release"""
    pk = np.abs(x).max(axis=1) if x.ndim > 1 else np.abs(x)
    a_n = max(1, int(attack_ms / 1000 * SR))
    env = uniform_filter1d(pk, size=a_n)  # attack smoothing
    r = np.exp(-1.0 / (release_ms / 1000 * SR))
    # release: y[n] = max(env[n], r*y[n-1]) approximated by maxfilter over release window then one-pole
    env = maximum_filter1d(env, size=int(release_ms / 1000 * SR) // 2 + 1)
    env = signal.lfilter([1 - r], [1, -r], env)
    return env


def compressor(x, thr_db=-18.0, ratio=4.0, attack_ms=20.0, release_ms=200.0, makeup_db=0.0, knee_db=6.0):
    env = env_follow(x, attack_ms, release_ms)
    lvl = 20 * np.log10(env + 1e-9)
    over = lvl - thr_db
    # soft knee
    gr = np.where(over <= -knee_db / 2, 0.0,
                  np.where(over >= knee_db / 2, over - over / ratio,
                           (over + knee_db / 2) ** 2 / (2 * knee_db) * (1 - 1 / ratio)))
    g = 10 ** ((-gr + makeup_db) / 20)
    return x * (g[:, None] if x.ndim > 1 else g), float(gr.max())


def saturate(x, drive=1.5, mix=1.0):
    """tape-ish: gentle pre-emphasis, tanh, de-emphasis"""
    y = shelf(x, 4000, 3.0, "high")
    y = np.tanh(drive * y) / np.tanh(drive)
    y = shelf(y, 4000, -3.0, "high")
    return x * (1 - mix) + y * mix


def load_ir(path, sr=SR, max_s=None, stereo=True):
    ir, isr = sf.read(path)
    if ir.ndim == 1:
        ir = np.stack([ir, ir], 1)
    if isr != sr:
        ir = signal.resample_poly(ir, sr, isr, axis=0)
    if max_s:
        ir = ir[: int(max_s * sr)]
    ir = ir / (np.sqrt((ir ** 2).sum(0)).max() + 1e-9)
    return ir if stereo else ir.mean(1)


def convolve(x, ir):
    """stereo in (or mono), stereo IR -> stereo out (true stereo: L*irL, R*irR)"""
    if x.ndim == 1:
        x = np.stack([x, x], 1)
    n = len(x)
    out = np.zeros((n, 2))
    for c in range(2):
        out[:, c] = signal.fftconvolve(x[:, c], ir[:, c])[:n]
    return out


def limiter(x, thr=0.95, lookahead_ms=3, release_ms=80):
    pk = np.abs(x).max(axis=1)
    la = max(1, int(lookahead_ms / 1000 * SR))
    env = maximum_filter1d(pk, size=2 * la + 1)
    env = maximum_filter1d(env, size=int(release_ms / 1000 * SR))
    env = uniform_filter1d(env, size=la)
    g = np.minimum(1.0, thr / np.maximum(env, 1e-9))
    return np.clip(x * g[:, None], -thr, thr)


def rms_db(x):
    return 20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-12)
