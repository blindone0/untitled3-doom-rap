"""Tiny numpy/scipy doom-metal synthesis engine (no samples, everything generated)."""
import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter1d, uniform_filter1d

SR = 44100
TL = 4096  # wavetable length


def midi2f(m):
    return 440.0 * 2.0 ** ((m - 69) / 12.0)


# ---------------- filters ----------------
def _butter(x, fc, kind, order=2):
    fc = min(fc, SR * 0.49)
    sos = signal.butter(order, fc, btype=kind, fs=SR, output="sos")
    return signal.sosfilt(sos, x, axis=0)


def lp(x, fc, order=2):
    return _butter(x, fc, "low", order)


def hp(x, fc, order=2):
    return _butter(x, fc, "high", order)


def peak(x, fc, gain_db, q=1.0):
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * fc / SR
    al = np.sin(w0) / (2 * q)
    b = np.array([1 + al * A, -2 * np.cos(w0), 1 - al * A])
    a = np.array([1 + al / A, -2 * np.cos(w0), 1 - al / A])
    return signal.lfilter(b / a[0], a / a[0], x, axis=0)


def shelf(x, fc, gain_db, kind="low", s=0.9):
    """RBJ low/high shelf"""
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * fc / SR
    al = np.sin(w0) / 2 * np.sqrt((A + 1 / A) * (1 / s - 1) + 2)
    c = np.cos(w0)
    sq = 2 * np.sqrt(A) * al
    if kind == "low":
        b = [A * ((A + 1) - (A - 1) * c + sq), 2 * A * ((A - 1) - (A + 1) * c), A * ((A + 1) - (A - 1) * c - sq)]
        a = [(A + 1) + (A - 1) * c + sq, -2 * ((A - 1) + (A + 1) * c), (A + 1) + (A - 1) * c - sq]
    else:
        b = [A * ((A + 1) + (A - 1) * c + sq), -2 * A * ((A - 1) + (A + 1) * c), A * ((A + 1) + (A - 1) * c - sq)]
        a = [(A + 1) - (A - 1) * c + sq, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - sq]
    b, a = np.array(b), np.array(a)
    return signal.lfilter(b / a[0], a / a[0], x, axis=0)


# ---------------- wavetables / oscillators ----------------
_tables = {}


def saw_table(f, rolloff=1.0):
    key = (round(f, 1), rolloff)
    if key not in _tables:
        n = np.arange(TL) / TL
        kmax = int(max(1, min(400, 0.45 * SR / f)))
        k = np.arange(1, kmax + 1)[:, None]
        t = (((-1.0) ** (k + 1)) * np.sin(2 * np.pi * k * n[None, :]) / k ** rolloff).sum(0)
        _tables[key] = t / np.abs(t).max()
    return _tables[key]


def _lookup(table, ph):
    idx = (ph % 1.0) * TL
    i0 = idx.astype(np.int64) % TL
    fr = idx - np.floor(idx)
    i1 = (i0 + 1) % TL
    return table[i0] * (1 - fr) + table[i1] * fr


def osc(f_arr, table, ph0=0.0):
    return _lookup(table, np.cumsum(f_arr / SR) + ph0)


def _fade_out(sig, ms=12):
    r = min(len(sig), int(ms / 1000 * SR))
    if r > 0:
        sig[-r:] *= np.linspace(1, 0, r)
    return sig


# ---------------- guitar ----------------
def string_note(f, n, rng, vel=1.0, mute=False):
    """plucked/distorted-guitar string: bright + dark band-limited saws with different decays"""
    t = np.arange(n) / SR
    cents = rng.normal(0, 2.5)
    fa = f * 2 ** (cents / 1200) * (1 + 0.006 * np.exp(-t / 0.02))
    ph = np.cumsum(fa / SR) + rng.uniform(0, 1)
    bright = _lookup(saw_table(f, 1.0), ph)
    dark = _lookup(saw_table(f, 2.2), ph)
    if mute:
        eb, ed = np.exp(-t / 0.03), np.exp(-t / 0.11)
    else:
        eb, ed = np.exp(-t / 0.45), np.exp(-t / 4.0)
    att = 1 - np.exp(-t / 0.003)
    sig = (0.55 * bright * eb + dark * ed) * att
    pick = hp(rng.standard_normal(n), 2500) * np.exp(-t / 0.0025) * 0.35
    return _fade_out((sig + pick) * vel)


def power_chord(root_midi, n, rng, vel=1.0, mute=False, intervals=(0, 7, 12), strum_ms=5.0):
    out = np.zeros(n)
    for i, iv in enumerate(intervals):
        d = int(i * strum_ms / 1000 * SR)
        if n - d < 100:
            continue
        out[d:] += string_note(midi2f(root_midi + iv), n - d, rng, vel * (1.0 if i == 0 else 0.8), mute)
    return out


def amp_guitar(x, gain=22.0, cab_lp=4800.0, presence=4.0, low_hp=70.0):
    x = hp(x, low_hp)
    x = peak(x, 800, 5.0, 0.7)  # overdrive pedal mid push
    y = np.tanh(gain * x)
    y = y + 0.22 * y * y  # asymmetry -> even harmonics
    y = lp(y, 9000)
    y = np.tanh(1.7 * y)  # 2nd stage
    y = hp(y, 45)
    y = lp(y, cab_lp, order=4)  # cabinet
    y = peak(y, 110, 1.5, 1.0)
    y = peak(y, 2300, presence, 1.2)
    y = peak(y, 450, -2.5, 1.0)
    return y


# ---------------- bass ----------------
def bass_note(f, n, rng, vel=1.0):
    t = np.arange(n) / SR
    ph = np.cumsum(np.full(n, f * 2 ** (rng.normal(0, 1.5) / 1200)) / SR)
    sw = _lookup(saw_table(f, 1.4), ph)
    sig = (0.6 * sw * np.exp(-t / 2.0) + 0.9 * np.sin(2 * np.pi * ph) * np.exp(-t / 5.0)) * (1 - np.exp(-t / 0.006))
    return _fade_out(sig * vel)


def amp_bass(x):
    y = np.tanh(2.5 * hp(x, 30))
    y = lp(y, 2200)
    y = peak(y, 70, 3.0, 0.8)
    y = peak(y, 700, 2.0, 1.0)
    return y


# ---------------- drums ----------------
_drum_cache = {}


def drum(name, rng=None):
    if name in _drum_cache:
        return _drum_cache[name]
    rng = rng or np.random.default_rng(7)
    if name == "kick":
        n = int(0.6 * SR)
        t = np.arange(n) / SR
        f = 45 + 130 * np.exp(-t / 0.028)
        body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.22)
        click = hp(rng.standard_normal(n), 1500) * np.exp(-t / 0.003) * 0.6
        s = np.tanh(2.2 * (1.3 * body + click))
        s = lp(s, 6000)
    elif name == "snare":
        n = int(0.5 * SR)
        t = np.arange(n) / SR
        tone = (np.sin(2 * np.pi * 180 * t) + 0.5 * np.sin(2 * np.pi * 330 * t)) * np.exp(-t / 0.07)
        noise = _butter(rng.standard_normal(n), 1200, "high") * np.exp(-t / 0.15)
        s = np.tanh(1.6 * (0.9 * tone + 0.9 * noise))
    elif name.startswith("tom"):
        f0 = {"tomlo": 85, "tommid": 120, "tomhi": 165}[name]
        n = int(0.6 * SR)
        t = np.arange(n) / SR
        f = f0 * (1 + 0.35 * np.exp(-t / 0.06))
        s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.3)
        s += hp(rng.standard_normal(n), 2000) * np.exp(-t / 0.004) * 0.3
        s = np.tanh(1.8 * s)
    elif name == "crash":
        n = int(3.5 * SR)
        t = np.arange(n) / SR
        noise = hp(rng.standard_normal(n), 2500) * np.exp(-t / 1.1)
        part = np.zeros(n)
        for fr in rng.uniform(2500, 11000, 40):
            part += np.sin(2 * np.pi * fr * t + rng.uniform(0, 6.28)) * np.exp(-t / rng.uniform(0.5, 2.0))
        s = 0.8 * noise + 0.15 * part / 6
        s = lp(s, 14000)
    elif name == "ride":
        n = int(1.2 * SR)
        t = np.arange(n) / SR
        noise = hp(rng.standard_normal(n), 4000) * np.exp(-t / 0.25) * 0.4
        part = np.zeros(n)
        for fr in rng.uniform(1800, 9000, 30):
            part += np.sin(2 * np.pi * fr * t + rng.uniform(0, 6.28)) * np.exp(-t / rng.uniform(0.15, 0.7))
        ping = np.sin(2 * np.pi * 3100 * t) * np.exp(-t / 0.5)
        s = noise + part / 8 + 0.3 * ping
        s = lp(s, 13000)
    elif name == "hat":
        n = int(0.12 * SR)
        t = np.arange(n) / SR
        s = hp(rng.standard_normal(n), 7000) * np.exp(-t / 0.03)
    else:
        raise ValueError(name)
    s = s / (np.abs(s).max() + 1e-9)
    _drum_cache[name] = _fade_out(s, 20)
    return _drum_cache[name]


# ---------------- lead ----------------
def lead_note(f, n, rng, vel=1.0, f_from=None, vib_depth=18.0, vib_rate=5.3):
    t = np.arange(n) / SR
    f_arr = np.full(n, f)
    if f_from is not None and f_from > 0:
        g = min(int(0.045 * SR), n)
        f_arr[:g] = f_from * (f / f_from) ** (np.linspace(0, 1, g) ** 0.8)
    vib_env = np.clip((t - 0.25) / 0.4, 0, 1)
    f_arr = f_arr * 2 ** (vib_depth * vib_env * np.sin(2 * np.pi * vib_rate * t + rng.uniform(0, 6.28)) / 1200)
    tb = saw_table(f * 1.01, 1.0)
    a = osc(f_arr * 2 ** (4 / 1200), tb, rng.uniform(0, 1))
    b = osc(f_arr * 2 ** (-4 / 1200), tb, rng.uniform(0, 1))
    env = (1 - np.exp(-t / 0.004)) * (0.45 + 0.55 * np.exp(-t / 2.5))
    pick = hp(rng.standard_normal(n), 3000) * np.exp(-t / 0.002) * 0.25
    return _fade_out(((a + b) * 0.5 * env + pick) * vel, 8)


def amp_lead(x):
    return amp_guitar(x, gain=28.0, cab_lp=5500.0, presence=5.0, low_hp=120.0)


# ---------------- fx ----------------
def delay_fx(x, time_s, fb=0.38, mix=0.32, damp=3200, repeats=7):
    D = int(time_s * SR)
    n = len(x)
    out = np.zeros((n + repeats * D, 2))
    out[:n, 0] += x
    out[:n, 1] += x
    tap = x.copy()
    g = mix
    for k in range(1, repeats + 1):
        tap = lp(tap, damp)
        out[k * D:k * D + n, k % 2] += tap * g
        g *= fb
    return out


def reverb_ir(rt60=2.8, seed=3, predelay_ms=25):
    rng = np.random.default_rng(seed)
    n = int(rt60 * SR)
    t = np.arange(n) / SR
    ir = rng.standard_normal((n, 2)) * np.exp(-6.9 * t / rt60)[:, None]
    lo = lp(ir, 1800)
    hi = ir - lo
    ir = lo + hi * np.exp(-2.5 * t / rt60)[:, None]
    ir[: int(predelay_ms / 1000 * SR)] = 0
    ir = hp(ir, 120)
    return ir / np.sqrt((ir ** 2).sum(0)).max()


def convolve_stereo(x, ir):
    if x.ndim == 1:
        x = np.stack([x, x], 1)
    out = np.zeros((len(x) + len(ir) - 1, 2))
    for c in range(2):
        out[:, c] = signal.fftconvolve(x[:, c], ir[:, c])
    return out


def limiter(x, thr=0.95, lookahead_ms=4, release_ms=90):
    pk = np.abs(x).max(axis=1)
    la = max(1, int(lookahead_ms / 1000 * SR))
    env = maximum_filter1d(pk, size=2 * la + 1)
    env = maximum_filter1d(env, size=int(release_ms / 1000 * SR))
    env = uniform_filter1d(env, size=la)
    g = np.minimum(1.0, thr / np.maximum(env, 1e-9))
    return np.clip(x * g[:, None], -thr, thr)


def add(buf, start, sig, gain=1.0):
    """add mono/stereo sig into buf (2ch) at sample offset start, clipping bounds"""
    if start < 0:
        sig = sig[-start:]
        start = 0
    end = min(len(buf), start + len(sig))
    if end <= start:
        return
    seg = sig[: end - start]
    if seg.ndim == 1:
        buf[start:end, 0] += seg * gain
        buf[start:end, 1] += seg * gain
    else:
        buf[start:end] += seg * gain
