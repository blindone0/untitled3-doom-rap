"""Sampler for Unreal Instruments 'Standard Guitar' (DI, SFZ): sustains, palm mutes, long sustains,
round robins, transposition, velocity -> level/brightness, pitch modulation (vibrato / legato slides)."""
import glob
import os
import re

import numpy as np
import soundfile as sf
from scipy import signal

SR = 44100
NOTE = {"c": 0, "c#": 1, "d": 2, "d#": 3, "e": 4, "f": 5, "f#": 6, "g": 7, "g#": 8, "a": 9, "a#": 10, "b": 11}


def name2midi(s):  # "a#2" -> 46 (c4 = 60)
    m = re.match(r"([a-g]#?)(-?\d)", s.lower())
    return NOTE[m.group(1)] + (int(m.group(2)) + 1) * 12


class StandardGuitar:
    ART = {"sus_down": "Sus_Down", "sus_up": "Sus_Up", "mute_down": "Mute_Down", "mute_up": "Mute_Up", "long": "Sus_Long"}

    def __init__(self, root):
        self.root = root
        self.table = {}
        for art, folder in self.ART.items():
            d = {}
            for f in glob.glob(os.path.join(root, folder, "*.flac")):
                m = re.match(r"([a-g]#?\d)_", os.path.basename(f))
                if m:
                    d.setdefault(name2midi(m.group(1)), []).append(f)
            for k in d:
                d[k].sort()
            self.table[art] = d
        self._cache = {}
        self._rr = {}

    def _load(self, f, chan):
        key = (f, chan)
        if key not in self._cache:
            x, sr = sf.read(f)
            if x.ndim > 1:
                x = x[:, chan] if chan is not None else x.mean(1)
            if sr != SR:
                x = signal.resample_poly(x, SR, sr)
            self._cache[key] = x.astype(np.float64)
        return self._cache[key]

    def pick(self, art, midi, rng):
        d = self.table[art]
        keys = np.array(sorted(d))
        k = int(keys[np.argmin(np.abs(keys - midi) + 0.1 * (keys < midi))])  # nearest, slight preference to pitch down
        files = d[k]
        rr = self._rr.get((art, k))
        if rr is None:
            rr = int(rng.integers(len(files)))
        f = files[rr % len(files)]
        self._rr[(art, k)] = rr + 1 + (int(rng.integers(2)) if len(files) > 3 else 0)
        return f, k

    def note(self, art, midi, n, vel, rng, chan=None, bend=None, glide_from=None, glide_ms=60.0, drift_cents=0.0):
        """Render n samples of a note. bend: per-sample semitone offsets (vibrato); glide_from: previous midi for
        a legato slide into this note; drift_cents: constant detune (double-tracking realism)."""
        f, k = self.pick(art, midi, rng)
        x = self._load(f, chan)
        semis = np.full(n, float(midi - k) + drift_cents / 100.0)
        if glide_from is not None:
            g = min(n, int(glide_ms / 1000 * SR))
            semis[:g] += (glide_from - midi) * (1 - np.linspace(0, 1, g)) ** 1.2
        if bend is not None:
            semis += bend[:n]
        if glide_from is None and bend is None and abs(midi - k) < 1e-9 and drift_cents == 0.0:
            y = x[:n].copy()
        else:
            ratio = 2 ** (semis / 12)
            pos = np.cumsum(ratio) - ratio[0]
            pos = pos[pos < len(x) - 1]
            y = np.interp(pos, np.arange(len(x)), x)
        if len(y) < n:
            y = np.pad(y, (0, n - len(y)))
        y = y * (vel ** 1.3)
        fc = 1500.0 * 2 ** (3.0 * vel)  # 1.5 kHz (soft) .. 12 kHz (hard)
        y = signal.sosfilt(signal.butter(1, min(fc, 18000), "low", fs=SR, output="sos"), y)
        r = min(n, int(0.012 * SR))
        if r > 0:
            y[-r:] *= np.linspace(1, 0, r)
        return y

    def power_chord(self, root, n, vel, rng, mute=False, up=False, chan=None, strum_ms=None, drift_cents=0.0, intervals=(0, 7, 12), long=False):
        art = "long" if (long and not mute) else ("mute_" if mute else "sus_") + ("up" if up else "down")
        if strum_ms is None:
            strum_ms = rng.uniform(4.0, 12.0)
        out = np.zeros(n)
        order = list(range(len(intervals)))
        if up:
            order = order[::-1]  # upstroke hits the high strings first
        for j, i in enumerate(order):
            d = int(j * strum_ms / 1000 * SR)
            if n - d < 200:
                continue
            v = float(np.clip(vel * rng.normal(1.0, 0.05) * (1.0 if i == 0 else 0.9), 0.05, 1.0))
            out[d:] += self.note(art, root + intervals[i], n - d, v, rng, chan=chan, drift_cents=drift_cents + rng.normal(0, 2.0))
        return out
