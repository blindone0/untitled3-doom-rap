"""Minimal SFZ drum sampler for the AVL Drumkits (velocity layers, pan, choke) with humanization."""
import os
import re

import numpy as np
import soundfile as sf
from scipy import signal

SR = 44100


class SFZKit:
    def __init__(self, path):
        self.base = os.path.dirname(path)
        self.regions = []  # dicts: key, lovel, hivel, sample, pan, group, off_by
        cur_group = {}
        cur = None
        txt = open(path, encoding="utf-8", errors="ignore").read()
        txt = re.sub(r"//[^\n]*", "", txt)
        tokens = re.findall(r"<[a-z]+>|[a-z_0-9]+=(?:\"[^\"]*\"|\S+)", txt)
        for tok in tokens:
            if tok == "<group>":
                self._flush(cur)
                cur = None
                cur_group = {}
            elif tok == "<region>":
                self._flush(cur)
                cur = dict(cur_group)
            elif tok.startswith("<"):
                self._flush(cur)
                cur = None
            else:
                k, v = tok.split("=", 1)
                v = v.strip('"')
                if cur is None:
                    cur_group[k] = v
                else:
                    cur[k] = v
        self._flush(cur)
        self._cache = {}

    def _flush(self, r):
        if r and "sample" in r:
            self.regions.append({
                "key": int(r.get("key", r.get("lokey", -1))),
                "lovel": int(r.get("lovel", 1)), "hivel": int(r.get("hivel", 127)),
                "sample": r["sample"], "pan": float(r.get("pan", 0)) / 100.0,
                "group": r.get("group"), "off_by": r.get("off_by"),
            })

    def keys(self):
        return sorted(set(r["key"] for r in self.regions))

    def _load(self, name):
        if name not in self._cache:
            x, sr = sf.read(os.path.join(self.base, name))
            if x.ndim > 1:
                x = x.mean(1)
            if sr != SR:
                x = signal.resample_poly(x, SR, sr)
            self._cache[name] = x.astype(np.float64)
        return self._cache[name]

    def hit(self, key, vel, rng, pitch_var=0.02):
        """vel in 0..1 -> (mono signal, pan). Random layer wobble at boundaries + slight pitch variation
        so repeated hits never sound identical."""
        v127 = int(np.clip(vel * 127, 1, 127))
        v127 = int(np.clip(v127 + rng.integers(-6, 7), 1, 127))
        regs = [r for r in self.regions if r["key"] == key and r["lovel"] <= v127 <= r["hivel"]]
        if not regs:
            regs = [r for r in self.regions if r["key"] == key]
            if not regs:
                raise KeyError(key)
        r = regs[0]
        x = self._load(r["sample"])
        ratio = 2 ** (rng.normal(0, pitch_var) )
        if abs(ratio - 1) > 1e-4:
            idx = np.arange(0, len(x) - 1, ratio)
            x = np.interp(idx, np.arange(len(x)), x)
        # velocity to gain: layers carry most of the timbre change, add a moderate level curve
        g = 0.35 + 0.65 * (vel ** 1.2)
        return x * g, r["pan"]


def place_stereo(buf, start, mono, pan, gain=1.0):
    """equal-power pan (pan in -1..1) into stereo buf"""
    th = (pan + 1) * np.pi / 4
    l, r = np.cos(th) * gain, np.sin(th) * gain
    if start < 0:
        mono = mono[-start:]
        start = 0
    end = min(len(buf), start + len(mono))
    if end <= start:
        return
    seg = mono[: end - start]
    buf[start:end, 0] += seg * l
    buf[start:end, 1] += seg * r


# Red Zeppelin 5pc mapping (AVL Drumkits 1.0)
RZ = {"kick": 36, "snare": 38, "rim": 40, "side": 37, "tom": 45, "tomedge": 47, "ftom1": 43, "ftom2": 41,
      "hhc": 42, "hho": 46, "hhs": 48, "hhp": 44, "crash1": 49, "crash2": 57, "ride": 51, "bell": 53,
      "shank": 59, "china": 60, "splash": 55}
