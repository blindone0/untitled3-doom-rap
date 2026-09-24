# -*- coding: utf-8 -*-
"""Section boundaries of the active track (track.py), derived from the actual phrases (flow json); shared by studio.py, mixvocal.py, karaoke.py.
A take of a section is trimmed to [first line - 0.3 s, last line end + 0.6 s], never past the next section's first line."""
import json
import re

from track import T

BT = json.load(open(T["bars"]))
BARS = {int(k): v for k, v in BT["bars"].items()}
FLOW = sorted(json.load(open(T["flow"], encoding="utf-8")), key=lambda l: l["t"])
LEAD_IN, TAIL, GAP, OVERLAP = 0.3, 0.6, 0.05, 0.12

# (name, first bar, bar after the last)  -> lyric lines with first_bar <= t < end_bar belong to the section
SECTIONS = [tuple(s) for s in T["sections"]]  # per track, see track.py


def _lines(a, b):
    return [l for l in FLOW if BARS[a] - 0.06 <= l["t"] < BARS[b] - 0.06]  # lyric times are rounded to 0.1 s


_raw = {}
for name, a, b in SECTIONS:
    ls = _lines(a, b)
    if ls:
        _raw[name] = (ls[0]["t"], ls[-1]["end"])
    else:
        _raw[name] = (BARS[a], BARS[b])

WINDOWS = {}
for name, a, b in SECTIONS:
    s, e = _raw[name]
    w0, w1 = s - LEAD_IN, e + TAIL
    later = [_raw[n][0] for (n, _, _) in SECTIONS if _raw[n][0] > s + 1e-6]
    if later:
        # sections butt straight against each other here, so a hard cut at the next one's first phrase clipped the
        # last consonant of every section (measured: 10-39 ms). Let a take run a little into the next section.
        w1 = max(e, min(w1, min(later) + OVERLAP))
    WINDOWS[name] = (round(w0, 3), round(w1, 3), round(s, 3))  # (window start, window end, first phrase start)


def section_of(take_name):
    return re.sub(r"_take\d+$", "", take_name)


def rec_params(name):
    """(part_start, duration) for recording: the beat plays until the window end"""
    w0, w1, s = WINDOWS[name]
    return s, round(w1 - s, 2)


def window_of(take_name):
    sec = section_of(take_name)
    return WINDOWS.get(sec)


if __name__ == "__main__":
    for name, _, _ in SECTIONS:
        w0, w1, s = WINDOWS[name]
        n = len(_lines(*[x for (nm, *x) in SECTIONS if nm == name][0]))
        print(f"{name:8s} phrases {n:2d}  sing from {s:6.1f}s  window {w0:6.1f} - {w1:6.1f}s  (record {rec_params(name)[1]:.1f}s)")
