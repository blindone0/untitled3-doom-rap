# -*- coding: utf-8 -*-
"""Section boundaries derived from the actual phrases (out/flow.json), shared by studio.py, mixvocal.py, karaoke.py.
A take of a section is trimmed to [first line - 0.3 s, last line end + 0.6 s], never past the next section's first line."""
import json
import re

BT = json.load(open("out/bar_times.json"))
BARS = {int(k): v for k, v in BT["bars"].items()}
FLOW = sorted(json.load(open("out/flow.json", encoding="utf-8")), key=lambda l: l["t"])
LEAD_IN, TAIL, GAP = 0.3, 0.6, 0.05

# (name, first bar, bar after the last)  -> lyric lines with first_bar <= t < end_bar belong to the section
SECTIONS = [("intro", 2, 8), ("verse1", 8, 24), ("hook_L", 24, 32), ("hook_R", 24, 32), ("verse2", 42, 58),
            ("verse3", 60, 72), ("ending", 75, 98)]


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
        w1 = min(w1, min(later) - GAP)
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
