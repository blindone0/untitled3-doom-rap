# -*- coding: utf-8 -*-
"""Syllable-level flow for the lyrics of the active track (see track.py): places every syllable on the 16th-note grid
of its bar. Ukrainian and English lyrics are supported.
Outputs (paths from track.py): flow json (for karaoke.py), printable flow sheet, guide audio (beat + tick per syllable)."""
import json
import re
import subprocess

import numpy as np
import soundfile as sf

from track import T

SR = 44100
VOW_UK = set("аеєиіїоуюяАЕЄИІЇОУЮЯ")
LYRICS = T["lyrics"]
BT = json.load(open(T["bars"]))
bar_t = sorted((int(k), v) for k, v in BT["bars"].items())
bar_starts = np.array([v for _, v in bar_t])
EN = T.get("lang") == "en"


def load_lines(path):
    lines, sec = [], ""
    for raw in open(path, encoding="utf-8"):
        line = raw.rstrip("\n")
        m = re.match(r"=== (.+?)\s+\((?:старт|start)", line)
        if m:
            sec = m.group(1).split(" —")[0].strip()
            continue
        m = re.match(r"\[(\d+):(\d+\.\d)\] (.*)", line)
        if m:
            lines.append({"t": int(m.group(1)) * 60 + float(m.group(2)), "text": m.group(3), "sec": sec})
    return lines


def syllabify_uk(w):
    """char spans of syllables inside a word (CV-based, good enough for karaoke)"""
    idx = [i for i, c in enumerate(w) if c in VOW_UK]
    if not idx:
        return []
    spans, start = [], 0
    for k, vi in enumerate(idx):
        if k == len(idx) - 1:
            end = len(w)
        else:
            gap = idx[k + 1] - vi - 1
            end = vi + 1 if gap <= 1 else vi + 2
        spans.append((start, end))
        start = end
    return spans


def syllabify_en(w):
    """English: vowel groups are nuclei; silent final -e / -ed / -es dropped; consonants split between nuclei"""
    lw = w.lower()
    core = re.sub(r"[^a-z']+$", "", lw)           # strip trailing punctuation
    core = re.sub(r"'s$", "", core)
    n = len(core)
    is_v = [(c in "aeiou") or (c == "y" and i > 0) for i, c in enumerate(core)]
    nuclei, i = [], 0
    while i < n:
        if is_v[i]:
            j = i
            while j + 1 < n and is_v[j + 1]:
                j += 1
            nuclei.append([i, j])
            i = j + 1
        else:
            i += 1
    if len(nuclei) > 1:
        a, b = nuclei[-1]
        if a == b == n - 1 and core[-1] == "e" and not is_v[n - 2]:
            if not (core.endswith("le") and n >= 3 and not is_v[n - 3]):
                nuclei.pop()                            # silent -e (stone, home) but not -ble/-tle
        elif a == b == n - 2 and core.endswith("ed") and core[-3] not in "td" and not is_v[n - 3]:
            nuclei.pop()                                # stopped, buried
        elif a == b == n - 2 and core.endswith("es") and core[-3] not in "sxz" and not core.endswith(("shes", "ches")):
            nuclei.pop()                                # trees, clothes
    if not nuclei:
        return []
    spans, start = [], 0
    for k, (a, b) in enumerate(nuclei):
        if k == len(nuclei) - 1:
            end = len(w)
        else:
            cons = core[b + 1:nuclei[k + 1][0]]
            if len(cons) <= 1 or cons.startswith(("th", "ch", "sh", "ph", "wh", "tch")):
                end = b + 1                                 # no-thing, wea-ther, wa-tching
            else:
                end = b + 2                                 # spea-kers -> speak-ers style split after one consonant
        spans.append((start, end))
        start = end
    return spans


def syllabify_word(w):
    return syllabify_en(w) if EN else syllabify_uk(w)


def syllables_of_line(text):
    """list of (char_start, char_end, word_index, first_in_word) for the sung part (stage directions in () skipped)"""
    out, wi, pending = [], 0, None  # pending = start of a vowel-less word (з, в, й ...) glued to the next syllable
    for m in re.finditer(r"[^\s-]+" if EN else r"\S+", text):  # English: hyphenated compounds count as two words
        tok = m.group(0)
        if tok.startswith("(") or tok.endswith(")"):
            continue
        spans = syllabify_word(tok)
        if not spans:
            if pending is None and any(c.isalpha() for c in tok):
                pending = m.start()
            continue
        for j, (a, b) in enumerate(spans):
            start = m.start() + a
            if j == 0 and pending is not None:
                start = pending
                pending = None
            out.append((start, m.start() + b, wi, j == 0))
        wi += 1
    return out


def place(n):
    """grid positions (in 16ths, 0..16) for n syllables in one bar"""
    if n == 0:
        return []
    if n <= 8:
        span = 2 * n if n < 8 else 15  # 8th-note chant feel, short lines end early
        pos = [i * span / n for i in range(n)]
    elif n <= 16:
        span = 15 if n < 16 else 16
        pos = [i * span / n for i in range(n)]
    else:
        return [i * 16 / n for i in range(n)]
    q, last = [], -1
    for p in pos:
        r = int(round(p))
        if r <= last:
            r = last + 1
        q.append(min(r, 15))
        last = r
    return [float(x) for x in q]


lines = load_lines(LYRICS)
flow = []
for i, l in enumerate(lines):
    k = int(np.searchsorted(bar_starts, l["t"] + 1e-3, side="right")) - 1
    D = float(bar_starts[k + 1] - bar_starts[k]) if k + 1 < len(bar_starts) else 240.0 / BT["bpm"]
    syl = syllables_of_line(l["text"])
    pos = place(len(syl))
    entries = []
    for (a, b, wi, first), p in zip(syl, pos):
        entries.append({"s": a, "e": b, "w": wi, "first": first, "slot": p, "t": round(l["t"] + p * D / 16, 3)})
    flow.append({"t": l["t"], "end": round(l["t"] + D, 3), "text": l["text"], "sec": l["sec"], "syl": entries, "n": len(syl)})
json.dump(flow, open(T["flow"], "w", encoding="utf-8"), ensure_ascii=False, indent=0)


# ---- printable sheet: syllables grouped under beats 1..4 ----
def mmss(t):
    return f"{int(t // 60)}:{t % 60:04.1f}"


if EN:
    sheet = ["FLOW SHEET: one line = 1 bar = 4 beats (|1| |2| |3| |4|), 4 sixteenths per beat.",
             "Syllables are hyphenated; what stands under one beat is said during that beat. A dot (·) = a sixteenth of rest.",
             f"Listen to {T['guide']}.mp3: a tick per syllable (lower tick = syllable on the beat), the beat plays quieter.", ""]
    syl_word = "syl"
else:
    sheet = ["ФЛОУ-ЛИСТ: кожен рядок = 1 бар = 4 долі (|1| |2| |3| |4|), в кожній долі 4 шістнадцяті.",
             "Склади через дефіс; те, що стоїть під однією долею, читається за цю долю. Крапка (·) = пауза на шістнадцяту.",
             f"Слухай {T['guide']}.mp3: тік на кожен склад (нижчий тік = склад на долю), біт грає тихіше.", ""]
    syl_word = "скл."
cur_sec = None
for l in flow:
    if l["sec"] != cur_sec:
        cur_sec = l["sec"]
        sheet += ["", f"===== {cur_sec} =====", ""]
    beats = [[] for _ in range(4)]
    prev_w = None
    for s in l["syl"]:
        b = min(3, int(s["slot"] // 4))
        piece = l["text"][s["s"]:s["e"]]
        if prev_w == s["w"] and beats[b]:
            beats[b][-1] += "-" + piece
        elif prev_w == s["w"]:
            beats[b].append("-" + piece)
        else:
            beats[b].append(piece)
        prev_w = s["w"]
    cells = " | ".join(f"{i + 1}: " + (" ".join(bt) if bt else "·") for i, bt in enumerate(beats))
    sheet.append(f"[{mmss(l['t'])}] ({l['n']:2d} {syl_word})  | {cells} |")
open(T["sheet"], "w", encoding="utf-8").write("\n".join(sheet))

# ---- guide audio: instrumental quieter + tick per syllable ----
beat, bsr = sf.read(T["beat"])
if beat.ndim == 1:
    beat = np.stack([beat, beat], 1)
g = beat * 0.4
N = len(g)
t = np.arange(int(0.012 * SR)) / SR
tick_hi = (np.random.default_rng(0).standard_normal(len(t)) * np.exp(-t / 0.002))
tick_hi = tick_hi / np.abs(tick_hi).max() * 0.6
t2 = np.arange(int(0.03 * SR)) / SR
tick_lo = np.sin(2 * np.pi * 900 * t2) * np.exp(-t2 / 0.008) * 0.8
for l in flow:
    for s in l["syl"]:
        tk = tick_lo if float(s["slot"]).is_integer() and int(s["slot"]) % 4 == 0 else tick_hi
        i = int(s["t"] * SR)
        e = min(N, i + len(tk))
        g[i:e, 0] += tk[: e - i]
        g[i:e, 1] += tk[: e - i]
g = np.clip(g, -0.98, 0.98)
sf.write(T["guide"] + ".wav", g.astype(np.float32), SR, subtype="PCM_16")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", T["guide"] + ".wav", "-codec:a", "libmp3lame", "-b:a", "192k", T["guide"] + ".mp3"], check=True)
tot = sum(l["n"] for l in flow)
print(f"{len(flow)} lines, {tot} syllables; max per bar {max(l['n'] for l in flow)}; wrote {T['flow']}, {T['sheet']}, {T['guide']}.mp3")
