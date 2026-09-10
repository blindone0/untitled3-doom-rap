# -*- coding: utf-8 -*-
"""Syllable-level flow for the lyrics: places every syllable on the 16th-note grid of its bar.
Outputs: out/flow.json (for karaoke.py), out/flow_sheet.txt (printable), out/guide_syllables.mp3 (beat + tick per syllable)."""
import json
import re
import subprocess

import numpy as np
import soundfile as sf

SR = 44100
VOW = set("аеєиіїоуюяАЕЄИІЇОУЮЯ")
LYRICS = "out/lyrics_oleni_doom_rap.txt"
BT = json.load(open("out/bar_times.json"))
bar_t = sorted((int(k), v) for k, v in BT["bars"].items())
bar_starts = np.array([v for _, v in bar_t])


def load_lines(path):
    lines, sec = [], ""
    for raw in open(path, encoding="utf-8"):
        line = raw.rstrip("\n")
        m = re.match(r"=== (.+?)\s+\(старт", line)
        if m:
            sec = m.group(1).split(" —")[0].strip()
            continue
        m = re.match(r"\[(\d+):(\d+\.\d)\] (.*)", line)
        if m:
            lines.append({"t": int(m.group(1)) * 60 + float(m.group(2)), "text": m.group(3), "sec": sec})
    return lines


def syllabify_word(w):
    """char spans of syllables inside a word (CV-based, good enough for karaoke)"""
    idx = [i for i, c in enumerate(w) if c in VOW]
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


def syllables_of_line(text):
    """list of (char_start, char_end, word_index, first_in_word) for the sung part (stage directions in () skipped)"""
    out, wi, pending = [], 0, None  # pending = start of a vowel-less word (з, в, й ...) glued to the next syllable
    for m in re.finditer(r"\S+", text):
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
    D = float(bar_starts[k + 1] - bar_starts[k]) if k + 1 < len(bar_starts) else 3.43
    syl = syllables_of_line(l["text"])
    pos = place(len(syl))
    entries = []
    for (a, b, wi, first), p in zip(syl, pos):
        entries.append({"s": a, "e": b, "w": wi, "first": first, "slot": p, "t": round(l["t"] + p * D / 16, 3)})
    flow.append({"t": l["t"], "end": round(l["t"] + D, 3), "text": l["text"], "sec": l["sec"], "syl": entries, "n": len(syl)})
json.dump(flow, open("out/flow.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)

# ---- printable sheet: syllables grouped under beats 1..4 ----
def mmss(t):
    return f"{int(t // 60)}:{t % 60:04.1f}"


sheet = ["ФЛОУ-ЛИСТ: кожен рядок = 1 бар = 4 долі (|1| |2| |3| |4|), в кожній долі 4 шістнадцяті.",
         "Склади через дефіс; те, що стоїть під однією долею, читається за цю долю. Крапка (·) = пауза на шістнадцяту.",
         "Слухай out/guide_syllables.mp3: тік на кожен склад (нижчий тік = склад на долю), біт грає тихіше.", ""]
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
    sheet.append(f"[{mmss(l['t'])}] ({l['n']:2d} скл.)  | {cells} |")
open("out/flow_sheet.txt", "w", encoding="utf-8").write("\n".join(sheet))

# ---- guide audio: instrumental quieter + tick per syllable ----
beat, bsr = sf.read("out/doom_cover_heavy_instrumental.wav")
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
sf.write("out/guide_syllables.wav", g.astype(np.float32), SR, subtype="PCM_16")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "out/guide_syllables.wav", "-codec:a", "libmp3lame", "-b:a", "192k", "out/guide_syllables.mp3"], check=True)
tot = sum(l["n"] for l in flow)
print(f"{len(flow)} lines, {tot} syllables; max per bar {max(l['n'] for l in flow)}; wrote out/flow.json, out/flow_sheet.txt, out/guide_syllables.mp3")
