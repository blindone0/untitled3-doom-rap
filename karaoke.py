# -*- coding: utf-8 -*-
"""Караоке-суфлер з підсвіткою по складах + запис вокалу в одному вікні.

  python karaoke.py --rehearse                             репетиція всієї пісні (біт грає, нічого не пишеться)
  python karaoke.py --rehearse --start 34.4 --dur 60       репетиція однієї секції
  python karaoke.py --name verse1 --start 34.4 --dur 60    запис секції (біт стартує за 7 с до входу)
  python karaoke.py --name full --full                     запис всієї пісні одним дублем
  python karaoke.py --guide ...                            замість чистого біту грає out/guide_syllables.wav (тік на кожен склад)
  python karaoke.py --list                                 список аудіопристроїв (--in N --out N щоб вибрати)

У вікні: ПРОБІЛ — старт, ESC — вихід (під час запису = скасувати). Дублі: vocals/<name>.wav (+ .json) -> python mixvocal.py
Потрібен out/flow.json (python flow.py) для підсвітки по складах.
"""
import argparse
import json
import os
import re

import numpy as np
import sounddevice as sd
import soundfile as sf

SR = 44100
ap = argparse.ArgumentParser()
ap.add_argument("--list", action="store_true")
ap.add_argument("--selftest", action="store_true")
ap.add_argument("--rehearse", action="store_true", help="тільки грати біт і показувати текст, без запису")
ap.add_argument("--guide", action="store_true", help="грати біт з тіками на кожен склад (out/guide_syllables.wav)")
ap.add_argument("--name", default=None)
ap.add_argument("--start", type=float, default=None)
ap.add_argument("--dur", type=float, default=60.0)
ap.add_argument("--pre", type=float, default=7.0, help="скільки секунд біту грає до входу (відлік)")
ap.add_argument("--full", action="store_true")
ap.add_argument("--beat", default="out/doom_cover_heavy_instrumental.wav")
ap.add_argument("--flow", default="out/flow.json")
ap.add_argument("--in", dest="dev_in", type=int, default=None)
ap.add_argument("--out", dest="dev_out", type=int, default=None)
ap.add_argument("--monitor", type=float, default=1.0, help="гучність біту 0..1")
ap.add_argument("--font", type=int, default=34)
args = ap.parse_args()

if args.list:
    print(sd.query_devices())
    raise SystemExit
if not (args.rehearse or args.name or args.selftest):
    ap.error("потрібно --rehearse або --name <назва_дубля>")

# ---------- текст по складах ----------
LINES = json.load(open(args.flow, encoding="utf-8"))
LINES.sort(key=lambda x: x["t"])
BT = json.load(open("out/bar_times.json"))
bar_t = sorted((int(k), v) for k, v in BT["bars"].items())
BEAT_T, BEAT_INFO = [], []
for (k, t0), (_, t1) in zip(bar_t[:-1], bar_t[1:]):
    for i in range(4):
        BEAT_T.append(t0 + i * (t1 - t0) / 4)
        BEAT_INFO.append((k, i + 1))
BEAT_T = np.array(BEAT_T)

# ---------- аудіо ----------
beat_path = "out/guide_syllables.wav" if args.guide else args.beat
beat, bsr = sf.read(beat_path)
assert bsr == SR
if beat.ndim == 1:
    beat = np.stack([beat, beat], 1)
if args.full or args.start is None:
    seg_start, seg_end, part_start = 0.0, len(beat) / SR, 0.0
else:
    seg_start = max(0.0, args.start - args.pre)
    seg_end = min(len(beat) / SR, args.start + args.dur)
    part_start = args.start
DATA = (beat[int(seg_start * SR):int(seg_end * SR)] * args.monitor).astype(np.float32)
RECORD = not args.rehearse


class Engine:
    def __init__(self):
        self.pos, self.rec, self.level, self.done, self.started, self.stream = 0, [], 0.0, False, False, None

    def cb(self, indata, outdata, frames, t, status):
        chunk = DATA[self.pos:self.pos + frames]
        outdata[:len(chunk)] = chunk
        outdata[len(chunk):] = 0
        if RECORD:
            self.rec.append(indata[:, 0].copy())
        self.level = float(np.abs(indata).max())
        self.pos += frames
        if self.pos >= len(DATA):
            self.done = True
            raise sd.CallbackStop

    def start(self):
        self.stream = sd.Stream(samplerate=SR, channels=(1, 2), device=(args.dev_in, args.dev_out), callback=self.cb, blocksize=1024)
        self.stream.start()
        self.started = True

    def stop(self):
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass

    def song_time(self):
        return seg_start + self.pos / SR

    def save(self):
        if not RECORD or not self.rec:
            return None
        os.makedirs("vocals", exist_ok=True)
        rec = np.concatenate(self.rec)
        sf.write(f"vocals/{args.name}.wav", rec, SR, subtype="FLOAT")
        json.dump({"song_start": seg_start, "part_start": part_start, "beat": args.beat}, open(f"vocals/{args.name}.json", "w"))
        return float(np.abs(rec).max())


def mmss(t):
    return f"{int(t // 60)}:{t % 60:04.1f}"


def line_at(t, look=0.10):
    idx = int(np.searchsorted([l["t"] for l in LINES], t + look, side="right")) - 1
    return idx


if args.selftest:
    print(f"flow: {len(LINES)} lines, {sum(l['n'] for l in LINES)} syllables; beats {len(BEAT_T)}; segment {seg_start:.1f}-{seg_end:.1f}s record={RECORD} beat={beat_path}")
    import tkinter as tk
    r = tk.Tk()
    r.withdraw()
    r.destroy()
    print("tk ok")
    raise SystemExit

# ---------- вікно ----------
import tkinter as tk
import tkinter.font as tkfont

BG = "#0b0b0d"
root = tk.Tk()
root.title("Олені біжать — суфлер")
root.configure(bg=BG)
root.geometry("1400x820")
BIG = tkfont.Font(family="Segoe UI", size=args.font, weight="bold")
MID = tkfont.Font(family="Segoe UI", size=int(args.font * 0.62))
SMALL = tkfont.Font(family="Segoe UI", size=int(args.font * 0.42))
STRIP = tkfont.Font(family="Segoe UI", size=int(args.font * 0.5), weight="bold")
HUGE = tkfont.Font(family="Segoe UI", size=int(args.font * 2.0), weight="bold")

top = tk.Frame(root, bg=BG)
top.pack(fill="x", padx=30, pady=(16, 0))
sec_lbl = tk.Label(top, text="", font=SMALL, fg="#8a8a95", bg=BG, anchor="w")
sec_lbl.pack(side="left")
time_lbl = tk.Label(top, text="", font=SMALL, fg="#8a8a95", bg=BG, anchor="e")
time_lbl.pack(side="right")

mid = tk.Frame(root, bg=BG)
mid.pack(fill="both", expand=True, padx=40)
count_lbl = tk.Label(mid, text="", font=HUGE, fg="#e0b341", bg=BG)
count_lbl.pack(pady=(6, 0))
prev_lbl = tk.Label(mid, text="", font=MID, fg="#3e3e46", bg=BG, wraplength=1300, justify="center")
prev_lbl.pack(pady=(4, 4))
cur_txt = tk.Text(mid, height=3, font=BIG, bg=BG, fg="#5c5c66", wrap="word", bd=0, highlightthickness=0, cursor="arrow", padx=10)
cur_txt.pack(fill="x", pady=4)
cur_txt.tag_configure("center", justify="center")
cur_txt.tag_configure("done", foreground="#e0b341")
cur_txt.tag_configure("now", foreground="#ffffff", underline=True)
cur_txt.tag_configure("todo", foreground="#5c5c66")
cur_txt.configure(state="disabled")
STRIP_W, STRIP_H = 1280, 84
strip = tk.Canvas(mid, width=STRIP_W, height=STRIP_H, bg="#111116", highlightthickness=0)
strip.pack(pady=(4, 8))
for i in range(17):
    x = i * STRIP_W / 16
    strip.create_line(x, 0, x, STRIP_H, fill="#3a3a44" if i % 4 == 0 else "#1f1f26", width=2 if i % 4 == 0 else 1)
for b in range(4):
    strip.create_text(b * STRIP_W / 4 + 6, 8, text=str(b + 1), fill="#8a8a95", font=SMALL, anchor="nw")
strip_items = []
playhead = strip.create_line(0, 0, 0, STRIP_H, fill="#ffffff", width=3)
next_lbl = tk.Label(mid, text="", font=MID, fg="#a9a9b3", bg=BG, wraplength=1300, justify="center")
next_lbl.pack(pady=4)
next2_lbl = tk.Label(mid, text="", font=MID, fg="#55555f", bg=BG, wraplength=1300, justify="center")
next2_lbl.pack(pady=2)

bottom = tk.Frame(root, bg=BG)
bottom.pack(fill="x", padx=30, pady=(0, 16))
beat_cv = tk.Canvas(bottom, width=140, height=40, bg=BG, highlightthickness=0)
beat_cv.pack(side="left")
beat_dots = [beat_cv.create_oval(10 + i * 32, 10, 30 + i * 32, 30, fill="#2a2a32", width=0) for i in range(4)]
status_lbl = tk.Label(bottom, text="ПРОБІЛ — старт    ESC — вихід", font=SMALL, fg="#8a8a95", bg=BG)
status_lbl.pack(side="left", padx=30)
lvl_cv = tk.Canvas(bottom, width=300, height=24, bg="#1a1a20", highlightthickness=0)
lvl_cv.pack(side="right")
lvl_bar = lvl_cv.create_rectangle(0, 0, 0, 24, fill="#3fbf6f", width=0)
tk.Label(bottom, text="мікрофон", font=SMALL, fg="#8a8a95", bg=BG).pack(side="right", padx=10)

eng = Engine()
state = {"finished": False, "peak_hold": 0.0, "line": -2, "syl": -2}
mode = "РЕПЕТИЦІЯ" if not RECORD else f"ЗАПИС «{args.name}»"
sec_lbl.config(text=f"{mode}   біт {mmss(seg_start)} → {mmss(seg_end)}   твій вхід {mmss(part_start)}")


def set_line_text(text):
    cur_txt.configure(state="normal")
    cur_txt.delete("1.0", "end")
    cur_txt.insert("1.0", text)
    cur_txt.tag_add("center", "1.0", "end")
    cur_txt.tag_add("todo", "1.0", "end")
    cur_txt.configure(state="disabled")


def draw_strip(line):
    for it in strip_items:
        strip.delete(it)
    strip_items.clear()
    if line is None:
        return
    for s in line["syl"]:
        x = s["slot"] / 16 * STRIP_W + 4
        piece = line["text"][s["s"]:s["e"]].strip(",.;:!?—«»")
        strip_items.append(strip.create_text(x, STRIP_H * 0.62, text=piece, fill="#e0b341" if s["first"] else "#c9c9d0", font=STRIP, anchor="w"))
    strip.tag_raise(playhead)


def paint_syllables(line, k):
    """k = індекс поточного складу (-1 = ще нічого не проспівано)"""
    cur_txt.configure(state="normal")
    for tag in ("done", "now", "todo"):
        cur_txt.tag_remove(tag, "1.0", "end")
    if k < 0:
        cur_txt.tag_add("todo", "1.0", "end")
    else:
        s = line["syl"][k]
        cur_txt.tag_add("done", "1.0", f"1.0+{s['s']}c")
        cur_txt.tag_add("now", f"1.0+{s['s']}c", f"1.0+{s['e']}c")
        cur_txt.tag_add("todo", f"1.0+{s['e']}c", "end")
    cur_txt.configure(state="disabled")


def tick():
    if not eng.started:
        set_line_text("Натисни ПРОБІЛ, коли будеш готовий")
        first = [l for l in LINES if l["t"] >= part_start]
        next_lbl.config(text=(first[0]["text"] if first else ""))
        root.after(50, tick)
        return
    t = eng.song_time()
    time_lbl.config(text=mmss(t))
    count_lbl.config(text=f"вхід через {part_start - t:0.1f}" if t < part_start - 0.05 else "")
    i = line_at(t)
    if i != state["line"]:
        state["line"], state["syl"] = i, -2
        if i >= 0:
            l = LINES[i]
            set_line_text(l["text"])
            draw_strip(l)
            prev_lbl.config(text=LINES[i - 1]["text"] if i > 0 else "")
            next_lbl.config(text=LINES[i + 1]["text"] if i + 1 < len(LINES) else "")
            next2_lbl.config(text=LINES[i + 2]["text"] if i + 2 < len(LINES) else "")
            sec_lbl.config(text=f"{mode}   {l['sec']}")
        else:
            set_line_text("…")
            draw_strip(None)
            first = [x for x in LINES if x["t"] >= t]
            prev_lbl.config(text="")
            next_lbl.config(text=first[0]["text"] if first else "")
            next2_lbl.config(text=first[1]["text"] if len(first) > 1 else "")
    if i >= 0:
        l = LINES[i]
        outside = l["t"] >= seg_end - 0.05  # next section's phrase: shown dim, not to be sung in this take
        k = -1 if outside else (int(np.searchsorted([s["t"] for s in l["syl"]], t + 0.05, side="right")) - 1 if l["syl"] else -1)
        if k != state["syl"]:
            state["syl"] = k
            paint_syllables(l, k)
            if outside:
                count_lbl.config(text="наступна секція — не співати")
        frac = float(np.clip((t - l["t"]) / max(0.1, l["end"] - l["t"]), 0, 1))
        x = frac * STRIP_W
        strip.coords(playhead, x, 0, x, STRIP_H)
    # такт
    kb = int(np.searchsorted(BEAT_T, t, side="right")) - 1
    if kb >= 0:
        bar, b = BEAT_INFO[kb]
        since = t - BEAT_T[kb]
        for j, d in enumerate(beat_dots):
            on = (j == b - 1) and since < 0.18
            beat_cv.itemconfig(d, fill=("#e0b341" if (on and b == 1) else "#f4f4f6" if on else "#2a2a32"))
        status_lbl.config(text=f"бар {bar}  доля {b}     ESC — {'скасувати' if RECORD else 'вихід'}")
    lv = eng.level
    state["peak_hold"] = max(lv, state["peak_hold"] * 0.9)
    lvl_cv.coords(lvl_bar, 0, 0, int(300 * min(1.0, state["peak_hold"])), 24)
    lvl_cv.itemconfig(lvl_bar, fill="#d9463b" if state["peak_hold"] > 0.95 else "#3fbf6f")
    if eng.done and not state["finished"]:
        state["finished"] = True
        eng.stop()
        pk = eng.save()
        if pk is None:
            set_line_text("Репетицію завершено. ESC — вихід")
        else:
            warn = "  (ЗАНАДТО ГУЧНО — зменш gain мікрофона)" if pk > 0.95 else ("  (дуже тихо — збільш gain)" if pk < 0.05 else "")
            set_line_text(f"Збережено vocals/{args.name}.wav   пік {20 * np.log10(pk + 1e-9):.0f} dBFS{warn}")
        next_lbl.config(text="далі:  python mixvocal.py" if pk is not None else "")
        next2_lbl.config(text="")
        count_lbl.config(text="")
        return
    root.after(30, tick)


def on_space(e=None):
    if not eng.started and not state["finished"]:
        count_lbl.config(text="")
        eng.start()


def on_esc(e=None):
    eng.stop()
    if eng.started and RECORD and not state["finished"]:
        print("скасовано, нічого не збережено")
    root.destroy()


root.bind("<space>", on_space)
root.bind("<Escape>", on_esc)
root.protocol("WM_DELETE_WINDOW", on_esc)
tick()
root.mainloop()
