# -*- coding: utf-8 -*-
"""Студія: записати дублі по секціях (кожна спроба зберігається окремо), вибрати кращий дубль,
прослухати дубль або весь трек із плейбаром, підігнати затримку, експортувати фінальний мікс.

  python studio.py
"""
import glob
import json
import os
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy import signal

SR = 44100
BEAT = "out/doom_cover_heavy_instrumental.wav"
FINAL = "out/oleni_doom_rap.wav"
VDIR = "vocals"
os.makedirs(VDIR, exist_ok=True)
PY = sys.executable

from sections import SECTIONS, WINDOWS, rec_params, window_of

SEC = {n: rec_params(n) for (n, _, _) in SECTIONS}  # name -> (sing start, record duration), phrase-based

beat, bsr = sf.read(BEAT)
assert bsr == SR
if beat.ndim == 1:
    beat = np.stack([beat, beat], 1)
beat = beat.astype(np.float32)
NBEAT = len(beat)
beat_rms_db = 20 * np.log10(np.sqrt((beat ** 2).mean()) + 1e-9)


def mmss(t):
    t = max(0.0, t)
    return f"{int(t // 60)}:{int(t % 60):02d}"


def latency_ms():
    p = os.path.join(VDIR, "latency.json")
    return json.load(open(p))["latency_ms"] if os.path.exists(p) else 0.0


def set_latency_ms(v):
    json.dump({"latency_ms": float(v)}, open(os.path.join(VDIR, "latency.json"), "w"))


def load_selection():
    p = os.path.join(VDIR, "selection.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def save_selection(sel):
    json.dump(sel, open(os.path.join(VDIR, "selection.json"), "w"), indent=1)


def takes_of(section):
    out = []
    for f in glob.glob(os.path.join(VDIR, f"{section}_take*.wav")):
        m = re.search(r"_take(\d+)\.wav$", f)
        if m:
            out.append((int(m.group(1)), f))
    return [f for _, f in sorted(out)]


def take_name(path):
    return os.path.splitext(os.path.basename(path))[0]


_cache = {}


def load_take(path):
    """mono float32, high-passed, levelled to the beat"""
    if path in _cache:
        return _cache[path]
    v, sr = sf.read(path)
    if v.ndim > 1:
        v = v.mean(1)
    if sr != SR:
        v = signal.resample_poly(v, SR, sr)
    v = v - v.mean()
    v = signal.sosfilt(signal.butter(2, 90, "high", fs=SR, output="sos"), v)
    act = np.abs(v) > 0.01
    rms_db = 20 * np.log10(np.sqrt((v[act] ** 2).mean()) + 1e-9) if act.any() else -60
    v = v * 10 ** ((beat_rms_db - 2.0 - rms_db) / 20)
    meta_p = os.path.splitext(path)[0] + ".json"
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {"song_start": 0.0}
    _cache[path] = (v.astype(np.float32), float(meta.get("song_start", 0.0)))
    return _cache[path]


def trim_to_window(v, s0, take):
    """keep only the take's own phrases: fade in/out at the section window (song seconds)"""
    w = window_of(take_name(take))
    if not w:
        return v
    w0, w1, _ = w
    v = v.copy()
    a = int((w0 - s0) * SR)
    b = int((w1 - s0) * SR)
    f = int(0.05 * SR)
    if a > 0:
        v[:max(0, a - f)] = 0
        v[max(0, a - f):a] *= np.linspace(0, 1, a - max(0, a - f))
    if b < len(v):
        b = max(b, 0)
        v[b + f:] = 0
        seg = v[b:b + f]
        seg *= np.linspace(1, 0, len(seg))
    return v


def build_mix(take_paths, beat_gain=0.8, vocal_db=0.0):
    mix = beat * beat_gain
    lat = latency_ms() / 1000.0
    for p in take_paths:
        v, s0 = load_take(p)
        v = trim_to_window(v, s0, p) * 10 ** (vocal_db / 20)
        s = int((s0 - lat) * SR)
        if s < 0:
            v = v[-s:]
            s = 0
        e = min(NBEAT, s + len(v))
        if e > s:
            mix[s:e, 0] += v[: e - s]
            mix[s:e, 1] += v[: e - s]
    return np.clip(mix, -0.98, 0.98)


class Player:
    def __init__(self):
        self.data = np.zeros((1, 2), np.float32)
        self.pos = 0
        self.stream = None
        self.playing = False

    def load(self, data, pos=0):
        self.pause()
        self.data = data
        self.pos = int(pos)

    def cb(self, outdata, frames, t, status):
        chunk = self.data[self.pos:self.pos + frames]
        outdata[:len(chunk)] = chunk
        outdata[len(chunk):] = 0
        self.pos += frames
        if self.pos >= len(self.data):
            self.playing = False
            raise sd.CallbackStop

    def play(self):
        if self.playing:
            return
        if self.pos >= len(self.data):
            self.pos = 0
        self.stream = sd.OutputStream(samplerate=SR, channels=2, callback=self.cb, blocksize=1024)
        self.stream.start()
        self.playing = True

    def pause(self):
        self.playing = False
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def seek(self, sec):
        self.pos = int(max(0, min(len(self.data) - 1, sec * SR)))


# ---------------- GUI ----------------
BG, FG, ACC, DIM = "#0b0b0d", "#f4f4f6", "#e0b341", "#8a8a95"
root = tk.Tk()
root.title("Олені біжать — студія")
root.configure(bg=BG)
root.geometry("1200x720")
F = tkfont.Font(family="Segoe UI", size=12)
FB = tkfont.Font(family="Segoe UI", size=12, weight="bold")
FS = tkfont.Font(family="Segoe UI", size=10)
player = Player()
selection = load_selection()
state = {"section": "verse1", "busy": False, "mode": "", "loaded_len": NBEAT}


def btn(parent, text, cmd, color="#22222a", **kw):
    return tk.Button(parent, text=text, command=cmd, font=F, bg=color, fg=FG, activebackground="#33333c", activeforeground=FG,
                     bd=0, padx=12, pady=6, cursor="hand2", **kw)


top = tk.Frame(root, bg=BG)
top.pack(fill="x", padx=16, pady=(12, 4))
tk.Label(top, text="СТУДІЯ", font=FB, fg=ACC, bg=BG).pack(side="left")
status = tk.Label(top, text="", font=FS, fg=DIM, bg=BG)
status.pack(side="left", padx=20)
tk.Label(top, text="затримка, мс:", font=FS, fg=DIM, bg=BG).pack(side="right")
lat_var = tk.StringVar(value=f"{latency_ms():.0f}")
lat_entry = tk.Entry(top, textvariable=lat_var, width=6, font=F, bg="#22222a", fg=FG, insertbackground=FG, bd=0, justify="center")
lat_entry.pack(side="right", padx=6)

body = tk.Frame(root, bg=BG)
body.pack(fill="both", expand=True, padx=16)

left = tk.Frame(body, bg=BG)
left.pack(side="left", fill="y", padx=(0, 12))
tk.Label(left, text="Секції", font=FB, fg=FG, bg=BG).pack(anchor="w")
sec_list = tk.Listbox(left, font=F, bg="#15151a", fg=FG, selectbackground=ACC, selectforeground="#000", bd=0, highlightthickness=0, width=34, height=9, activestyle="none")
sec_list.pack(fill="y", pady=6)
guide_var = tk.BooleanVar(value=False)
tk.Checkbutton(left, text="записувати під біт з тіками", variable=guide_var, font=FS, fg=DIM, bg=BG, selectcolor="#15151a", activebackground=BG, activeforeground=FG).pack(anchor="w")
btn(left, "▶ Репетиція секції (з підсвіткою)", lambda: launch_karaoke(rehearse=True)).pack(fill="x", pady=(8, 4))
btn(left, "● Записати новий дубль", lambda: launch_karaoke(rehearse=False), color="#5a1f1f").pack(fill="x", pady=4)

right = tk.Frame(body, bg=BG)
right.pack(side="left", fill="both", expand=True)
take_hdr = tk.Label(right, text="Дублі", font=FB, fg=FG, bg=BG)
take_hdr.pack(anchor="w")
take_list = tk.Listbox(right, font=F, bg="#15151a", fg=FG, selectbackground="#33333c", selectforeground=FG, bd=0, highlightthickness=0, height=9, activestyle="none")
take_list.pack(fill="both", expand=True, pady=6)
tb = tk.Frame(right, bg=BG)
tb.pack(fill="x")
btn(tb, "★ Використати цей дубль", lambda: use_take()).pack(side="left", padx=(0, 6))
btn(tb, "▶ Прослухати дубль", lambda: play_take()).pack(side="left", padx=6)
btn(tb, "✕ Видалити дубль", lambda: delete_take(), color="#3a1a1a").pack(side="right")

transport = tk.Frame(root, bg="#111116")
transport.pack(fill="x", padx=16, pady=12)
tr1 = tk.Frame(transport, bg="#111116")
tr1.pack(fill="x", padx=10, pady=(10, 0))
btn(tr1, "▶ / ❚❚", lambda: toggle_play()).pack(side="left")
btn(tr1, "■", lambda: stop_play()).pack(side="left", padx=6)
time_lbl = tk.Label(tr1, text="0:00 / 0:00", font=F, fg=FG, bg="#111116", width=12)
time_lbl.pack(side="left", padx=10)
mode_lbl = tk.Label(tr1, text="", font=FS, fg=DIM, bg="#111116")
mode_lbl.pack(side="left")
btn(tr1, "Експорт фінального міксу", lambda: export_mix(), color="#1f3a2a").pack(side="right")
btn(tr1, "▶ Весь трек (швидкий прев'ю)", lambda: play_full()).pack(side="right", padx=6)
tk.Label(tr1, text="гучність біту", font=FS, fg=DIM, bg="#111116").pack(side="right", padx=(10, 4))
beat_gain = tk.DoubleVar(value=0.8)
tk.Scale(tr1, from_=0.0, to=1.0, resolution=0.05, orient="horizontal", variable=beat_gain, length=120, showvalue=False, bg="#111116", fg=FG, troughcolor="#22222a", highlightthickness=0, bd=0).pack(side="right")
seek = tk.Scale(transport, from_=0, to=NBEAT / SR, resolution=0.1, orient="horizontal", showvalue=False, length=1100, bg="#111116", fg=FG, troughcolor="#22222a", highlightthickness=0, bd=0, sliderlength=18)
seek.pack(fill="x", padx=10, pady=(4, 10))
seeking = {"drag": False}
seek.bind("<ButtonPress-1>", lambda e: seeking.update(drag=True))


def on_seek_release(e):
    seeking["drag"] = False
    player.seek(seek.get())


seek.bind("<ButtonRelease-1>", on_seek_release)


# ---------------- логіка ----------------
def set_status(msg):
    status.config(text=msg)


def refresh_sections(keep=True):
    cur = state["section"]
    sec_list.delete(0, "end")
    for (n, a, b) in SECTIONS:
        s, d = SEC[n]
        tk_ = takes_of(n)
        used = selection.get(n)
        mark = f"★ {used.split('_take')[-1]}" if used else ("—" if not tk_ else "?")
        sec_list.insert("end", f"{n:8s} {mmss(s)}  {len(tk_)} дубл.  {mark}")
    names = [n for (n, _, _) in SECTIONS]
    if cur in names:
        sec_list.selection_set(names.index(cur))


def refresh_takes():
    take_list.delete(0, "end")
    sec = state["section"]
    for f in takes_of(sec):
        info = sf.info(f)
        star = "★ " if selection.get(sec) == take_name(f) else "   "
        when = time.strftime("%H:%M", time.localtime(os.path.getmtime(f)))
        take_list.insert("end", f"{star}{take_name(f)}   {info.duration:5.1f} с   {when}")
    take_hdr.config(text=f"Дублі — {sec}  (старт {mmss(SEC[sec][0])}, {SEC[sec][1]:.0f} с)")


def on_section_select(e=None):
    if sec_list.curselection():
        state["section"] = SECTIONS[sec_list.curselection()[0]][0]
        refresh_takes()


sec_list.bind("<<ListboxSelect>>", on_section_select)


def current_take():
    if not take_list.curselection():
        return None
    return takes_of(state["section"])[take_list.curselection()[0]]


def use_take():
    p = current_take()
    if not p:
        return set_status("вибери дубль у списку")
    selection[state["section"]] = take_name(p)
    save_selection(selection)
    refresh_sections()
    refresh_takes()
    set_status(f"для {state['section']} використовується {take_name(p)}")


def delete_take():
    p = current_take()
    if not p:
        return
    if not messagebox.askyesno("Видалити", f"Видалити {take_name(p)} назавжди?"):
        return
    os.remove(p)
    jp = os.path.splitext(p)[0] + ".json"
    if os.path.exists(jp):
        os.remove(jp)
    _cache.pop(p, None)
    if selection.get(state["section"]) == take_name(p):
        selection.pop(state["section"], None)
        save_selection(selection)
    refresh_sections()
    refresh_takes()


def selected_paths():
    out = []
    for (n, _, _) in SECTIONS:
        if n in selection:
            p = os.path.join(VDIR, selection[n] + ".wav")
            if os.path.exists(p):
                out.append(p)
    return out


def apply_latency():
    try:
        set_latency_ms(float(lat_var.get()))
    except ValueError:
        pass


def play_take():
    p = current_take()
    if not p:
        return set_status("вибери дубль у списку")
    apply_latency()
    v, s0 = load_take(p)
    mix = build_mix([p], beat_gain=beat_gain.get())
    player.load(mix, pos=max(0, int((s0 - 0.5) * SR)))
    state["mode"] = f"дубль {take_name(p)}"
    mode_lbl.config(text=state["mode"])
    player.play()


def play_full():
    apply_latency()
    paths = selected_paths()
    mix = build_mix(paths, beat_gain=beat_gain.get())
    player.load(mix, pos=0)
    state["mode"] = f"весь трек, дублів: {len(paths)}"
    mode_lbl.config(text=state["mode"])
    player.play()


def toggle_play():
    if player.playing:
        player.pause()
    else:
        player.play()


def stop_play():
    player.pause()
    player.seek(0)


def launch_karaoke(rehearse):
    if state["busy"]:
        return set_status("зачекай, попередній процес ще працює")
    player.pause()
    sec = state["section"]
    s, d = SEC[sec]
    n = len(takes_of(sec)) + 1
    name = f"{sec}_take{n}"
    cmd = [PY, "karaoke.py", "--start", f"{s}", "--dur", f"{d}"]
    cmd += ["--rehearse"] if rehearse else ["--name", name]
    if guide_var.get() or rehearse:
        cmd.append("--guide")
    state["busy"] = True
    set_status(("репетиція " if rehearse else "запис ") + sec + " … (вікно суфлера відкрите)")

    def run():
        subprocess.run(cmd, cwd=os.getcwd())
        state["busy"] = False
        root.after(0, after_record, name if not rehearse else None)

    threading.Thread(target=run, daemon=True).start()


def after_record(name):
    refresh_sections()
    refresh_takes()
    if name and os.path.exists(os.path.join(VDIR, name + ".wav")):
        selection[state["section"]] = name
        save_selection(selection)
        refresh_sections()
        refresh_takes()
        set_status(f"записано {name} і вибрано як робочий дубль (можна змінити)")
    else:
        set_status("готово")


def export_mix():
    if state["busy"]:
        return set_status("зачекай, інший процес ще працює")
    apply_latency()
    player.pause()
    state["busy"] = True
    set_status("експорт: mixvocal.py працює (≈30 с) …")

    def run():
        r = subprocess.run([PY, "mixvocal.py"], cwd=os.getcwd(), capture_output=True, text=True)
        state["busy"] = False
        root.after(0, after_export, r.returncode, (r.stdout or "") + (r.stderr or ""))

    threading.Thread(target=run, daemon=True).start()


def after_export(code, log):
    if code != 0 or not os.path.exists(FINAL):
        set_status("експорт не вдався, дивись консоль")
        print(log)
        return
    y, _ = sf.read(FINAL)
    player.load(y.astype(np.float32), pos=0)
    state["mode"] = "фінальний мікс out/oleni_doom_rap.mp3"
    mode_lbl.config(text=state["mode"])
    set_status("експортовано out/oleni_doom_rap.mp3 і .wav — граю фінал")
    player.play()


def poll():
    n = len(player.data)
    if not seeking["drag"]:
        seek.config(to=n / SR)
        seek.set(player.pos / SR)
    time_lbl.config(text=f"{mmss(player.pos / SR)} / {mmss(n / SR)}")
    root.after(100, poll)


def on_close():
    player.pause()
    apply_latency()
    root.destroy()


root.protocol("WM_DELETE_WINDOW", on_close)
root.bind("<space>", lambda e: toggle_play() if root.focus_get() is not lat_entry else None)
refresh_sections()
refresh_takes()
poll()
if "--selftest" in sys.argv:
    print("sections:", {k: v for k, v in SEC.items()})
    print("takes:", {n: len(takes_of(n)) for (n, _, _) in SECTIONS})
    m = build_mix(selected_paths())
    print("mix ok", m.shape, float(np.abs(m).max()))
    root.destroy()
else:
    root.mainloop()
