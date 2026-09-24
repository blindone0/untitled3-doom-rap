# -*- coding: utf-8 -*-
"""STATIC — original English lyrics (BONES-style: deadpan, low, lo-fi, nothing to prove) with bar timestamps.
Reads out/static_bar_times.json (written by static_beat.py) -> out/lyrics_static.md (+ .txt for flow.py)."""
import json

T = json.load(open("out/static_bar_times.json"))
bars = {int(k): v for k, v in T["bars"].items()}


def ts(bar):
    s = bars[bar]
    return f"{int(s // 60)}:{s % 60:04.1f}"


INTRO = [
    "(spoken) Yeah. Channel three. Nothing on.",
    "(spoken) Leave it.",
]
HOOK = [
    "Static in the speakers, static in my head",
    "Cold house, no lights, I been sleeping with the dead",
    "Don't call my phone, I don't pick up no more",
    "Left the key in the dirt, buried under the floor",
    "Static in the speakers, static in my chest",
    "Cheap smoke, black clouds, put the rest to rest",
    "Nothing on the screen but I'm watching it still",
    "Ghost in a grey hoodie on the windowsill",
]
VERSE1 = [
    "Woke up on the floor again, TV still on",
    "Channel three, no picture, just the snow and the hum",
    "Everybody I know is a name on a stone",
    "Or a voice on my phone that I let ring alone",
    "I don't want your money, I don't want your advice",
    "Got a hole in my chest where the weather turns to ice",
    "Drive at night with the headlights off",
    "Dead-end road, same trees, same fog",
    "They say get better, I say get lost",
    "Bought my box off the rack at a discount cost",
    "Rain on the hood of the car like applause",
    "I'm a rerun, a ghost, a signal with no cause",
]
VERSE2 = [
    "Dial tone, dial tone, nobody's home",
    "I talk to the static, it's the only thing I own",
    "Grey sky, grey skin, grey water in the tub",
    "Mold on the ceiling spelling out my name, no love",
    "Every friend I had turned into a rumor",
    "Every night's a funeral and I'm the only mourner",
    "Thrift-store jacket, smells like somebody's dad",
    "Dead man's clothes fit better than anything I had",
    "Nothing on the roof, nothing in the basement",
    "Just the hum of the fridge and the cracks in the pavement",
    "When they find me I'll be smiling with the volume down",
    "Static on the screen, static in the ground",
]
OUTRO = [
    "Static, static, static, static",
    "Turn it up, turn it up, turn it up, turn it off",
    "(whisper) I'm still here",
    "(whisper) Nobody is",
]

sections = [
    ("INTRO — spoken, not rapped", 2, INTRO,
     "Only the loop is playing. Say it flat, close to the mic, like you are talking to nobody. One line per bar."),
    ("HOOK 1", 4, HOOK,
     "Deadpan, low, no energy, a hair behind the beat. Every line lands with the snare on beat 3 ('static in my HEAD'). "
     "Record it twice (hook1_L and hook1_R) — the mix pans the doubles."),
    ("VERSE 1", 12, VERSE1,
     "Monotone, lazy 8ths — about 10-12 syllables per bar, let the words drag. Do not push volume; the mic does the work. "
     "Last bar: the drums drop out — let the line hang."),
    ("HOOK 2", 24, HOOK, "Same as hook 1, doubled (hook2_L / hook2_R)."),
    ("VERSE 2", 32, VERSE2,
     "Even lower and slower than verse 1. Half-whisper the last two lines."),
    ("HOOK 3", 44, HOOK, "Busier hats under you now; keep the voice exactly as flat as before. Doubled (hook3_L / hook3_R)."),
    ("OUTRO", 52, OUTRO,
     "First two lines chanted on the beat; the record is dying underneath. Last two lines: whisper, then silence."),
]

md = ["# STATIC — lyrics (English)", "",
      f"68 BPM, 4/4, one bar = {bars[1] - bars[0]:.2f} s. Times are positions in `out/static_instrumental.wav` (minus your sound-card latency). "
      "Original text. Themes: numbness, dead TV channels, an empty house, being a ghost while still alive.", ""]
txt = []
for title, bar0, lines, note in sections:
    md += [f"## {title}", f"_bars {bar0}–{bar0 + len(lines) - 1}, start {ts(bar0)}_", "", f"> {note}", "", "| bar | time | line |", "|---|---|---|"]
    txt += ["", f"=== {title}  (start {ts(bar0)}) ===", note, ""]
    for i, line in enumerate(lines):
        b = bar0 + i
        md.append(f"| {b} | {ts(b)} | {line} |")
        txt.append(f"[{ts(b)}] {line}")
    md.append("")
md += ["## How to record", "",
       "1. `python track.py static` — switches all the vocal tools to this track (`python track.py oleni` switches back).",
       "2. `python flow.py` — syllable grid, flow sheet and guide audio with a tick per syllable.",
       "3. `python studio.py` — pick a section, record takes (the prompter highlights every syllable), choose the best take, export.",
       "   Or by hand: `python karaoke.py --name verse1 --start " + f"{bars[12]:.1f}" + " --dur 45`.",
       f"4. Hooks twice: `hook1_L` and `hook1_R` (start {ts(4)}), same for hook2 ({ts(24)}) and hook3 ({ts(44)}).",
       "5. `python mixvocal.py` — mixes the takes into `out/static.mp3` (auto-tune key is E minor; `--voice clean` for a raw take).",
       "", "Section starts:"] + [f"- {title.split(' —')[0]}: {ts(b0)}" for (title, b0, _, _) in sections]
open("out/lyrics_static.md", "w", encoding="utf-8").write("\n".join(md))
open("out/lyrics_static.txt", "w", encoding="utf-8").write("\n".join(txt))
total = sum(len(l) for (_, _, l, _) in sections)
print(f"wrote out/lyrics_static.md / .txt ({total} lines); hook {ts(4)}, verse 1 {ts(12)}, hook 2 {ts(24)}, verse 2 {ts(32)}, hook 3 {ts(44)}, outro {ts(52)}")
