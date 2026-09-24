# -*- coding: utf-8 -*-
"""STATIC — original English lyrics (BONES-style delivery: deadpan, low, lo-fi, nothing to prove) with bar timestamps.
Themes: numbness, a cold ex, fake friends, syrup and smoke, TV static, not answering the phone. No death, no graves.
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
    "Cold house, no lights, blue screen on the bed",
    "Don't hit my line, I don't pick up no more",
    "Left your key on the table by the door",
    "Static in the speakers, static in my chest",
    "Cheap smoke, grey clouds, don't care about the rest",
    "Nothing on the screen but I'm watching it still",
    "Same hoodie, same couch, same windowsill",
]
VERSE1 = [
    "Woke up on the couch again, TV still on",
    "Channel three, no picture, just the snow and the hum",
    "Everybody I know is a name on my phone",
    "That I let ring out 'cause I like it alone",
    "I don't want your money, I don't want your advice",
    "Got a hole in my chest where the weather turns to ice",
    "Drive at night with the headlights off",
    "Empty road, same trees, same fog",
    "They say get better, I say get lost",
    "Bought my jacket off the rack at a discount cost",
    "Rain on the hood of the car like applause",
    "I'm a rerun, same show, same flaws",
]
VERSE2 = [
    "Dial tone, dial tone, nobody's home",
    "I talk to the static, it's the only thing I own",
    "You said forever, then you said goodnight",
    "Now the only thing that's loyal is the TV light",
    "Every friend I had turned into a rumor",
    "I heard I'm doing bad, I heard it from a computer",
    "Thrift-store jacket, smells like somebody's dad",
    "Fits me better than the love you said you had",
    "Nothing on the roof, nothing in the basement",
    "Just the hum of the fridge and the cracks in the pavement",
    "Syrup in the cup and the volume turned down",
    "Static on the screen, that's the only sound",
]
OUTRO = [
    "Static, static, static, static",
    "Turn it up, turn it up, turn it up, turn it off",
    "(whisper) I'm still here",
    "(whisper) Leave it",
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
      "Original text. Themes: numbness, a cold ex, fake friends, syrup and smoke, TV static, not answering the phone.", ""]
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
