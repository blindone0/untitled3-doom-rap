# -*- coding: utf-8 -*-
"""STATIC — original English lyrics in a strict metre: EVERY line is exactly 16 syllables, one per sixteenth note,
so the reading is a straight even machine pulse with no gaps anywhere in the song.
Themes: numbness, a cold ex, fake friends, syrup and smoke, TV static, not answering the phone. No death, no graves.
Reads out/static_bar_times.json (written by static_beat.py) -> out/lyrics_static.md (+ .txt for flow.py).
The syllable count of every line is checked here with the same splitter the flow and the vocal use."""
import json
import sys

from syllables import count

PER_BAR = 16

T = json.load(open("out/static_bar_times.json"))
bars = {int(k): v for k, v in T["bars"].items()}


def ts(bar):
    s = bars[bar]
    return f"{int(s // 60)}:{s % 60:04.1f}"


INTRO = [
    "Channel number three is on and there is nothing on the screen now",
    "I am not asleep, I am only waiting for the sound to end",
]
HOOK = [
    "Static in the speakers and there is static in my head tonight",
    "Nobody is calling and there is nobody at home tonight",
    "Cold house, empty table, and a cheap smoke in the yellow lamp light",
    "Blue screen in the corner with a nothing on it, that is alright",
    "Static in the speakers and there is static in my chest tonight",
    "I am never picking up the phone again, no I never will",
    "Nothing on the channel but I watch it till it is morning light",
    "Same old hoodie, same old couch, and the same old window, same old night",
]
VERSE1 = [
    "I woke up on the carpet with the television running on",
    "Channel number three, no picture, only snow and only the hum",
    "Everybody that I know is just a number on the glass",
    "I am letting all of it be ringing till it all goes away",
    "I don't want your money and I do not want your advice at all",
    "I got a hole inside my chest and all the weather turns to ice",
    "I'm driving in the middle of the night without the headlights on",
    "Empty road, the same old trees, and the same old fog, the same old road",
    "Everybody says get better and I am saying get away",
    "I bought the jacket off the rack for almost nothing, it's alright",
    "The rain is on the metal like a crowd is clapping in the dark",
    "I am a rerun of a rerun, same old show, the same old flaw",
]
VERSE2 = [
    "Dial tone and dial tone and there is still nobody at home tonight",
    "I'm talking to the static 'cause the static's all I really own",
    "You were saying it forever and then you said it with a goodnight",
    "And now the only loyal thing I got is the television light",
    "Every friend I ever had is just a rumour on a screen",
    "And I heard that I am doing badly, heard it from a small machine",
    "A thrift store jacket smelling like it came from somebody's dad",
    "It fits me better than the love you said you had for me back then",
    "Nothing on the roof and there is nothing in the basement here",
    "Only humming from the fridge and cracking pavement in my ear",
    "Syrup in the cup and I am turning all the volume way down",
    "Static on the screen and it is the only sound around tonight",
]
OUTRO = [
    "Static, static, static, static, static, static, static, static",
    "Turn it up and turn it up and turn it up and turn it all off",
    "I am still here, I am still here, I am still here, I am still here",
    "Nobody is, nobody is, nobody is, nobody is",
]

FLAT = "Flat, even, one syllable per sixteenth note, no pauses inside the line, the same speed and the same tone from " \
       "the first word to the last. Do not act the words."
sections = [
    ("INTRO", 2, INTRO, FLAT + " The loop plays alone under it."),
    ("HOOK 1", 4, HOOK, FLAT),
    ("VERSE 1", 12, VERSE1, FLAT),
    ("HOOK 2", 24, HOOK, FLAT),
    ("VERSE 2", 32, VERSE2, FLAT),
    ("HOOK 3", 44, HOOK, FLAT + " The hats are busier here; the voice does not change."),
    ("OUTRO", 52, OUTRO, FLAT + " The record dies underneath; the voice keeps the same pulse to the end."),
]

bad = [(t, i, l, count(l)) for (t, _, lines, _) in sections for i, l in enumerate(lines) if count(l) != PER_BAR]
if bad:
    for t, i, l, c in bad:
        print(f"{t} line {i + 1}: {c} syllables, not {PER_BAR}: {l}", file=sys.stderr)
    raise SystemExit(f"{len(bad)} lines break the metre")

step = (bars[1] - bars[0]) / PER_BAR
md = ["# STATIC — lyrics (English)", "",
      f"68 BPM, 4/4, one bar = {bars[1] - bars[0]:.2f} s. **Every line is exactly {PER_BAR} syllables — one per sixteenth note "
      f"({step * 1000:.0f} ms each) — for the whole song**, so the reading is one straight even pulse with no gaps. "
      "Times are positions in `out/static_instrumental.wav` (minus your sound-card latency). Original text.", ""]
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
       "   Or let the machine read it: `python static_vocal.py`.",
       "4. `python mixvocal.py --voice clean --vocal-db 4 --carve-db 5 --duck-db 1.5 --carve-lo 500 --carve-hi 3500 --dry --soft 1 --glue-db -15`"
       " — mixes the takes into `out/static.mp3`.",
       "", "Section starts:"] + [f"- {title}: {ts(b0)}" for (title, b0, _, _) in sections]
open("out/lyrics_static.md", "w", encoding="utf-8").write("\n".join(md))
open("out/lyrics_static.txt", "w", encoding="utf-8").write("\n".join(txt))
total = sum(len(l) for (_, _, l, _) in sections)
print(f"wrote out/lyrics_static.md / .txt — {total} lines, all exactly {PER_BAR} syllables ({step * 1000:.0f} ms per syllable); "
      f"hook {ts(4)}, verse 1 {ts(12)}, hook 2 {ts(24)}, verse 2 {ts(32)}, hook 3 {ts(44)}, outro {ts(52)}")
