# -*- coding: utf-8 -*-
"""Active track for the vocal tools (flow.py, sections.py, karaoke.py, studio.py, mixvocal.py, record.py).

  python track.py            show the active track
  python track.py static     switch to STATIC (BONES-style, English)
  python track.py oleni      switch back to «Олені» (doom-rap, Ukrainian)

The choice is stored in out/track.json; without it the original «Олені» track is used, so every old command keeps working.
Each track has its own beat, bar grid, lyric sheet, flow files, takes folder, final mix name, auto-tune key and
recording sections (name, first bar, bar after the last; names ending in _L/_R are panned doubles)."""
import json
import os
import sys

TRACKS = {
    "oleni": {
        "title": "Олені біжать", "beat": "out/doom_cover_heavy_instrumental.wav", "bars": "out/bar_times.json",
        "lyrics": "out/lyrics_oleni_doom_rap.txt", "flow": "out/flow.json", "sheet": "out/flow_sheet.txt",
        "guide": "out/guide_syllables", "takes": "vocals", "final": "out/oleni_doom_rap", "scale": "Am", "lang": "uk",
        "sections": [["intro", 2, 8], ["verse1", 8, 24], ["hook_L", 24, 32], ["hook_R", 24, 32], ["verse2", 42, 58],
                     ["verse3", 60, 72], ["ending", 75, 98]],
    },
    "static": {
        "title": "STATIC", "beat": "out/static_instrumental.wav", "bars": "out/static_bar_times.json",
        "lyrics": "out/lyrics_static.txt", "flow": "out/flow_static.json", "sheet": "out/flow_sheet_static.txt",
        "guide": "out/guide_syllables_static", "takes": "vocals_static", "final": "out/static", "scale": "Em", "lang": "en",
        # one centred take per section: a doubled robot voice combs against itself and the words stop being words
        "sections": [["intro", 2, 4], ["hook1", 4, 12], ["verse1", 12, 24], ["hook2", 24, 32],
                     ["verse2", 32, 44], ["hook3", 44, 52], ["outro", 52, 56]],
    },
}
STATE = "out/track.json"


def active_name():
    if os.path.exists(STATE):
        try:
            n = json.load(open(STATE)).get("name", "oleni")
            if n in TRACKS:
                return n
        except (ValueError, OSError):
            pass
    return "oleni"


NAME = active_name()
T = TRACKS[NAME]

if __name__ == "__main__":
    if len(sys.argv) > 1:
        n = sys.argv[1]
        if n not in TRACKS:
            raise SystemExit(f"unknown track {n!r}; known: {', '.join(TRACKS)}")
        os.makedirs("out", exist_ok=True)
        json.dump({"name": n}, open(STATE, "w"), indent=1)
        NAME, T = n, TRACKS[n]
        os.makedirs(T["takes"], exist_ok=True)
        # the mic latency is a property of the sound card, share it between tracks
        lat = "vocals/latency.json"
        if os.path.exists(lat) and not os.path.exists(os.path.join(T["takes"], "latency.json")):
            json.dump(json.load(open(lat)), open(os.path.join(T["takes"], "latency.json"), "w"))
    print(f"active track: {NAME} — {T['title']}")
    for k in ("beat", "bars", "lyrics", "flow", "takes", "final", "scale"):
        print(f"  {k:7s} {T[k]}")
