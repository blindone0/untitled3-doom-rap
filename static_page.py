# -*- coding: utf-8 -*-
"""Build the listening page for the active track from its lyric sheet: out/static_page.html.
The page is published as an Artifact together with out/static.mp3 and out/static_instrumental.mp3."""
import html
import json
import re

from track import T

LY = T["lyrics"]
BT = json.load(open(T["bars"]))
BARS = {int(k): v for k, v in BT["bars"].items()}
DUR = max(BARS.values())

sections = []
for raw in open(LY, encoding="utf-8"):
    line = raw.rstrip("\n")
    m = re.match(r"=== (.+?)\s+\(start", line)
    if m:
        sections.append({"name": m.group(1).strip(), "lines": []})
        continue
    m = re.match(r"\[(\d+):(\d+\.\d)\] (.*)", line)
    if m and sections:
        t = int(m.group(1)) * 60 + float(m.group(2))
        sections[-1]["lines"].append((t, m.group(3)))


def mmss(t):
    return f"{int(t // 60)}:{t % 60:04.1f}"


body = []
for s in sections:
    a = s["lines"][0][0]
    body.append('<section class="part">')
    body.append(f'<h2 class="part-name">{html.escape(s["name"])}'
                f'<span class="part-meta">{mmss(a)} · {len(s["lines"])} bars</span></h2>')
    body.append('<ol class="lines">')
    for t, text in s["lines"]:
        body.append(f'<li><button class="line" data-t="{t:.1f}">'
                    f'<span class="t">{mmss(t)}</span><span class="w">{html.escape(text)}</span></button></li>')
    body.append("</ol></section>")
LYRICS = "\n".join(body)
TOTAL = mmss(197.6)

PAGE = """<title>STATIC</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@600;800&family=IBM+Plex+Mono:wght@400;500&family=Newsreader:opsz,wght@6..72,300;6..72,400&display=swap">
<style>
  /* One deliberate world: a dead television at four in the morning. Every colour is painted, so the page holds
     whichever ground the viewer's theme paints behind it. */
  :root {
    --ground: #0f1214;
    --surface: #171b1f;
    --edge: #262c31;
    --ink: #dfe4e7;
    --muted: #79838b;
    --screen: #3d76ad;   /* the blue of a channel with nothing on it */
    --lamp: #d8a54a;     /* the yellow lamp light from the hook, spent only on the playhead */
    --mono: "IBM Plex Mono", ui-monospace, "Cascadia Mono", Consolas, monospace;
    --display: "Archivo", "Helvetica Neue", Arial, sans-serif;
    --body: "Newsreader", Georgia, "Times New Roman", serif;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--ground);
    color: var(--ink);
    font-family: var(--body);
    -webkit-font-smoothing: antialiased;
  }
  .wrap { max-width: 44rem; margin: 0 auto; padding-inline: 20px; padding-block: 0 72px; }

  /* ---------- header ---------- */
  header { position: relative; padding-block: 72px 28px; }
  #snow {
    position: absolute; inset: -8px -20px auto -20px; height: 260px; width: calc(100% + 40px);
    opacity: .16; pointer-events: none; mix-blend-mode: screen;
    -webkit-mask-image: linear-gradient(#000, transparent 86%);
    mask-image: linear-gradient(#000, transparent 86%);
  }
  .eyebrow {
    position: relative; font-family: var(--mono); font-size: .72rem; letter-spacing: .22em;
    text-transform: uppercase; color: var(--screen); margin: 0 0 14px;
  }
  h1 {
    position: relative; font-family: var(--display); font-weight: 800; font-size: clamp(3.4rem, 17vw, 6.6rem);
    line-height: .86; letter-spacing: -.035em; margin: 0; text-wrap: balance;
  }
  .spec {
    position: relative; font-family: var(--mono); font-size: .78rem; color: var(--muted);
    margin: 20px 0 0; line-height: 1.9;
  }
  .spec b { color: var(--ink); font-weight: 500; }

  /* ---------- player ---------- */
  .player {
    position: sticky; top: 0; z-index: 5; display: flex; align-items: center; gap: 14px;
    background: color-mix(in srgb, var(--ground) 88%, transparent);
    backdrop-filter: blur(8px);
    border-block: 1px solid var(--edge); padding: 14px 0; margin-block: 8px 34px;
  }
  .play {
    flex: none; width: 52px; height: 52px; border-radius: 50%; border: 1px solid var(--edge);
    background: var(--surface); color: var(--ink); cursor: pointer; display: grid; place-items: center;
    transition: border-color .15s ease, background .15s ease;
  }
  .play:hover { border-color: var(--screen); background: #1b2128; }
  .play svg { width: 17px; height: 17px; fill: currentColor; }
  .bars { flex: 1 1 auto; min-width: 0; }
  #seek { width: 100%; margin: 0; accent-color: var(--lamp); height: 20px; cursor: pointer; }
  .clock {
    display: flex; justify-content: space-between; font-family: var(--mono); font-size: .72rem;
    color: var(--muted); font-variant-numeric: tabular-nums; margin-top: 2px;
  }
  .clock .on { color: var(--lamp); }

  /* ---------- lyrics ---------- */
  .part { margin-block: 0 38px; }
  .part-name {
    display: flex; align-items: baseline; gap: 12px; font-family: var(--mono); font-weight: 500;
    font-size: .74rem; letter-spacing: .2em; text-transform: uppercase; color: var(--screen);
    margin: 0 0 14px; padding-bottom: 8px; border-bottom: 1px solid var(--edge);
  }
  .part-meta { margin-left: auto; color: var(--muted); letter-spacing: .08em; font-weight: 400; }
  .lines { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 2px; }
  .line {
    display: flex; gap: 14px; width: 100%; text-align: left; background: none; border: 0;
    border-left: 2px solid transparent; padding: 5px 6px 5px 10px; margin: 0; cursor: pointer;
    color: var(--muted); font-family: var(--body); font-size: 1.06rem; line-height: 1.45;
    transition: color .12s ease, border-color .12s ease;
  }
  .line .t {
    flex: none; font-family: var(--mono); font-size: .7rem; color: #4d565d; padding-top: .38em;
    font-variant-numeric: tabular-nums;
  }
  .line:hover { color: var(--ink); }
  .line.on { color: var(--ink); border-left-color: var(--lamp); }
  .line.on .t { color: var(--lamp); }
  .line.done { color: #565f66; }
  :focus-visible { outline: 2px solid var(--screen); outline-offset: 2px; }

  /* ---------- footer ---------- */
  footer { border-top: 1px solid var(--edge); padding-top: 26px; }
  footer h3 {
    font-family: var(--mono); font-weight: 500; font-size: .74rem; letter-spacing: .2em;
    text-transform: uppercase; color: var(--screen); margin: 0 0 12px;
  }
  footer p { color: var(--muted); font-size: .98rem; line-height: 1.7; margin: 0 0 14px; max-width: 62ch; }
  footer audio { width: 100%; margin-bottom: 18px; }
  .colophon { font-family: var(--mono); font-size: .72rem; color: #5b646b; line-height: 1.9; }

  @media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
  @media (max-width: 420px) { .line { font-size: 1rem; gap: 10px; } .line .t { font-size: .64rem; } }
</style>

<div class="wrap">
  <header>
    <canvas id="snow" aria-hidden="true"></canvas>
    <p class="eyebrow">Track 02 · dead channel</p>
    <h1>STATIC</h1>
    <p class="spec">
      68 BPM half-time · E minor · __TOTAL__<br>
      Every line is <b>16 syllables</b>, one per sixteenth note — <b>221 ms</b> apart, start to finish.<br>
      Read by a Windows speech synthesizer, one word at a time.
    </p>
  </header>

  <div class="player">
    <button class="play" id="play" aria-label="Play">
      <svg id="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z"></path></svg>
    </button>
    <div class="bars">
      <input type="range" id="seek" min="0" max="1000" value="0" step="1" aria-label="Seek">
      <div class="clock"><span class="on" id="now">0:00</span><span id="dur">__TOTAL__</span></div>
    </div>
  </div>
  <audio id="song" src="static.mp3" preload="metadata"></audio>

__LYRICS__

  <footer>
    <h3>The beat on its own</h3>
    <audio controls src="static_instrumental.mp3"></audio>
    <p>
      The loop is a guitar swell pad, a low pulse and a slow descending line, played three semitones up and slowed
      down like a record, then run through wow and flutter, tape saturation and vinyl crackle. Under it: a snare and
      hand clap, swung hats, a reversed cymbal into every hook, and an 808 with a distorted layer so it survives a
      phone speaker.
    </p>
    <p>
      The voice is levelled to one note by playback speed rather than by a vocoder — a vocoder smears the stops and
      turns <em>static</em> into <em>sad</em>. Each word is synthesized alone at a speed that makes it the length of
      its syllables, then placed so its first vowel lands on the beat.
    </p>
    <p class="colophon">
      Written, produced and read by machine in Claude Code.<br>
      Original composition and lyrics. Click any line to jump to it.
    </p>
  </footer>
</div>

<script>
(function () {
  var song = document.getElementById("song");
  var play = document.getElementById("play");
  var icon = document.getElementById("icon");
  var seek = document.getElementById("seek");
  var now = document.getElementById("now");
  var durEl = document.getElementById("dur");
  var lines = Array.prototype.slice.call(document.querySelectorAll(".line"));
  var times = lines.map(function (el) { return parseFloat(el.dataset.t); });
  var PLAY = "M8 5v14l11-7z";
  var PAUSE = "M6 5h4v14H6zM14 5h4v14h-4z";
  var scrubbing = false, cur = -1;

  function clock(t) {
    if (!isFinite(t)) return "0:00";
    var m = Math.floor(t / 60), s = Math.floor(t % 60);
    return m + ":" + (s < 10 ? "0" : "") + s;
  }
  function mark(t) {
    var i = -1;
    for (var k = 0; k < times.length; k++) { if (times[k] <= t + 0.08) i = k; else break; }
    if (i === cur) return;
    cur = i;
    lines.forEach(function (el, k) {
      el.classList.toggle("on", k === i);
      el.classList.toggle("done", k < i);
    });
    if (i >= 0 && !scrubbing) {
      var r = lines[i].getBoundingClientRect();
      if (r.top < 120 || r.bottom > window.innerHeight - 40) {
        lines[i].scrollIntoView({ block: "center", behavior: "smooth" });
      }
    }
  }
  play.addEventListener("click", function () {
    if (song.paused) { song.play(); } else { song.pause(); }
  });
  song.addEventListener("play", function () { icon.firstElementChild.setAttribute("d", PAUSE); play.setAttribute("aria-label", "Pause"); });
  song.addEventListener("pause", function () { icon.firstElementChild.setAttribute("d", PLAY); play.setAttribute("aria-label", "Play"); });
  song.addEventListener("loadedmetadata", function () { durEl.textContent = clock(song.duration); });
  song.addEventListener("timeupdate", function () {
    if (!scrubbing && song.duration) seek.value = String(Math.round(1000 * song.currentTime / song.duration));
    now.textContent = clock(song.currentTime);
    mark(song.currentTime);
  });
  seek.addEventListener("input", function () {
    scrubbing = true;
    if (song.duration) { var t = song.duration * seek.value / 1000; now.textContent = clock(t); mark(t); }
  });
  seek.addEventListener("change", function () {
    if (song.duration) song.currentTime = song.duration * seek.value / 1000;
    scrubbing = false;
  });
  lines.forEach(function (el) {
    el.addEventListener("click", function () {
      song.currentTime = parseFloat(el.dataset.t);
      mark(song.currentTime);
      if (song.paused) song.play();
    });
  });

  // Channel three with nothing on it. Faint, and it lifts a little while the track runs.
  var cv = document.getElementById("snow"), cx = cv.getContext("2d");
  var still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var W = 0, H = 0, raf = 0;
  function size() {
    W = cv.clientWidth; H = cv.clientHeight;
    cv.width = Math.max(1, Math.floor(W / 2)); cv.height = Math.max(1, Math.floor(H / 2));
  }
  function frame() {
    var img = cx.createImageData(cv.width, cv.height);
    var d = img.data, lift = song.paused ? 0 : 42;
    for (var i = 0; i < d.length; i += 4) {
      var v = (Math.random() * (150 + lift)) | 0;
      d[i] = v; d[i + 1] = v; d[i + 2] = v; d[i + 3] = 255;
    }
    cx.putImageData(img, 0, 0);
    if (!still) raf = requestAnimationFrame(function () { setTimeout(frame, 70); });
  }
  window.addEventListener("resize", size);
  size(); frame();
  document.addEventListener("visibilitychange", function () {
    if (document.hidden) { cancelAnimationFrame(raf); } else if (!still) { frame(); }
  });
})();
</script>
"""

open("out/static_page.html", "w", encoding="utf-8").write(
    PAGE.replace("__LYRICS__", LYRICS).replace("__TOTAL__", TOTAL))
print(f"wrote out/static_page.html — {sum(len(s['lines']) for s in sections)} lines in {len(sections)} sections")
