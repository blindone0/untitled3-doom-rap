"""Record your vocal over the beat, section by section.

  python record.py --list                              list audio devices
  python record.py --calibrate                         measure round-trip latency (use speakers, or hold a headphone cup to the mic)
  python record.py --name verse1 --start 34.3 --dur 62 record one section: the beat plays from `pre` seconds before `start`
  python record.py --name full --full                  record the whole song in one pass

Wear closed headphones so the beat does not bleed into the mic. Takes are saved in vocals/<name>.wav (+ .json with the position).
Then run: python mixvocal.py
"""
import argparse
import json
import os
import time

import numpy as np
import sounddevice as sd
import soundfile as sf

SR = 44100
ap = argparse.ArgumentParser()
ap.add_argument("--list", action="store_true")
ap.add_argument("--calibrate", action="store_true")
ap.add_argument("--name", default=None)
ap.add_argument("--start", type=float, default=0.0, help="song position (s) where your part begins")
ap.add_argument("--dur", type=float, default=60.0, help="how long to record after start (s)")
ap.add_argument("--pre", type=float, default=7.0, help="seconds of beat played before start (count-in)")
ap.add_argument("--full", action="store_true")
ap.add_argument("--beat", default="out/doom_cover_heavy_instrumental.wav")
ap.add_argument("--in", dest="dev_in", type=int, default=None, help="input device index (see --list)")
ap.add_argument("--out", dest="dev_out", type=int, default=None, help="output device index (see --list)")
ap.add_argument("--monitor", type=float, default=1.0, help="beat playback level 0..1")
args = ap.parse_args()

if args.list:
    print(sd.query_devices())
    raise SystemExit

os.makedirs("vocals", exist_ok=True)
device = (args.dev_in, args.dev_out)

if args.calibrate:
    from scipy import signal as sps
    n = 9 * SR
    sig = np.zeros(n)
    burst = np.random.default_rng(0).standard_normal(int(0.03 * SR)) * np.hanning(int(0.03 * SR))
    times = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    for t in times:
        sig[int(t * SR):int(t * SR) + len(burst)] += burst
    sig = sig / np.abs(sig).max() * 0.9
    print("CALIBRATION: nothing to do, stay quiet for 9 seconds. You should hear 8 loud clicks; the mic must hear them too")
    print("(speakers on, or hold one headphone cup right up to the mic).")
    rec = sd.playrec(np.stack([sig, sig], 1).astype(np.float32), samplerate=SR, channels=1, device=device, blocking=True)[:, 0]
    rec = rec - rec.mean()
    band = sps.sosfilt(sps.butter(4, [2000, 9000], btype="band", fs=SR, output="sos"), rec)
    env = np.abs(band)
    floor = np.median(env) + 1e-9
    lat, hits = [], 0
    for t in times:
        s = int(t * SR)
        win = env[s:s + int(1.0 * SR)]
        pk = int(np.argmax(win))
        if win[pk] > 8 * floor:  # a real click stands well above the room noise
            hits += 1
            lat.append(pk / SR * 1000)
    print(f"mic level: peak {20 * np.log10(np.abs(rec).max() + 1e-9):.0f} dBFS, clicks detected: {hits}/8")
    if hits >= 4 and (max(lat) - min(lat)) <= 10:
        lat_ms = float(np.median(lat))
        json.dump({"latency_ms": lat_ms}, open("vocals/latency.json", "w"))
        print(f"OK: latency {lat_ms:.1f} ms (values {[round(x, 1) for x in lat]}) saved to vocals/latency.json")
    else:
        print(f"FAILED: values {[round(x, 1) for x in lat]} are not consistent -> the mic did not hear the clicks clearly.")
        print("Turn the speakers up / bring the mic closer and run again, or skip this: record anyway and fix timing later with")
        print("  python mixvocal.py --offset-ms <number>   (bigger number = your voice moves earlier)")
    raise SystemExit

if not args.name:
    ap.error("--name is required (e.g. --name verse1)")

beat, bsr = sf.read(args.beat)
assert bsr == SR, "beat must be 44.1 kHz"
if beat.ndim == 1:
    beat = np.stack([beat, beat], 1)
if args.full:
    seg_start, seg_end = 0.0, len(beat) / SR
else:
    seg_start = max(0.0, args.start - args.pre)
    seg_end = min(len(beat) / SR, args.start + args.dur)
seg = beat[int(seg_start * SR):int(seg_end * SR)] * args.monitor
print(f"take '{args.name}': beat from {seg_start:.1f}s to {seg_end:.1f}s ({seg_end - seg_start:.0f}s). Your part starts at {args.start:.1f}s.")
for i in (3, 2, 1):
    print(f"  starting in {i} ...")
    time.sleep(1)
print("  RECORDING - go. (Ctrl+C aborts)")
t0 = time.time()
try:
    rec = sd.playrec(seg.astype(np.float32), samplerate=SR, channels=1, device=device, blocking=True)[:, 0]
except KeyboardInterrupt:
    sd.stop()
    print("aborted, nothing saved")
    raise SystemExit
pk = float(np.abs(rec).max())
print(f"done in {time.time() - t0:.0f}s, peak {20 * np.log10(pk + 1e-9):.1f} dBFS" + ("  (TOO HOT - lower the mic gain)" if pk > 0.95 else "") + ("  (very quiet - raise the mic gain)" if pk < 0.05 else ""))
sf.write(f"vocals/{args.name}.wav", rec, SR, subtype="FLOAT")
json.dump({"song_start": seg_start, "part_start": args.start, "beat": args.beat}, open(f"vocals/{args.name}.json", "w"))
print(f"saved vocals/{args.name}.wav  -> run: python mixvocal.py")
