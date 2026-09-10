"""Render note events through a SoundFont with tinysoundfont, with sample-accurate-ish scheduling (64-sample blocks)
and optional per-note pitch-bend automation (vibrato / slides)."""
import numpy as np
import tinysoundfont as tsf

SR = 44100
BLOCK = 64


def render(sf2, preset, events, n_samples, bank=0, bend_range=2.0, gain_db=-6.0):
    """events: list of dicts {s: start_sample, n: length_samples, m: midi, v: 0..1, bend: optional array of
    semitone offsets sampled every BLOCK samples for the note's duration}. Returns mono float64."""
    synth = tsf.Synth(samplerate=SR, gain=gain_db)
    sid = synth.sfload(sf2)
    synth.program_select(0, sid, bank, preset)
    synth.pitchbend_range(0, bend_range)
    synth.start()
    ons = sorted(((e["s"] // BLOCK) * BLOCK, i) for i, e in enumerate(events))
    offs = sorted((((e["s"] + e["n"]) // BLOCK) * BLOCK, i) for i, e in enumerate(events))
    total_blocks = (n_samples + BLOCK - 1) // BLOCK
    out = np.zeros(total_blocks * BLOCK, dtype=np.float32)
    oi = fi = 0
    active = {}  # idx -> (start_block, bend array)
    for b in range(total_blocks):
        pos = b * BLOCK
        while fi < len(offs) and offs[fi][0] <= pos:
            i = offs[fi][1]
            synth.noteoff(0, events[i]["m"])
            active.pop(i, None)
            fi += 1
        while oi < len(ons) and ons[oi][0] <= pos:
            i = ons[oi][1]
            e = events[i]
            if e.get("bend") is not None:
                active[i] = (b, e["bend"])
                synth.pitchbend(0, int(np.clip(8192 + e["bend"][0] / bend_range * 8191, 0, 16383)))
            else:
                synth.pitchbend(0, 8192)
            synth.noteon(0, e["m"], int(np.clip(e["v"] * 127, 1, 127)))
            oi += 1
        if active:
            # one channel: apply the most recent note's bend
            i = max(active, key=lambda k: active[k][0])
            sb, bend = active[i]
            k = min(b - sb, len(bend) - 1)
            synth.pitchbend(0, int(np.clip(8192 + bend[k] / bend_range * 8191, 0, 16383)))
        buf = synth.generate(BLOCK)
        arr = np.frombuffer(bytes(buf), dtype=np.float32)
        out[pos:pos + BLOCK] = arr.reshape(-1, 2).mean(1)[:BLOCK]
    synth.stop()
    return out[:n_samples].astype(np.float64)


if __name__ == "__main__":
    import soundfile as sf

    n = 6 * SR
    ev = [{"s": 0, "n": SR, "m": 33, "v": 0.9}, {"s": SR, "n": SR, "m": 40, "v": 0.9},
          {"s": 2 * SR, "n": 2 * SR, "m": 45, "v": 0.9, "bend": np.sin(np.linspace(0, 2 * np.pi * 5 * 2, 2 * SR // BLOCK)) * 0.3},
          {"s": 4 * SR, "n": int(1.5 * SR), "m": 57, "v": 0.8}]
    for preset, name in [(33, "fingerbass"), (27, "cleangtr"), (28, "mutedgtr")]:
        y = render("assets/GeneralUser-GS.sf2", preset, ev, n)
        print(name, "peak", np.abs(y).max().round(3), "rms", np.sqrt((y ** 2).mean()).round(4))
        sf.write(f"out/test_sf_{name}.wav", y, SR)
