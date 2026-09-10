"""Neural amp/pedal captures (GuitarML Proteus LSTM-40 json models) + cabinet impulse responses."""
import json

import numpy as np
import soundfile as sf
import torch
from scipy import signal

torch.set_num_threads(max(1, torch.get_num_threads()))


class Proteus:
    """Single-input LSTM capture: y = lin(lstm(x)) + x (Automated-GuitarAmpModelling 'SimpleRNN', skip=1)."""

    def __init__(self, path):
        d = json.load(open(path))
        sd = d["state_dict"]
        H = d["model_data"]["hidden_size"]
        assert d["model_data"]["input_size"] == 1, "conditioned (knob) models not supported"
        self.lstm = torch.nn.LSTM(1, H, batch_first=True)
        self.lin = torch.nn.Linear(H, 1)
        with torch.no_grad():
            self.lstm.weight_ih_l0.copy_(torch.tensor(sd["rec.weight_ih_l0"], dtype=torch.float32))
            self.lstm.weight_hh_l0.copy_(torch.tensor(sd["rec.weight_hh_l0"], dtype=torch.float32))
            self.lstm.bias_ih_l0.copy_(torch.tensor(sd["rec.bias_ih_l0"], dtype=torch.float32).flatten())
            self.lstm.bias_hh_l0.copy_(torch.tensor(sd["rec.bias_hh_l0"], dtype=torch.float32).flatten())
            self.lin.weight.copy_(torch.tensor(sd["lin.weight"], dtype=torch.float32))
            self.lin.bias.copy_(torch.tensor(sd["lin.bias"], dtype=torch.float32).flatten())
        self.lstm.eval()
        self.lin.eval()

    def process(self, x, chunk=1 << 18):
        x = np.asarray(x, dtype=np.float32)
        out = np.empty_like(x)
        h = None
        with torch.no_grad():
            for i in range(0, len(x), chunk):
                seg = torch.from_numpy(x[i:i + chunk]).view(1, -1, 1)
                y, h = self.lstm(seg, h)
                y = self.lin(y) + seg
                out[i:i + chunk] = y.view(-1).numpy()
        return out.astype(np.float64)


def load_ir(path, sr=44100, max_ms=200.0):
    ir, isr = sf.read(path)
    if ir.ndim > 1:
        ir = ir.mean(1)
    if isr != sr:
        ir = signal.resample_poly(ir, sr, isr)
    ir = ir[: int(max_ms / 1000 * sr)]
    ir = ir / (np.sqrt((ir ** 2).sum()) + 1e-9)  # unit energy
    return ir


def cab(x, ir):
    return signal.fftconvolve(x, ir)[: len(x)]


if __name__ == "__main__":
    import sys
    import time

    sr = 44100
    t = np.arange(10 * sr) / sr
    x = 0.3 * np.sin(2 * np.pi * 110 * t) * np.exp(-t / 3)
    for name in sys.argv[1:] or ["assets/proteus/6505Plus_Red_DirectOut.json"]:
        m = Proteus(name)
        t0 = time.time()
        y = m.process(x)
        print(f"{name}: 10 s in {time.time() - t0:.2f} s, out peak {np.abs(y).max():.3f} rms {np.sqrt((y ** 2).mean()):.3f}")
