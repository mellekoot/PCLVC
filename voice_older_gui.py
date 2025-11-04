import textwrap, json, os, sys, pathlib

code = r'''
"""
Real-time "Older Voice" Changer with GUI (PyQt5) + Optional Upscale Processing
-----------------------------------------------------------------------------
Features
- Live mic -> processed voice -> speakers/headphones (full-duplex).
- "Older voice" effect blends: slight pitch drop, high-frequency rolloff, tremolo (shakiness), breath noise.
- Adjustable "Age" amount (0–100).
- Device pickers for Input/Output.
- Optional internal 2× upscaling of processing (oversampling) for smoother filters.
- Simple input level meter.

Notes
- This is a prototype: artifacts may occur depending on system/latency.
- Use headphones to avoid feedback loops. Start with low volume.
- Tested with Python 3.9–3.12.

Run:
    pip install -r requirements.txt
    python voice_older_gui.py
"""

import sys
import threading
import queue
import time
import math
import numpy as np

from scipy.signal import butter, lfilter, resample_poly
import sounddevice as sd

from PyQt5 import QtWidgets, QtCore

# -------------------------- DSP Helpers --------------------------

def db_to_lin(db):
    return 10 ** (db / 20.0)

def butter_lowpass(cutoff_hz, fs, order=4):
    nyq = 0.5 * fs
    norm = min(cutoff_hz / nyq, 0.99)
    b, a = butter(order, norm, btype='low', analog=False)
    return b, a

class OnePoleLP:
    def __init__(self, fs, cutoff_hz=5000.0):
        self.fs = fs
        self.set_cutoff(cutoff_hz)
        self.z = 0.0
    def set_cutoff(self, cutoff_hz):
        # simple one-pole low-pass
        rc = 1.0 / (2.0 * math.pi * max(1.0, cutoff_hz))
        self.alpha = (1.0 / self.fs) / (rc + (1.0 / self.fs))
    def process(self, x):
        y = np.empty_like(x)
        z = self.z
        a = self.alpha
        for i, xi in enumerate(x):
            z = z + a * (xi - z)
            y[i] = z
        self.z = z
        return y

class Tremolo:
    def __init__(self, fs, rate_hz=6.0, depth=0.05):
        self.fs = fs
        self.phase = 0.0
        self.set_params(rate_hz, depth)
    def set_params(self, rate_hz, depth):
        self.rate = rate_hz
        self.depth = max(0.0, min(depth, 0.95))
    def process(self, x):
        n = len(x)
        t = (np.arange(n) + self.phase) / self.fs
        mod = 1.0 - self.depth * (0.5 * (1.0 + np.sin(2*np.pi*self.rate*t)))
        self.phase = (self.phase + n) % self.fs
        return x * mod.astype(x.dtype)

class JitterVibrato:
    """Very shallow random pitch wobble using delay modulation (chorus-like)."""
    def __init__(self, fs, max_ms=8.0, rate_hz=3.0):
        self.fs = fs
        self.max_samps = int(max_ms * 1e-3 * fs)
        self.rate = rate_hz
        self.phase = 0.0
        self.buf = np.zeros(self.max_samps + 4096, dtype=np.float32)
        self.widx = 0
    def process(self, x):
        n = len(x)
        out = np.zeros_like(x)
        # write into circular buffer
        for i in range(n):
            self.buf[self.widx] = x[i]
            self.widx = (self.widx + 1) % len(self.buf)
            t = (self.phase + i) / self.fs
            # small random walk in LFO rate for shakiness
            lfo = (np.sin(2*np.pi*self.rate*t) + 0.3*np.sin(2*np.pi*0.5*self.rate*t + 1.1))
            delay = 0.5*self.max_samps*(1 + lfo)  # 0..max
            ridx = int((self.widx - 1 - delay) % len(self.buf))
            ridx2 = (ridx - 1) % len(self.buf)
            frac = delay - int(delay)
            out[i] = (1-frac)*self.buf[ridx] + frac*self.buf[ridx2]
        self.phase = (self.phase + n) % self.fs
        return out

def simple_pitch_shift_block(x, semitones, fs):
    """Approximate pitch shift by resampling each block.
    Negative semitones lowers pitch -> "older". This is crude but fast.
    """
    if abs(semitones) < 1e-3:
        return x
    factor = 2 ** (semitones / 12.0)  # speed factor
    # change speed, then bring back to original length using resample_poly
    # Upsample to change speed
    # First: speed change via interpolation
    idx = np.arange(0, len(x), factor)
    if len(idx) < 2:
        return x
    sped = np.interp(idx, np.arange(len(x)), x).astype(x.dtype)
    # then resample back to original length
    y = resample_poly(sped, up=len(x), down=len(sped)).astype(x.dtype)
    if len(y) > len(x):
        y = y[:len(x)]
    elif len(y) < len(x):
        y = np.pad(y, (0, len(x)-len(y)))
    return y

def add_breath_noise(x, level=0.01):
    noise = np.random.randn(*x.shape).astype(x.dtype) * level
    return x + noise

# -------------------------- Audio Engine --------------------------

class VoiceAgerState:
    def __init__(self, fs=48000, channels=1):
        self.fs = fs
        self.channels = channels
        self.age_amount = 50  # 0..100
        self.enable_upscale = False
        self.target_internal_fs = fs  # updated when upscale toggled
        self.lock = threading.Lock()
        # per-channel processors
        self.trem = [Tremolo(fs, rate_hz=6.0, depth=0.05) for _ in range(channels)]
        self.vib = [JitterVibrato(fs, max_ms=6.0, rate_hz=3.0) for _ in range(channels)]
        self.lp = [OnePoleLP(fs, cutoff_hz=6000.0) for _ in range(channels)]

    def set_fs(self, fs):
        with self.lock:
            self.fs = fs
            self.target_internal_fs = 2*fs if self.enable_upscale else fs
            self.trem = [Tremolo(self.target_internal_fs, rate_hz=6.0, depth=0.05) for _ in range(self.channels)]
            self.vib = [JitterVibrato(self.target_internal_fs, max_ms=6.0, rate_hz=3.0) for _ in range(self.channels)]
            self.lp = [OnePoleLP(self.target_internal_fs, cutoff_hz=6000.0) for _ in range(self.channels)]

    def set_age(self, amount):
        with self.lock:
            self.age_amount = max(0, min(100, int(amount)))

    def set_upscale(self, enabled):
        with self.lock:
            self.enable_upscale = bool(enabled)
            self.target_internal_fs = 2*self.fs if self.enable_upscale else self.fs
            # rebuild processors at new fs
            self.trem = [Tremolo(self.target_internal_fs, rate_hz=6.0, depth=0.05) for _ in range(self.channels)]
            self.vib = [JitterVibrato(self.target_internal_fs, max_ms=6.0, rate_hz=3.0) for _ in range(self.channels)]
            self.lp = [OnePoleLP(self.target_internal_fs, cutoff_hz=6000.0) for _ in range(self.channels)]

    def process(self, inbuf):
        # inbuf shape: (frames, channels), float32
        with self.lock:
            age = self.age_amount / 100.0
            do_up = self.enable_upscale
            fs_in = self.fs
            fs_internal = self.target_internal_fs

        # Upscale (oversample) if requested
        x = inbuf
        if do_up and fs_internal != fs_in:
            x = resample_poly(x, up=2, down=1, axis=0).astype(np.float32)

        # Per-channel processing
        y = np.zeros_like(x, dtype=np.float32)
        for ch in range(x.shape[1]):
            sig = x[:, ch]

            # 1) Slight pitch drop based on age: 0..-4 semitones
            semi = -4.0 * age
            sig = simple_pitch_shift_block(sig, semi, fs_internal)

            # 2) Tremolo depth scales with age (adds shakiness)
            self.trem[ch].set_params(rate_hz=5.5 + 2.0*age, depth=0.02 + 0.10*age)
            sig = self.trem[ch].process(sig)

            # 3) Tiny vibrato/chorus jitter for frail tone (very subtle)
            sig = 0.8*sig + 0.2*self.vib[ch].process(sig)

            # 4) Low-pass: cut more highs with age (8k -> 3.5k)
            cutoff = 8000.0 - 4500.0*age
            self.lp[ch].set_cutoff(max(1500.0, cutoff))
            sig = self.lp[ch].process(sig)

            # 5) Sandiness: a touch of breath noise with age
            sig = add_breath_noise(sig, level=0.002 + 0.01*age)

            # 6) Gentle gain trim to avoid overload
            sig = np.tanh(1.2*sig).astype(np.float32)

            y[:, ch] = sig

        # Downscale if oversampled
        if do_up and fs_internal != fs_in:
            y = resample_poly(y, up=1, down=2, axis=0).astype(np.float32)

        return y

# -------------------------- GUI App --------------------------

class VoiceAgerGUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-time Older Voice Changer")
        self.state = VoiceAgerState()
        self.stream = None
        self.running = False
        self.blocksize = 1024
        self.latency = 'low'

        self.init_ui()
        self.refresh_devices()

        # level meter update timer
        self.level_timer = QtCore.QTimer(self)
        self.level_timer.timeout.connect(self.update_level_meter)
        self.level_timer.start(50)
        self.last_level = 0.0

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Device selectors
        dev_layout = QtWidgets.QGridLayout()
        self.in_dev_combo = QtWidgets.QComboBox()
        self.out_dev_combo = QtWidgets.QComboBox()
        dev_layout.addWidget(QtWidgets.QLabel("Input (Mic):"), 0, 0)
        dev_layout.addWidget(self.in_dev_combo, 0, 1)
        dev_layout.addWidget(QtWidgets.QLabel("Output (Speakers/HP):"), 1, 0)
        dev_layout.addWidget(self.out_dev_combo, 1, 1)

        # Sample rate
        self.fs_combo = QtWidgets.QComboBox()
        for fs in [16000, 24000, 32000, 44100, 48000]:
            self.fs_combo.addItem(f"{fs} Hz", fs)
        self.fs_combo.setCurrentText("48000 Hz")
        dev_layout.addWidget(QtWidgets.QLabel("Sample Rate:"), 2, 0)
        dev_layout.addWidget(self.fs_combo, 2, 1)

        layout.addLayout(dev_layout)

        # Age slider
        self.age_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.age_slider.setRange(0, 100)
        self.age_slider.setValue(60)
        self.age_label = QtWidgets.QLabel("Age amount: 60")

        self.age_slider.valueChanged.connect(lambda v: self.on_age_changed(v))
        layout.addWidget(self.age_label)
        layout.addWidget(self.age_slider)

        # Upscale checkbox
        self.upscale_check = QtWidgets.QCheckBox("Upscale processing 2× (smoother filters)")
        self.upscale_check.setChecked(False)
        self.upscale_check.stateChanged.connect(self.on_upscale_changed)
        layout.addWidget(self.upscale_check)

        # Level meter
        self.level_bar = QtWidgets.QProgressBar()
        self.level_bar.setRange(0, 100)
        layout.addWidget(QtWidgets.QLabel("Input level"))
        layout.addWidget(self.level_bar)

        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        layout.addLayout(btn_layout)

        self.start_btn.clicked.connect(self.start_stream)
        self.stop_btn.clicked.connect(self.stop_stream)

        # Footer
        foot = QtWidgets.QLabel("Tip: Use headphones to avoid feedback. Lower your speaker volume if you hear echo.")
        foot.setWordWrap(True)
        layout.addWidget(foot)

    def on_age_changed(self, v):
        self.age_label.setText(f"Age amount: {v}")
        self.state.set_age(v)

    def on_upscale_changed(self, s):
        self.state.set_upscale(self.upscale_check.isChecked())

    def refresh_devices(self):
        self.in_dev_combo.clear()
        self.out_dev_combo.clear()
        devices = sd.query_devices()
        default_in = sd.default.device[0]
        default_out = sd.default.device[1]

        for i, d in enumerate(devices):
            name = f"{i}: {d['name']}"
            if d['max_input_channels'] > 0:
                self.in_dev_combo.addItem(name, i)
                if i == default_in:
                    self.in_dev_combo.setCurrentIndex(self.in_dev_combo.count()-1)
            if d['max_output_channels'] > 0:
                self.out_dev_combo.addItem(name, i)
                if i == default_out:
                    self.out_dev_combo.setCurrentIndex(self.out_dev_combo.count()-1)

    def audio_callback(self, indata, outdata, frames, time_info, status):
        if status:
            # print(status, file=sys.stderr)
            pass
        # mono/stereo handling: take min channels
        in_ch = indata.shape[1]
        out_ch = outdata.shape[1]
        ch = min(in_ch, out_ch)
        x = indata[:, :ch].copy()

        # track input level (RMS)
        lvl = float(np.sqrt(np.mean(x**2) + 1e-9))
        self.last_level = max(0.0, min(1.0, lvl * 10.0))

        # update state sample rates if changed
        self.state.set_fs(int(self.fs_combo.currentData()))

        # process
        y = self.state.process(x)

        # if output has more channels, duplicate
        if out_ch > ch:
            y = np.tile(y, (1, out_ch)) if y.ndim == 1 else np.tile(y, (1, int(out_ch/ch)))
        elif out_ch < ch:
            y = y[:, :out_ch]
        outdata[:] = y

    def start_stream(self):
        if self.running:
            return
        try:
            in_dev = int(self.in_dev_combo.currentData())
            out_dev = int(self.out_dev_combo.currentData())
            fs = int(self.fs_combo.currentData())
            self.state.set_fs(fs)

            self.stream = sd.Stream(
                device=(in_dev, out_dev),
                samplerate=fs,
                blocksize=self.blocksize,
                dtype='float32',
                channels=1,  # process mono for stability; duplicated if needed
                latency=self.latency,
                callback=self.audio_callback
            )
            self.stream.start()
            self.running = True
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error starting audio", str(e))

    def stop_stream(self):
        if not self.running:
            return
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass
        self.stream = None
        self.running = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def update_level_meter(self):
        self.level_bar.setValue(int(self.last_level * 100))

    def closeEvent(self, event):
        self.stop_stream()
        event.accept()

def main():
    app = QtWidgets.QApplication(sys.argv)
    gui = VoiceAgerGUI()
    gui.resize(520, 320)
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
'''

reqs = """\
sounddevice>=0.4.6
numpy>=1.23
scipy>=1.10
PyQt5>=5.15
"""

base = pathlib.Path("/mnt/data")
(base / "voice_older_gui.py").write_text(code, encoding="utf-8")
(base / "requirements.txt").write_text(reqs, encoding="utf-8")

print("Created files:")
print("- /mnt/data/voice_older_gui.py")
print("- /mnt/data/requirements.txt")
