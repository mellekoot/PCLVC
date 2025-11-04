# main.py
"""
Older Voice Studio — Permeable Cage (WORKING)
---------------------------------------------
- Intro splash (fade + gentle scale), then the main GUI.
- Real-time low-artifact “~40-year-old male” tone.
- Louder by default (input +4 dB, output +8 dB) with sliders.
- Start → Record (fullscreen) → Preview → Save/Discard.
- Optional Hugging Face “Make Older (HF)” in Preview (store your key locally).

Run:
    pip install sounddevice numpy scipy PyQt5 soundfile requests
    python main.py
"""

import os, sys, math, json, datetime, threading, io
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
import sounddevice as sd
import soundfile as sf
import requests

from PyQt5 import QtWidgets, QtCore, QtGui

APP_TITLE = "Older Voice Studio — Permeable Cage"
SPLASH_TEXT = "Permeable Cage"
CFG_DIR = Path.home() / ".older_voice_studio"
CFG_DIR.mkdir(parents=True, exist_ok=True)
CFG_PATH = CFG_DIR / "config.json"

DEFAULT_CONFIG = {
    "hf_api_key": "",
    "hf_model_id": "speechbrain/mtl-mimic-voicebank",  # generic enhancement; optional
    "last_save_dir": str(Path.cwd()),
    "output_gain_db": 8.0,
    "input_gain_db": 4.0,
    "last_fs": 48000
}

# ---------------- Config ----------------
def load_config():
    if CFG_PATH.exists():
        try:
            with open(CFG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            for k, v in DEFAULT_CONFIG.items():
                cfg.setdefault(k, v)
            return cfg
        except Exception:
            pass
    return DEFAULT_CONFIG.copy()

def save_config(cfg):
    try:
        with open(CFG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        pass

CONFIG = load_config()

# ---------------- DSP: clean older tone ----------------
def db_to_lin(db: float) -> float:
    return float(10.0 ** (db/20.0))

def simple_pitch_shift_block(x, semitones):
    if abs(semitones) < 1e-3:
        return x
    factor = 2 ** (semitones / 12.0)
    idx = np.arange(0, len(x), factor)
    if len(idx) < 2:
        return x
    sped = np.interp(idx, np.arange(len(x)), x).astype(x.dtype)
    y = resample_poly(sped, up=len(x), down=len(sped)).astype(x.dtype)
    if len(y) > len(x): y = y[:len(x)]
    elif len(y) < len(x): y = np.pad(y, (0, len(x)-len(y)))
    return y

def biquad_shelf_low(fc, fs, gain_db, S=1.0):
    A = 10**(gain_db/40)
    w0 = 2*np.pi*fc/fs
    alpha = np.sin(w0)/2*np.sqrt((A + 1/A)*(1/S - 1) + 2)
    cosw = np.cos(w0)
    b0 =    A*((A+1) - (A-1)*cosw + 2*np.sqrt(A)*alpha)
    b1 =  2*A*((A-1) - (A+1)*cosw)
    b2 =    A*((A+1) - (A-1)*cosw - 2*np.sqrt(A)*alpha)
    a0 =       (A+1) + (A-1)*cosw + 2*np.sqrt(A)*alpha
    a1 =   -2*((A-1) + (A+1)*cosw)
    a2 =       (A+1) + (A-1)*cosw - 2*np.sqrt(A)*alpha
    return np.array([b0/a0, b1/a0, b2/a0]), np.array([1.0, a1/a0, a2/a0])

def biquad_shelf_high(fc, fs, gain_db, S=1.0):
    A = 10**(gain_db/40)
    w0 = 2*np.pi*fc/fs
    alpha = np.sin(w0)/2*np.sqrt((A + 1/A)*(1/S - 1) + 2)
    cosw = np.cos(w0)
    b0 =    A*((A+1) + (A-1)*cosw + 2*np.sqrt(A)*alpha)
    b1 = -2*A*((A-1) + (A+1)*cosw)
    b2 =    A*((A+1) + (A-1)*cosw - 2*np.sqrt(A)*alpha)
    a0 =       (A+1) - (A-1)*cosw + 2*np.sqrt(A)*alpha
    a1 =    2*((A-1) - (A+1)*cosw)
    a2 =       (A+1) - (A-1)*cosw - 2*np.sqrt(A)*alpha
    return np.array([b0/a0, b1/a0, b2/a0]), np.array([1.0, a1/a0, a2/a0])

class Biquad:
    def __init__(self, b, a):
        self.b = b; self.a = a
        self.z1 = 0.0; self.z2 = 0.0
    def process(self, x):
        y = np.empty_like(x)
        b0,b1,b2 = self.b; a0,a1,a2 = self.a
        z1 = self.z1; z2 = self.z2
        for i, xi in enumerate(x):
            yi = b0*xi + z1
            z1n = b1*xi - a1*yi + z2
            z2 = b2*xi - a2*yi
            z1 = z1n
            y[i] = yi
        self.z1 = z1; self.z2 = z2
        return y

class SmoothLimiter:
    def __init__(self, thresh=0.98, release=0.0025):
        self.thresh = thresh; self.env = 0.0; self.rel = release
    def process(self, x):
        y = x.copy()
        for i in range(len(y)):
            v = abs(float(y[i])); self.env = max(v, self.env - self.rel)
            if self.env > self.thresh:
                g = self.thresh / (self.env + 1e-9)
                y[i] *= g
        return y

class Voice40Chain:
    """Natural ~40 y/o male tone: slight depth, warmth, de-brighten, limiter, adjustable gains."""
    def __init__(self, fs):
        self.fs = fs
        self.pitch_semitones = -2.0
        bL, aL = biquad_shelf_low(180.0, fs, +2.5, S=0.9)
        bH, aH = biquad_shelf_high(6500.0, fs, -2.0, S=0.9)
        self.low = Biquad(bL, aL)
        self.high = Biquad(bH, aH)
        self.lim = SmoothLimiter(thresh=0.98, release=0.003)
        self.input_gain = db_to_lin(CONFIG.get("input_gain_db", 4.0))
        self.output_gain = db_to_lin(CONFIG.get("output_gain_db", 8.0))
    def set_fs(self, fs):
        self.__init__(fs)
    def set_input_gain_db(self, db):
        self.input_gain = db_to_lin(db); CONFIG["input_gain_db"] = float(db); save_config(CONFIG)
    def set_output_gain_db(self, db):
        self.output_gain = db_to_lin(db); CONFIG["output_gain_db"] = float(db); save_config(CONFIG)
    def process(self, x_mono):
        x = (x_mono * self.input_gain).astype(np.float32)
        x = simple_pitch_shift_block(x, self.pitch_semitones)
        x = self.low.process(x)
        x = self.high.process(x)
        x = self.lim.process(x)
        x = (x * self.output_gain).astype(np.float32)
        mx = np.max(np.abs(x)) + 1e-9
        if mx > 1.0: x = x / mx
        return x

# ---------------- Hugging Face (optional) ----------------
def hf_make_older(api_key: str, model_id: str, wav_bytes: bytes, timeout=120) -> bytes:
    if not api_key:
        raise RuntimeError("No Hugging Face API key set (Settings).")
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    files = {"file": ("input.wav", wav_bytes, "audio/wav")}
    r = requests.post(url, headers=headers, files=files, timeout=timeout)
    if r.status_code == 415:
        files = {"audio": ("input.wav", wav_bytes, "audio/wav")}
        r = requests.post(url, headers=headers, files=files, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"HF error {r.status_code}: {r.text[:300]}")
    return r.content

# ---------------- UI theme ----------------
DARK_QSS = """
* { font-family: 'Segoe UI', 'Inter', 'Helvetica Neue', Arial; font-size: 12.5pt; }
QWidget { background-color: #0f1115; color: #E6E8EE; }
QLabel { color: #C9CDD7; }
QPushButton {
    background-color: #1b1f2a; border: 1px solid #2b3245; padding: 10px 16px; border-radius: 10px;
}
QPushButton:hover { background-color: #242a3a; }
QPushButton:pressed { background-color: #1a2030; }
QComboBox, QLineEdit {
    background-color: #131722; border: 1px solid #2b3245; border-radius: 8px; padding: 6px 10px; color: #E6E8EE;
}
QProgressBar {
    border: 1px solid #2b3245; border-radius: 8px; text-align: center; background: #131722;
}
QProgressBar::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #4f7cff, stop:1 #7c4dff); border-radius: 8px; }
QGroupBox {
    border: 1px solid #2b3245; border-radius: 12px; margin-top: 16px; padding: 12px; background: #10141f;
}
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; color: #9aa4c7; }
"""

CARD_CSS = """
background: rgba(255,255,255,0.03);
border: 1px solid rgba(255,255,255,0.06);
border-radius: 16px;
padding: 16px;
"""

class GlassCard(QtWidgets.QFrame):
    def __init__(self, title=None, parent=None):
        super().__init__(parent)
        self.setStyleSheet(CARD_CSS)
        v = QtWidgets.QVBoxLayout(self); v.setContentsMargins(16,16,16,16)
        if title:
            lbl = QtWidgets.QLabel(title); f = lbl.font(); f.setPointSize(13); f.setBold(True); lbl.setFont(f)
            lbl.setStyleSheet("color:#a7b2d9;"); v.addWidget(lbl)
        self.body = QtWidgets.QVBoxLayout(); v.addLayout(self.body)

class ApiKeyDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings — API Key")
        self.setModal(True)
        self.setStyleSheet(DARK_QSS)
        lay = QtWidgets.QVBoxLayout(self); lay.setContentsMargins(18,18,18,18)
        lay.addWidget(QtWidgets.QLabel("Hugging Face API Key (stored locally):"))
        self.key_edit = QtWidgets.QLineEdit(CONFIG.get("hf_api_key",""))
        self.key_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        lay.addWidget(self.key_edit)
        lay.addWidget(QtWidgets.QLabel("Model (auto-chosen):"))
        self.model_edit = QtWidgets.QLineEdit(CONFIG.get("hf_model_id","speechbrain/mtl-mimic-voicebank"))
        lay.addWidget(self.model_edit)
        row = QtWidgets.QHBoxLayout()
        save_btn = QtWidgets.QPushButton("Save"); close_btn = QtWidgets.QPushButton("Close")
        row.addWidget(save_btn); row.addStretch(1); row.addWidget(close_btn)
        lay.addLayout(row)
        save_btn.clicked.connect(self.save); close_btn.clicked.connect(self.close)
    def save(self):
        CONFIG["hf_api_key"] = self.key_edit.text().strip()
        CONFIG["hf_model_id"] = self.model_edit.text().strip() or DEFAULT_CONFIG["hf_model_id"]
        save_config(CONFIG)
        QtWidgets.QMessageBox.information(self, "Saved", "Settings saved locally.")

# ---------------- Splash ----------------
class Splash(QtWidgets.QWidget):
    """Pretty splash: fade + gentle scale, then close and open main app."""
    def __init__(self):
        super().__init__()
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.resize(760, 280)

        root = QtWidgets.QVBoxLayout(self); root.setContentsMargins(0,0,0,0)
        self.container = QtWidgets.QFrame()
        self.container.setStyleSheet("background-color: rgba(15,17,21,240); border-radius: 22px;")
        inner = QtWidgets.QVBoxLayout(self.container); inner.setContentsMargins(30,30,30,30)

        title = QtWidgets.QLabel(SPLASH_TEXT)
        tf = title.font(); tf.setPointSize(40); tf.setBold(True); title.setFont(tf)
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("color:#a7b2d9;")
        sub = QtWidgets.QLabel("voice studio")
        sfnt = sub.font(); sfnt.setPointSize(14); sub.setFont(sfnt)
        sub.setAlignment(QtCore.Qt.AlignCenter)
        sub.setStyleSheet("color:#8e96b6;")

        inner.addStretch(1)
        inner.addWidget(title)
        inner.addWidget(sub)
        inner.addStretch(1)
        root.addWidget(self.container)

        screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
        self.move(screen.center() - self.rect().center())

        # Fade
        self.fx = QtWidgets.QGraphicsOpacityEffect(self.container)
        self.container.setGraphicsEffect(self.fx)
        self.fx.setOpacity(0.0)
        self.fade = QtCore.QPropertyAnimation(self.fx, b"opacity", self)
        self.fade.setDuration(900)
        self.fade.setStartValue(0.0)
        self.fade.setEndValue(1.0)
        self.fade.setEasingCurve(QtCore.QEasingCurve.InOutQuad)

        # Gentle scale (simulate by geometry)
        self.scale_anim = QtCore.QPropertyAnimation(self.container, b"geometry", self)
        start_rect = QtCore.QRect(self.container.x()+20, self.container.y()+20,
                                  self.container.width()-40, self.container.height()-40)
        end_rect = QtCore.QRect(self.container.x(), self.container.y(),
                                self.container.width(), self.container.height())
        self.scale_anim.setDuration(900)
        self.scale_anim.setStartValue(start_rect)
        self.scale_anim.setEndValue(end_rect)
        self.scale_anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)

        self.fade.finished.connect(self._hold_then_close)
        QtCore.QTimer.singleShot(100, self.fade.start)
        QtCore.QTimer.singleShot(100, self.scale_anim.start)

    def _hold_then_close(self):
        QtCore.QTimer.singleShot(700, self.close)

# ---------------- Preview dialog ----------------
class PreviewDialog(QtWidgets.QDialog):
    def __init__(self, mono_audio: np.ndarray, fs: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preview Recording")
        self.setModal(True); self.setStyleSheet(DARK_QSS)
        self.mono = mono_audio.astype(np.float32); self.fs = fs
        self.playing = False

        lay = QtWidgets.QVBoxLayout(self); lay.setContentsMargins(18,18,18,18)

        lay.addWidget(QtWidgets.QLabel("Preview your audio:"))
        r = QtWidgets.QHBoxLayout()
        self.play_btn = QtWidgets.QPushButton("▶ Play")
        self.stop_btn = QtWidgets.QPushButton("■ Stop"); self.stop_btn.setEnabled(False)
        r.addWidget(self.play_btn); r.addWidget(self.stop_btn); r.addStretch(1)
        lay.addLayout(r)

        hfrow = QtWidgets.QHBoxLayout()
        self.hf_btn = QtWidgets.QPushButton("Make Older (HF)")
        tip = QtWidgets.QLabel("Uses your HF key & model (Settings). Replaces preview on success.")
        tip.setStyleSheet("color:#8e96b6;")
        hfrow.addWidget(self.hf_btn); hfrow.addWidget(tip); hfrow.addStretch(1)
        lay.addLayout(hfrow)

        lay.addSpacing(8)
        self.name_edit = QtWidgets.QLineEdit("voice_sample")
        self.dir_label = QtWidgets.QLabel(CONFIG.get("last_save_dir", str(Path.cwd())))
        self.dir_btn = QtWidgets.QPushButton("Choose Save Folder")
        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("Name"), 0, 0); grid.addWidget(self.name_edit, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Folder"), 1, 0); grid.addWidget(self.dir_btn, 1, 1)
        grid.addWidget(self.dir_label, 2, 0, 1, 2)
        lay.addLayout(grid)

        f = QtWidgets.QHBoxLayout()
        self.disc_btn = QtWidgets.QPushButton("✖ Discard")
        self.save_btn = QtWidgets.QPushButton("💾 Save")
        f.addStretch(1); f.addWidget(self.disc_btn); f.addWidget(self.save_btn)
        lay.addLayout(f)

        self.play_btn.clicked.connect(self._play)
        self.stop_btn.clicked.connect(self._stop)
        self.save_btn.clicked.connect(self._save)
        self.disc_btn.clicked.connect(self.reject)
        self.dir_btn.clicked.connect(self._choose_dir)
        self.hf_btn.clicked.connect(self._hf_make_older)

    def _choose_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose save directory", self.dir_label.text())
        if d:
            self.dir_label.setText(d)
            CONFIG["last_save_dir"] = d; save_config(CONFIG)

    def _safe_name(self, s):
        for c in '<>:"/\\|?*': s = s.replace(c, '_')
        return s.strip() or "voice"

    def _save(self):
        out_dir = Path(self.dir_label.text().strip() or os.getcwd())
        out_dir.mkdir(parents=True, exist_ok=True)
        name = self._safe_name(self.name_edit.text())
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"{name}_{ts}.wav"
        mx = float(np.max(np.abs(self.mono)) + 1e-9)
        wav = self.mono / mx if mx > 1.0 else self.mono
        sf.write(str(out_path), wav, self.fs)
        QtWidgets.QMessageBox.information(self, "Saved", f"Saved to:\n{out_path}")
        self.accept()

    def _play(self):
        if self.playing: return
        self.playing = True
        self.play_btn.setEnabled(False); self.stop_btn.setEnabled(True)
        def run():
            try:
                sd.play(self.mono, self.fs, blocking=True)
            finally:
                self.playing = False
                self.play_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        threading.Thread(target=run, daemon=True).start()

    def _stop(self):
        sd.stop()

    def _hf_make_older(self):
        api_key = CONFIG.get("hf_api_key","").strip()
        model_id = CONFIG.get("hf_model_id","").strip()
        if not api_key or not model_id:
            QtWidgets.QMessageBox.warning(self, "Settings needed", "Set your Hugging Face API key & model in Settings.")
            return
        bio = io.BytesIO()
        sf.write(bio, self.mono, self.fs, format="WAV")
        wav_bytes = bio.getvalue()
        try:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            out_bytes = hf_make_older(api_key, model_id, wav_bytes)
        except Exception as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(self, "HF Error", str(e)); return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        try:
            data, fs2 = sf.read(io.BytesIO(out_bytes), dtype="float32")
            if data.ndim > 1: data = np.mean(data, axis=1)
            chain = Voice40Chain(fs2); chain.pitch_semitones = -2.5
            data = chain.process(data)
            if fs2 != self.fs:
                data = resample_poly(data, up=self.fs, down=fs2).astype(np.float32)
            self.mono = data
            QtWidgets.QMessageBox.information(self, "HF Done", "Preview replaced with HF-aged audio.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Parse Error", f"Could not read HF output: {e}")

# ---------------- Fullscreen recording overlay ----------------
class FullscreenRecord(QtWidgets.QWidget):
    stopped = QtCore.pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.Window)
        self.setStyleSheet(DARK_QSS)
        lay = QtWidgets.QVBoxLayout(self); lay.setContentsMargins(40,40,40,40)
        title = QtWidgets.QLabel("Recording…")
        f = title.font(); f.setPointSize(28); f.setBold(True); title.setFont(f)
        title.setAlignment(QtCore.Qt.AlignCenter)
        subtitle = QtWidgets.QLabel("Speak normally — effect is applied in real time.")
        subtitle.setAlignment(QtCore.Qt.AlignCenter); subtitle.setStyleSheet("color:#8e96b6;")
        lay.addStretch(1); lay.addWidget(title); lay.addWidget(subtitle); lay.addSpacing(20)
        self.stop_btn = QtWidgets.QPushButton("● STOP")
        stopf = self.stop_btn.font(); stopf.setPointSize(22); stopf.setBold(True); self.stop_btn.setFont(stopf)
        self.stop_btn.setStyleSheet("""
            QPushButton { background-color: #b3261e; color:#fff; border: 1px solid #ed6a5e; border-radius: 14px; padding: 14px 26px; }
            QPushButton:hover { background-color: #d0342c; }
            QPushButton:pressed { background-color: #8a1c17; }
        """)
        self.stop_btn.setFixedWidth(260)
        self.stop_btn.clicked.connect(self._stop)
        self.stop_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        lay.addWidget(self.stop_btn, alignment=QtCore.Qt.AlignCenter)
        lay.addStretch(2)
    def _stop(self):
        self.stopped.emit(); self.close()

# ---------------- Main App ----------------
class MainApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(900, 560)
        self.setStyleSheet(DARK_QSS)

        self.fs = int(CONFIG.get("last_fs", 48000))
        self.chain = Voice40Chain(self.fs)
        self.stream = None; self.running = False
        self.last_level = 0.0
        self.recording = False; self.record_buf = []

        self._build()
        self._wire()
        self._devices()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(60)

    def _build(self):
        root = QtWidgets.QVBoxLayout(self); root.setContentsMargins(18,18,18,10); root.setSpacing(14)
        hdr = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Older Voice Studio")
        f = title.font(); f.setPointSize(20); f.setBold(True); title.setFont(f)
        hint = QtWidgets.QLabel("Use headphones to avoid feedback.")
        hint.setStyleSheet("color:#8e96b6;")
        self.settings_btn = QtWidgets.QPushButton("Settings")
        hdr.addWidget(title); hdr.addStretch(1); hdr.addWidget(hint); hdr.addWidget(self.settings_btn)
        root.addLayout(hdr)

        row = QtWidgets.QHBoxLayout(); row.setSpacing(14)

        card_io = GlassCard("Audio I/O")
        g = QtWidgets.QGridLayout()
        self.in_combo = QtWidgets.QComboBox(); self.out_combo = QtWidgets.QComboBox()
        self.fs_combo = QtWidgets.QComboBox()
        for fs in [16000, 24000, 32000, 44100, 48000]:
            self.fs_combo.addItem(f"{fs} Hz", fs)
        # Select the saved fs if present
        idx_saved = [16000,24000,32000,44100,48000].index(self.fs) if self.fs in [16000,24000,32000,44100,48000] else 4
        self.fs_combo.setCurrentIndex(idx_saved)
        g.addWidget(QtWidgets.QLabel("Input (Mic)"), 0, 0); g.addWidget(self.in_combo, 0, 1)
        g.addWidget(QtWidgets.QLabel("Output"), 1, 0); g.addWidget(self.out_combo, 1, 1)
        g.addWidget(QtWidgets.QLabel("Sample rate"), 2, 0); g.addWidget(self.fs_combo, 2, 1)
        card_io.body.addLayout(g)

        card_ctrl = GlassCard("Controls")
        self.level = QtWidgets.QProgressBar(); self.level.setRange(0,100)
        gr = QtWidgets.QGridLayout()
        self.in_gain = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.in_gain.setRange(-12, +24)
        self.out_gain = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.out_gain.setRange(-6, +24)
        self.in_gain.setValue(int(CONFIG.get("input_gain_db", 4.0)))
        self.out_gain.setValue(int(CONFIG.get("output_gain_db", 8.0)))
        gr.addWidget(QtWidgets.QLabel("Mic Gain (dB)"),0,0); gr.addWidget(self.in_gain,0,1)
        gr.addWidget(QtWidgets.QLabel("Output Gain (dB)"),1,0); gr.addWidget(self.out_gain,1,1)

        btnrow = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("● Start")
        self.stop_btn = QtWidgets.QPushButton("■ Stop"); self.stop_btn.setEnabled(False)
        self.rec_btn = QtWidgets.QPushButton("● Record")
        btnrow.addWidget(self.start_btn); btnrow.addWidget(self.stop_btn); btnrow.addStretch(1); btnrow.addWidget(self.rec_btn)

        card_ctrl.body.addWidget(QtWidgets.QLabel("Input level")); card_ctrl.body.addWidget(self.level)
        card_ctrl.body.addLayout(gr)
        card_ctrl.body.addLayout(btnrow)

        row.addWidget(card_io, 1); row.addWidget(card_ctrl, 1)
        root.addLayout(row)

        foot = QtWidgets.QLabel("Permeable Cage")
        foot.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        foot.setStyleSheet("color: rgba(167,178,217,150); padding: 6px;")
        root.addWidget(foot)

    def _wire(self):
        self.settings_btn.clicked.connect(self._open_settings)
        self.start_btn.clicked.connect(self.start_stream)
        self.stop_btn.clicked.connect(self.stop_stream)
        self.rec_btn.clicked.connect(self.record_fullscreen)
        self.fs_combo.currentIndexChanged.connect(self._change_fs)
        self.in_gain.valueChanged.connect(lambda v: self.chain.set_input_gain_db(v))
        self.out_gain.valueChanged.connect(lambda v: self.chain.set_output_gain_db(v))

    def _devices(self):
        self.in_combo.clear(); self.out_combo.clear()
        try:
            devs = sd.query_devices()
            has_in = has_out = False
            for i, d in enumerate(devs):
                label = f"{i}: {d['name']}"
                if d['max_input_channels'] > 0:
                    self.in_combo.addItem(label, i); has_in = True
                if d['max_output_channels'] > 0:
                    self.out_combo.addItem(label, i); has_out = True
            if not has_in or not has_out:
                QtWidgets.QMessageBox.warning(self, "No audio device",
                    "No suitable input/output device found. Plug in a mic or headset.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Audio error", f"Device query failed:\n{e}")

    def _change_fs(self):
        self.fs = int(self.fs_combo.currentData())
        CONFIG["last_fs"] = int(self.fs); save_config(CONFIG)
        self.chain.set_fs(self.fs)

    def _tick(self):
        self.level.setValue(int(self.last_level*100))

    def _open_settings(self):
        ApiKeyDialog(self).exec_()

    # audio stream
    def _audio_cb(self, indata, outdata, frames, time_info, status):
        if status:
            # ignore under/overruns quietly
            pass
        # mono in:
        x = indata[:, 0].copy() if indata.shape[1] >= 1 else np.zeros(frames, dtype=np.float32)
        lvl = float(np.sqrt(np.mean(x**2) + 1e-9)); self.last_level = max(0.0, min(1.0, lvl*8.0))
        y = self.chain.process(x.astype(np.float32))
        # mono out:
        outdata[:, 0] = y
        if self.recording:
            self.record_buf.append(y.copy())

    def _pick_device(self, combo, want_input=True):
        data = combo.currentData()
        if data is not None:
            return int(data)
        # Fallback to system default
        try:
            d_in, d_out = sd.default.device
            return int(d_in if want_input else d_out)
        except Exception:
            # fallback to index 0 if exists
            return 0

    def start_stream(self):
        if self.running: return
        try:
            in_dev = self._pick_device(self.in_combo, want_input=True)
            out_dev = self._pick_device(self.out_combo, want_input=False)
            self.fs = int(self.fs_combo.currentData())
            self.chain.set_fs(self.fs)
            self.stream = sd.Stream(device=(in_dev, out_dev), samplerate=self.fs, blocksize=512,
                                    channels=1, dtype='float32', callback=self._audio_cb)
            self.stream.start(); self.running = True
            self.start_btn.setEnabled(False); self.stop_btn.setEnabled(True)
        except Exception as e:
            # fallback hint for Windows devices that dislike 48k
            QtWidgets.QMessageBox.warning(self, "Stream error",
                f"{e}\n\nTip: try changing 'Sample rate' to 44100 Hz, then Start again.")

    def stop_stream(self):
        if not self.running: return
        try:
            self.stream.stop(); self.stream.close()
        except Exception:
            pass
        self.stream = None; self.running = False
        self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False)

    # recording flow
    def record_fullscreen(self):
        if not self.running:
            QtWidgets.QMessageBox.warning(self, "Not running", "Start the audio stream first."); return
        self.record_buf = []; self.recording = True
        self.full = FullscreenRecord(self); self.full.stopped.connect(self._stop_record)
        self.full.showFullScreen()

    def _stop_record(self):
        self.recording = False
        mono = np.concatenate(self.record_buf).astype(np.float32) if self.record_buf else np.zeros(1, dtype=np.float32)
        PreviewDialog(mono, self.fs, self).exec_()

# ---------------- Boot: splash then main ----------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    app.setStyleSheet(DARK_QSS)

    splash = Splash()
    splash.show()

    def open_main():
        w = MainApp(); w.show()

    # Use a singleShot to open the window just after splash closes (robust on Windows)
    splash.destroyed.connect(lambda: QtCore.QTimer.singleShot(50, open_main))

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

