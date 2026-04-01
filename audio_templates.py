import os
import shutil
import tempfile
import threading
import wave
from pathlib import Path
from typing import Optional

import numpy as np
from pvrecorder import PvRecorder
from PySide6.QtWidgets import QDialog, QPlainTextEdit, QPushButton, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit, QMessageBox
from window_context import get_resource_path

try:
    from python_speech_features import mfcc
    from fastdtw import fastdtw
    from scipy.spatial.distance import cosine
    from scipy import signal
    import scipy.io.wavfile as wav
    AUDIO_MATCH_AVAILABLE = True
except Exception:
    AUDIO_MATCH_AVAILABLE = False


def resolve_template_dir(template_dir: str = "templates") -> Path:
    original = Path(template_dir)
    if original.is_absolute():
        target = original
    else:
        parts = original.parts
        if parts and parts[0].lower() == "templates":
            base = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "pySiberMan"
            target = base / Path(*parts)
        else:
            target = Path.cwd() / original
    target.mkdir(parents=True, exist_ok=True)
    source = get_resource_path(original.as_posix())
    if source.exists() and source.is_dir():
        has_user_wavs = any(target.glob("*.wav"))
        if not has_user_wavs:
            for wav_file in source.glob("*.wav"):
                dst = target / wav_file.name
                if not dst.exists():
                    try:
                        shutil.copy2(wav_file, dst)
                    except Exception:
                        continue
    return target


class AudioTemplateMatcher:
    def __init__(self, template_dir="templates"):
        self.template_dir = resolve_template_dir(str(template_dir))
        self.templates = {}
        self.template_signatures = {}
        self.max_dtw_candidates = 8
        if AUDIO_MATCH_AVAILABLE:
            self.reload_templates()

    def _signature(self, feat: np.ndarray) -> Optional[np.ndarray]:
        if feat is None or feat.size == 0:
            return None
        try:
            mu = np.mean(feat, axis=0).astype(np.float32)
            sd = np.std(feat, axis=0).astype(np.float32)
            sig = np.concatenate([mu, sd]).astype(np.float32)
            norm = float(np.linalg.norm(sig))
            if norm <= 1e-9:
                return None
            return sig / norm
        except Exception:
            return None

    def _preprocess(self, x: np.ndarray, rate: int) -> Optional[np.ndarray]:
        if x is None:
            return None
        if x.ndim > 1:
            x = x[:, 0]
        if x.dtype != np.float32:
            if np.issubdtype(x.dtype, np.integer):
                x = x.astype(np.float32) / 32768.0
            else:
                x = x.astype(np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            sos = signal.butter(4, [80.0, 3800.0], btype="bandpass", fs=rate, output="sos")
            x = signal.sosfilt(sos, x)
        except Exception:
            pass

        if x.size >= 2:
            x = np.concatenate([x[:1], x[1:] - 0.97 * x[:-1]])

        frame_len = int(0.02 * rate)
        hop = int(0.01 * rate)
        if frame_len <= 0 or hop <= 0 or x.size < frame_len:
            return None

        n_frames = 1 + (x.size - frame_len) // hop
        frames = np.lib.stride_tricks.as_strided(
            x,
            shape=(n_frames, frame_len),
            strides=(x.strides[0] * hop, x.strides[0]),
            writeable=False,
        )
        win = np.hanning(frame_len).astype(np.float32)
        framed = frames * win[None, :]
        rms = np.sqrt(np.mean(framed * framed, axis=1) + 1e-12)
        if rms.size == 0:
            return None

        k = min(10, rms.size)
        noise_idx = np.argsort(rms)[:k]
        noise_level = float(np.median(rms[noise_idx]))
        thr = max(noise_level * 2.5, float(np.percentile(rms, 25)) * 1.5, 0.008)

        active = np.where(rms >= thr)[0]
        if active.size:
            start_f = max(0, int(active[0]) - 2)
            end_f = min(rms.size - 1, int(active[-1]) + 2)
            start = start_f * hop
            end = min(x.size, end_f * hop + frame_len)
            x = x[start:end]

        peak = float(np.max(np.abs(x))) if x.size else 0.0
        if peak > 1e-6:
            x = x / peak
        return x

    def _extract(self, x: np.ndarray, rate: int) -> Optional[np.ndarray]:
        if x is None or x.size == 0:
            return None
        try:
            feat = mfcc(
                x,
                samplerate=rate,
                winlen=0.025,
                winstep=0.01,
                numcep=13,
                nfilt=26,
                nfft=512,
                appendEnergy=True,
            ).astype(np.float32)
        except Exception:
            return None

        if feat.size == 0:
            return None
        mu = np.mean(feat, axis=0)
        sd = np.std(feat, axis=0)
        sd = np.where(sd < 1e-6, 1.0, sd)
        feat = (feat - mu) / sd
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        return feat

    def reload_templates(self):
        if not AUDIO_MATCH_AVAILABLE:
            return
        self.templates = {}
        self.template_signatures = {}
        for f in self.template_dir.glob("*.wav"):
            try:
                rate, sig = wav.read(str(f))
                x = self._preprocess(np.asarray(sig), int(rate))
                feat = self._extract(x, int(rate))
                if feat is None:
                    continue
                self.templates[f.stem] = feat
                signature = self._signature(feat)
                if signature is not None:
                    self.template_signatures[f.stem] = signature
            except Exception:
                continue

    def match(self, pcm_data, threshold: float):
        if not AUDIO_MATCH_AVAILABLE or not self.templates:
            return None, None

        rate = 16000
        x = self._preprocess(np.asarray(pcm_data, dtype=np.int16), rate)
        feat = self._extract(x, rate)
        if feat is None:
            return None, None
        query_signature = self._signature(feat)

        candidate_names = list(self.templates.keys())
        if query_signature is not None and len(candidate_names) > self.max_dtw_candidates:
            ranked = []
            for name in candidate_names:
                sig = self.template_signatures.get(name)
                if sig is None:
                    score = -1.0
                else:
                    score = float(np.dot(query_signature, sig))
                ranked.append((score, name))
            ranked.sort(key=lambda x: x[0], reverse=True)
            candidate_names = [name for _, name in ranked[:self.max_dtw_candidates]]

        best_dist = float("inf")
        best_name = None
        for name in candidate_names:
            t_feat = self.templates.get(name)
            if t_feat is None:
                continue
            try:
                distance, path = fastdtw(feat, t_feat, dist=cosine)
                denom = max(1, len(path))
                avg_dist = float(distance) / float(denom)
            except Exception:
                continue
            if avg_dist < best_dist:
                best_dist = avg_dist
                best_name = name

        if best_name is None:
            return None, None
        if best_dist <= float(threshold):
            return best_name, best_dist
        return None, best_dist


class RecorderDialog(QDialog):
    def __init__(self, parent=None, device_index=-1, template_dir="templates", scene_name="音频指令"):
        super().__init__(parent)
        self.setWindowTitle(f"录制{scene_name}")
        self.device_index = device_index
        self.template_dir = resolve_template_dir(str(template_dir))
        self.scene_name = scene_name
        self.recorder = None
        self.recorder_lock = threading.Lock()
        self.audio_data = []
        self.is_recording = False
        self._matcher = AudioTemplateMatcher()
        self.thread = None
        
        self.status_label = QLabel("按住按钮开始录音 (2-3秒)")
        self.btn_record = QPushButton("按住说话")
        self.btn_play = QPushButton("播放回放")
        self.btn_save = QPushButton("保存")
        self.btn_play.setEnabled(False)
        self.btn_save.setEnabled(False)
        
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText(f"输入{scene_name}名称")
        
        layout = QVBoxLayout()
        layout.addWidget(self.status_label)
        layout.addWidget(self.name_edit)
        layout.addWidget(self.btn_record)
        layout.addWidget(self.btn_play)
        layout.addWidget(self.btn_save)
        self.setLayout(layout)
        
        self.btn_record.pressed.connect(self.start_recording)
        self.btn_record.released.connect(self.stop_recording)
        self.btn_play.clicked.connect(self.play_audio)
        self.btn_save.clicked.connect(self.save_audio)

    def _create_recorder_with_fallback(self):
        devices = []
        try:
            devices = PvRecorder.get_available_devices()
        except Exception:
            devices = []
        candidate_indices = []
        try:
            pref = int(self.device_index)
        except Exception:
            pref = -1
        if pref >= 0:
            candidate_indices.append(pref)
        candidate_indices.append(-1)
        for idx in range(len(devices)):
            if idx not in candidate_indices:
                candidate_indices.append(idx)
        last_error = None
        for idx in candidate_indices:
            try:
                recorder = PvRecorder(device_index=idx, frame_length=512)
                recorder.start()
                self.device_index = idx
                return recorder, idx, devices
            except Exception as e:
                last_error = e
                continue
        raise RuntimeError(f"Failed to initialize PvRecorder: {last_error}")

    def start_recording(self):
        try:
            self.recorder, used_index, devices = self._create_recorder_with_fallback()
            self.is_recording = True
            self.audio_data = []
            if used_index >= 0 and used_index < len(devices):
                self.status_label.setText(f"正在录音...({devices[used_index]})")
            else:
                self.status_label.setText("正在录音...(系统默认设备)")
            self.thread = threading.Thread(target=self._record_loop, daemon=True)
            self.thread.start()
        except Exception as e:
            msg = str(e)
            if "Failed to initialize PvRecorder" in msg:
                QMessageBox.critical(
                    self,
                    "错误",
                    "无法启动录音: 无法初始化录音设备。\n"
                    "请检查系统麦克风权限、输入设备是否可用，或关闭占用麦克风的软件后重试。"
                )
            else:
                QMessageBox.critical(self, "错误", f"无法启动录音: {msg}")

    def _record_loop(self):
        while self.is_recording:
            try:
                with self.recorder_lock:
                    recorder = self.recorder
                    if not recorder:
                        break
                    pcm = recorder.read()
                if pcm:
                    self.audio_data.extend(pcm)
            except Exception:
                break
            
    def stop_recording(self):
        self.is_recording = False
        try:
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=1.0)
        except Exception:
            pass
        with self.recorder_lock:
            recorder = self.recorder
            self.recorder = None
        if recorder:
            try:
                recorder.stop()
            except Exception:
                pass
            try:
                recorder.delete()
            except Exception:
                pass

        raw = np.asarray(self.audio_data, dtype=np.int16)
        x = None
        try:
            x = self._matcher._preprocess(raw, 16000)
        except Exception:
            x = None
        if x is not None and x.size:
            x_i16 = np.clip(x * 32767.0, -32768.0, 32767.0).astype(np.int16)
            self.audio_data = x_i16.tolist()

        self.status_label.setText(f"录音结束, 长度: {len(self.audio_data) / 16000:.2f}s")
        self.btn_play.setEnabled(True)
        self.btn_save.setEnabled(True)
        
    def play_audio(self):
        if not self.audio_data: return
        try:
            import winsound
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                fname = f.name
            
            with wave.open(fname, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(np.array(self.audio_data, dtype=np.int16).tobytes())
            
            winsound.PlaySound(fname, winsound.SND_FILENAME)
            os.unlink(fname)
        except Exception as e:
            QMessageBox.warning(self, "播放失败", str(e))

    def save_audio(self):
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "提示", "请输入名称")
            return
            
        try:
            path = self.template_dir / f"{name}.wav"
            with wave.open(str(path), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(np.array(self.audio_data, dtype=np.int16).tobytes())
            QMessageBox.information(self, "成功", f"已保存到 {path}")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))


class TemplateManageDialog(QDialog):
    def __init__(self, parent=None, device_index=-1, template_dir="templates", scene_name="音频指令"):
        super().__init__(parent)
        self.device_index = device_index
        self.template_dir = resolve_template_dir(str(template_dir))
        self.scene_name = scene_name
        self.setWindowTitle(f"{scene_name}管理")
        self.resize(400, 300)
        
        self.list_widget = QPlainTextEdit()
        self.list_widget.setReadOnly(True)
        
        self.btn_refresh = QPushButton("刷新列表")
        self.btn_record = QPushButton("录制新指令")
        
        layout = QVBoxLayout()
        layout.addWidget(QLabel(f"已录制的指令 ({self.template_dir})"))
        layout.addWidget(self.list_widget)
        
        btns = QHBoxLayout()
        btns.addWidget(self.btn_refresh)
        btns.addWidget(self.btn_record)
        layout.addLayout(btns)
        
        self.setLayout(layout)
        
        self.btn_refresh.clicked.connect(self.refresh_list)
        self.btn_record.clicked.connect(self.record_new)
        
        self.refresh_list()
        
    def refresh_list(self):
        self.list_widget.clear()
        files = list(self.template_dir.glob("*.wav"))
        for f in files:
            self.list_widget.appendPlainText(f.name)
            
    def record_new(self):
        dlg = RecorderDialog(self, device_index=self.device_index, template_dir=str(self.template_dir), scene_name=self.scene_name)
        dlg.exec()
        self.refresh_list()
