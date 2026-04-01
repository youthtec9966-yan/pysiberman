import sys
from pathlib import Path
import json
import threading
import traceback
from typing import Optional
import re
import difflib

# PySide6 是 Qt 框架的 Python 绑定，用于创建图形用户界面 (GUI)
from PySide6.QtCore import QObject, QThread, Signal, Qt, QTimer
from PySide6.QtWidgets import QApplication, QSystemTrayIcon, QMenu, QDialog, QFormLayout, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPushButton, QHBoxLayout, QVBoxLayout, QLabel, QFileDialog, QMessageBox, QPlainTextEdit, QWidget, QProgressBar, QSlider
from PySide6.QtGui import QIcon, QPixmap, QAction, QPainter, QPen, QColor

import keyring # 用于安全地存储密码（如 API Key）
import psutil # 用于获取系统进程和硬件信息


from pvrecorder import PvRecorder # 用于录音
import numpy as np
import asyncio
import websockets # 用于 WebSocket 通信
import socket
import subprocess
import os
import tempfile
import wave
import hashlib
from openai import AsyncOpenAI # OpenAI 官方 SDK，用于调用兼容 OpenAI 接口的大模型
import time
import faulthandler # 用于调试崩溃（Faults）
import edge_tts

try:
    from python_speech_features import mfcc
    from fastdtw import fastdtw
    from scipy.spatial.distance import cosine
    from scipy import signal
    import scipy.io.wavfile as wav
    AUDIO_MATCH_AVAILABLE = True
except Exception:
    AUDIO_MATCH_AVAILABLE = False

# 导入阿里云Fun-ASR SDK
from dashscope.audio.asr import Recognition, RecognitionCallback

# 简单的回调类实现
class SimpleRecognitionCallback(RecognitionCallback):
    def __init__(self):
        self.results = []
    
    def on_result(self, result):
        self.results.append(result)
    
    def on_error(self, error):
        pass
    
    def on_close(self):
        pass
    
    def on_complete(self):
        pass

class AudioTemplateMatcher:
    def __init__(self, template_dir="templates"):
        self.template_dir = Path(template_dir)
        self.template_dir.mkdir(exist_ok=True)
        self.templates = {}
        if AUDIO_MATCH_AVAILABLE:
            self.reload_templates()

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
        for f in self.template_dir.glob("*.wav"):
            try:
                rate, sig = wav.read(str(f))
                x = self._preprocess(np.asarray(sig), int(rate))
                feat = self._extract(x, int(rate))
                if feat is None:
                    continue
                self.templates[f.stem] = feat
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

        best_dist = float("inf")
        best_name = None
        for name, t_feat in self.templates.items():
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
        self.template_dir = Path(template_dir)
        self.template_dir.mkdir(parents=True, exist_ok=True)
        self.scene_name = scene_name
        self.recorder = None
        self.audio_data = []
        self.is_recording = False
        self._matcher = AudioTemplateMatcher()
        
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
        
    def start_recording(self):
        try:
            self.recorder = PvRecorder(device_index=self.device_index, frame_length=512)
            self.recorder.start()
            self.is_recording = True
            self.audio_data = []
            self.status_label.setText("正在录音...")
            self.thread = threading.Thread(target=self._record_loop)
            self.thread.start()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法启动录音: {e}")

    def _record_loop(self):
        while self.is_recording and self.recorder:
            try:
                pcm = self.recorder.read()
                self.audio_data.extend(pcm)
            except:
                break
            
    def stop_recording(self):
        self.is_recording = False
        if self.recorder:
            self.recorder.stop()
            self.recorder.delete()
            self.recorder = None

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
        self.template_dir = Path(template_dir)
        self.template_dir.mkdir(parents=True, exist_ok=True)
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

import ctypes
from ctypes import wintypes

class WindowContextManager:
    """
    Manages the context of the currently active window to allow for seamless switching
    between the digital human and other applications (e.g., video players).
    It handles:
    1. Capturing the current active window handle.
    2. Pausing media playback (via global media keys).
    3. Exiting fullscreen mode (via Esc key).
    4. Restoring the previous window to the foreground.
    5. Restoring fullscreen mode (via F11 key).
    6. Resuming media playback.
    """
    def __init__(self):
        self.last_hwnd = None
        self.user32 = ctypes.windll.user32
        
        # Virtual Key Codes
        self.VK_MEDIA_PLAY_PAUSE = 0xB3
        self.VK_SPACE = 0x20
        self.VK_ESCAPE = 0x1B
        self.VK_F11 = 0x7A 
        self.VK_RETURN = 0x0D
        self.VK_MENU = 0x12 # Alt key
        
        # Event Flags
        self.KEYEVENTF_KEYUP = 0x0002

    def _send_key(self, vk_code):
        """Simulate a single key press and release."""
        self.user32.keybd_event(vk_code, 0, 0, 0)
        self.user32.keybd_event(vk_code, 0, self.KEYEVENTF_KEYUP, 0)

    def capture_context(self):
        """
        Capture the current foreground window, pause media, and exit fullscreen.
        Must be called BEFORE the digital human window takes focus.
        """
        self.last_hwnd = self.user32.GetForegroundWindow()
        
        # Only capture if it's a valid window and not our own tray app (simplistic check)
        if not self.last_hwnd:
            return

        print(f"[WindowManager] Captured hwnd: {self.last_hwnd}")

        # 1. Pause Media (Try Space, then Media Key as backup)
        # Many web players use Space for play/pause
        self._send_key(self.VK_SPACE)
        # self._send_key(self.VK_MEDIA_PLAY_PAUSE)
        
        # 2. Exit Fullscreen (Try ESC)
        # Most players/browsers exit fullscreen with Esc.
        self._send_key(self.VK_ESCAPE)

    def restore_context(self):
        """
        Restore the previously captured window, re-enter fullscreen, and resume media.
        """
        if not self.last_hwnd:
            print("[WindowManager] No context to restore.")
            return

        print(f"[WindowManager] Restoring hwnd: {self.last_hwnd}")
        
        # 1. Restore Window State (if minimized)
        if self.user32.IsIconic(self.last_hwnd):
             self.user32.ShowWindow(self.last_hwnd, 9) # SW_RESTORE

        # 2. Bring to Foreground
        # Note: This might fail if the OS blocks it, but since we are the foreground process now,
        # handing off focus usually works.
        self.user32.SetForegroundWindow(self.last_hwnd)
        
        # Give a tiny delay for focus to switch
        time.sleep(0.2)
        
        # 3. Resume Media (Space) - Moved before fullscreen as requested
        self._send_key(self.VK_SPACE)
        
        time.sleep(0.2)

        # 4. Enter Fullscreen (Enter) - Changed from F11 to Enter as requested
        self._send_key(self.VK_RETURN)
        
        # Reset
        self.last_hwnd = None


def get_resource_path(relative_path: str) -> Path:
    """
    获取资源的绝对路径。
    这个函数是为了兼容开发环境和 PyInstaller 打包后的环境。
    PyInstaller 会把资源解压到一个临时目录 (sys._MEIPASS)。
    """
    try:
        # PyInstaller 创建临时文件夹并将路径存储在 _MEIPASS 中
        base_path = sys._MEIPASS
    except Exception:
        base_path = Path.cwd()
        # 如果是打包后的 exe 运行
        if getattr(sys, 'frozen', False):
            base_path = Path(sys.executable).parent
            # 处理 _internal 目录结构（PyInstaller 的一种打包模式）
            if (base_path / "_internal").exists():
                base_path = base_path / "_internal"

    p = Path(base_path) / relative_path
    if not p.exists():
        # 开发模式下的回退机制，或者当前工作目录不同时
        if (Path.cwd() / relative_path).exists():
            return Path.cwd() / relative_path
    return p


class ConfigManager:
    """
    配置管理器
    负责加载和保存应用程序的配置信息到 JSON 文件。
    """
    def __init__(self):
        # 使用 LOCALAPPDATA 目录存储配置，避免 Program Files 目录的权限问题
        local_app_data = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
        self.config_dir = Path(local_app_data) / "pySiberMan" / "config"
        self.config_path = self.config_dir / "config.json"
        self.config = {}
        self._ensure()
        self._load()

    def _ensure(self):
        """确保配置目录和文件存在，如果不存在则创建默认配置"""
        if not self.config_dir.exists():
            self.config_dir.mkdir(parents=True, exist_ok=True)
        if not self.config_path.exists():
            self.config_path.write_text(json.dumps({
                "aliyun_appkey": "",
                "chat_engine": "llm",
                "device_index": -1,
                "llm_base_url": "",
                "llm_model": "",
                "wake_engine": "asr",
                "asr_wake_phrases": "小石警官",
                "asr_model_size": "small",
                "asr_compute_type": "int8",
                "asr_device": "cpu",
                "asr_model_dir": "",
                "audio_match_enabled": True,
                "audio_match_threshold": 0.45,
                "asr_profile_mode": "smart",
                "asr_standby_noise_margin": 0.015,
                "asr_speaking_noise_margin": 0.05,
                "asr_standby_energy_ratio": 1.35,
                "asr_speaking_energy_ratio": 2.2,
                "asr_interrupt_peak": 0.085,
                "asr_interrupt_rms": 0.015
            }, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load(self):
        """从文件加载配置"""
        try:
            self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        except Exception:
            # 加载失败时使用默认配置
            self.config = {
                "aliyun_appkey": "",
                "chat_engine": "llm",
                "device_index": -1,
                "llm_base_url": "",
                "llm_model": "",
                "wake_engine": "asr",
                "asr_wake_phrases": "小石警官",
                "asr_model_size": "small",
                "asr_compute_type": "int8",
                "asr_device": "cpu",
                "asr_model_dir": "",
                "audio_match_enabled": True,
                "audio_match_threshold": 0.45,
                "asr_profile_mode": "smart",
                "asr_standby_noise_margin": 0.015,
                "asr_speaking_noise_margin": 0.05,
                "asr_standby_energy_ratio": 1.35,
                "asr_speaking_energy_ratio": 2.2,
                "asr_interrupt_peak": 0.085,
                "asr_interrupt_rms": 0.015
            }

    def save(self):
        """保存当前配置到文件"""
        self.config_path.write_text(json.dumps(self.config, ensure_ascii=False, indent=2), encoding="utf-8")

    def get(self, key: str, default=None):
        """获取配置项"""
        return self.config.get(key, default)

    def set(self, key: str, value):
        """设置配置项（需要手动调用 save 才能持久化）"""
        self.config[key] = value


class SecretStore:
    """
    密钥存储
    使用 keyring 库将敏感信息（如 API Key）存储在操作系统的凭据管理器中，
    而不是明文保存在 JSON 文件里。
    """
    def __init__(self):
        self.service = "pySiberMan"

    def get(self, name: str, default: str = None) -> Optional[str]:
        """获取密钥"""
        try:
            val = keyring.get_password(self.service, name)
            return val if val is not None else default
        except Exception:
            return default

    def set(self, name: str, value: str):
        """保存密钥"""
        try:
            keyring.set_password(self.service, name, value)
        except Exception:
            pass

class AudioWakeWorkerASR(QObject):
    """
    ASR (语音识别) 唤醒工作线程
    该类在后台线程中运行，持续录制音频，并使用 faster-whisper 模型进行识别。
    如果识别到的文本中包含唤醒词（如“小石警官”），则触发 wake_detected 信号。
    """
    wake_detected = Signal() # 唤醒信号
    error = Signal(str)      # 错误信号
    log = Signal(str)        # 日志信号
    volume = Signal(float)   # 音量信号（0.0-1.0）
    chat_input_detected = Signal(str) # 新增信号：用于传输对话内容的ASR结果
    interrupt_detected = Signal(str)

    def __init__(self, phrases: str, device_index: int, api_key: str, audio_match_enabled: bool = True, audio_match_threshold: float = 0.45):
        super().__init__()
        # 处理唤醒词列表
        self.phrases = [p.strip() for p in (phrases or "").split(",") if p.strip()]
        self.phrases_norm = [self._normalize(p) for p in self.phrases]
        
        self.device_index = device_index
        self.api_key = api_key
        
        self.wake_matcher = AudioTemplateMatcher("templates/wake")
        self.interrupt_matcher = AudioTemplateMatcher("templates/interrupt")
        self.audio_match_enabled = bool(audio_match_enabled)
        self.audio_match_threshold = float(audio_match_threshold)
        self.wake_audio_match_threshold = float(audio_match_threshold)
        
        self._running = False
        self._paused = False
        self._mode = "wake" # 模式："wake" (唤醒模式) 或 "chat" (对话模式)
        self._thread = None
        # 注意：此处不保留 recorder 实例，防止跨线程访问
        
        # 动态降噪相关参数
        self.noise_floor = 0.01  # 初始噪音阈值
        self.vol_history = []     # 音量历史记录
        self.max_vol_history = [] # 最大音量历史记录
        self.NOISE_MARGIN = 0.02  # 噪音阈值与语音的差距
        self.HISTORY_SIZE = 1000    # 历史记录大小 (约32秒，每帧32ms)
        self._speaking_state = False
        self.standby_noise_margin = 0.015
        self.speaking_noise_margin = 0.05
        self.standby_energy_ratio = 1.35
        self.speaking_energy_ratio = 2.2
        self.speaking_interrupt_peak = 0.085
        self.speaking_interrupt_rms = 0.015
        self.last_asr_peak = 0.0
        self.last_asr_rms = 0.0

    def set_speaking_state(self, speaking: bool):
        self._speaking_state = bool(speaking)
        self.log.emit(f"ASR speaking_state={self._speaking_state}")

    def apply_dynamic_profile(self, profile: dict):
        if not profile:
            return
        self.standby_noise_margin = float(profile.get("standby_noise_margin", self.standby_noise_margin))
        self.speaking_noise_margin = float(profile.get("speaking_noise_margin", self.speaking_noise_margin))
        self.standby_energy_ratio = float(profile.get("standby_energy_ratio", self.standby_energy_ratio))
        self.speaking_energy_ratio = float(profile.get("speaking_energy_ratio", self.speaking_energy_ratio))
        self.speaking_interrupt_peak = float(profile.get("speaking_interrupt_peak", self.speaking_interrupt_peak))
        self.speaking_interrupt_rms = float(profile.get("speaking_interrupt_rms", self.speaking_interrupt_rms))

    def get_vad_snapshot(self):
        avg_max_vol = sum(self.max_vol_history) / len(self.max_vol_history) if self.max_vol_history else 0.0
        if self._speaking_state:
            gate = max(
                self.noise_floor + self.speaking_noise_margin,
                avg_max_vol * self.speaking_energy_ratio,
                self.speaking_interrupt_peak
            )
        else:
            gate = max(
                self.noise_floor + self.standby_noise_margin,
                avg_max_vol * self.standby_energy_ratio
            )
        return {
            "noise_floor": float(self.noise_floor),
            "avg_max_vol": float(avg_max_vol),
            "gate": float(gate),
            "speaking": bool(self._speaking_state)
        }

    def set_mode(self, mode: str):
        """设置工作模式: 'wake' 或 'chat'"""
        if mode in ["wake", "chat"]:
            self._mode = mode
            self.log.emit(f"Switched to {mode} mode")
            # 切换模式时，如果处于暂停状态，且切换到 chat 模式，通常希望立即开始监听
            # 但具体的暂停控制交给上层逻辑调用 resume() 更安全
            
    def reload_audio_templates(self):
        if self.wake_matcher:
            self.wake_matcher.reload_templates()
        if self.interrupt_matcher:
            self.interrupt_matcher.reload_templates()

    def pause(self):
        """暂停监听 (但不释放模型资源)"""
        self._paused = True

    def resume(self):
        """恢复监听"""
        self._paused = False

    def _normalize(self, s: str) -> str:
        """
        文本标准化：去除标点符号，并纠正常见的唤醒词误识别。
        例如将“小时警官”、“消失警官”纠正为“小石警官”。
        """
        t = re.sub(r"[\s,.!?;:，。！？；：、（）()《》〈〉「」『』“”‘’—…·\-]+", "", s or "")
        for v in ["小时","消失","小是","小识","小事","肖石","晓石","晓诗","萧石","小十", "消石", "销石", "小实", "小视", "孝石"]:
            t = t.replace(v, "小石")
        for v in ["景观","尽管","警管","井关","金冠","尽关","经管","景觀","尽古","頂關","景瓜","金关","里关", "敬官", "静观", "警馆", "井官", "经官", "竞管", "境关"]:
            t = t.replace(v, "警官")
        return t

    def _partial_ratio(self, a: str, b: str) -> float:
        """
        计算两个字符串的相似度（模糊匹配）。
        用于容错，当识别结果不完全一致但很接近时也视为唤醒。
        """
        if not a or not b:
            return 0.0
        if len(a) > len(b):
            a, b = b, a
        L = len(a)
        best = 0.0
        step = max(1, L // 3)
        pad = 4
        for i in range(0, max(1, len(b) - L + pad), step):
            j = min(len(b), i + L + pad)
            r = difflib.SequenceMatcher(None, a, b[i:j]).ratio()
            if r > best:
                best = r
            if best >= 0.9:
                break
        return best

    def _match_wake_audio_command(self, pcm_window):
        if not (self.audio_match_enabled and self.wake_matcher):
            return False
        match_name, match_dist = self.wake_matcher.match(pcm_window, threshold=self.wake_audio_match_threshold)
        if not match_name:
            return False
        if match_dist is not None:
            self.log.emit(f"Wake Audio Match: {match_name} ({match_dist:.3f})")
        else:
            self.log.emit(f"Wake Audio Match: {match_name}")
        self.wake_detected.emit()
        self._paused = True
        self.log.emit("wake listening paused")
        return True

    def _match_interrupt_audio_command(self, pcm_window):
        if not (self.audio_match_enabled and self.interrupt_matcher):
            return False
        match_name, match_dist = self.interrupt_matcher.match(pcm_window, threshold=self.audio_match_threshold)
        if not match_name:
            return False
        if match_dist is not None:
            self.log.emit(f"Interrupt Audio Match: {match_name} ({match_dist:.3f})")
        else:
            self.log.emit(f"Interrupt Audio Match: {match_name}")
        self.interrupt_detected.emit(match_name)
        return True

    def start(self):
        """启动后台录音识别线程"""
        if self._running:
            return
        self._running = True
        # 必须设置为 daemon=True，防止主程序退出时线程还在后台写日志导致崩溃
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """
        停止录音并同步等待线程结束。
        这是防止闪退的关键：必须确保旧线程彻底死掉，释放了麦克风和显存，
        然后才能允许开启新线程。
        """
        if not self._running:
            return
            
        self._running = False
        
        # 这里的 wait/join 是解决闪退的核心
        if self._thread and self._thread.is_alive():
            # 这里的 timeout 稍微设长一点，因为 transcribe 可能正在运行
            # 如果超过 5 秒还没停，说明卡死了，那也没办法，只能由操作系统回收
            self._thread.join(timeout=5.0) 
        
        self._thread = None

    def pause(self):
        """暂停监听 (但不释放模型资源)"""
        self._paused = True

    def resume(self):
        """恢复监听"""
        self._paused = False

    def _update_noise_floor(self, arr):
        """动态更新噪音阈值"""
        current_max = np.max(np.abs(arr))
        
        # 更新音量历史记录
        self.vol_history.append(current_max)
        if len(self.vol_history) > self.HISTORY_SIZE:
            self.vol_history.pop(0)
        
        if self.vol_history:
            sorted_hist = sorted(self.vol_history)
            idx = max(0, int(len(sorted_hist) * 0.1) - 1)
            self.noise_floor = sorted_hist[idx] * 1.15

    def _is_speech(self, arr):
        """判断是否有语音（基于动态噪音阈值）"""
        current_max = np.max(np.abs(arr))
        current_rms = float(np.sqrt(np.mean(np.square(arr)))) if len(arr) else 0.0
        
        # 更新最大音量历史记录
        self.max_vol_history.append(current_max)
        if len(self.max_vol_history) > self.HISTORY_SIZE:
            self.max_vol_history.pop(0)
        
        # 计算平均最大音量
        avg_max_vol = sum(self.max_vol_history) / len(self.max_vol_history) if self.max_vol_history else 0

        if self._speaking_state:
            dynamic_gate = max(
                self.noise_floor + self.speaking_noise_margin,
                avg_max_vol * self.speaking_energy_ratio,
                self.speaking_interrupt_peak
            )
            return current_max > dynamic_gate and current_rms > self.speaking_interrupt_rms
        return current_max > (self.noise_floor + self.standby_noise_margin) and current_max > (avg_max_vol * self.standby_energy_ratio)

    def _process_asr(self, pcm_data):
        """处理累积的语音数据进行识别"""
        if not pcm_data or not self._running:
            return

        # 1. 将音频数据转换为WAV格式的临时文件
        arr = np.asarray(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
        self.last_asr_peak = float(np.max(np.abs(arr))) if len(arr) else 0.0
        self.last_asr_rms = float(np.sqrt(np.mean(np.square(arr)))) if len(arr) else 0.0
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            temp_filename = temp_wav.name
        
        try:
            with wave.open(temp_filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes((arr * 32768).astype(np.int16).tobytes())

            # 2. 调用阿里云Fun-ASR API
            # 创建回调对象
            callback = SimpleRecognitionCallback()
            
            # 初始化Recognition对象
            recognition = Recognition(
                model="paraformer-realtime-v1",
                callback=callback,
                format="wav",
                sample_rate=16000,
                api_key=self.api_key
            )
            
            # 调用识别API
            result = recognition.call(file=temp_filename)
            
            # 3. 解析识别结果
            full_text = ""
            if result:
                # 简化结果解析逻辑
                try:
                    # 尝试通用的解析路径
                    candidates = []
                    if hasattr(result, 'get'):
                        candidates.extend(result.get('sentence', []))
                        candidates.extend(result.get('sentences', []))
                        
                        output = result.get('output', {})
                        if isinstance(output, dict):
                            candidates.extend(output.get('sentence', []))
                            candidates.extend(output.get('sentences', []))
                    
                    for item in candidates:
                        if hasattr(item, 'text'):
                            full_text += item.text
                        elif isinstance(item, dict) and 'text' in item:
                            full_text += item['text']
                except Exception:
                    pass

            # 备用：从回调获取
            if not full_text and callback.results:
                try:
                    for cb_result in callback.results:
                         if hasattr(cb_result, 'text'):
                             full_text += cb_result.text
                except Exception:
                    pass

            # 清理临时文件
            try:
                os.unlink(temp_filename)
            except:
                pass

            if not full_text:
                return

            norm_text = self._normalize(full_text)
            self.log.emit(f"asr: {full_text}")

            # --- 模式分支 ---
            if self._mode == "chat":
                if full_text.strip():
                    self.chat_input_detected.emit(full_text)
            else:
                # 唤醒模式
                hit = False
                for p in self.phrases_norm:
                    if not p: continue
                    if p in norm_text or (len(norm_text) > 2 and p in norm_text) or self._partial_ratio(p, norm_text) >= 0.75:
                        hit = True
                        break
                
                if hit:
                    self.log.emit("wake: hit")
                    self.wake_detected.emit()
                    self._paused = True
                    self.log.emit("wake listening paused")

        except Exception as e:
            self.error.emit(f"ASR Process Error: {e}")
            try:
                if os.path.exists(temp_filename):
                    os.unlink(temp_filename)
            except:
                pass

    def _run(self):
        """后台线程主逻辑 - 基于 VAD 的动态分段"""
        recorder = None
        try:
            faulthandler.enable()
            self.log.emit("asr engine ready")

            # frame_length=512 (32ms)
            recorder = PvRecorder(device_index=self.device_index, frame_length=512)
            recorder.start()
            
            self.log.emit("wake listening started")

            # VAD 状态机变量
            speech_frames = []
            silence_counter = 0
            in_speech = False
            interrupt_ring = []
            interrupt_tick = 0
            last_interrupt_ts = 0.0
            
            # 参数配置
            MAX_SILENCE_FRAMES_WAKE = 25  # 约 800ms 静音判定为结束
            MAX_SILENCE_FRAMES_CHAT = 40  # 约 1.28s 静音判定为结束（更抗停顿，减少误切段）
            MIN_SPEECH_FRAMES = 15   # 约 500ms 最短语音长度
            MIN_MATCH_FRAMES = 6     # 约 200ms 最短匹配长度
            MAX_SPEECH_DURATION_FRAMES = 468 # 约 15秒 最长语音
            
            while self._running:
                try:
                    pcm = recorder.read()
                    if self._paused:
                        # 暂停时清空状态
                        speech_frames = []
                        in_speech = False
                        silence_counter = 0
                        continue
                    
                    if not pcm:
                        continue

                    # 计算音量与 VAD
                    arr = np.array(pcm, dtype=np.int16)
                    arr_float = arr.astype(np.float32) / 32768.0
                    volume_level = np.max(np.abs(arr_float))
                    current_rms = float(np.sqrt(np.mean(np.square(arr_float)))) if len(arr_float) else 0.0
                    self.volume.emit(min(1.0, volume_level))
                    
                    # 只有在非语音状态下才更新噪音基底，防止把语音当噪音
                    if not in_speech:
                         self._update_noise_floor(arr_float)

                    is_speech_frame = self._is_speech(arr_float)
                    interrupt_ring.extend(pcm)
                    if len(interrupt_ring) > 16000:
                        interrupt_ring = interrupt_ring[-16000:]
                    interrupt_tick += 1
                    if interrupt_tick % 8 == 0:
                        now_ts = time.monotonic()
                        if (now_ts - last_interrupt_ts) > 0.8:
                            base_gate = self.speaking_noise_margin if self._speaking_state else self.standby_noise_margin
                            vol_gate = max(float(self.noise_floor) + float(base_gate), 0.02)
                            rms_gate = float(self.speaking_interrupt_rms) if self._speaking_state else 0.006
                            if volume_level >= vol_gate or current_rms >= rms_gate:
                                if self._match_interrupt_audio_command(interrupt_ring):
                                    last_interrupt_ts = time.monotonic()
                                    interrupt_ring = []
                                    speech_frames = []
                                    in_speech = False
                                    silence_counter = 0
                                    continue

                    if in_speech:
                        speech_frames.extend(pcm)
                        
                        if not is_speech_frame:
                            silence_counter += 1
                        else:
                            silence_counter = 0 # 重置静音计数

                        if len(speech_frames) >= MIN_MATCH_FRAMES * 512 and len(speech_frames) % (8 * 512) == 0:
                            window_pcm = speech_frames[-16000:] if len(speech_frames) > 16000 else speech_frames
                            if self._mode != "chat" and self._match_wake_audio_command(window_pcm):
                                speech_frames = []
                                in_speech = False
                                silence_counter = 0
                                interrupt_ring = []
                                continue
                            if self._match_interrupt_audio_command(window_pcm):
                                last_interrupt_ts = time.monotonic()
                                speech_frames = []
                                in_speech = False
                                silence_counter = 0
                                interrupt_ring = []
                                continue
                        
                        # 检查是否结束
                        # 1. 静音超时 (说话结束)
                        # 2. 语音过长 (强制截断)
                        max_silence_frames = MAX_SILENCE_FRAMES_CHAT if self._mode == "chat" else MAX_SILENCE_FRAMES_WAKE
                        if silence_counter >= max_silence_frames or len(speech_frames) >= MAX_SPEECH_DURATION_FRAMES * 512:
                            # 剔除末尾的静音帧 (如果是静音超时造成的)
                            valid_len = len(speech_frames)
                            if silence_counter >= max_silence_frames:
                                valid_len -= (silence_counter * 512)
                            
                            valid_pcm = speech_frames[:valid_len]
                            
                            # 过滤太短的语音 (可能是噪音)
                            if len(valid_pcm) > MIN_MATCH_FRAMES * 512:
                                self.log.emit(f"VAD: Speech detected ({len(valid_pcm)/16000:.2f}s), processing...")

                                if len(valid_pcm) > MIN_SPEECH_FRAMES * 512:
                                    self._process_asr(valid_pcm)
                            
                            # 重置状态
                            speech_frames = []
                            in_speech = False
                            silence_counter = 0
                            
                    else: # Silence state
                        if is_speech_frame:
                            in_speech = True
                            speech_frames.extend(pcm)
                            silence_counter = 0
                            self.log.emit("VAD: Speech started")
                        else:
                            # 可选：保留少量前导帧以防止切头 (这里暂不实现，因为 frame 很短)
                            pass

                except Exception as e:
                    self.error.emit(f"Mic/Loop Error: {e}")
                    # 防止死循环刷屏，稍微 sleep
                    time.sleep(0.1)

        except Exception as e:
            self.error.emit(f"Worker Exception: {str(e)}")
        finally:
            if recorder is not None:
                try:
                    recorder.stop()
                    recorder.delete()
                except: pass
            import gc
            gc.collect()
            self.log.emit("wake listening stopped")


class AudioCurveWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._volume_points = []
        self._gate_points = []
        self._floor_points = []
        self._speaking = False
        self.max_points = 180
        self.setMinimumHeight(120)

    def append_sample(self, volume: float, gate: float, floor: float, speaking: bool):
        v = max(0.0, min(1.0, float(volume)))
        g = max(0.0, min(1.0, float(gate)))
        f = max(0.0, min(1.0, float(floor)))
        self._volume_points.append(v)
        self._gate_points.append(g)
        self._floor_points.append(f)
        self._speaking = bool(speaking)
        if len(self._volume_points) > self.max_points:
            self._volume_points.pop(0)
        if len(self._gate_points) > self.max_points:
            self._gate_points.pop(0)
        if len(self._floor_points) > self.max_points:
            self._floor_points.pop(0)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        rect = self.rect()
        bg = QColor(245, 248, 255) if self._speaking else QColor(248, 248, 248)
        painter.fillRect(rect, bg)
        painter.setPen(QPen(QColor(210, 210, 210), 1))
        painter.drawRect(rect.adjusted(0, 0, -1, -1))
        if len(self._volume_points) < 2:
            return

        width = max(1, rect.width() - 2)
        height = max(1, rect.height() - 2)
        step = width / max(1, self.max_points - 1)

        def y_of(v: float):
            return int(rect.bottom() - 1 - max(0.0, min(1.0, v)) * height)

        floor_y = y_of(self._floor_points[-1])
        painter.setPen(QPen(QColor(90, 170, 90), 1))
        painter.drawLine(1, floor_y, rect.right() - 1, floor_y)

        gate_y = y_of(self._gate_points[-1])
        painter.setPen(QPen(QColor(255, 170, 0), 1))
        painter.drawLine(1, gate_y, rect.right() - 1, gate_y)

        painter.setPen(QPen(QColor(60, 120, 255), 2))
        last_x = 1
        last_y = y_of(self._volume_points[0])
        for i in range(1, len(self._volume_points)):
            x = int(1 + i * step)
            y = y_of(self._volume_points[i])
            painter.drawLine(last_x, last_y, x, y)
            last_x, last_y = x, y


class SettingsDialog(QDialog):
    """
    设置对话框
    
    用于配置:
    1. LLM 连接信息 (Base URL, API Key, Model)
    2. 唤醒引擎设置 (ASR/VAD)
    3. ASR 模型参数 (大小, 设备, 目录)
    """
    def __init__(self, cfg: ConfigManager, secrets: SecretStore, app=None):
        """
        初始化设置界面
        
        Args:
            cfg: 全局配置管理器实例
            secrets: 密钥存储实例
        """
        super().__init__()
        self.cfg = cfg
        self.secrets = secrets
        self.app = app
        self.setWindowTitle("设置")
        self.resize(760, 620)
        
        self.engine_combo = QComboBox()
        self.engine_combo.addItem("ASR 识别触发", "asr")
        eng = self.cfg.get("wake_engine", "asr")

        

        self.asr_phrases = QLineEdit("小石警官")
        try:
            self.asr_phrases.setReadOnly(True)
            self.asr_phrases.setEnabled(False)
        except Exception:
            pass
        self.asr_model = QComboBox()
        for name in ["tiny", "base", "small", "medium"]:
            self.asr_model.addItem(name, name)
        ms = self.cfg.get("asr_model_size", "small")
        i = ["tiny", "base", "small", "medium"].index(ms) if ms in ["tiny", "base", "small", "medium"] else 2
        self.asr_model.setCurrentIndex(i)

        self.device_combo = QComboBox()
        try:
            devices = PvRecorder.get_available_devices()
            for i, name in enumerate(devices):
                self.device_combo.addItem(name, i)
            idx = self.cfg.get("device_index", -1)
            if idx is not None and idx >= 0 and idx < len(devices):
                self.device_combo.setCurrentIndex(idx)
        except Exception as e:
            self.device_combo.addItem(f"获取设备失败: {str(e)}", -1)

        self.asr_model_dir = QLineEdit(self.cfg.get("asr_model_dir", ""))
        try:
            self.asr_model_dir.setPlaceholderText("选择 Systran/faster-whisper-<size> 模型目录")
        except Exception:
            pass
        btn_browse = QPushButton("选择目录")
        btn_browse.clicked.connect(self._browse_asr_model_dir)
        w = QWidget()
        row = QHBoxLayout()
        row.setContentsMargins(0,0,0,0)
        row.addWidget(self.asr_model_dir)
        row.addWidget(btn_browse)
        w.setLayout(row)

        # 定义 LLM 输入框
        self.llm_base_url = QLineEdit(self.cfg.get("llm_base_url", "https://api.deepseek.com/v1"))
        self.llm_api_key = QLineEdit(self.secrets.get("llm_api_key", "sk-8394fdbc7380424eab4633fed976a6fe"))
        self.llm_api_key.setEchoMode(QLineEdit.Password)
        self.llm_model = QLineEdit(self.cfg.get("llm_model", "deepseek-chat"))

        form = QFormLayout()
        form.addRow("LLM Base URL", self.llm_base_url)
        form.addRow("LLM API Key", self.llm_api_key)
        form.addRow("LLM 模型", self.llm_model)
        form.addRow("唤醒方式", self.engine_combo)
        form.addRow("ASR 唤醒短语", self.asr_phrases)
        form.addRow("ASR 模型大小", self.asr_model)
        form.addRow("设备", self.device_combo)
        form.addRow("ASR 模型目录", w)
        
        self.chat_font_size = QSpinBox()
        self.chat_font_size.setRange(12, 72)
        self.chat_font_size.setValue(int(self.cfg.get("chat_font_size", 36)))
        form.addRow("会话字体大小(px)", self.chat_font_size)

        self.audio_match_enabled = QComboBox()
        self.audio_match_enabled.addItem("启用", True)
        self.audio_match_enabled.addItem("禁用", False)
        ame = bool(self.cfg.get("audio_match_enabled", True))
        self.audio_match_enabled.setCurrentIndex(0 if ame else 1)
        form.addRow("音频指令/唤醒", self.audio_match_enabled)

        self.audio_match_threshold = QDoubleSpinBox()
        self.audio_match_threshold.setRange(0.05, 1.50)
        self.audio_match_threshold.setSingleStep(0.01)
        try:
            self.audio_match_threshold.setDecimals(2)
        except Exception:
            pass
        self.audio_match_threshold.setValue(float(self.cfg.get("audio_match_threshold", 0.45)))
        form.addRow("音频指令阈值", self.audio_match_threshold)
        self.asr_profile_mode = QComboBox()
        self.asr_profile_mode.addItem("智能（推荐）", "smart")
        self.asr_profile_mode.addItem("安静环境（更灵敏）", "sensitive")
        self.asr_profile_mode.addItem("嘈杂环境（更稳健）", "robust")
        mode = str(self.cfg.get("asr_profile_mode", "smart") or "smart")
        for idx in range(self.asr_profile_mode.count()):
            if self.asr_profile_mode.itemData(idx) == mode:
                self.asr_profile_mode.setCurrentIndex(idx)
                break
        form.addRow("识别灵敏度", self.asr_profile_mode)
        self.asr_profile_tip = QLabel("")
        form.addRow("智能说明", self.asr_profile_tip)
        self.btn_profile_calibrate = QPushButton("按当前环境自动校准")
        form.addRow("", self.btn_profile_calibrate)
        self.profile_override = None
        self.asr_profile_mode.currentIndexChanged.connect(self._on_profile_mode_changed)
        self.btn_profile_calibrate.clicked.connect(self._auto_calibrate_profile)
        self._on_profile_mode_changed(self.asr_profile_mode.currentIndex())

        self.audio_curve = AudioCurveWidget()
        self.audio_curve_status = QLabel("待机")
        self.audio_curve_legend = QLabel("蓝线=输入音量 橙线=触发门限 绿线=噪声底（参数自动调整）")
        form.addRow("音频曲线", self.audio_curve)
        form.addRow("曲线状态", self.audio_curve_status)
        form.addRow("曲线说明", self.audio_curve_legend)


        self.btn_test = QPushButton("测试 LLM")
        self.btn_test.setVisible(True)
        btn_ok = QPushButton("保存")
        btn_cancel = QPushButton("取消")
        self.btn_test.clicked.connect(self._test_llm_connection)
        btn_ok.clicked.connect(self._save)
        btn_cancel.clicked.connect(self.reject)
        btns = QHBoxLayout()
        btns.addWidget(self.btn_test)
        btns.addStretch()
        btns.addWidget(btn_ok)
        btns.addWidget(btn_cancel)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addLayout(btns)
        self.setLayout(layout)

    

    def _save(self):
        """
        保存设置
        
        将界面上的配置写入配置文件和密钥存储，并关闭对话框。
        """
        self.cfg.set("llm_base_url", self.llm_base_url.text())
        self.cfg.set("llm_model", self.llm_model.text())
        self.cfg.set("device_index", self.device_combo.currentIndex())
        self.cfg.set("wake_engine", self.engine_combo.currentData())
        self.cfg.set("asr_wake_phrases", "小石警官")
        self.cfg.set("asr_model_size", self.asr_model.currentData())
        self.cfg.set("asr_model_dir", self.asr_model_dir.text())
        self.cfg.set("chat_font_size", self.chat_font_size.value())
        self.cfg.set("audio_match_enabled", self.audio_match_enabled.currentData())
        self.cfg.set("audio_match_threshold", float(self.audio_match_threshold.value()))
        self.cfg.set("asr_profile_mode", self.asr_profile_mode.currentData())
        profile = self._current_profile()
        self.cfg.set("asr_standby_noise_margin", profile["standby_noise_margin"])
        self.cfg.set("asr_speaking_noise_margin", profile["speaking_noise_margin"])
        self.cfg.set("asr_standby_energy_ratio", profile["standby_energy_ratio"])
        self.cfg.set("asr_speaking_energy_ratio", profile["speaking_energy_ratio"])
        self.cfg.set("asr_interrupt_peak", profile["speaking_interrupt_peak"])
        self.cfg.set("asr_interrupt_rms", profile["speaking_interrupt_rms"])
        self.cfg.save()
        self.secrets.set("llm_api_key", self.llm_api_key.text())
        self.accept()

    def _current_profile(self):
        if self.profile_override:
            return dict(self.profile_override)
        mode = self.asr_profile_mode.currentData()
        base = self._profile_template(mode)
        if mode != "smart":
            return base
        snapshot = self._worker_snapshot()
        if not snapshot:
            return base
        floor = float(snapshot.get("noise_floor", 0.0) or 0.0)
        base["standby_noise_margin"] = self._clamp(max(base["standby_noise_margin"], floor * 1.8), 0.010, 0.080)
        base["speaking_noise_margin"] = self._clamp(max(base["speaking_noise_margin"], floor * 3.2), 0.025, 0.180)
        base["speaking_interrupt_peak"] = self._clamp(max(base["speaking_interrupt_peak"], floor * 6.0), 0.060, 0.220)
        base["speaking_interrupt_rms"] = self._clamp(max(base["speaking_interrupt_rms"], floor * 2.5), 0.008, 0.055)
        return base

    def _on_profile_mode_changed(self, _):
        self.profile_override = None
        mode = self.asr_profile_mode.currentData()
        if mode == "sensitive":
            self.asr_profile_tip.setText("安静环境下更灵敏，更容易听到小声说话。")
        elif mode == "robust":
            self.asr_profile_tip.setText("嘈杂环境下更稳健，减少误触发。")
        else:
            self.asr_profile_tip.setText("根据实时噪声自动调节，通用场景推荐。")
        if not self.app:
            return
        try:
            self.app._on_settings_profile_changed(self._current_profile())
        except Exception:
            pass

    def _auto_calibrate_profile(self):
        mode = self.asr_profile_mode.currentData()
        base = self._profile_template(mode)
        snapshot = self._worker_snapshot()
        if snapshot:
            floor = float(snapshot.get("noise_floor", 0.0) or 0.0)
            gate = float(snapshot.get("gate", 0.0) or 0.0)
            base["standby_noise_margin"] = self._clamp(max(base["standby_noise_margin"], floor * 2.0), 0.010, 0.090)
            base["speaking_noise_margin"] = self._clamp(max(base["speaking_noise_margin"], floor * 3.4), 0.025, 0.200)
            base["speaking_interrupt_peak"] = self._clamp(max(base["speaking_interrupt_peak"], gate * 0.55), 0.060, 0.260)
            base["speaking_interrupt_rms"] = self._clamp(max(base["speaking_interrupt_rms"], floor * 2.8), 0.008, 0.060)
        self.profile_override = dict(base)
        self.asr_profile_tip.setText("已按当前环境自动校准。保存后长期生效。")
        if self.app:
            try:
                self.app._on_settings_profile_changed(self.profile_override)
            except Exception:
                pass

    def _profile_template(self, mode: str):
        if mode == "sensitive":
            return {
                "standby_noise_margin": 0.012,
                "speaking_noise_margin": 0.040,
                "standby_energy_ratio": 1.25,
                "speaking_energy_ratio": 1.85,
                "speaking_interrupt_peak": 0.075,
                "speaking_interrupt_rms": 0.012,
            }
        if mode == "robust":
            return {
                "standby_noise_margin": 0.020,
                "speaking_noise_margin": 0.068,
                "standby_energy_ratio": 1.55,
                "speaking_energy_ratio": 2.55,
                "speaking_interrupt_peak": 0.105,
                "speaking_interrupt_rms": 0.020,
            }
        return {
            "standby_noise_margin": 0.015,
            "speaking_noise_margin": 0.050,
            "standby_energy_ratio": 1.35,
            "speaking_energy_ratio": 2.20,
            "speaking_interrupt_peak": 0.085,
            "speaking_interrupt_rms": 0.015,
        }

    def _worker_snapshot(self):
        if not self.app:
            return None
        worker = getattr(self.app, "worker", None)
        if not worker:
            return None
        try:
            return worker.get_vad_snapshot()
        except Exception:
            return None

    def _clamp(self, value: float, lo: float, hi: float):
        return max(lo, min(hi, float(value)))

    def update_audio_curve(self, volume: float, gate: float, floor: float, speaking: bool):
        self.audio_curve.append_sample(volume, gate, floor, speaking)
        self.audio_curve_status.setText("说话中（强降噪）" if speaking else "待机（高灵敏）")

    def _test_llm_connection(self):
        """
        测试 LLM 连接
        
        尝试使用当前的 Base URL 和 API Key 发起一个简单的 Chat Completion 请求，
        以验证连接是否通畅。
        """
        base_url = self.llm_base_url.text()
        api_key = self.llm_api_key.text()
        model_name = self.llm_model.text()
        if not base_url or not api_key or not model_name:
            QMessageBox.warning(self, "信息不完整", "请填写完整的 LLM Base URL, API Key 和模型。")
            return
        original_text = self.btn_test.text()
        self.btn_test.setEnabled(False)
        self.btn_test.setText("测试中...")
        QApplication.processEvents()
        try:
            client = AsyncOpenAI(base_url=base_url, api_key=api_key)
            async def do_test():
                await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=1,
                )
            asyncio.run(do_test())
            QMessageBox.information(self, "成功", "连接成功！已切换为 LLM 模式")
            self.cfg.set("chat_engine", "llm")
        except Exception as e:
            QMessageBox.warning(self, "失败", f"连接失败: {traceback.format_exc()}")
        finally:
            self.btn_test.setEnabled(True)
            self.btn_test.setText(original_text)

    def _browse_asr_model_dir(self):
        """打开目录选择对话框以选择本地 ASR 模型目录"""
        try:
            d = QFileDialog.getExistingDirectory(self, "选择 ASR 模型目录")
            if d:
                self.asr_model_dir.setText(d)
        except Exception:
            pass


class LogWindow(QDialog):
    """
    日志显示窗口
    
    使用 QPlainTextEdit 显示应用程序运行日志。
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("日志")
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        layout = QVBoxLayout()
        layout.addWidget(self.text)
        self.setLayout(layout)

    def append(self, line: str):
        """向日志窗口追加一行文本"""
        self.text.appendPlainText(line)


class ASRMonitorDialog(QDialog):
    """
    ASR 监听监视窗口
    
    显示当前 ASR 识别引擎的状态和实时识别到的文本。
    包含一个进度条（显示音量或置信度）和一个文本框。
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ASR 监听")
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        lay = QVBoxLayout()
        lay.addWidget(self.bar)
        lay.addWidget(self.text)
        self.setLayout(lay)
    def append(self, line: str):
        """追加识别日志"""
        self.text.appendPlainText(line)
    def set_progress(self, v: int):
        """更新进度条值"""
        try:
            self.bar.setValue(int(max(0, min(100, v))))
        except Exception:
            pass

class ControlWindow(QDialog):
    """
    数字人控制面板 (主窗口)
    
    提供对数字人应用的主要控制功能：
    - 开始/停止监听
    - 一键唤醒
    - 打开设置
    - 查看日志
    - 显示当前状态信息
    """
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.setWindowTitle("数字人控制面板")
        self.setFixedWidth(500) # 固定窗口宽度
        self.lbl_status = QLabel("未运行")
        self.lbl_wake = QLabel("")
        self.lbl_device = QLabel("")
        self.lbl_volume = QLabel("音量: 0%")
        self.volume_bar = QProgressBar()
        self.volume_bar.setRange(0, 100)
        self.volume_bar.setValue(0)
        btn_start = QPushButton("开始监听")
        btn_stop = QPushButton("停止监听")
        btn_close_player = QPushButton("关闭数字人")
        btn_wake = QPushButton("一键唤醒")
        btn_settings = QPushButton("设置")
        btn_logs = QPushButton("日志")
        btn_stop.setEnabled(False)
        btn_start.clicked.connect(self.app.start_listening)
        btn_stop.clicked.connect(self.app.stop_listening)
        btn_close_player.clicked.connect(self.app.close_player)
        btn_wake.clicked.connect(self.app.manual_wake)
        btn_settings.clicked.connect(self.app.open_settings)
        btn_logs.clicked.connect(self.app.show_logs)
        self._btn_start = btn_start
        self._btn_stop = btn_stop
        form = QFormLayout()
        form.addRow("状态", self.lbl_status)
        form.addRow("唤醒词", self.lbl_wake)
        form.addRow("设备", self.lbl_device)
        form.addRow("音量", self.lbl_volume)
        form.addRow("音量条", self.volume_bar)
        row = QHBoxLayout()
        row.addWidget(btn_start)
        row.addWidget(btn_stop)
        row.addWidget(btn_close_player)
        row.addWidget(btn_wake)
        row.addWidget(btn_settings)
        row.addWidget(btn_logs)
        lay = QVBoxLayout()
        lay.addLayout(form)
        lay.addLayout(row)
        self.setLayout(lay)
        self.refresh_info()

    def closeEvent(self, event):
        """
        重写关闭事件
        
        点击关闭按钮时，不退出程序，而是隐藏窗口到系统托盘。
        """
        event.ignore()
        self.hide()
        try:
            self.app.tray.showMessage("运行中", "数字人已最小化到托盘，双击托盘图标可重新打开。", QSystemTrayIcon.Information, 2000)
        except Exception:
            pass

    def refresh_info(self):
        """刷新界面上显示的配置信息（唤醒词、设备等）"""
        eng = self.app.cfg.get("wake_engine", "asr")
        kp = self.app.cfg.get("asr_wake_phrases", "")
        self.lbl_wake.setText("ASR:" + (kp if kp else "未选择"))
        devices = PvRecorder.get_available_devices()
        idx = int(self.app.cfg.get("device_index", -1))
        name = devices[idx] if 0 <= idx < len(devices) else "未选择"
        self.lbl_device.setText(name)

    def set_running(self, running: bool):
        """更新开始/停止按钮的可用状态"""
        self._btn_start.setEnabled(not running)
        self._btn_stop.setEnabled(running)
        self.lbl_status.setText("运行中" if running else "未运行")

    def update_status(self, s: str):
        """更新状态栏文本"""
        self.lbl_status.setText(s)
    
    def update_volume(self, volume: float):
        """更新音量显示"""
        volume_percent = int(volume * 100)
        self.lbl_volume.setText(f"音量: {volume_percent}%")
        self.volume_bar.setValue(volume_percent)


class TrayApp(QApplication):
    """
    主应用程序类
    
    管理整个应用程序的生命周期，包括:
    1. 系统托盘图标和菜单
    2. 子窗口管理 (控制面板, 设置, 日志等)
    3. 后台线程管理 (HTTP Server, WebSocket Server, ASR Worker)
    4. 核心业务逻辑 (唤醒处理, 浏览器进程管理)
    """
    log_requested = Signal(str)
    status_updated = Signal(str)
    asr_text_received = Signal(str)
    close_player_signal = Signal()
    start_listening_signal = Signal()
    stop_listening_signal = Signal()
    ensure_chat_mode_signal = Signal()
    tts_start_signal = Signal()
    tts_end_signal = Signal()
    def __init__(self, argv):
        """
        初始化应用程序
        
        初始化配置、密钥、UI 组件、系统托盘以及后台服务定时器。
        """
        super().__init__(argv)
        self.setQuitOnLastWindowClosed(False)
        self.cfg = ConfigManager()
        self.secrets = SecretStore()
        self.log_win = LogWindow()
        # self.asr_monitor = ASRMonitorDialog() # Removed as requested
        self.ctrl = ControlWindow(self)
        icon_path = get_resource_path("res") / "image" / "logo.ico"
        tray_icon = QIcon(str(icon_path)) if icon_path.exists() else QIcon(QPixmap(16, 16))
        self.tray = QSystemTrayIcon(tray_icon, self)
        menu = QMenu()
        self.act_start = QAction("开始监听")
        self.act_stop = QAction("停止监听")
        self.act_wake = QAction("一键唤醒")
        self.act_settings = QAction("设置")
        self.act_audio_wake_cmd = QAction("管理唤醒指令")
        self.act_audio_interrupt_cmd = QAction("管理打断指令")
        self.act_logs = QAction("日志")
        self.act_exit = QAction("退出")
        self.act_stop.setEnabled(False)
        menu.addAction(self.act_start)
        menu.addAction(self.act_stop)
        menu.addSeparator()
        menu.addAction(self.act_wake)
        menu.addAction(self.act_settings)
        menu.addAction(self.act_audio_wake_cmd)
        menu.addAction(self.act_audio_interrupt_cmd)
        menu.addAction(self.act_logs)
        menu.addSeparator()
        menu.addAction(self.act_exit)
        self.tray.setContextMenu(menu)
        self.tray.setToolTip("数字人唤醒监听")
        self.tray.show()
        try:
            self.ctrl.setWindowIcon(tray_icon)
            self.log_win.setWindowIcon(tray_icon)
            # self.asr_monitor.setWindowIcon(tray_icon) # ASR Monitor removed
        except Exception:
            pass
        try:
            self.tray.showMessage("数字人唤醒", "托盘应用已启动。右键菜单可设置并开始监听。", QSystemTrayIcon.Information, 5000)
        except Exception:
            pass
        try:
            print("托盘应用已启动，请查看系统托盘图标（任务栏右下角）")
        except Exception:
            pass
        
        self.ctrl.show()

        self.act_start.triggered.connect(self.start_listening)
        self.act_stop.triggered.connect(self.stop_listening)
        self.act_wake.triggered.connect(self.manual_wake)
        self.act_settings.triggered.connect(self.open_settings)
        self.act_audio_wake_cmd.triggered.connect(self.open_wake_audio_manager)
        self.act_audio_interrupt_cmd.triggered.connect(self.open_interrupt_audio_manager)
        self.act_logs.triggered.connect(self.show_logs)
        self.act_exit.triggered.connect(self.exit_app)
        self.tray.activated.connect(self._on_tray_activated)
        self.close_player_signal.connect(self.close_player)
        self.start_listening_signal.connect(self.start_listening)
        self.stop_listening_signal.connect(self.stop_listening)
        self.ensure_chat_mode_signal.connect(self._ensure_chat_mode)
        self.tts_start_signal.connect(self._mark_tts_start)
        self.tts_end_signal.connect(self._mark_tts_end)

        self.worker = None
        self.http_thread = None
        self.ws_thread = None
        self.ws_clients = set()
        self.ws_loop = None
        self.browser_pid = None
        self.chat_history = []
        self.pending_intro = None # 待播放的开场白
        self.llm_lock = None
        self.stop_generation = False # Flag to stop LLM generation
        self.is_speaking = False # 是否正在播放 TTS
        self.tts_timeout_ms = 15000
        self.tts_guard_timer = QTimer(self)
        self.tts_guard_timer.setSingleShot(True)
        self.tts_guard_timer.timeout.connect(self._on_tts_timeout)
        self.pending_user_text = None
        self.user_input_debounce_ms = 700
        self.user_input_timer = QTimer(self)
        self.user_input_timer.setSingleShot(True)
        self.user_input_timer.timeout.connect(self._commit_pending_user_input)
        self.last_wake_ts = 0.0
        self.wake_grace_seconds = 8.0
        self.last_tts_start_ts = 0.0
        self.idle_prompt_seconds = 10.0
        self.idle_close_wait_seconds = 3.0
        self._idle_waiting_for_close = False
        self._idle_prompt_sent = False
        self.idle_prompt_timer = QTimer(self)
        self.idle_prompt_timer.setSingleShot(True)
        self.idle_prompt_timer.timeout.connect(self._on_idle_prompt_timeout)
        self.idle_close_timer = QTimer(self)
        self.idle_close_timer.setSingleShot(True)
        self.idle_close_timer.timeout.connect(self._on_idle_close_timeout)
        self.player_active = False
        self._awakened = False
        self.window_manager = WindowContextManager()
        self.tts_cache_dir = get_resource_path("player") / "tts_cache"
        self.tts_lock = threading.Lock()
        self.tts_voice = "zh-CN-liaoning-XiaobeiNeural"
        self.tts_rate = "+10%"
        self.tts_volume = "+0%"
        self.tts_pitch = "+0Hz"
        self.tts_play_proc = None
        self.tts_play_proc_lock = threading.Lock()
        self.tts_mci_alias = None
        self.tts_order_cond = threading.Condition()
        self.tts_submit_index = 0
        self.tts_play_index = 1
        self.tts_cancel_seq = 0
        self.settings_dlg = None
        try:
            self.tts_cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        self.log_requested.connect(self.log_win.append)
        self.status_updated.connect(self.ctrl.update_status)
        # self.asr_text_received.connect(self.asr_monitor.append) # ASR Monitor removed

        self._log_fp = None
        self._init_file_log()
        try:
            sys.excepthook = self._excepthook
        except Exception:
            pass

        try:
            self.start_http()
        except Exception:
            pass
        try:
            self.start_ws()
        except Exception:
            pass

        try:
            self.browser_timer = QTimer(self)
            self.browser_timer.setInterval(1000)
            self.browser_timer.timeout.connect(self._check_browser)
            self.browser_timer.start()
        except Exception:
            pass

        try:
            self.health_timer = QTimer(self)
            self.health_timer.setInterval(5000)
            self.health_timer.timeout.connect(self._health_tick)
            self.health_timer.start()
        except Exception:
            pass
        
        # 启动程序时自动开始监听并最小化，等待语音唤醒
        try:
            def _auto_run():
                self.start_listening()
                self.ctrl.hide()
            QTimer.singleShot(1000, _auto_run)
        except Exception:
            pass

    def _on_tray_activated(self, reason):
        """
        托盘图标激活事件处理
        
        当用户双击托盘图标时，显示并激活控制面板窗口。
        """
        if reason == QSystemTrayIcon.DoubleClick:
            self.ctrl.show()
            self.ctrl.raise_()
            self.ctrl.activateWindow()

    def _open_template_manager(self, title: str, template_dir: str):
        device_index = int(self.cfg.get("device_index", -1))
        dlg = TemplateManageDialog(None, device_index=device_index, template_dir=template_dir, scene_name=title)
        dlg.exec()
        if self.worker:
            self.worker.reload_audio_templates()

    def open_wake_audio_manager(self):
        self._open_template_manager("唤醒指令", "templates/wake")

    def open_interrupt_audio_manager(self):
        self._open_template_manager("打断指令", "templates/interrupt")

    def show_logs(self):
        """显示日志窗口"""
        self.log_win.show()
        self.log_win.raise_()

    def open_settings(self):
        """打开设置对话框，并在保存后刷新主界面信息"""
        try:
            dlg = SettingsDialog(self.cfg, self.secrets, app=self)
            self.settings_dlg = dlg
            try:
                self._on_worker_volume(float(self.ctrl.volume_bar.value()) / 100.0)
            except Exception:
                pass
            if dlg.exec() == QDialog.Accepted:
                try:
                    self.ctrl.refresh_info()
                except Exception:
                    pass
                try:
                    self._apply_asr_profile_to_worker()
                except Exception:
                    pass
        except Exception as e:
            QMessageBox.critical(None, "错误", f"无法打开设置窗口: {str(e)}\n{traceback.format_exc()}")
            self._on_error(f"打开设置窗口失败: {str(e)}")
        finally:
            self.settings_dlg = None

    def _get_asr_profile_from_config(self):
        return {
            "standby_noise_margin": float(self.cfg.get("asr_standby_noise_margin", 0.015)),
            "speaking_noise_margin": float(self.cfg.get("asr_speaking_noise_margin", 0.05)),
            "standby_energy_ratio": float(self.cfg.get("asr_standby_energy_ratio", 1.35)),
            "speaking_energy_ratio": float(self.cfg.get("asr_speaking_energy_ratio", 2.2)),
            "speaking_interrupt_peak": float(self.cfg.get("asr_interrupt_peak", 0.085)),
            "speaking_interrupt_rms": float(self.cfg.get("asr_interrupt_rms", 0.015)),
        }

    def _apply_asr_profile_to_worker(self):
        if not self.worker:
            return
        self.worker.apply_dynamic_profile(self._get_asr_profile_from_config())

    def _on_settings_profile_changed(self, profile: dict):
        if self.worker:
            try:
                self.worker.apply_dynamic_profile(profile)
            except Exception:
                pass
        try:
            self._on_worker_volume(float(self.ctrl.volume_bar.value()) / 100.0)
        except Exception:
            pass

    def _on_worker_volume(self, volume: float):
        self.ctrl.update_volume(volume)
        dlg = self.settings_dlg
        if not dlg:
            return
        if not dlg.isVisible():
            return
        noise_floor = 0.0
        gate = 0.0
        speaking = bool(self.is_speaking)
        if self.worker:
            try:
                snap = self.worker.get_vad_snapshot()
                noise_floor = float(snap.get("noise_floor", 0.0))
                gate = float(snap.get("gate", 0.0))
                speaking = bool(snap.get("speaking", speaking))
            except Exception:
                pass
        dlg.update_audio_curve(volume, gate, noise_floor, speaking)

    def start_listening(self):
        """
        开始监听
        
        1. 获取并配置音频设备
        2. 创建并启动 AudioWakeWorkerASR 线程
        3. 更新 UI 状态
        """
        # 每次启动监听时清理上一轮会话
        self.chat_history = []
        self.ws_broadcast({"type": "CHAT_CLEAR"})

        # 如果 Worker 已经存在且正在运行，说明是热启动（恢复），直接 resume 即可
        if self.worker and getattr(self.worker, '_running', False):
            self.worker.resume()
            self._on_log("恢复监听 (Hot Resume)")
            self.act_start.setEnabled(False)
            self.act_stop.setEnabled(True)
            try:
                self.ctrl.set_running(True)
            except Exception:
                pass
            return

        devices = PvRecorder.get_available_devices()
        if not devices:
            try:
                QMessageBox.warning(None, "麦克风不可用", "未检测到音频设备")
            except Exception:
                pass
            return
        device_index = int(self.cfg.get("device_index", -1))
        if device_index < 0 or device_index >= len(devices):
            device_index = 0
            try:
                self.cfg.set("device_index", device_index)
                self.cfg.save()
            except Exception:
                pass
        phrases = self.cfg.get("asr_wake_phrases", "")
        api_key = "sk-28313ea70a8f47d09a6cd1cab51c477e"  # 使用用户提供的API密钥
        audio_match_enabled = bool(self.cfg.get("audio_match_enabled", True))
        audio_match_threshold = float(self.cfg.get("audio_match_threshold", 0.45))
        self.worker = AudioWakeWorkerASR(phrases, device_index, api_key, audio_match_enabled=audio_match_enabled, audio_match_threshold=audio_match_threshold)
        self.worker.wake_detected.connect(self.manual_wake)
        self.worker.interrupt_detected.connect(self.handle_audio_interrupt_command)
        self.worker.chat_input_detected.connect(self.handle_backend_asr)
        self.worker.error.connect(self._on_error)
        self.worker.log.connect(self._on_log)
        self.worker.log.connect(self._on_asr_log)
        self.worker.error.connect(self._on_asr_error)
        self.worker.volume.connect(self._on_worker_volume)
        self._apply_asr_profile_to_worker()
        self.worker.reload_audio_templates()
        self.worker.set_speaking_state(self.is_speaking)
        self.worker.start()
        self.act_start.setEnabled(False)
        self.act_stop.setEnabled(True)
        self._on_log("开始监听")
        try:
            self.ctrl.set_running(True)
        except Exception:
            pass
        try:
            self._awakened = False
            # self.asr_monitor.show() # Removed
            # self.asr_monitor.raise_() # Removed
            try:
                # self.asr_monitor.append("初始化识别引擎...") # Removed
                self._on_log("正在初始化识别引擎...")
            except Exception:
                pass
        except Exception:
            pass

    def _ensure_chat_mode(self):
        if self.worker:
            self.worker.set_mode("chat")
            self.worker.resume()
            self._on_log("ASR 切换到 chat 模式")

    def _on_tts_timeout(self):
        if self.is_speaking:
            self._on_log("TTS 状态超时重置")
            self.is_speaking = False
            try:
                if self.worker:
                    self.worker.set_speaking_state(False)
            except Exception:
                pass

    def _mark_tts_start(self):
        self.is_speaking = True
        self.last_tts_start_ts = time.monotonic()
        self._cancel_idle_timers()
        try:
            if self.worker:
                self.worker.set_speaking_state(True)
        except Exception:
            pass
        try:
            self.tts_guard_timer.start(self.tts_timeout_ms)
        except Exception:
            pass

    def _mark_tts_end(self):
        self.is_speaking = False
        self.last_tts_start_ts = 0.0
        try:
            if self.worker:
                self.worker.set_speaking_state(False)
        except Exception:
            pass
        try:
            self.tts_guard_timer.stop()
        except Exception:
            pass
        self._schedule_idle_prompt_if_possible()

    def stop_listening(self):
        """
        停止监听
        
        停止 ASR 工作线程，清理资源，并更新 UI 状态。
        """
        # 禁用按钮防止重复点击
        self.act_start.setEnabled(False)
        self.act_stop.setEnabled(False)
        try:
            self._clear_pending_user_input()
        except Exception:
            pass
        
        if self.worker:
            try:
                # 这现在会阻塞直到线程完全结束，不再会有僵尸线程
                self.worker.stop()
            except Exception as e:
                print(f"Stop error: {e}")
        
        self.worker = None
        
        # 恢复按钮状态
        self.act_start.setEnabled(True)
        self.act_stop.setEnabled(False)
        
        self._on_log("监听已停止")
        try:
            self.ctrl.set_running(False)
        except: pass
        self._awakened = False

    def _clear_pending_user_input(self):
        self.pending_user_text = None
        try:
            self.user_input_timer.stop()
        except Exception:
            pass

    def _merge_user_text(self, old_text: str, new_text: str) -> str:
        old_text = (old_text or "").strip()
        new_text = (new_text or "").strip()
        if not old_text:
            return new_text
        if not new_text:
            return old_text
        if new_text.startswith(old_text) or old_text in new_text:
            return new_text
        if new_text in old_text:
            return old_text
        joiner = ""
        try:
            if old_text[-1].isascii() or new_text[0].isascii():
                joiner = " "
        except Exception:
            joiner = " "
        return old_text + joiner + new_text

    def _queue_user_input(self, text: str):
        clean_text = "".join(c for c in (text or "") if c.isalnum() or '\u4e00' <= c <= '\u9fff')
        if len(clean_text) < 2:
            return
        self._note_activity()
        self.pending_user_text = self._merge_user_text(self.pending_user_text, text)
        try:
            self.user_input_timer.start(int(self.user_input_debounce_ms))
        except Exception:
            self._commit_pending_user_input()

    def _heuristic_should_ignore_user_input(self, text: str) -> bool:
        raw = (text or "").strip()
        if not raw:
            return True
        clean = "".join(c for c in raw if c.isalnum() or '\u4e00' <= c <= '\u9fff')
        trivial = {
            "嗯", "啊", "哦", "哈", "哈哈", "呵呵", "好的", "行", "可以", "对", "是", "不是",
            "知道", "明白", "没事", "谢谢", "谢谢你", "再见", "拜拜"
        }
        if clean in trivial:
            return True
        if any(x in raw for x in ["？", "?", "请问", "麻烦", "帮我", "咨询", "怎么", "如何", "为什么", "咋", "怎么办"]):
            return False
        if len(clean) < 4:
            return True
        key_hits = 0
        for k in ["小石", "警官", "报警", "派出所", "身份证", "户口", "居住证", "驾驶证", "驾照", "车辆", "违章", "诈骗", "被骗", "转账", "银行卡", "微信", "法律", "违法", "立案", "报案"]:
            if k in raw:
                key_hits += 1
                break
        if key_hits > 0:
            return False
        if len(clean) < 10:
            return True
        return False

    async def _llm_gate_user_input(self, text: str) -> str:
        base_url = self.cfg.get("llm_base_url", "")
        api_key = self.secrets.get("llm_api_key", "")
        model = self.cfg.get("llm_model", "")
        if not base_url or not api_key or not model:
            return "__OK__"
        try:
            client = AsyncOpenAI(base_url=base_url, api_key=api_key)
            system_prompt = (
                "你是语音输入过滤器。你只能输出两种结果之一：__IGNORE__ 或 __OK__。\n"
                "当输入明显是环境里别人聊天的片段、无对象的碎句、口头禅/语气词、无法形成提问或求助的内容时，输出 __IGNORE__。\n"
                "当输入是在向“小石警官/警官”提问、求助、咨询、办理业务、报警报案，或包含明确问题/指令时，输出 __OK__。\n"
                "不要输出任何其他字符。"
            )
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                max_tokens=4,
                temperature=0,
                stream=False,
            )
            out = ""
            try:
                out = (resp.choices[0].message.content or "").strip()
            except Exception:
                out = ""
            if out.startswith("__IGNORE__"):
                return "__IGNORE__"
            if out.startswith("__OK__"):
                return "__OK__"
        except Exception:
            return "__OK__"
        return "__OK__"

    async def _process_committed_user_input(self, text: str):
        if self._heuristic_should_ignore_user_input(text):
            return
        verdict = await self._llm_gate_user_input(text)
        if verdict == "__IGNORE__":
            return
        await self.ask_llm(text)

    def _commit_pending_user_input(self):
        text = (self.pending_user_text or "").strip()
        self.pending_user_text = None
        clean_text = "".join(c for c in text if c.isalnum() or '\u4e00' <= c <= '\u9fff')
        if len(clean_text) < 2:
            return
        if hasattr(self, 'ws_loop') and self.ws_loop and self.ws_loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(self._process_committed_user_input(text), self.ws_loop)
            except Exception:
                pass
        else:
            self._on_log("Error: WS Loop not running, cannot ask LLM")

    def exit_app(self):
        """
        完全安全的退出逻辑
        
        1. 停止监听
        2. 关闭所有窗口
        3. 退出应用程序
        """
        # 1. 先停止监听
        self.stop_listening()
        
        # 2. 停止 http 和 ws 线程 (虽然它们是 daemon，但显式清理更好)
        # 这里主要依靠 Daemon 属性由 Python 自动回收，
        # 但我们要确保 audio worker 必须死透。
        
        # 3. 关闭所有窗口
        self.ctrl.close()
        self.log_win.close()
        # self.asr_monitor.close()
        
        # 4. 退出
        self.quit()

    def handle_audio_interrupt_command(self, command_name: str):
        self._note_activity()
        self._on_log(f"Audio interrupt command: {command_name}")
        self.stop_generation = True
        try:
            self.tts_cancel_seq += 1
        except Exception:
            pass
        try:
            self.ws_broadcast({"type": "STOP_GENERATION"})
        except Exception:
            pass
        try:
            self._stop_backend_tts_playback()
        except Exception:
            pass
        try:
            self.tts_end_signal.emit()
        except Exception:
            pass
        try:
            self._force_release_speaking_state()
        except Exception:
            pass
        try:
            with self.tts_order_cond:
                self.tts_play_index = self.tts_submit_index + 1
                self.tts_order_cond.notify_all()
        except Exception:
            pass

    def handle_backend_asr(self, text: str):
        """处理后台 ASR 识别到的对话内容"""
        self._note_activity()
        self._on_log(f"Backend ASR: {text}")

        # 如果没有 WebSocket 客户端连接，说明前端可能已关闭
        # 此时如果在 chat 模式收到输入，应检查是否是唤醒词，或者强制切回 wake 模式
        if not self.ws_clients:
            self._on_log("Warn: Chat input received but no clients connected. Checking for wake word...")
            
            # 尝试作为唤醒词处理
            norm_text = text
            if self.worker:
                norm_text = self.worker._normalize(text)
            
            hit = False
            # 尝试获取 worker 的 phrases_norm，如果获取不到则重新计算
            phrases_norm = []
            if self.worker and hasattr(self.worker, 'phrases_norm'):
                phrases_norm = self.worker.phrases_norm
            else:
                phrases = [p.strip() for p in (self.cfg.get("asr_wake_phrases", "") or "").split(",") if p.strip()]
                # 简单的 fallback normalize
                phrases_norm = [p for p in phrases] 

            for p in phrases_norm:
                if p and p in norm_text:
                    hit = True
                    break
            
            if hit:
                self._on_log("Wake word detected in chat mode (disconnected). Waking up...")
                self.manual_wake()
                return
            
            # 如果不是唤醒词，且无客户端，切换回 wake 模式以免后续一直 stuck
            self._on_log("No clients and not wake word. Switching back to wake mode.")
            if self.worker:
                self.worker.set_mode("wake")
            return

        try:
            if text and (not self.is_speaking):
                self.ws_broadcast({"type": "USER_INPUT", "text": text})
        except Exception:
            pass
        
        # --- 核心逻辑：如果在说话，只允许打断，不允许新对话 ---
        if self.is_speaking:
            clean_text = "".join(c for c in text if c.isalnum() or '\u4e00' <= c <= '\u9fff')
            interrupt_keywords = ["停", "停下", "别说", "闭嘴", "安静", "打断", "不听","等等","等会","我打断你一下","好啦","别说了"]
            speaking_stale = self._is_speaking_stale()
            peak = 0.0
            try:
                if self.worker:
                    peak = float(getattr(self.worker, "last_asr_peak", 0.0) or 0.0)
            except Exception:
                peak = 0.0
            loud_interrupt = False
            try:
                if self.worker:
                    loud_interrupt = peak >= float(getattr(self.worker, "speaking_interrupt_peak", 0.085) or 0.085)
            except Exception:
                loud_interrupt = False
            force_interrupt_after_wake = self._in_wake_grace() and len(clean_text) >= 2
            if any(k in clean_text for k in interrupt_keywords) or loud_interrupt or force_interrupt_after_wake or speaking_stale:
                if speaking_stale:
                    self._on_log("Detected stale speaking state, force reset")
                    self._force_release_speaking_state()
                self._on_log(f"Detected interrupt keyword while speaking: {text}")
                self.stop_generation = True
                try:
                    self.ws_broadcast({"type": "STOP_GENERATION"})
                except Exception:
                    pass
                if text and len(clean_text) >= 2:
                    self._queue_user_input(text)
            else:
                self._on_log(f"Ignored input while speaking: {text}")
            
            return
        # ---------------------------------------------------
        self._queue_user_input(text)

    def _in_wake_grace(self) -> bool:
        if self.last_wake_ts <= 0:
            return False
        return (time.monotonic() - self.last_wake_ts) <= float(self.wake_grace_seconds)

    def _is_speaking_stale(self) -> bool:
        if not self.is_speaking:
            return False
        if self.last_tts_start_ts <= 0:
            return True
        return (time.monotonic() - self.last_tts_start_ts) > 6.0

    def _force_release_speaking_state(self):
        self.is_speaking = False
        self.last_tts_start_ts = 0.0
        try:
            self.tts_guard_timer.stop()
        except Exception:
            pass
        try:
            if self.worker:
                self.worker.set_speaking_state(False)
        except Exception:
            pass

    def _note_activity(self):
        self._idle_waiting_for_close = False
        self._idle_prompt_sent = False
        self._cancel_idle_timers()

    def _cancel_idle_timers(self):
        try:
            self.idle_prompt_timer.stop()
        except Exception:
            pass
        try:
            self.idle_close_timer.stop()
        except Exception:
            pass

    def _schedule_idle_prompt_if_possible(self):
        if self.is_speaking:
            return
        if not self.ws_clients:
            return
        if not self.worker or getattr(self.worker, "_mode", "wake") != "chat":
            return
        if self._idle_waiting_for_close:
            return
        if self._idle_prompt_sent:
            try:
                self.idle_close_timer.start(int(float(self.idle_close_wait_seconds) * 1000))
                self._idle_waiting_for_close = True
            except Exception:
                pass
            return
        try:
            self.idle_prompt_timer.start(int(float(self.idle_prompt_seconds) * 1000))
        except Exception:
            pass

    def _on_idle_prompt_timeout(self):
        if self.is_speaking or not self.ws_clients:
            self._schedule_idle_prompt_if_possible()
            return
        if not self.worker or getattr(self.worker, "_mode", "wake") != "chat":
            return
        text = "请问还有其他可以帮您的吗？"
        self._idle_prompt_sent = True
        try:
            self.chat_history.append({"role": "assistant", "content": text})
        except Exception:
            pass
        try:
            self.ws_broadcast({"type": "IDLE_PROMPT", "text": text})
        except Exception:
            pass

    def _on_idle_close_timeout(self):
        if self.is_speaking:
            self._idle_waiting_for_close = False
            self._schedule_idle_prompt_if_possible()
            return
        if not self.ws_clients:
            self._idle_waiting_for_close = False
            return
        if not self.worker or getattr(self.worker, "_mode", "wake") != "chat":
            self._idle_waiting_for_close = False
            return
        self._idle_waiting_for_close = False
        self._idle_prompt_sent = False
        try:
            self.close_player_signal.emit()
        except Exception:
            pass

    def _on_wake(self):
        """
        唤醒回调
        
        当检测到唤醒词时触发：
        1. 启动数字人播放器 (如果未启动)
        2. 广播 "唤醒成功" 消息
        3. 停止监听 (避免自己对话触发自己)
        """
        self._on_log("唤醒命中")
        try:
            if not self.player_active:
                self.launch_player()
                self.ws_broadcast({"type": "CHAT_APPEND", "role": "assistant", "text": "唤醒成功"})
            try:
                self.stop_listening()
            except Exception:
                pass
        except Exception as e:
            self._on_error(str(e))

    def _on_error(self, msg: str):
        """处理错误信息：记录日志"""
        self._on_log("错误: " + msg)

    def _on_log(self, msg: str):
        """
        统一日志处理
        
        将日志发送到：
        1. 日志窗口 (LogWindow)
        2. 状态栏 (ControlWindow)
        3. 日志文件 (pySiberMan.log)
        """
        self.log_requested.emit(msg)
        self.status_updated.emit(msg)
        try:
            self._file_log(msg)
        except Exception:
            pass

    def _on_asr_log(self, msg: str):
        """
        处理 ASR 引擎日志
        
        1. 忽略 asr_progress
        2. 解析 asr: TEXT 识别内容
           - 记录到日志
           - 进行文本标准化
           - 检查唤醒词是否命中
        3. 处理引擎状态消息 (ready, listening started) - 记录到日志
        """
        try:
            if msg.startswith("asr_progress:"):
                # 忽略进度条更新
                return
            if msg.startswith("asr: "):
                t = msg[5:].strip()
                tn = self._normalize_text(t)
                
                # 记录识别内容到日志
                self._on_log(f"ASR识别: {t}")
                
                if not self._awakened:
                    phrases = [p.strip() for p in (self.cfg.get("asr_wake_phrases", "") or "").split(",") if p.strip()]
                    if any(tn == self._normalize_text(p) for p in phrases):
                        self._awakened = True
                        # self.manual_wake()
                if t:
                    try:
                        if getattr(self, 'ws_loop', None) and len(self.ws_clients) > 0:
                            import asyncio as _asyncio
                            # _asyncio.run_coroutine_threadsafe(self.ask_llm(t), self.ws_loop)
                    except Exception:
                        pass
            else:
                if msg == "asr engine ready":
                    self._on_log("ASR引擎就绪")
                elif msg == "wake listening started":
                    self._on_log("开始监听唤醒词...")
                elif msg == "wake listening stopped":
                    self._on_log("监听线程已停止")
                else:
                    self._on_log(f"ASR消息: {msg}")
        except Exception:
            pass

    def _on_asr_error(self, msg: str):
        """处理 ASR 错误：记录日志"""
        try:
            self._on_log("ASR错误: " + msg)
        except Exception:
            pass

    def _on_external_asr_text(self, text: str):
        """
        处理外部传入的 ASR 文本 (例如来自浏览器)
        
        用于支持浏览器端的语音识别结果也能触发唤醒。
        """
        try:
            self.asr_text_received.emit(text)
            self._on_log(f"浏览器识别: {text}") # Log external text
            if not self._awakened:
                phrases = [p.strip() for p in (self.cfg.get("asr_wake_phrases", "") or "").split(",") if p.strip()]
                if any(text == p for p in phrases):
                    self._awakened = True
                    QTimer.singleShot(0, self.manual_wake)
        except Exception:
            pass

    def ensure_port_free(self, port: int) -> bool:
        """检查端口是否被占用，如果未被占用则返回 True"""
        s = socket.socket()
        try:
            s.bind(("127.0.0.1", port))
            s.close()
            return True
        except Exception:
            return False

    def start_http(self):
        """
        启动内置 HTTP 服务器
        
        在 3400 端口提供静态文件服务，主要用于托管数字人前端页面 (player/index.html)。
        """
        if self.http_thread and self.http_thread.is_alive():
            return
        def run():
            from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
            try:
                # Use helper to find where 'player' folder is, then serve its parent directory
                player_path = get_resource_path("player")
                root = player_path.parent
                os.chdir(str(root))
            except Exception:
                pass
            srv = ThreadingHTTPServer(("127.0.0.1", 3400), SimpleHTTPRequestHandler)
            srv.serve_forever()
        self.http_thread = threading.Thread(target=run, daemon=True)
        self.http_thread.start()

    async def _edge_tts_save_async(self, text: str, out_mp3: Path) -> bool:
        communicate = edge_tts.Communicate(
            text=text,
            voice=self.tts_voice,
            rate=self.tts_rate,
            volume=self.tts_volume,
            pitch=self.tts_pitch
        )
        await communicate.save(str(out_mp3))
        return out_mp3.exists() and out_mp3.stat().st_size > 256

    def _synthesize_tts_edge_file(self, text: str) -> Optional[Path]:
        try:
            t = (text or "").strip()
            if not t:
                return None
            t = t[:220]
            key = hashlib.md5(t.encode("utf-8")).hexdigest()[:16]
            mp3_path = self.tts_cache_dir / f"tts_{key}.mp3"
            if mp3_path.exists() and mp3_path.stat().st_size > 256:
                return mp3_path
            with self.tts_lock:
                if mp3_path.exists() and mp3_path.stat().st_size > 256:
                    return mp3_path
                asyncio.run(self._edge_tts_save_async(t, mp3_path))
            if mp3_path.exists() and mp3_path.stat().st_size > 256:
                return mp3_path
            return None
        except Exception as e:
            try:
                self._on_log(f"EdgeTTS synth error: {e}")
            except Exception:
                pass
            return None

    def _stop_backend_tts_playback(self):
        try:
            alias = None
            with self.tts_play_proc_lock:
                alias = self.tts_mci_alias
                self.tts_mci_alias = None
            if alias:
                try:
                    ctypes.windll.winmm.mciSendStringW(f"stop {alias}", None, 0, 0)
                except Exception:
                    pass
                try:
                    ctypes.windll.winmm.mciSendStringW(f"close {alias}", None, 0, 0)
                except Exception:
                    pass
        except Exception:
            pass
        try:
            with self.tts_play_proc_lock:
                p = self.tts_play_proc
                self.tts_play_proc = None
            if p and p.poll() is None:
                p.terminate()
        except Exception:
            pass

    def _mci_send(self, command: str) -> tuple[int, str]:
        try:
            buf = ctypes.create_unicode_buffer(260)
            rc = ctypes.windll.winmm.mciSendStringW(command, buf, len(buf), 0)
            if rc != 0:
                err_buf = ctypes.create_unicode_buffer(260)
                ctypes.windll.winmm.mciGetErrorStringW(rc, err_buf, len(err_buf))
                return rc, err_buf.value
            return 0, buf.value
        except Exception as e:
            return -1, str(e)

    def _play_tts_mp3_backend(self, mp3_path: Path, cancel_seq: int) -> bool:
        try:
            if not mp3_path or (not mp3_path.exists()):
                return False
            alias = f"tts_{int(time.time() * 1000)}"
            safe_path = str(mp3_path).replace('"', '""')
            rc, msg = self._mci_send(f'open "{safe_path}" type mpegvideo alias {alias}')
            if rc == 0:
                with self.tts_play_proc_lock:
                    self.tts_mci_alias = alias
                rc, msg = self._mci_send(f"play {alias}")
                if rc == 0:
                    while True:
                        if cancel_seq != self.tts_cancel_seq:
                            self._mci_send(f"stop {alias}")
                            break
                        rc, mode = self._mci_send(f"status {alias} mode")
                        if rc != 0 or (mode or "").strip().lower() != "playing":
                            break
                        time.sleep(0.03)
                    self._mci_send(f"close {alias}")
                    with self.tts_play_proc_lock:
                        if self.tts_mci_alias == alias:
                            self.tts_mci_alias = None
                    return True
                self._mci_send(f"close {alias}")
                with self.tts_play_proc_lock:
                    if self.tts_mci_alias == alias:
                        self.tts_mci_alias = None
            try:
                self._on_log(f"MCI 播放失败，改用 PowerShell: {msg}")
            except Exception:
                pass
            ps_path = str(mp3_path).replace("'", "''")
            script = (
                "Add-Type -AssemblyName presentationCore;"
                "$p = New-Object System.Windows.Media.MediaPlayer;"
                "$script:ended = $false;"
                "$p.MediaEnded += { $script:ended = $true };"
                "$p.MediaFailed += { $script:ended = $true };"
                f"$p.Open([Uri]('{ps_path}'));"
                "$p.Play();"
                "while (-not $script:ended) { Start-Sleep -Milliseconds 40 };"
                "$p.Stop();"
                "$p.Close();"
            )
            p = subprocess.Popen(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            with self.tts_play_proc_lock:
                self.tts_play_proc = p
            while True:
                if cancel_seq != self.tts_cancel_seq:
                    try:
                        p.terminate()
                    except Exception:
                        pass
                    break
                rc = p.poll()
                if rc is not None:
                    break
                time.sleep(0.02)
            with self.tts_play_proc_lock:
                if self.tts_play_proc is p:
                    self.tts_play_proc = None
            return True
        except Exception as e:
            try:
                self._on_log(f"Backend play error: {e}")
            except Exception:
                pass
            return False

    def _handle_tts_request(self, order_index: int, req_id: str, tts_text: str, cancel_seq: int):
        should_advance = False
        try:
            text = (tts_text or "").strip()
            if not text:
                self.ws_broadcast({"type": "TTS_AUDIO", "req_id": req_id, "url": "", "text": "", "backend_played": False})
                should_advance = True
                return
            mp3_path = self._synthesize_tts_edge_file(text)
            if cancel_seq != self.tts_cancel_seq:
                self.ws_broadcast({"type": "TTS_AUDIO", "req_id": req_id, "url": "", "text": text, "backend_played": False})
                should_advance = True
                return
            with self.tts_order_cond:
                while order_index != self.tts_play_index:
                    if cancel_seq != self.tts_cancel_seq:
                        self.ws_broadcast({"type": "TTS_AUDIO", "req_id": req_id, "url": "", "text": text, "backend_played": False})
                        should_advance = True
                        return
                    self.tts_order_cond.wait(0.03)
            if not mp3_path:
                self.ws_broadcast({"type": "TTS_AUDIO", "req_id": req_id, "url": "", "text": text, "backend_played": False})
                should_advance = True
                return
            if cancel_seq != self.tts_cancel_seq:
                self.ws_broadcast({"type": "TTS_AUDIO", "req_id": req_id, "url": "", "text": text, "backend_played": False})
                should_advance = True
                return
            self.tts_start_signal.emit()
            self.ws_broadcast({"type": "TTS_PLAYBACK_START", "req_id": req_id, "text": text})
            played = self._play_tts_mp3_backend(mp3_path, cancel_seq)
            self.tts_end_signal.emit()
            self.ws_broadcast({"type": "TTS_PLAYBACK_END", "req_id": req_id, "text": text})
            self.ws_broadcast({"type": "TTS_AUDIO", "req_id": req_id, "url": "", "text": text, "backend_played": bool(played)})
            should_advance = True
        except Exception as e:
            try:
                self._on_log(f"TTS request error: {e}")
            except Exception:
                pass
            try:
                self.tts_end_signal.emit()
            except Exception:
                pass
            self.ws_broadcast({"type": "TTS_AUDIO", "req_id": req_id, "url": "", "text": tts_text or "", "backend_played": False})
            should_advance = True
        finally:
            if should_advance:
                with self.tts_order_cond:
                    if order_index >= self.tts_play_index:
                        self.tts_play_index = order_index + 1
                    self.tts_order_cond.notify_all()

    async def ws_handler(self, websocket, path=None):
        """
        WebSocket 连接处理器
        
        处理与前端页面的实时通信：
        1. 发送聊天历史记录
        2. 接收前端消息 (ASR 结果, 状态控制等)
        3. 转发 ASR 结果给 LLM
        """
        self.ws_clients.add(websocket)
        try:
            # 新客户端连接，自动切换到 chat 模式
            if self.worker:
                self.worker.set_mode("chat")
                # 确保处于监听状态 (resume 可能会在其他地方被调用，但这里调用是安全的)
                self.worker.resume()

            # 检查是否有待播放的开场白
            intro_msg = None
            if self.pending_intro:
                intro_msg = self.pending_intro
                self.pending_intro = None # 消费掉
                self._on_log(f"New client connected, found pending intro: {intro_msg[:10]}...")
            else:
                self._on_log("New client connected, no pending intro.")

            # 1. 先同步历史记录（全部静默显示）
            for msg in self.chat_history:
                if msg.get("role") != "system":
                    await websocket.send(json.dumps({
                        "type": "CHAT_HISTORY_ITEM",
                        "role": msg["role"],
                        "text": msg["content"],
                    }))

            # 2. 如果有 pending_intro，作为新消息发送（触发 TTS），并追加到历史
            if intro_msg:
                # 追加到历史
                self.chat_history.append({"role": "assistant", "content": intro_msg})
                # 发送给前端播放
                await websocket.send(json.dumps({
                    "type": "CHAT_APPEND",
                    "role": "assistant",
                    "text": intro_msg,
                }))
            async for message in websocket:
                try:
                    data = json.loads(message)
                    try:
                        self._file_log("ws recv: " + str(data.get("type")))
                    except Exception:
                        pass
                    if data.get("type") == "ASR_RESULT" and data.get("text"):
                        try:
                            self.ws_broadcast({"type": "ASR_TEXT", "text": data["text"]})
                        except Exception:
                            pass
                        try:
                            self._on_external_asr_text(data["text"])
                        except Exception:
                            pass
                        try:
                            QTimer.singleShot(0, lambda t=data["text"]: self.handle_backend_asr(t))
                        except Exception:
                            pass
                    elif data.get("type") == "ASR_START_LOCAL":
                        try:
                            self.start_listening_signal.emit()
                            self.ensure_chat_mode_signal.emit()
                        except Exception as e:
                            self._on_error(str(e))
                    elif data.get("type") == "ASR_STOP_LOCAL":
                        try:
                            self.stop_listening_signal.emit()
                        except Exception:
                            pass
                    elif data.get("type") == "CLOSE_PLAYER":
                        self._on_log("收到前端关闭信号 (CLOSE_PLAYER)")
                        try:
                            self.close_player_signal.emit()
                        except Exception as e:
                            self._on_error(f"触发关闭失败: {e}")
                    elif data.get("type") == "STOP_GENERATION":
                        self._on_log("收到打断信号，停止生成")
                        self.stop_generation = True
                        try:
                            self.tts_cancel_seq += 1
                        except Exception:
                            pass
                        self._stop_backend_tts_playback()
                        self.tts_end_signal.emit()
                        with self.tts_order_cond:
                            self.tts_play_index = self.tts_submit_index + 1
                            self.tts_order_cond.notify_all()
                    elif data.get("type") == "LOG":
                        # 处理前端发来的日志消息
                        log_text = data.get("text", "")
                        if log_text:
                            self._on_log(f"前端日志: {log_text}")
                    elif data.get("type") == "TTS_START":
                        self.tts_start_signal.emit()
                    elif data.get("type") == "TTS_END":
                        self.tts_end_signal.emit()
                    elif data.get("type") == "TTS_SYNTH":
                        req_id = data.get("req_id")
                        tts_text = data.get("text", "")
                        with self.tts_order_cond:
                            self.tts_submit_index += 1
                            order_index = self.tts_submit_index
                        cancel_seq = int(getattr(self, "tts_cancel_seq", 0) or 0)
                        asyncio.create_task(asyncio.to_thread(self._handle_tts_request, order_index, req_id, tts_text, cancel_seq))
                except Exception as e:
                    self._on_error(f"Error processing ws message: {e}")
        finally:
            try:
                self.ws_clients.remove(websocket)
            except Exception:
                pass
            
            # 如果所有客户端都断开，自动切换回唤醒模式
            if not self.ws_clients:
                self._on_log("All clients disconnected, switching to wake mode")
                if self.worker:
                    self.worker.set_mode("wake")
                
                try:
                    self.window_manager.restore_context()
                except Exception as e:
                    self._on_log(f"Context restore failed: {e}")

    async def ask_llm(self, text: str, hidden_input: bool = False):
        """
        请求 LLM
        
        Args:
            text: 用户输入的文本
            hidden_input: 是否隐藏输入 (不显示在聊天历史中)，通常用于系统指令
            
        流程:
        1. 组装系统提示词与对话上下文
        2. 发送请求并流式返回
        3. 更新聊天历史并广播给前端
        """
        if not self.llm_lock:
            return
        async with self.llm_lock:
            self.stop_generation = False # Reset flag
            try:
                self._file_log(f"llm ask (hidden={hidden_input})")
            except Exception:
                pass
            
            # --- 语音唤醒特殊处理 ---
            # 如果是简短的唤醒词，直接回复固定开场白，不消耗 LLM
            clean_text = "".join(c for c in text if c.isalnum() or '\u4e00' <= c <= '\u9fff')
            wake_keywords = ["小石警官", "小石", "警官"]
            is_wake_command = False
            # 如果包含唤醒词且长度较短（比如小于10个字），视为纯唤醒指令
            if len(clean_text) < 10 and any(k in clean_text for k in wake_keywords):
                is_wake_command = True
            
            if is_wake_command:
                intro_text = "你好，我是数字民警小石警官。很高兴为您服务！请问有什么我可以帮您的吗？"
                if not hidden_input:
                    self.chat_history.append({"role": "user", "content": text})

                if self.ws_clients:
                    self.chat_history.append({"role": "assistant", "content": intro_text})
                    self.ws_broadcast({"type": "CHAT_APPEND", "role": "assistant", "text": intro_text})
                else:
                    self.pending_intro = intro_text
                return
            # ----------------------

            try:
                base_url = self.cfg.get("llm_base_url", "https://api.openai.com/v1")
                api_key = self.secrets.get("llm_api_key", "")
                model = self.cfg.get("llm_model", "gpt-3.5-turbo")
                
                if not base_url or not api_key:
                    self.ws_broadcast({"type": "CHAT_APPEND", "role": "assistant", "text": "请先在设置中配置 LLM API 信息"})
                    return

                client = AsyncOpenAI(base_url=base_url, api_key=api_key)
                
                system_prompt = "你是一位包头公安局石拐分局的数字人民警，负责为群众解答警务、法律方面的问题，你的回答必须专业、诚挚、热情，绝对不能有任何不耐烦，指责意味的回答，为了保证对话的连贯性，回答内容控制在200字以内。注意：你叫“小石警官”或者有时被人误叫成“小时景观”或者其他发音为“xiao shi jin（g） guan”这都是在呼唤你，不要搞错了。"
                
                messages = [{"role": "system", "content": system_prompt}]
                for msg in self.chat_history[-10:]:
                    messages.append({"role": msg["role"], "content": msg["content"]})
                messages.append({"role": "user", "content": text})

                if not hidden_input:
                    self.chat_history.append({"role": "user", "content": text})

                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=500,
                    stream=True
                )
                
                full_content = ""
                buffer = ""
                first_chunk = True
                last_tts_flush_ts = time.monotonic()
                sent_any_tts = False
                def _flush_tts(force: bool = False):
                    nonlocal buffer, last_tts_flush_ts, sent_any_tts
                    if not buffer:
                        return
                    buf = buffer
                    min_len = 22 if not sent_any_tts else 48
                    now = time.monotonic()
                    if not force:
                        if len(buf) < min_len and (now - last_tts_flush_ts) < 0.9:
                            return
                    cut = -1
                    for p in "。！？；.!?;\n":
                        i = buf.rfind(p)
                        if i > cut:
                            cut = i
                    if cut < 0:
                        for p in "，,、:：":
                            i = buf.rfind(p)
                            if i > cut:
                                cut = i
                    if cut < 0:
                        i = buf.rfind(" ")
                        if i >= 0:
                            cut = i
                    if force:
                        out = buf.strip()
                        buffer = ""
                    else:
                        if cut >= 0:
                            out = buf[:cut + 1].strip()
                            buffer = buf[cut + 1:]
                        else:
                            out = buf.strip()
                            buffer = ""
                    if out and not self.stop_generation:
                        self.ws_broadcast({"type": "TTS_CHUNK", "text": out})
                        sent_any_tts = True
                        last_tts_flush_ts = time.monotonic()
                async for chunk in response:
                    if self.stop_generation:
                        self._file_log("Generation interrupted by user")
                        break
                    content = chunk.choices[0].delta.content
                    if content:
                        full_content += content
                        buffer += content
                        _flush_tts(force=any(p in content for p in "。！？；.!?;\n"))
                        
                        if first_chunk:
                            self.ws_broadcast({"type": "CHAT_START", "role": "assistant", "text": ""})
                            first_chunk = False
                        self.ws_broadcast({"type": "CHAT_PARTIAL", "text": content})
                
                if buffer and not self.stop_generation:
                    _flush_tts(force=True)
                
                self.ws_broadcast({"type": "STREAM_END"})
                        
                self.chat_history.append({"role": "assistant", "content": full_content})
            except Exception as e:
                err_msg = f"LLM 请求失败: {str(e)}"
                self._on_error(err_msg)
                self.ws_broadcast({"type": "CHAT_APPEND", "role": "assistant", "text": "抱歉，我遇到了一些问题，请稍后再试。"})

    def start_ws(self):
        """
        启动 WebSocket 服务器
        
        在 3399 端口启动 WS 服务，运行在单独的线程中 (使用 asyncio 事件循环)。
        """
        if self.ws_thread and self.ws_thread.is_alive():
            return
        def run():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self.ws_loop = loop
                self.llm_lock = asyncio.Lock()
                async def start():
                    async with websockets.serve(self.ws_handler, "127.0.0.1", 3399):
                        await asyncio.Future()
                loop.run_until_complete(start())
            except Exception as e:
                self._file_log(f"WS Error: {e}")
        self.ws_thread = threading.Thread(target=run, daemon=True)
        self.ws_thread.start()

    def ws_broadcast(self, obj: dict):
        """向所有连接的 WebSocket 客户端广播消息"""
        try:
            import json as _json
            data = _json.dumps(obj)
            loop = getattr(self, 'ws_loop', None)
            if loop and loop.is_running():
                # 关键修复：使用 list(self.ws_clients) 创建副本进行遍历
                # 防止在发送过程中有客户端断开导致 "Set changed size during iteration" 错误
                clients_copy = list(self.ws_clients)
                if clients_copy:
                    import asyncio as _asyncio
                    async def send_all():
                        for ws in clients_copy:
                            try:
                                await ws.send(data)
                            except:
                                pass # 发送失败忽略，ws_handler 会处理移除
                    _asyncio.run_coroutine_threadsafe(send_all(), loop)
        except Exception:
            pass

    def _check_browser(self):
        """
        定时检查浏览器进程状态
        
        如果浏览器进程消失且没有 WebSocket 连接，则认为播放器已关闭，
        此时重新开始 ASR 监听。
        """
        try:
            if self.player_active:
                window_ok = self._has_player_window()
                pid_ok = False
                try:
                    pid_ok = self.browser_pid and psutil.pid_exists(self.browser_pid)
                except Exception:
                    pid_ok = False
                if (not pid_ok) and (not window_ok) and (len(self.ws_clients) == 0):
                    self.player_active = False
                    self.browser_pid = None
                    try:
                        self._on_log("播放器已关闭，正在自动重新启动监听...")
                    except Exception:
                        pass
                    try:
                        self.start_listening()
                    except Exception:
                        pass
        except Exception:
            pass

    def _has_player_window(self) -> bool:
        try:
            import ctypes
            found = [False]

            def _enum_window_callback(hwnd, _):
                try:
                    if not ctypes.windll.user32.IsWindowVisible(hwnd):
                        return True
                    length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
                    if length <= 0:
                        return True
                    buff = ctypes.create_unicode_buffer(length + 1)
                    ctypes.windll.user32.GetWindowTextW(hwnd, buff, length + 1)
                    title = buff.value
                    if "Live2D Player" in title:
                        found[0] = True
                        return False
                except Exception:
                    pass
                return True

            WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
            ctypes.windll.user32.EnumWindows(WNDENUMPROC(_enum_window_callback), 0)
            return found[0]
        except Exception:
            return False

    def manual_wake(self):
        """
        手动/自动唤醒触发函数
        
        1. 启动播放器 (launch_player)
        2. 广播 "唤醒成功"
        3. 触发开场白
        """
        # text = "你好，我是数字民警小石警官。很高兴为您服务！"
        # self.chat_history.append({"role": "assistant", "content": text})            
        # self.pending_intro = text                 
    
        # self.ws_broadcast({"type": "CHAT_APPEND", "role": "assistant", "text": text})
        # 1. 确保服务已启动 (端口检查逻辑移到此处，避免 launch_player 耗时)
        try:
            if self.ensure_port_free(3400):
                self.start_http()
            if self.ensure_port_free(3399):
                self.start_ws()
        except Exception:
            pass

        # 2. 立即启动播放器 (如果尚未启动)，实现“一听到口令立刻弹出”
        should_launch = False
        if not self.ws_clients:
            if self._has_player_window():
                self.player_active = True
                should_launch = False
            elif self.player_active:
                pid_ok = False
                try:
                    pid_ok = bool(self.browser_pid) and psutil.pid_exists(self.browser_pid)
                except Exception:
                    pid_ok = False
                if not pid_ok:
                    self.player_active = False
                    should_launch = True
            else:
                should_launch = True
        
        if should_launch:
            self.launch_player()
            self._on_log("唤醒命中：正在启动播放器窗口...")

        try:
            self._on_log("手动唤醒")
            self._on_log(f"状态检查: ws_clients={len(self.ws_clients)}, player_active={self.player_active}")
            # Capture context if we are starting fresh
            if not self.ws_clients:
                 try:
                     self.window_manager.capture_context()
                 except Exception as e:
                     self._on_error(f"Context capture failed: {e}")
            
            self.process_wake_response()
            return

        except Exception as e:
            self._on_error(str(e))

    def process_wake_response(self):
        """统一处理唤醒后的响应逻辑"""
        self.last_wake_ts = time.monotonic()
        self._force_release_speaking_state()
        self._note_activity()
        intro_text = "你好，我是数字民警小石警官。很高兴为您服务！请问有什么我可以帮您的吗？"
        
        if self.ws_clients:
            # 如果已连接，直接发送唤醒信号和欢迎语
            self.ws_broadcast({"type": "WAKE_EXISTING"})
            self.chat_history.append({"role": "assistant", "content": intro_text})
            self.ws_broadcast({"type": "CHAT_APPEND", "role": "assistant", "text": intro_text})
        else:
            # 如果未连接（正在启动中），设置 pending，等待连接后发送
            # 注意：ws_handler 会在连接建立时将 pending_intro 添加到 chat_history 并发送
            self.pending_intro = intro_text
            
        # 切换 ASR 状态
        if self.worker:
            self.worker.set_mode("chat")
            self.worker.resume()

    def find_edge(self) -> Optional[str]:
        """查找 Microsoft Edge 浏览器的可执行文件路径"""
        paths = [
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
        ]
        for p in paths:
            if Path(p).exists():
                return p
        return None

    def launch_player(self):
        """
        启动数字人播放器
        
        1. 确保服务已运行
        2. 启动 Edge 浏览器
        """
        self.player_active = True
        font_size = self.cfg.get("chat_font_size", 36)
        url = f"http://localhost:3400/player/index.html?fontsize={font_size}"
        exe = self.find_edge()
        if exe:
            try:
                # --app=URL 以应用模式启动（无地址栏等），--kiosk 全屏模式（无最大化按钮，只能 Alt+F4 或 Esc 配合 JS 关闭）
                # 必须使用独立的 user-data-dir，否则如果后台有 Edge 进程，新窗口会合并到现有进程，导致 --kiosk 参数失效
                user_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "edge_kiosk_data")
                
                args = [
                    exe, 
                    "--new-window", 
                    # "--kiosk", 
                    "--edge-kiosk-type=fullscreen",
                    "--app=" + url, 
                    f"--user-data-dir={user_data_dir}",
                    "--autoplay-policy=no-user-gesture-required", 
                    "--use-fake-ui-for-media-stream", 
                    "--no-first-run"
                ]
                p = subprocess.Popen(args, close_fds=True)
                self.browser_pid = p.pid
                
                # 强制置顶逻辑
                def _bring_to_front():
                    # 轮询 10 次，每次间隔 0.5 秒，查找该 PID 对应的窗口
                    found = False
                    for _ in range(10):
                        time.sleep(0.5)
                        
                        def _enum_cb(hwnd, _):
                            nonlocal found
                            try:
                                process_id = ctypes.c_ulong()
                                ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
                                if process_id.value == self.browser_pid:
                                    # 再次检查标题，确保是我们的页面
                                    length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
                                    buff = ctypes.create_unicode_buffer(length + 1)
                                    ctypes.windll.user32.GetWindowTextW(hwnd, buff, length + 1)
                                    # Edge 的主窗口通常包含 "Edge" 或页面标题，这里简单处理，只要是该PID的可视窗口
                                    if ctypes.windll.user32.IsWindowVisible(hwnd):
                                        # 强制置顶
                                        # HWND_TOPMOST = -1
                                        # SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW = 0x0003 | 0x0040 = 0x0043
                                        ctypes.windll.user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0043)
                                        ctypes.windll.user32.SetForegroundWindow(hwnd)
                                        found = True
                                        return False # Stop enumeration
                            except:
                                pass
                            return True

                        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
                        ctypes.windll.user32.EnumWindows(WNDENUMPROC(_enum_cb), 0)
                        
                        if found:
                            break
                            
                threading.Thread(target=_bring_to_front, daemon=True).start()

            except Exception:
                os.startfile(url)
        else:
            try:
                os.startfile(url)
            except Exception:
                pass
        
        self.player_active = True
        
        # 启动时直接设置自我介绍
        # 注意：这里只设置 pending_intro，不加入历史记录，也不广播
        # 等到前端连接时 (ws_handler)，再作为新消息发送并加入历史
        self.process_wake_response()

    def close_player(self):
        """
        手动关闭数字人播放器进程
        
        策略：
        1. [优先] 查找标题包含 "Live2D Player" 的窗口并发送关闭消息 (最优雅)
        2. [后备] 使用系统命令 (wmic/taskkill) 强制关闭包含 localhost:3400 的进程
        """
        self._on_log("正在执行关闭操作...")
        self._note_activity()
        
        # 切换 ASR 回到唤醒模式
        if self.worker:
            self.worker.set_mode("wake")
            self.worker.resume()
        
        # --- 策略 1: Windows API 关闭窗口 ---
        try:
            import ctypes
            found_window = [False] # 使用列表在闭包中修改
            
            def _enum_window_callback(hwnd, _):
                try:
                    length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
                    if length > 0:
                        buff = ctypes.create_unicode_buffer(length + 1)
                        ctypes.windll.user32.GetWindowTextW(hwnd, buff, length + 1)
                        title = buff.value
                        # 匹配 index.html 中的 <title>Live2D Player</title>
                        if "Live2D Player" in title:
                            self._on_log(f"找到窗口 [{title}] (HWND: {hwnd})，发送关闭信号...")
                            # WM_CLOSE = 0x0010
                            ctypes.windll.user32.PostMessageW(hwnd, 0x0010, 0, 0)
                            found_window[0] = True
                except:
                    pass
                return True # 继续枚举

            WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
            ctypes.windll.user32.EnumWindows(WNDENUMPROC(_enum_window_callback), 0)
            
            if found_window[0]:
                # 给一点时间让窗口关闭
                time.sleep(1)
        except Exception as e:
            self._on_log(f"窗口操作失败: {e}")

        # --- 策略 2: 进程强制查杀 (补刀) ---
        target_url_snippet = "localhost:3400"
        closed_count = 0
        
        # 2.1 关闭记录的 PID
        if self.browser_pid:
            try:
                if psutil.pid_exists(self.browser_pid):
                    p = psutil.Process(self.browser_pid)
                    children = p.children(recursive=True)
                    for child in children:
                        try: child.kill() 
                        except: pass
                    p.kill()
                    closed_count += 1
            except Exception:
                pass
        
        # 2.2 使用 WMIC 系统命令查杀
        try:
            # 查找命令行包含 localhost:3400 的 msedge.exe 进程
            cmd = f"wmic process where \"name='msedge.exe' and commandline like '%{target_url_snippet}%'\" call terminate"
            subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
 
        # 2.3 Python psutil 再次扫描清理
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['name'] and 'msedge' in proc.info['name'].lower():
                        cmdline = proc.info['cmdline']
                        if cmdline:
                            cmd_str = " ".join(cmdline)
                            if target_url_snippet in cmd_str:
                                proc.kill()
                                closed_count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
        except Exception:
            pass

        self._on_log("关闭操作已完成")

        # 清理状态
        self.player_active = False
        self.browser_pid = None
        
        # 手动触发重启监听
        try:
            self.start_listening()
        except Exception:
            pass

    

    def _init_file_log(self):
        """
        初始化文件日志
        
        日志文件位置: LOCALAPPDATA/pySiberMan/logs/pySiberMan.log
        同时启用 faulthandler 以捕获崩溃信息。
        """
        try:
            base = os.getenv("LOCALAPPDATA") or str(Path.home())
            d = Path(base) / "pySiberMan" / "logs"
            d.mkdir(parents=True, exist_ok=True)
            p = d / "pySiberMan.log"
            self._log_fp = open(p, "a", encoding="utf-8")
            try:
                faulthandler.enable(self._log_fp)
            except Exception:
                pass
        except Exception:
            self._log_fp = None

    def _file_log(self, s: str):
        """写入一行日志到文件"""
        try:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            if self._log_fp:
                self._log_fp.write(ts + " " + s + "\n")
                self._log_fp.flush()
        except Exception:
            pass

    def _excepthook(self, etype, value, tb):
        """全局异常捕获钩子，将未捕获异常写入日志文件"""
        try:
            self._file_log("EXC " + "".join(traceback.format_exception(etype, value, tb)))
        except Exception:
            pass

    def _health_tick(self):
        """定期健康检查，记录内存使用和线程数"""
        try:
            p = psutil.Process(os.getpid())
            mem = p.memory_info().rss
            th = p.num_threads()
            self._file_log(f"HEALTH rss={mem} threads={th} ws={len(self.ws_clients)} player={self.player_active}")
        except Exception:
            pass

    def _normalize_text(self, s: str) -> str:
        """
        文本标准化 (用于 ASR 结果匹配)
        
        1. 去除标点符号
        2. 修正常见的同音错别字 (如 "小时" -> "小石")
        """
        t = re.sub(r"[\s,.!?;:，。！？；：、（）()《》〈〉「」『』“”‘’—…·\-]+", "", s or "")
        for v in ["小时","消失","小是","小识","小事","肖石","晓石","晓诗","萧石","小十"]:
            t = t.replace(v, "小石")
        for v in ["景观","尽管","警管","井关","金冠","尽关","经管","景觀","尽古","頂關","景瓜","金关","里关"]:
            t = t.replace(v, "警官")
        return t

    def _clean_markdown(self, text: str) -> str:
        """
        清除 Markdown 格式 (用于 TTS 朗读)
        
        去除加粗、斜体、代码块、标题、链接、图片等标记，只保留纯文本。
        """
        if not text:
            return ""
        if not isinstance(text, str):
            text = str(text)
        # Remove bold/italic markers
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'__(.*?)__', r'\1', text)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        text = re.sub(r'_(.*?)_', r'\1', text)
        # Remove code ticks
        text = re.sub(r'`(.*?)`', r'\1', text)
        # Remove headers
        text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
        # Remove links [text](url) -> text
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        # Remove images ![text](url) -> text
        text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', r'\1', text)
        return text


if __name__ == "__main__":
    """
    程序入口点
    
    1. 处理 PyInstaller noconsole 模式下的 stdout/stderr 重定向
    2. 清理残留进程 (占用 3399/3400 端口的旧实例)
    3. 启动 TrayApp 应用程序
    """
    # Fix for PyInstaller noconsole mode where sys.stdout/stderr are None
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")

    # Try to clean up existing instances or processes holding ports
    try:
        current_pid = os.getpid()
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                # Check if process is using port 3399 or 3400
                for conn in proc.connections():
                    if conn.laddr.port in [3399, 3400]:
                        if proc.pid != current_pid:
                            proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
    except Exception:
        pass

    app = TrayApp(sys.argv)
    sys.exit(app.exec())
