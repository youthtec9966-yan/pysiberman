import asyncio
from array import array
import os
import tempfile
import time
import traceback
import wave

from dashscope.audio.asr import Recognition, RecognitionCallback
from openai import AsyncOpenAI
from pvrecorder import PvRecorder
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPen, QColor
from PySide6.QtWidgets import QApplication, QSystemTrayIcon, QDialog, QFormLayout, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPushButton, QHBoxLayout, QVBoxLayout, QLabel, QFileDialog, QMessageBox, QPlainTextEdit, QWidget, QProgressBar, QScrollArea, QCheckBox

from config_store import ConfigManager, SecretStore


class SettingsAsrTestCallback(RecognitionCallback):
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
            stored_name = str(self.cfg.get("device_name", "") or "").strip()
            name_pos = self.device_combo.findText(stored_name)
            if name_pos >= 0:
                self.device_combo.setCurrentIndex(name_pos)
            elif idx is not None and idx >= 0 and idx < len(devices):
                pos = self.device_combo.findData(int(idx))
                if pos >= 0:
                    self.device_combo.setCurrentIndex(pos)
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
        self.llm_base_url = QLineEdit(self.cfg.get("llm_base_url", "https://ark.cn-beijing.volces.com/api/v3"))
        self.llm_api_key = QLineEdit(self.secrets.get("llm_api_key", "43bdf90d-590e-442e-8f68-5207a88d3052"))
        self.llm_api_key.setEchoMode(QLineEdit.Password)
        self.asr_api_key = QLineEdit(self.secrets.get("asr_api_key", self.cfg.get("aliyun_appkey", "sk-3b0b4ca3a53a4cf489a5a294eda0aff0")))
        self.asr_api_key.setEchoMode(QLineEdit.Password)
        self.porcupine_access_key = QLineEdit(self.secrets.get("porcupine_access_key", self.cfg.get("porcupine_access_key", "")))
        self.porcupine_access_key.setEchoMode(QLineEdit.Password)
        self.kws_enabled = QCheckBox("启用 Porcupine KWS 唤醒")
        self.kws_enabled.setChecked(bool(self.cfg.get("kws_enabled", False)))
        self.llm_model = QLineEdit(self.cfg.get("llm_model", "doubao-seed-1-6-flash-250828"))
        self.tts_model = QComboBox()
        self.tts_model.setEditable(True)
        for name in ["cosyvoice-v3-flash", "cosyvoice-v1"]:
            self.tts_model.addItem(name, name)
        tts_model = str(self.cfg.get("aliyun_tts_model", "cosyvoice-v3-flash") or "cosyvoice-v3-flash").strip()
        if self.tts_model.findText(tts_model) < 0:
            self.tts_model.addItem(tts_model, tts_model)
        self.tts_model.setCurrentText(tts_model)
        try:
            self.tts_model.setInsertPolicy(QComboBox.NoInsert)
        except Exception:
            pass
        self.tts_voice = QComboBox()
        self.tts_voice.setEditable(True)
        for name in ["longxiang", "longyuan", "longfei", "longtong", "longshuo", "longshu", "longlaotie", "longxiaocheng", "longxiaochun"]:
            self.tts_voice.addItem(name, name)
        tts_voice = str(self.cfg.get("aliyun_tts_voice", "longxiang") or "longxiang").strip()
        if self.tts_voice.findText(tts_voice) < 0:
            self.tts_voice.addItem(tts_voice, tts_voice)
        self.tts_voice.setCurrentText(tts_voice)
        try:
            self.tts_voice.setInsertPolicy(QComboBox.NoInsert)
        except Exception:
            pass

        form = QFormLayout()
        form.addRow("LLM Base URL", self.llm_base_url)
        form.addRow("LLM API Key", self.llm_api_key)
        form.addRow("ASR API Key", self.asr_api_key)
        form.addRow("KWS 唤醒", self.kws_enabled)
        form.addRow("Porcupine AccessKey", self.porcupine_access_key)
        form.addRow("LLM 模型", self.llm_model)
        form.addRow("TTS 模型", self.tts_model)
        form.addRow("TTS 音色（男声）", self.tts_voice)
        form.addRow("唤醒方式", self.engine_combo)
        form.addRow("ASR 唤醒短语", self.asr_phrases)
        form.addRow("ASR 模型大小", self.asr_model)
        form.addRow("设备", self.device_combo)
        form.addRow("ASR 模型目录", w)
        
        self.chat_font_size = QSpinBox()
        self.chat_font_size.setRange(12, 72)
        self.chat_font_size.setValue(int(self.cfg.get("chat_font_size", 36)))
        form.addRow("会话字体大小(px)", self.chat_font_size)
        self.force_exit_check_interval_ms = QSpinBox()
        self.force_exit_check_interval_ms.setRange(1000, 60000)
        self.force_exit_check_interval_ms.setSingleStep(500)
        self.force_exit_check_interval_ms.setValue(int(self.cfg.get("force_exit_check_interval_ms", 10000)))
        form.addRow("兜底检测间隔(ms)", self.force_exit_check_interval_ms)
        self.force_exit_timeout_seconds = QDoubleSpinBox()
        self.force_exit_timeout_seconds.setRange(1.0, 300.0)
        self.force_exit_timeout_seconds.setSingleStep(0.1)
        self.force_exit_timeout_seconds.setDecimals(1)
        self.force_exit_timeout_seconds.setValue(float(self.cfg.get("force_exit_timeout_seconds", 23.6)))
        form.addRow("兜底强退阈值(s)", self.force_exit_timeout_seconds)
        self.idle_prompt_seconds = QDoubleSpinBox()
        self.idle_prompt_seconds.setRange(1.0, 120.0)
        self.idle_prompt_seconds.setSingleStep(0.5)
        self.idle_prompt_seconds.setDecimals(1)
        self.idle_prompt_seconds.setValue(float(self.cfg.get("idle_prompt_seconds", 5.0)))
        form.addRow("规则四等待(s)", self.idle_prompt_seconds)
        self.idle_close_wait_seconds = QDoubleSpinBox()
        self.idle_close_wait_seconds.setRange(1.0, 120.0)
        self.idle_close_wait_seconds.setSingleStep(0.5)
        self.idle_close_wait_seconds.setDecimals(1)
        self.idle_close_wait_seconds.setValue(float(self.cfg.get("idle_close_wait_seconds", 5.0)))
        form.addRow("规则五等待(s)", self.idle_close_wait_seconds)

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
        self.interrupt_audio_match_threshold = QDoubleSpinBox()
        self.interrupt_audio_match_threshold.setRange(0.05, 1.50)
        self.interrupt_audio_match_threshold.setSingleStep(0.01)
        self.interrupt_audio_match_threshold.setDecimals(2)
        self.interrupt_audio_match_threshold.setValue(float(self.cfg.get("interrupt_audio_match_threshold", self.cfg.get("audio_match_threshold", 0.45))))
        form.addRow("打断匹配阈值", self.interrupt_audio_match_threshold)
        self.interrupt_audio_window_ms = QSpinBox()
        self.interrupt_audio_window_ms.setRange(200, 3000)
        self.interrupt_audio_window_ms.setSingleStep(100)
        self.interrupt_audio_window_ms.setValue(int(self.cfg.get("interrupt_audio_window_ms", 1000)))
        form.addRow("打断匹配窗口(ms)", self.interrupt_audio_window_ms)
        self.interrupt_audio_check_interval_frames = QSpinBox()
        self.interrupt_audio_check_interval_frames.setRange(1, 30)
        self.interrupt_audio_check_interval_frames.setValue(int(self.cfg.get("interrupt_audio_check_interval_frames", 8)))
        form.addRow("打断检测步长(帧)", self.interrupt_audio_check_interval_frames)
        self.interrupt_audio_cooldown_ms = QSpinBox()
        self.interrupt_audio_cooldown_ms.setRange(100, 5000)
        self.interrupt_audio_cooldown_ms.setSingleStep(100)
        self.interrupt_audio_cooldown_ms.setValue(int(self.cfg.get("interrupt_audio_cooldown_ms", 800)))
        form.addRow("打断冷却(ms)", self.interrupt_audio_cooldown_ms)
        self.interrupt_audio_standby_rms_gate = QDoubleSpinBox()
        self.interrupt_audio_standby_rms_gate.setRange(0.001, 0.100)
        self.interrupt_audio_standby_rms_gate.setSingleStep(0.001)
        self.interrupt_audio_standby_rms_gate.setDecimals(4)
        self.interrupt_audio_standby_rms_gate.setValue(float(self.cfg.get("interrupt_audio_standby_rms_gate", 0.006)))
        form.addRow("打断待机RMS门限", self.interrupt_audio_standby_rms_gate)
        self.asr_interrupt_enabled = QCheckBox("启用 ASR 打断")
        self.asr_interrupt_enabled.setChecked(bool(self.cfg.get("asr_interrupt_enabled", False)))
        form.addRow("ASR打断", self.asr_interrupt_enabled)
        self.audio_command_interrupt_enabled = QCheckBox("启用音频指令打断")
        self.audio_command_interrupt_enabled.setChecked(bool(self.cfg.get("audio_command_interrupt_enabled", False)))
        form.addRow("音频指令打断", self.audio_command_interrupt_enabled)
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
        self.interrupt_audio_match_threshold.valueChanged.connect(self._on_interrupt_config_changed)
        self.interrupt_audio_window_ms.valueChanged.connect(self._on_interrupt_config_changed)
        self.interrupt_audio_check_interval_frames.valueChanged.connect(self._on_interrupt_config_changed)
        self.interrupt_audio_cooldown_ms.valueChanged.connect(self._on_interrupt_config_changed)
        self.interrupt_audio_standby_rms_gate.valueChanged.connect(self._on_interrupt_config_changed)
        self.asr_interrupt_enabled.toggled.connect(self._on_interrupt_feature_flags_changed)
        self.audio_command_interrupt_enabled.toggled.connect(self._on_interrupt_feature_flags_changed)
        self.kws_enabled.toggled.connect(self._update_kws_controls)
        self._update_kws_controls(self.kws_enabled.isChecked())
        self._on_profile_mode_changed(self.asr_profile_mode.currentIndex())

        self.audio_curve = AudioCurveWidget()
        self.audio_curve_status = QLabel("待机")
        self.audio_curve_legend = QLabel("蓝线=输入音量 橙线=触发门限 绿线=噪声底（参数自动调整）")
        form.addRow("音频曲线", self.audio_curve)
        form.addRow("曲线状态", self.audio_curve_status)
        form.addRow("曲线说明", self.audio_curve_legend)


        self.btn_test = QPushButton("测试 LLM")
        self.btn_test_asr = QPushButton("测试 ASR")
        self.btn_clear_tts_cache = QPushButton("清空TTS缓存")
        self.btn_test.setVisible(True)
        btn_ok = QPushButton("保存")
        btn_cancel = QPushButton("取消")
        self.btn_test.clicked.connect(self._test_llm_connection)
        self.btn_test_asr.clicked.connect(self._test_asr_connection)
        self.btn_clear_tts_cache.clicked.connect(self._clear_tts_cache)
        btn_ok.clicked.connect(self._save)
        btn_cancel.clicked.connect(self.reject)
        btns = QHBoxLayout()
        btns.addWidget(self.btn_test)
        btns.addWidget(self.btn_test_asr)
        btns.addWidget(self.btn_clear_tts_cache)
        btns.addStretch()
        btns.addWidget(btn_ok)
        btns.addWidget(btn_cancel)

        form_container = QWidget()
        form_container.setLayout(form)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(form_container)
        layout = QVBoxLayout()
        layout.addWidget(scroll)
        layout.addLayout(btns)
        self.setLayout(layout)

    

    def _save(self):
        """
        保存设置
        
        将界面上的配置写入配置文件和密钥存储，并关闭对话框。
        """
        self.cfg.set("llm_base_url", self.llm_base_url.text())
        self.cfg.set("llm_model", self.llm_model.text())
        selected_index = self.device_combo.currentData()
        try:
            selected_index = int(selected_index)
        except Exception:
            selected_index = self.device_combo.currentIndex()
        self.cfg.set("device_index", selected_index)
        self.cfg.set("device_name", self.device_combo.currentText())
        self.cfg.set("wake_engine", self.engine_combo.currentData())
        self.cfg.set("asr_wake_phrases", "小石警官")
        self.cfg.set("asr_model_size", self.asr_model.currentData())
        self.cfg.set("asr_model_dir", self.asr_model_dir.text())
        self.cfg.set("aliyun_tts_model", self.tts_model.currentText().strip())
        self.cfg.set("aliyun_tts_voice", self.tts_voice.currentText().strip())
        self.cfg.set("chat_font_size", self.chat_font_size.value())
        self.cfg.set("force_exit_check_interval_ms", int(self.force_exit_check_interval_ms.value()))
        self.cfg.set("force_exit_timeout_seconds", float(self.force_exit_timeout_seconds.value()))
        self.cfg.set("idle_prompt_seconds", float(self.idle_prompt_seconds.value()))
        self.cfg.set("idle_close_wait_seconds", float(self.idle_close_wait_seconds.value()))
        self.cfg.set("audio_match_enabled", self.audio_match_enabled.currentData())
        self.cfg.set("audio_match_threshold", float(self.audio_match_threshold.value()))
        self.cfg.set("interrupt_audio_match_threshold", float(self.interrupt_audio_match_threshold.value()))
        self.cfg.set("interrupt_audio_window_ms", int(self.interrupt_audio_window_ms.value()))
        self.cfg.set("interrupt_audio_check_interval_frames", int(self.interrupt_audio_check_interval_frames.value()))
        self.cfg.set("interrupt_audio_cooldown_ms", int(self.interrupt_audio_cooldown_ms.value()))
        self.cfg.set("interrupt_audio_standby_rms_gate", float(self.interrupt_audio_standby_rms_gate.value()))
        self.cfg.set("asr_interrupt_enabled", bool(self.asr_interrupt_enabled.isChecked()))
        self.cfg.set("audio_command_interrupt_enabled", bool(self.audio_command_interrupt_enabled.isChecked()))
        self.cfg.set("asr_profile_mode", self.asr_profile_mode.currentData())
        profile = self._current_profile()
        self.cfg.set("asr_standby_noise_margin", profile["standby_noise_margin"])
        self.cfg.set("asr_speaking_noise_margin", profile["speaking_noise_margin"])
        self.cfg.set("asr_standby_energy_ratio", profile["standby_energy_ratio"])
        self.cfg.set("asr_speaking_energy_ratio", profile["speaking_energy_ratio"])
        self.cfg.set("asr_interrupt_peak", profile["speaking_interrupt_peak"])
        self.cfg.set("asr_interrupt_rms", profile["speaking_interrupt_rms"])
        asr_api_key = self.asr_api_key.text().strip()
        self.cfg.set("aliyun_appkey", asr_api_key)
        self.cfg.set("kws_enabled", bool(self.kws_enabled.isChecked()))
        self.cfg.set("porcupine_access_key", self.porcupine_access_key.text().strip())
        self.cfg.save()
        self.secrets.set("llm_api_key", self.llm_api_key.text())
        self.secrets.set("asr_api_key", asr_api_key)
        self.secrets.set("porcupine_access_key", self.porcupine_access_key.text().strip())
        self.accept()

    def _update_kws_controls(self, enabled: bool):
        enabled = bool(enabled)
        self.porcupine_access_key.setEnabled(enabled)
        if enabled:
            self.porcupine_access_key.setPlaceholderText("启用 KWS 后请输入 Porcupine AccessKey")
        else:
            self.porcupine_access_key.setPlaceholderText("KWS 关闭时无需填写 AccessKey")

    def _interrupt_runtime_config(self):
        return {
            "interrupt_audio_match_threshold": float(self.interrupt_audio_match_threshold.value()),
            "interrupt_audio_window_ms": int(self.interrupt_audio_window_ms.value()),
            "interrupt_audio_check_interval_frames": int(self.interrupt_audio_check_interval_frames.value()),
            "interrupt_audio_cooldown_ms": int(self.interrupt_audio_cooldown_ms.value()),
            "interrupt_audio_standby_rms_gate": float(self.interrupt_audio_standby_rms_gate.value()),
        }

    def _on_interrupt_config_changed(self, _):
        if not self.app:
            return
        try:
            self.app._on_settings_interrupt_config_changed(self._interrupt_runtime_config())
        except Exception:
            pass

    def _on_interrupt_feature_flags_changed(self, _):
        if not self.app:
            return
        try:
            self.app._on_settings_interrupt_feature_flags_changed(
                bool(self.asr_interrupt_enabled.isChecked()),
                bool(self.audio_command_interrupt_enabled.isChecked())
            )
        except Exception:
            pass

    def _clear_tts_cache(self):
        if not self.app:
            QMessageBox.warning(self, "提示", "应用实例不可用")
            return
        try:
            removed, failed = self.app.clear_tts_cache()
            if failed > 0:
                QMessageBox.warning(self, "完成", f"已删除 {removed} 个缓存文件，失败 {failed} 个")
            else:
                QMessageBox.information(self, "完成", f"已删除 {removed} 个缓存文件")
        except Exception as e:
            QMessageBox.warning(self, "失败", f"清理缓存失败: {e}")

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

    def _test_asr_connection(self):
        api_key = self.asr_api_key.text().strip()
        if not api_key:
            QMessageBox.warning(self, "信息不完整", "请先填写 ASR API Key。")
            return
        selected_index = self.device_combo.currentData()
        try:
            selected_index = int(selected_index)
        except Exception:
            selected_index = -1
        recorder = None
        temp_filename = ""
        original_text = self.btn_test_asr.text()
        self.btn_test_asr.setEnabled(False)
        self.btn_test_asr.setText("测试中...")
        QApplication.processEvents()
        try:
            recorder = PvRecorder(device_index=selected_index, frame_length=512)
            recorder.start()
            pcm_data = []
            end_at = time.monotonic() + 3.0
            while time.monotonic() < end_at:
                pcm_data.extend(recorder.read())
            recorder.stop()
            recorder.delete()
            recorder = None
            if len(pcm_data) < 1600:
                QMessageBox.warning(self, "失败", "录音数据过短，请检查麦克风权限和输入设备。")
                return
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                temp_filename = temp_wav.name
            pcm_array = array("h", pcm_data)
            with wave.open(temp_filename, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(pcm_array.tobytes())
            callback = SettingsAsrTestCallback()
            recognition = Recognition(
                model="paraformer-realtime-v1",
                callback=callback,
                format="wav",
                sample_rate=16000,
                api_key=api_key
            )
            result = recognition.call(file=temp_filename)
            text = ""
            if result and hasattr(result, "get"):
                candidates = []
                candidates.extend(result.get("sentence", []))
                candidates.extend(result.get("sentences", []))
                output = result.get("output", {})
                if isinstance(output, dict):
                    candidates.extend(output.get("sentence", []))
                    candidates.extend(output.get("sentences", []))
                for item in candidates:
                    if hasattr(item, "text"):
                        text += item.text
                    elif isinstance(item, dict) and "text" in item:
                        text += str(item["text"])
            if not text and callback.results:
                for cb_result in callback.results:
                    if hasattr(cb_result, "text"):
                        text += cb_result.text
            text = (text or "").strip()
            if text:
                QMessageBox.information(self, "ASR测试成功", f"识别结果：{text}")
            else:
                QMessageBox.warning(self, "ASR测试完成", "已调用阿里云ASR，但未识别到文本，请对着麦克风清晰说话后重试。")
        except Exception as e:
            QMessageBox.warning(self, "ASR测试失败", f"{e}\n\n{traceback.format_exc()}")
        finally:
            if recorder:
                try:
                    recorder.stop()
                except Exception:
                    pass
                try:
                    recorder.delete()
                except Exception:
                    pass
            if temp_filename:
                try:
                    if os.path.exists(temp_filename):
                        os.unlink(temp_filename)
                except Exception:
                    pass
            self.btn_test_asr.setEnabled(True)
            self.btn_test_asr.setText(original_text)

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
        cfg_name = str(self.app.cfg.get("device_name", "") or "").strip()
        if cfg_name and cfg_name in devices:
            name = cfg_name
        else:
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
