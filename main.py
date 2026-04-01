import sys
from pathlib import Path
import json
import threading
import traceback
from typing import Optional
import re
import ctypes
import base64
import queue

# PySide6 是 Qt 框架的 Python 绑定，用于创建图形用户界面 (GUI)
from PySide6.QtCore import QObject, QThread, Signal, Qt, QTimer
from PySide6.QtWidgets import QApplication, QSystemTrayIcon, QMenu, QDialog, QMessageBox
from PySide6.QtGui import QIcon, QPixmap, QAction

import psutil # 用于获取系统进程和硬件信息


from pvrecorder import PvRecorder # 用于录音
import asyncio
import websockets # 用于 WebSocket 通信
import socket
from urllib.parse import urlparse
import subprocess
import os
from openai import AsyncOpenAI # OpenAI 官方 SDK，用于调用兼容 OpenAI 接口的大模型
import time
import faulthandler # 用于调试崩溃（Faults）
import dashscope
from dashscope.audio.tts_v2 import AudioFormat, ResultCallback, SpeechSynthesizerObjectPool

from config_store import ConfigManager, SecretStore
from window_context import WindowContextManager, get_resource_path
from audio_templates import TemplateManageDialog
from asr_worker import AudioWakeWorkerASR
from ui_components import SettingsDialog, LogWindow, ControlWindow


class AliyunTtsStreamCallback(ResultCallback):
    def __init__(self):
        super().__init__()
        self.audio_queue = queue.Queue()
        self.done_event = threading.Event()
        self.error_message = None

    def on_data(self, data: bytes) -> None:
        if data:
            self.audio_queue.put(bytes(data))

    def on_complete(self) -> None:
        self.done_event.set()

    def on_error(self, message) -> None:
        self.error_message = str(message)
        self.done_event.set()

    def on_close(self) -> None:
        self.done_event.set()

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
        self._refresh_wake_audio_manager_action()

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
        self.post_answer_tts_wait_seconds = 12.0
        self._awaiting_tts_start = False
        self.tts_start_wait_timer = QTimer(self)
        self.tts_start_wait_timer.setSingleShot(True)
        self.tts_start_wait_timer.timeout.connect(self._on_tts_start_wait_timeout)
        self.pending_user_text = None
        self.manual_command_armed_until_ts = 0.0
        self.manual_command_timeout_seconds = 12.0
        self.user_input_debounce_ms = 700
        self.user_input_timer = QTimer(self)
        self.user_input_timer.setSingleShot(True)
        self.user_input_timer.timeout.connect(self._commit_pending_user_input)
        self.awaiting_response = False
        self.asr_fallback_during_tts = True
        self.close_after_answer = False
        self.single_turn_close_pending = False
        self.single_turn_close_requested_ts = 0.0
        self.last_wake_wallclock = ""
        self.last_session_start_ts = 0.0
        self.last_speaking_start_ts = 0.0
        self.last_interrupt_ts = 0.0
        self.last_wake_ts = 0.0
        self.last_successful_response_ts = 0.0
        self.last_successful_response_text = ""
        self.force_exit_watch_started_ts = 0.0
        self.wake_grace_seconds = 8.0
        self.last_tts_start_ts = 0.0
        self.force_exit_check_interval_ms = 10000
        self.force_exit_timeout_seconds = 23.6
        self._reload_force_exit_guard_config()
        self.idle_prompt_seconds = 5.0
        self.idle_close_wait_seconds = 5.0
        self._reload_idle_prompt_config()
        self._idle_waiting_for_close = False
        self._idle_prompt_sent = False
        self.idle_prompt_timer = QTimer(self)
        self.idle_prompt_timer.setSingleShot(True)
        self.idle_prompt_timer.timeout.connect(self._on_idle_prompt_timeout)
        self.idle_close_timer = QTimer(self)
        self.idle_close_timer.setSingleShot(True)
        self.idle_close_timer.timeout.connect(self._on_idle_close_timeout)
        self.player_active = False
        self.player_connect_timeout_seconds = 12.0
        self.player_max_relaunch_attempts = 2
        self.player_launch_started_ts = 0.0
        self.player_relaunch_count = 0
        self._awakened = False
        self._suppress_interrupt_toggle = False
        self.window_manager = WindowContextManager()
        local_appdata = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
        self.runtime_data_dir = local_appdata / "pySiberMan"
        self.tts_lock = threading.Lock()
        self.tts_pool_lock = threading.Lock()
        self.tts_pool = None
        self.tts_model = str(self.cfg.get("aliyun_tts_model", "cosyvoice-v3-flash") or "cosyvoice-v3-flash").strip()
        self.tts_voice = str(self.cfg.get("aliyun_tts_voice", "longxiang") or "longxiang").strip()
        self.tts_rate = self._normalize_aliyun_tts_rate(self.cfg.get("aliyun_tts_rate", 1.1))
        self.tts_volume = self._normalize_aliyun_tts_volume(self.cfg.get("aliyun_tts_volume", 50))
        self.tts_pitch = self._normalize_aliyun_tts_pitch(self.cfg.get("aliyun_tts_pitch", 1.0))
        self.tts_play_proc = None
        self.tts_play_proc_lock = threading.Lock()
        self.tts_mci_alias = None
        self.tts_order_cond = threading.Condition()  
        
        self.tts_submit_index = 0
        self.tts_play_index = 1
        self.tts_cancel_seq = 0
        self.tts_played_events = {}
        self.settings_dlg = None
        try:
            self.runtime_data_dir.mkdir(parents=True, exist_ok=True)
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
        try:
            self.force_exit_guard_timer = QTimer(self)
            self.force_exit_guard_timer.setInterval(int(self.force_exit_check_interval_ms))
            self.force_exit_guard_timer.timeout.connect(self._force_exit_guard_tick)
            self.force_exit_guard_timer.start()
        except Exception:
            pass
        try:
            QTimer.singleShot(1500, lambda: threading.Thread(target=self._run_startup_diagnostics, daemon=True).start())
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
        device_index, _ = self._resolve_preferred_device_index()
        dlg = TemplateManageDialog(None, device_index=device_index, template_dir=template_dir, scene_name=title)
        dlg.exec()
        if self.worker:
            self.worker.reload_audio_templates()

    def _is_kws_enabled(self) -> bool:
        return bool(self.cfg.get("kws_enabled", False))

    def _get_kws_runtime_signature(self):
        return (
            self._is_kws_enabled(),
            str(self.secrets.get("porcupine_access_key", self.cfg.get("porcupine_access_key", "")) or "").strip(),
            str(self.cfg.get("porcupine_keyword_path", "./sjg_zh_windows_v4_0_0.ppn") or "").strip(),
            str(self.cfg.get("porcupine_model_path", "./porcupine_params_zh.pv") or "").strip(),
            float(self.cfg.get("porcupine_sensitivity", 0.7) or 0.7),
        )

    def _refresh_wake_audio_manager_action(self):
        kws_enabled = self._is_kws_enabled()
        self.act_audio_wake_cmd.setEnabled(not kws_enabled)

    def open_wake_audio_manager(self):
        self._refresh_wake_audio_manager_action()
        if self._is_kws_enabled():
            try:
                QMessageBox.information(None, "提示", "当前已启用 KWS 唤醒，音频模板唤醒已停用。关闭 KWS 后可继续管理唤醒指令。")
            except Exception:
                pass
            return
        try:
            self._open_template_manager("唤醒指令", "templates/wake")
        except Exception:
            pass

    def open_interrupt_audio_manager(self):
        self._open_template_manager("打断指令", "templates/interrupt")


    def show_logs(self):
        """显示日志窗口"""
        self.log_win.show()
        self.log_win.raise_()

    def open_settings(self):
        """打开设置对话框，并在保存后刷新主界面信息"""
        try:
            kws_signature_before = self._get_kws_runtime_signature()
            was_running = bool(self.worker)
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
                    self._reload_tts_config()
                except Exception:
                    pass
                try:
                    self._reload_force_exit_guard_config()
                except Exception:
                    pass
                try:
                    self._reload_idle_prompt_config()
                except Exception:
                    pass
                try:
                    self._apply_asr_profile_to_worker()
                except Exception:
                    pass
                try:
                    self._refresh_wake_audio_manager_action()
                except Exception:
                    pass
                try:
                    kws_signature_after = self._get_kws_runtime_signature()
                    if was_running and kws_signature_before != kws_signature_after:
                        self._on_log("KWS 设置已变更，正在重启监听以应用新配置")
                        self.stop_listening()
                        self.start_listening()
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

    def _get_interrupt_match_config_from_config(self):
        return {
            "interrupt_audio_match_threshold": float(self.cfg.get("interrupt_audio_match_threshold", self.cfg.get("audio_match_threshold", 0.45))),
            "interrupt_audio_window_ms": float(self.cfg.get("interrupt_audio_window_ms", 1000)),
            "interrupt_audio_check_interval_frames": int(self.cfg.get("interrupt_audio_check_interval_frames", 8)),
            "interrupt_audio_cooldown_ms": float(self.cfg.get("interrupt_audio_cooldown_ms", 800)),
            "interrupt_audio_standby_rms_gate": float(self.cfg.get("interrupt_audio_standby_rms_gate", 0.006)),
        }

    def _get_kws_config_from_config(self):
        enabled = bool(self.cfg.get("kws_enabled", False))
        raw_access_key = self.cfg.get("porcupine_access_key", "")
        raw_keyword_path = self.cfg.get("porcupine_keyword_path", "./sjg_zh_windows_v4_0_0.ppn")
        raw_model_path = self.cfg.get("porcupine_model_path", "./porcupine_params_zh.pv")
        raw_sensitivity = self.cfg.get("porcupine_sensitivity", 0.7)
        raw_interrupt_fallback_enabled = self.cfg.get("kws_interrupt_fallback_enabled", True)
        raw_interrupt_cooldown_ms = self.cfg.get("kws_interrupt_cooldown_ms", self.cfg.get("interrupt_audio_cooldown_ms", 800))
        access_key = str(raw_access_key or "").strip()
        keyword_path = str(raw_keyword_path or "").strip()
        model_path = str(raw_model_path or "").strip()
        interrupt_fallback_enabled = bool(raw_interrupt_fallback_enabled)
        try:
            interrupt_cooldown_ms = float(raw_interrupt_cooldown_ms if raw_interrupt_cooldown_ms is not None else 800.0)
        except Exception:
            interrupt_cooldown_ms = 800.0
        secret_access_key = str(self.secrets.get("porcupine_access_key", "") or "").strip()
        env_access_key = str(os.getenv("PICOVOICE_ACCESS_KEY", "") or "").strip()
        env_access_key_alt = str(os.getenv("PORCUPINE_ACCESS_KEY", "") or "").strip()
        try:
            sensitivity = float(raw_sensitivity if raw_sensitivity is not None else 0.7)
        except Exception:
            sensitivity = 0.7
        def _is_probably_path(v: str) -> bool:
            if not v:
                return False
            s = str(v).strip().lower()
            if s.endswith(".ppn") or s.endswith(".pv") or s.endswith(".wav"):
                return True
            if "\\" in s:
                return True
            if s.startswith("./") or s.startswith("../") or s.startswith("models/"):
                return True
            if s.startswith("."):
                return True
            if len(s) >= 2 and s[1] == ":":
                return True
            return False
        def _is_probably_key(v: str) -> bool:
            if not v:
                return False
            s = str(v).strip()
            if _is_probably_path(s):
                return False
            return len(s) >= 24
        def _resolve_kws_path(v: str, fallback_name: str) -> str:
            s = str(v or "").strip()
            candidate = s or fallback_name
            p = Path(candidate)
            if p.is_absolute() and p.exists():
                return str(p)
            try:
                rp = get_resource_path(candidate)
                if rp.exists():
                    return str(rp)
            except Exception:
                pass
            try:
                rp2 = get_resource_path(fallback_name)
                if rp2.exists():
                    return str(rp2)
            except Exception:
                pass
            return candidate
        for candidate in [env_access_key, env_access_key_alt, secret_access_key, access_key]:
            if _is_probably_key(candidate):
                access_key = str(candidate).strip()
                break
        missing_any = (not access_key) or (not keyword_path) or (not model_path)
        access_invalid = _is_probably_path(access_key) or (not _is_probably_key(access_key))
        if missing_any or access_invalid:
            try:
                import wakeup as wakeup_cfg
                wk = str(getattr(wakeup_cfg, "ACCESS_KEY", "") or "").strip()
                wp = str(getattr(wakeup_cfg, "KEYWORD_PATH", "") or "").strip()
                wm = str(getattr(wakeup_cfg, "MODEL_PATH", "") or "").strip()
                ws = getattr(wakeup_cfg, "SENSITIVITY", sensitivity)
                if _is_probably_key(wk):
                    access_key = wk
                if wp:
                    keyword_path = wp
                if wm:
                    model_path = wm
                try:
                    sensitivity = float(ws if ws is not None else sensitivity)
                except Exception:
                    pass
            except Exception:
                pass
        swapped = False
        if _is_probably_path(access_key) and _is_probably_key(keyword_path):
            access_key, keyword_path = keyword_path, access_key
            swapped = True
        if _is_probably_path(access_key) and _is_probably_key(model_path):
            access_key, model_path = model_path, access_key
            swapped = True
        should_persist = swapped or missing_any or access_invalid
        if should_persist:
            try:
                self.cfg.set("porcupine_access_key", access_key)
                self.cfg.set("porcupine_keyword_path", keyword_path)
                self.cfg.set("porcupine_model_path", model_path)
                self.cfg.set("porcupine_sensitivity", sensitivity)
                self.cfg.save()
            except Exception:
                pass
            if enabled:
                try:
                    self._on_log("KWS配置已自动纠正（检测到 AccessKey/模型路径异常或缺失）")
                except Exception:
                    pass
        keyword_path = _resolve_kws_path(keyword_path, "sjg_zh_windows_v4_0_0.ppn")
        model_path = _resolve_kws_path(model_path, "porcupine_params_zh.pv")
        return {
            "enabled": enabled,
            "access_key": access_key,
            "keyword_path": keyword_path,
            "model_path": model_path,
            "sensitivity": sensitivity,
            "interrupt_fallback_enabled": interrupt_fallback_enabled,
            "interrupt_cooldown_ms": interrupt_cooldown_ms,
        }

    def _apply_asr_profile_to_worker(self):
        if not self.worker:
            return
        self.worker.apply_dynamic_profile(self._get_asr_profile_from_config())
        self.worker.apply_interrupt_match_config(self._get_interrupt_match_config_from_config())

    def _reload_force_exit_guard_config(self):
        try:
            interval_ms = int(float(self.cfg.get("force_exit_check_interval_ms", 10000)))
        except Exception:
            interval_ms = 10000
        interval_ms = max(1000, min(60000, interval_ms))
        try:
            timeout_sec = float(self.cfg.get("force_exit_timeout_seconds", 23.6))
        except Exception:
            timeout_sec = 23.6
        timeout_sec = max(1.0, min(300.0, timeout_sec))
        self.force_exit_check_interval_ms = interval_ms
        self.force_exit_timeout_seconds = timeout_sec
        try:
            timer = getattr(self, "force_exit_guard_timer", None)
            if timer:
                timer.setInterval(int(interval_ms))
        except Exception:
            pass

    def _reload_idle_prompt_config(self):
        try:
            prompt_sec = float(self.cfg.get("idle_prompt_seconds", 5.0))
        except Exception:
            prompt_sec = 5.0
        try:
            close_wait_sec = float(self.cfg.get("idle_close_wait_seconds", 5.0))
        except Exception:
            close_wait_sec = 5.0
        self.idle_prompt_seconds = max(1.0, min(120.0, prompt_sec))
        self.idle_close_wait_seconds = max(1.0, min(120.0, close_wait_sec))

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

    def _on_settings_interrupt_config_changed(self, cfg: dict):
        if self.worker:
            try:
                self.worker.apply_interrupt_match_config(cfg)
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

    def _resolve_preferred_device_index(self):
        devices = []
        try:
            devices = PvRecorder.get_available_devices()
        except Exception as e:
            self._on_log(f"枚举麦克风失败，将尝试系统默认设备: {e}")
        configured_name = str(self.cfg.get("device_name", "") or "").strip()
        try:
            configured_index = int(self.cfg.get("device_index", -1))
        except Exception:
            configured_index = -1
        if devices:
            if configured_name and configured_name in devices:
                device_index = devices.index(configured_name)
            elif 0 <= configured_index < len(devices):
                device_index = configured_index
            else:
                device_index = 0
        else:
            device_index = -1
        try:
            self.cfg.set("device_index", int(device_index))
            if devices and 0 <= device_index < len(devices):
                self.cfg.set("device_name", str(devices[device_index]))
            self.cfg.save()
        except Exception:
            pass
        return device_index, devices

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
        self._set_single_turn_close_pending(False)

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

        device_index, devices = self._resolve_preferred_device_index()
        try:
            if devices and 0 <= device_index < len(devices):
                self._on_log(f"监听设备: {devices[device_index]} (index={device_index})")
            else:
                self._on_log("未检测到可枚举麦克风，改为尝试系统默认录音设备")
        except Exception:
            pass
        phrases = self.cfg.get("asr_wake_phrases", "")
        default_asr_key = "sk-28313ea70a8f47d09a6cd1cab51c477e"
        api_key = (self.secrets.get("asr_api_key", self.cfg.get("aliyun_appkey", default_asr_key)) or "").strip()
        if not api_key:
            api_key = default_asr_key
        audio_match_enabled = bool(self.cfg.get("audio_match_enabled", True))
        audio_match_threshold = float(self.cfg.get("audio_match_threshold", 0.45))
        kws_config = self._get_kws_config_from_config()
        self.worker = AudioWakeWorkerASR(
            phrases,
            device_index,
            api_key,
            audio_match_enabled=audio_match_enabled,
            audio_match_threshold=audio_match_threshold,
            kws_config=kws_config
        )
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
        self._to_wake_mode("程序启动")
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
            self._to_chat_idle_mode("确保对话模式")
            self._on_log("ASR 切换到 chat 模式")

    def _on_tts_timeout(self):
        self._awaiting_tts_start = False
        self.force_exit_watch_started_ts = 0.0
        try:
            self.tts_start_wait_timer.stop()
        except Exception:
            pass
        if self.is_speaking:
            self._on_log("TTS 状态超时重置")
            self.is_speaking = False
            self.awaiting_response = False
            self.force_exit_watch_started_ts = time.time()
            self._to_chat_idle_mode("TTS超时")
            self._schedule_idle_prompt_if_possible()

    def _on_tts_start_wait_timeout(self):
        if not self._awaiting_tts_start:
            return
        if self.is_speaking:
            self._awaiting_tts_start = False
            return
        self._awaiting_tts_start = False
        self.force_exit_watch_started_ts = 0.0
        if self._should_close_after_answer():
            self._on_log("回答后等待TTS启动超时，当前为单轮会话，立即退出数字人")
            QTimer.singleShot(0, self._try_close_player_after_answer)
            return
        self._on_log("回答后等待TTS启动超时，转为空闲计时")
        self._to_chat_idle_mode("等待TTS启动超时")

    def _enter_waiting_tts_start(self, reason: str):
        self._awaiting_tts_start = True
        self.force_exit_watch_started_ts = time.time()
        self._cancel_idle_timers()
        self._to_speaking_mode(reason)
        try:
            self.tts_start_wait_timer.start(int(float(self.post_answer_tts_wait_seconds) * 1000))
        except Exception:
            pass
        self._on_log(f"回答已生成，等待TTS启动: {float(self.post_answer_tts_wait_seconds):.1f}s")

    def _mark_tts_start(self):
        self._awaiting_tts_start = False
        self.force_exit_watch_started_ts = time.time()
        try:
            self.tts_start_wait_timer.stop()
        except Exception:
            pass
        self.is_speaking = True
        self.last_speaking_start_ts = time.monotonic()
        self.last_tts_start_ts = time.monotonic()
        self._cancel_idle_timers()
        try:
            if self.worker:
                self.worker.set_speaking_state(True)
                if self._suppress_interrupt_toggle:
                    self._to_chat_idle_mode("打断确认TTS")
                    self._suppress_interrupt_toggle = False
                else:
                    self._to_speaking_mode("数字人回答中")
        except Exception:
            pass
        try:
            self.tts_guard_timer.start(self.tts_timeout_ms)
        except Exception:
            pass

    def _mark_tts_end(self):
        self._awaiting_tts_start = False
        self.force_exit_watch_started_ts = 0.0
        try:
            self.tts_start_wait_timer.stop()
        except Exception:
            pass
        self.is_speaking = False
        self.last_tts_start_ts = 0.0
        self.awaiting_response = False
        keep_rule5_stage = bool(self._idle_prompt_sent)
        self._idle_waiting_for_close = False
        if not keep_rule5_stage:
            self._idle_prompt_sent = False
        self._cancel_idle_timers()
        try:
            if self.worker:
                self.worker.set_speaking_state(False)
                if keep_rule5_stage:
                    self._to_chat_idle_mode("规则九-语音播放完成进入规则五计时")
                else:
                    self._to_chat_idle_mode("规则九-语音播放完成进入规则四")
        except Exception:
            pass
        try:
            self.tts_guard_timer.stop()
        except Exception:
            pass
        if self._should_close_after_answer():
            self._on_log("回答播报完成，准备立即退出数字人")
            QTimer.singleShot(120, self._try_close_player_after_answer)
            return
        if keep_rule5_stage:
            self._on_log("规则九: 语音播放完成，保持规则五阶段计时")
        else:
            self._on_log("规则九: 语音播放完成，立即进入规则四计时")
        self._schedule_idle_prompt_if_possible()

    def _try_close_player_after_answer(self):
        if not self._should_close_after_answer():
            return
        if self.awaiting_response or self.is_speaking or self._awaiting_tts_start:
            QTimer.singleShot(120, self._try_close_player_after_answer)
            return
        pending_tts = False
        pending_playback_events = 0
        try:
            with self.tts_order_cond:
                pending_tts = bool(self.tts_play_index <= self.tts_submit_index)
        except Exception:
            pending_tts = False
        try:
            with self.tts_lock:
                pending_playback_events = len(self.tts_played_events)
        except Exception:
            pending_playback_events = 0
        force_close = False
        elapsed = 0.0
        try:
            if self.single_turn_close_requested_ts > 0:
                elapsed = time.monotonic() - float(self.single_turn_close_requested_ts)
                force_close = elapsed >= 2.5
        except Exception:
            elapsed = 0.0
            force_close = False
        if pending_tts or pending_playback_events > 0:
            if not force_close:
                QTimer.singleShot(120, self._try_close_player_after_answer)
                return
            self._on_log(
                f"回答结束后关闭等待超时，强制退出数字人: pending_tts={pending_tts}, "
                f"pending_playback_events={pending_playback_events}, elapsed={elapsed:.2f}s"
            )
        self._set_single_turn_close_pending(False)
        if not self.player_active:
            return
        self._on_log("回答已结束，立即退出数字人")
        try:
            self.close_player_signal.emit()
        except Exception:
            self.close_player()

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
        self._set_single_turn_close_pending(False)

    def _clear_pending_user_input(self):
        self.pending_user_text = None
        try:
            self.user_input_timer.stop()
        except Exception:
            pass
        try:
            self.ws_broadcast({"type": "USER_INPUT_REVOKE"})
        except Exception:
            pass

    def _clear_manual_command_state(self):
        self.manual_command_armed_until_ts = 0.0

    def _arm_manual_command_state(self):
        self.manual_command_armed_until_ts = time.monotonic() + float(self.manual_command_timeout_seconds)

    def _is_manual_command_active(self) -> bool:
        if not self.player_active:
            return False
        armed_until = float(self.manual_command_armed_until_ts or 0.0)
        if armed_until <= 0:
            return False
        if time.monotonic() > armed_until:
            self.manual_command_armed_until_ts = 0.0
            return False
        return True

    def _looks_like_manual_attention_call(self, text: str) -> bool:
        normalized = self._normalize_text(text)
        if not normalized:
            return False
        aliases = ["小石警官", "小石", "警官"]
        suffixes = {"", "啊", "呀", "哎", "诶", "欸", "喂", "在吗", "在嘛", "在不在", "在没在", "出来", "出来下", "出来一下"}
        for alias in aliases:
            if normalized == alias:
                return True
            if normalized.startswith(alias):
                suffix = normalized[len(alias):]
                if suffix in suffixes or len(suffix) <= 2:
                    return True
        return False

    def _is_wake_intro_text(self, text: str) -> bool:
        normalized = self._normalize_text(text)
        if not normalized:
            return False
        return normalized in {"你好请讲", "您好请讲"}

    def _should_suppress_short_ack(self) -> bool:
        if not self._in_wake_grace():
            return False
        return self._is_wake_intro_text(self.last_successful_response_text)

    def _should_force_interrupt_after_wake(self, text: str) -> bool:
        normalized = self._normalize_text(text)
        if not normalized:
            return False
        if not self._in_wake_grace():
            return False
        if not self._is_wake_intro_text(self.last_successful_response_text):
            return False
        compact = "".join(c for c in normalized if c.isalnum() or '\u4e00' <= c <= '\u9fff')
        if len(compact) < 4:
            return False
        if self._heuristic_should_ignore_user_input(compact):
            return False
        return True

    def _emit_manual_ack(self):
        if self._should_suppress_short_ack():
            self._on_log("当前处于唤醒欢迎语阶段，跳过托底“嗯”应答")
            return
        text = "嗯"
        try:
            self.chat_history.append({"role": "assistant", "content": text})
        except Exception:
            pass
        try:
            self._enter_waiting_tts_start("托底应答播报")
            self.ws_broadcast({"type": "CHAT_APPEND", "role": "assistant", "text": text})
        except Exception:
            pass
        self._mark_successful_response(text)

    def _heuristic_is_exit_command(self, text: str) -> bool:
        normalized = self._normalize_text(text)
        if not normalized:
            return False
        keywords = [
            "退出", "关闭", "退下", "关掉", "关了", "结束", "下去", "消失", "隐藏", "最小化",
            "回托盘", "回到托盘", "退到托盘", "返回托盘", "收起来", "收起", "撤下", "下班"
        ]
        if any(k in normalized for k in keywords):
            return True
        patterns = [
            r"(你|你先|你就)?退下吧",
            r"(你|你先|你就)?退出吧",
            r"(你|你先|你就)?关闭吧",
            r"回(到)?托盘(区)?吧?",
            r"退(到)?托盘(区)?吧?",
        ]
        return any(re.search(p, normalized) for p in patterns)

    async def _llm_classify_manual_command(self, text: str) -> str:
        text = self._normalize_text(text)
        if self._heuristic_is_exit_command(text):
            return "__EXIT__"
        if self._heuristic_should_ignore_user_input(text):
            heuristic_default = "__IGNORE__"
        else:
            heuristic_default = "__CHAT__"
        base_url = self.cfg.get("llm_base_url", "")
        api_key = self.secrets.get("llm_api_key", "")
        model = self.cfg.get("llm_model", "")
        if not base_url or not api_key or not model:
            return heuristic_default
        try:
            client = AsyncOpenAI(base_url=base_url, api_key=api_key)
            system_prompt = (
                "你是数字人会话控制分类器。你只能输出三种结果之一：__EXIT__、__CHAT__、__IGNORE__。\n"
                "当前上下文是：用户先呼叫了“小石警官”，数字人已回应“嗯”，正在等待下一句。\n"
                "如果用户表达的是让数字人退出、关闭、退下、隐藏、最小化、回托盘区，输出 __EXIT__。\n"
                "如果用户是在继续提问、咨询、求助、办理业务、报警报案，输出 __CHAT__。\n"
                "如果只是语气词、环境噪声、无意义碎句，输出 __IGNORE__。\n"
                "不要输出任何其他字符。"
            )
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                max_tokens=6,
                temperature=0,
                stream=False,
            )
            out = ""
            try:
                out = (resp.choices[0].message.content or "").strip()
            except Exception:
                out = ""
            if out.startswith("__EXIT__"):
                return "__EXIT__"
            if out.startswith("__CHAT__"):
                return "__CHAT__"
            if out.startswith("__IGNORE__"):
                return "__IGNORE__"
        except Exception:
            return heuristic_default
        return heuristic_default

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
        if self.awaiting_response or self._is_single_turn_locked():
            return
        self.pending_user_text = self._merge_user_text(self.pending_user_text, text)
        try:
            self.ws_broadcast({"type": "USER_INPUT_DRAFT", "text": self.pending_user_text})
        except Exception:
            pass
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
        text = self._normalize_text(text)
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
        if self._is_manual_command_active():
            verdict = await self._llm_classify_manual_command(text)
            if verdict == "__EXIT__":
                self._clear_manual_command_state()
                self.awaiting_response = False
                self.force_exit_watch_started_ts = 0.0
                try:
                    self.ws_broadcast({"type": "USER_INPUT_REVOKE"})
                except Exception:
                    pass
                self._on_log(f"托底指令判定为退出: {text}")
                try:
                    self.close_player_signal.emit()
                except Exception:
                    pass
                return
            if verdict == "__IGNORE__":
                self.awaiting_response = False
                self.force_exit_watch_started_ts = 0.0
                try:
                    self.ws_broadcast({"type": "USER_INPUT_REVOKE"})
                except Exception:
                    pass
                self._to_chat_idle_mode("托底控制无效输入")
                self._on_log(f"托底指令判定无效: {text}")
                return
            self._clear_manual_command_state()
        if self._heuristic_should_ignore_user_input(text):
            self.awaiting_response = False
            self._awaiting_tts_start = False
            self.force_exit_watch_started_ts = time.time()
            try:
                self.ws_broadcast({"type": "USER_INPUT_REVOKE"})
            except Exception:
                pass
            if self._idle_prompt_sent:
                self._keep_rule5_countdown("规则八无效问题")
            self._to_chat_idle_mode("问题无效-规则八过滤")
            self._on_log(f"问题判定无效: {text}")
            return
        verdict = await self._llm_gate_user_input(text)
        if verdict == "__IGNORE__":
            self.awaiting_response = False
            self._awaiting_tts_start = False
            self.force_exit_watch_started_ts = time.time()
            try:
                self.ws_broadcast({"type": "USER_INPUT_REVOKE"})
            except Exception:
                pass
            if self._idle_prompt_sent:
                self._keep_rule5_countdown("LLM无效问题")
            self._to_chat_idle_mode("问题无效-LLM过滤")
            self._on_log(f"问题判定无效(LLM): {text}")
            return
        self.awaiting_response = True
        self._note_activity()
        self.force_exit_watch_started_ts = time.time()
        self._to_speaking_mode("规则二-LLM判定有效问题")
        try:
            self.ws_broadcast({"type": "USER_INPUT_CONFIRM", "text": text})
        except Exception:
            pass
        self._on_log(f"问题已接收并准备回答: {text}")
        self._set_single_turn_close_pending(True)
        await self.ask_llm(text)

    def _commit_pending_user_input(self):
        text = (self.pending_user_text or "").strip()
        self.pending_user_text = None
        clean_text = "".join(c for c in text if c.isalnum() or '\u4e00' <= c <= '\u9fff')
        if len(clean_text) < 2:
            try:
                self.ws_broadcast({"type": "USER_INPUT_REVOKE"})
            except Exception:
                pass
            return
        if self.awaiting_response or self.is_speaking or self._awaiting_tts_start or self._is_single_turn_locked():
            try:
                self.ws_broadcast({"type": "USER_INPUT_REVOKE"})
            except Exception:
                pass
            return
        if self._is_manual_command_active() and (not hasattr(self, 'ws_loop') or not self.ws_loop or not self.ws_loop.is_running()):
            verdict = "__EXIT__" if self._heuristic_is_exit_command(text) else "__IGNORE__"
            if verdict == "__EXIT__":
                self._clear_manual_command_state()
                self._on_log(f"托底指令(本地)判定为退出: {text}")
                try:
                    self.close_player_signal.emit()
                except Exception:
                    pass
            else:
                try:
                    self.ws_broadcast({"type": "USER_INPUT_REVOKE"})
                except Exception:
                    pass
            return
        if hasattr(self, 'ws_loop') and self.ws_loop and self.ws_loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(self._process_committed_user_input(text), self.ws_loop)
            except Exception:
                self.awaiting_response = False
                self._awaiting_tts_start = False
                self.force_exit_watch_started_ts = time.time()
                self._to_chat_idle_mode("提问提交失败")
                try:
                    self.ws_broadcast({"type": "USER_INPUT_REVOKE"})
                except Exception:
                    pass
        else:
            self._on_log("Error: WS Loop not running, cannot ask LLM")
            self.awaiting_response = False
            self._awaiting_tts_start = False
            self.force_exit_watch_started_ts = time.time()
            self._to_chat_idle_mode("提问提交失败")
            try:
                self.ws_broadcast({"type": "USER_INPUT_REVOKE"})
            except Exception:
                pass

    def exit_app(self):
        """
        完全安全的退出逻辑
        
        1. 停止监听
        2. 关闭所有窗口
        3. 退出应用程序
        """
        try:
            self._cancel_idle_timers()
        except Exception:
            pass
        try:
            self.tts_guard_timer.stop()
        except Exception:
            pass
        try:
            with self.tts_pool_lock:
                pool = self.tts_pool
                self.tts_pool = None
            if pool:
                pool.shutdown()
        except Exception:
            pass
        worker = self.worker
        self.worker = None
        if worker:
            def _stop_worker():
                try:
                    worker.stop()
                except Exception:
                    pass
            threading.Thread(target=_stop_worker, daemon=True).start()
        try:
            self.ctrl.close()
        except Exception:
            pass
        try:
            self.log_win.close()
        except Exception:
            pass
        try:
            self.quit()
        finally:
            def _force_exit():
                time.sleep(2.0)
                try:
                    os._exit(0)
                except Exception:
                    pass
            threading.Thread(target=_force_exit, daemon=True).start()

    def handle_audio_interrupt_command(self, command_name: str):
        self._note_activity()
        self._set_single_turn_close_pending(False)
        was_speaking = bool(self.is_speaking)
        source = "audio_template"
        try:
            if command_name == "asr_fallback":
                source = "asr_fallback"
            elif command_name == "kws_fallback":
                source = "kws_fallback"
            elif str(command_name).startswith("frontend"):
                source = "frontend"
        except Exception:
            source = "audio_template"
        self._on_log(f"打断命中可视化日志（来源={source}）")
        self._on_log(f"Audio interrupt command: {command_name}")
        self.stop_generation = True
        self.awaiting_response = False
        self._awaiting_tts_start = False
        self.last_interrupt_ts = time.monotonic()
        try:
            if self.worker:
                self._to_chat_idle_mode("规则三-收到打断")
        except Exception:
            pass
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
        if self.ws_clients and was_speaking and (not self._should_suppress_short_ack()):
            text = "嗯"
            self._suppress_interrupt_toggle = True
            try:
                self.chat_history.append({"role": "assistant", "content": text})
            except Exception:
                pass
            try:
                self._enter_waiting_tts_start("打断确认播报")
                self.ws_broadcast({"type": "CHAT_APPEND", "role": "assistant", "text": text})
            except Exception:
                pass
            self._mark_successful_response(text)
            self._on_log("规则三: 发声时收到打断，已回复“嗯”并启动规则四计时")
            self._schedule_idle_prompt_if_possible()
        elif self.ws_clients and was_speaking:
            self._on_log("发声时收到打断，但当前为唤醒欢迎语阶段，跳过“嗯”确认")

    def handle_backend_asr(self, text: str):
        """处理后台 ASR 识别到的对话内容"""
        text = self._normalize_text(text)
        was_idle_prompt_stage = bool(self._idle_prompt_sent)
        self._note_activity()
        self._on_log(f"Backend ASR: {text}")
        if self.player_active and self._looks_like_manual_attention_call(text):
            sent_ack = False
            if self.is_speaking or self.awaiting_response or self._awaiting_tts_start:
                sent_ack = bool(self.is_speaking and self.ws_clients)
                try:
                    self.handle_audio_interrupt_command("asr_fallback")
                except Exception:
                    self._force_release_speaking_state()
                    self.awaiting_response = False
                    self._awaiting_tts_start = False
                    self.force_exit_watch_started_ts = time.time()
            self._arm_manual_command_state()
            if self.worker:
                self._to_chat_idle_mode("托底口令唤起")
            if not sent_ack:
                self._emit_manual_ack()
            self._on_log(f"托底口令唤起成功: {text}")
            return

        if not self.ws_clients:
            self._on_log("Warn: Chat input received but no clients connected. Switching back to wake mode.")
            if self.worker:
                self.worker.set_mode("wake")
            return

        clean_text = "".join(c for c in text if c.isalnum() or '\u4e00' <= c <= '\u9fff')
        wake_grace_active = self._in_wake_grace()
        if wake_grace_active and self._is_wake_intro_text(clean_text):
            self._on_log(f"忽略唤醒欢迎语回声: {text}")
            return
        exit_keywords = ["没有", "没了", "不用了", "不需要", "退下", "退出", "结束", "关闭"]
        if was_idle_prompt_stage and any(k in clean_text for k in exit_keywords):
            self._on_log(f"规则六: 追问后命中结束关键词，关闭数字人: {text}")
            try:
                self.close_player_signal.emit()
            except Exception:
                pass
            return
        interrupt_keywords = ["停", "停下", "闭嘴", "可以了", "别说", "安静", "打断", "不听", "等等", "等会", "我打断你一下", "好啦", "别说了"]
        speaking_stale = self._is_speaking_stale()
        peak = 0.0
        try:
            if self.worker:
                peak = float(getattr(self.worker, "last_asr_peak", 0.0) or 0.0)
        except Exception:
            peak = 0.0
        rms = 0.0
        try:
            if self.worker:
                rms = float(getattr(self.worker, "last_asr_rms", 0.0) or 0.0)
        except Exception:
            rms = 0.0
        peak_threshold = 0.085
        rms_threshold = 0.015
        try:
            if self.worker:
                peak_threshold = float(getattr(self.worker, "speaking_interrupt_peak", 0.085) or 0.085)
                rms_threshold = float(getattr(self.worker, "speaking_interrupt_rms", 0.015) or 0.015)
        except Exception:
            peak_threshold = 0.085
            rms_threshold = 0.015
        force_interrupt_after_wake = self._should_force_interrupt_after_wake(clean_text)
        matched_keywords = [k for k in interrupt_keywords if k in clean_text]
        has_interrupt_keyword = len(matched_keywords) > 0
        loud_interrupt_raw = (peak >= peak_threshold) and (rms >= rms_threshold)
        loud_interrupt = False
        should_try_interrupt = self.is_speaking or self.awaiting_response
        self._on_log(
            "ASR托底打断判定详情: "
            f"text={text!r} clean={clean_text!r} "
            f"is_speaking={self.is_speaking} awaiting_response={self.awaiting_response} "
            f"should_try_interrupt={should_try_interrupt} "
            f"has_interrupt_keyword={has_interrupt_keyword} matched_keywords={matched_keywords} "
            f"peak={peak:.4f}/{peak_threshold:.4f} rms={rms:.4f}/{rms_threshold:.4f} "
            f"loud_interrupt_raw={loud_interrupt_raw} loud_interrupt={loud_interrupt} "
            f"wake_grace_active={wake_grace_active} text_len={len(clean_text)} "
            f"force_interrupt_after_wake={force_interrupt_after_wake} "
            f"speaking_stale={speaking_stale}"
        )
        if should_try_interrupt and (has_interrupt_keyword or force_interrupt_after_wake or speaking_stale):
            if speaking_stale:
                self._on_log("Detected stale speaking state, force reset")
                self._force_release_speaking_state()
            self._on_log(
                "ASR托底打断触发原因: "
                f"keyword={has_interrupt_keyword} "
                f"wake_grace={force_interrupt_after_wake} stale={speaking_stale}"
            )
            self._on_log(f"ASR托底打断触发: {text}")
            self.handle_audio_interrupt_command("asr_fallback")
            return
        if self.is_speaking or self.awaiting_response or self._awaiting_tts_start or self._is_single_turn_locked():
            self._on_log(f"Ignored input during single-turn lock: {text}")
            return
        self._queue_user_input(text)

    def _in_wake_grace(self) -> bool:
        if self.last_wake_ts <= 0:
            return False
        return (time.monotonic() - self.last_wake_ts) <= float(self.wake_grace_seconds)

    def _set_single_turn_close_pending(self, enabled: bool):
        enabled = bool(enabled)
        self.close_after_answer = enabled
        self.single_turn_close_pending = enabled
        if enabled:
            if self.single_turn_close_requested_ts <= 0:
                self.single_turn_close_requested_ts = time.monotonic()
        else:
            self.single_turn_close_requested_ts = 0.0

    def _should_close_after_answer(self) -> bool:
        return bool(self.close_after_answer or self.single_turn_close_pending)

    def _is_single_turn_locked(self) -> bool:
        return self._should_close_after_answer()

    def _is_speaking_stale(self) -> bool:
        if not self.is_speaking:
            return False
        if self.last_tts_start_ts <= 0:
            return True
        return (time.monotonic() - self.last_tts_start_ts) > 6.0

    def _force_release_speaking_state(self):
        self.is_speaking = False
        self.last_tts_start_ts = 0.0
        self._awaiting_tts_start = False
        try:
            self.tts_guard_timer.stop()
        except Exception:
            pass
        try:
            if self.worker:
                self.worker.set_speaking_state(False)
        except Exception:
            pass

    def _apply_listener_state(self, mode: str, asr_enabled: bool, wake_enabled: bool, interrupt_enabled: bool, reason: str):
        if not self.worker:
            return
        try:
            self.worker.set_mode(mode)
            self.worker.set_asr_enabled(bool(asr_enabled))
            self.worker.set_wake_listener_enabled(bool(wake_enabled))
            self.worker.set_interrupt_listener_enabled(bool(interrupt_enabled))
            self.worker.resume()
            self._on_log(
                f"监听状态切换[{reason}] mode={mode} asr={bool(asr_enabled)} wake_audio={bool(wake_enabled)} interrupt_audio={bool(interrupt_enabled)}"
            )
        except Exception as e:
            self._on_log(f"监听状态切换失败[{reason}]: {e}")

    def _to_wake_mode(self, reason: str):
        self.awaiting_response = False
        self._awaiting_tts_start = False
        self.force_exit_watch_started_ts = 0.0
        self._apply_listener_state("wake", False, True, False, reason)

    def _to_chat_idle_mode(self, reason: str):
        if not self.awaiting_response and not self._awaiting_tts_start:
            self.force_exit_watch_started_ts = time.time()
        self._apply_listener_state("chat", True, False, bool(self.awaiting_response or self._awaiting_tts_start), reason)
        if not self.awaiting_response and not self._awaiting_tts_start:
            self._schedule_idle_prompt_if_possible()

    def _to_speaking_mode(self, reason: str):
        self._apply_listener_state("chat", self.asr_fallback_during_tts, False, True, reason)

    def _test_tcp(self, host: str, port: int, timeout: float = 1.8) -> bool:
        try:
            with socket.create_connection((host, int(port)), timeout=timeout):
                return True
        except Exception:
            return False

    def _run_startup_diagnostics(self):
        try:
            net_ok = self._test_tcp("1.1.1.1", 53, 1.5) or self._test_tcp("223.5.5.5", 53, 1.5)
            self._on_log(f"启动自检: 网络连通={'OK' if net_ok else 'FAIL'}")
        except Exception as e:
            self._on_log(f"启动自检: 网络检测异常 {e}")
        try:
            base_url = (self.cfg.get("llm_base_url", "") or "").strip()
            model = (self.cfg.get("llm_model", "") or "").strip()
            api_key = (self.secrets.get("llm_api_key", "") or "").strip()
            if not base_url or not model or not api_key:
                self._on_log("启动自检: LLM配置不完整")
            else:
                u = urlparse(base_url)
                host = (u.hostname or "").strip()
                port = u.port or (443 if (u.scheme or "https").lower() == "https" else 80)
                llm_ok = bool(host) and self._test_tcp(host, int(port), 2.0)
                self._on_log(f"启动自检: LLM连通={'OK' if llm_ok else 'FAIL'} host={host}:{port}")
        except Exception as e:
            self._on_log(f"启动自检: LLM检测异常 {e}")
        try:
            asr_key = "sk-28313ea70a8f47d09a6cd1cab51c477e"
            asr_ok = bool(asr_key and len(asr_key) > 10)
            self._on_log(f"启动自检: ASR配置={'OK' if asr_ok else 'FAIL'}")
        except Exception as e:
            self._on_log(f"启动自检: ASR检测异常 {e}")
        try:
            tts_ok = bool(self._get_aliyun_tts_api_key())
            self._on_log(f"启动自检: 阿里云TTS配置={'OK' if tts_ok else 'FAIL'} model={self.tts_model} voice={self.tts_voice}")
        except Exception as e:
            self._on_log(f"启动自检: 阿里云TTS检测异常 {e}")

    def _note_activity(self):
        self._awaiting_tts_start = False
        self.force_exit_watch_started_ts = time.time()
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
        try:
            self.tts_start_wait_timer.stop()
        except Exception:
            pass

    def _schedule_idle_prompt_if_possible(self):
        if self.is_speaking:
            return
        if self._should_close_after_answer():
            QTimer.singleShot(0, self._try_close_player_after_answer)
            return
        if (not self.ws_clients) and (not self.player_active):
            return
        if not self.worker or getattr(self.worker, "_mode", "wake") != "chat":
            return
        if self._idle_waiting_for_close:
            return
        try:
            if self._idle_prompt_sent:
                wait_seconds = float(self.idle_close_wait_seconds)
                self._on_log(f"规则五计时启动: {wait_seconds:.1f}s")
                self.idle_close_timer.start(int(wait_seconds * 1000))
            else:
                wait_seconds = float(self.idle_prompt_seconds)
                self._on_log(f"规则四计时启动: {wait_seconds:.1f}s")
                self.idle_prompt_timer.start(int(wait_seconds * 1000))
            self._idle_waiting_for_close = True
        except Exception:
            pass

    def _keep_rule5_countdown(self, reason: str):
        if not self._idle_prompt_sent:
            return
        try:
            if not self.idle_close_timer.isActive():
                wait_seconds = float(self.idle_close_wait_seconds)
                self._on_log(f"规则五计时启动: {wait_seconds:.1f}s")
                self.idle_close_timer.start(int(wait_seconds * 1000))
            self._idle_waiting_for_close = True
            self._on_log(f"规则五阶段保持: {reason}，仅有效问题可取消关闭")
        except Exception:
            pass

    def _on_idle_prompt_timeout(self):
        if self.is_speaking:
            self._idle_waiting_for_close = False
            self._schedule_idle_prompt_if_possible()
            return
        if not self.worker or getattr(self.worker, "_mode", "wake") != "chat":
            self._idle_waiting_for_close = False
            return
        self._idle_waiting_for_close = False
        self._idle_prompt_sent = True
        if self.ws_clients:
            text = "请问还有其他的问题吗？"
            self._on_log("规则四: 首次等待超时，发起追问并进入第二次计时")
            try:
                self.chat_history.append({"role": "assistant", "content": text})
            except Exception:
                pass
            try:
                self._enter_waiting_tts_start("追问播报")
                self.ws_broadcast({"type": "IDLE_PROMPT", "text": text})
            except Exception:
                pass
            self._mark_successful_response(text)
        else:
            self._on_log("规则四: 无前端连接，跳过追问，直接进入规则五计时")
        self._schedule_idle_prompt_if_possible()

    def _on_idle_close_timeout(self):
        if self.is_speaking:
            self._idle_waiting_for_close = False
            return
        if not self.worker or getattr(self.worker, "_mode", "wake") != "chat":
            self._idle_waiting_for_close = False
            return
        self._idle_waiting_for_close = False
        self._idle_prompt_sent = False
        self._on_log("规则五: 第二次等待超时，执行最小化并回到唤醒监听")
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
        lowered = str(msg or "").lower()
        if ("recorder init failed" in lowered) or ("failed to initialize pvrecorder" in lowered):
            try:
                self.act_start.setEnabled(True)
                self.act_stop.setEnabled(False)
            except Exception:
                pass
            try:
                self.ctrl.set_running(False)
            except Exception:
                pass
            try:
                QMessageBox.warning(
                    None,
                    "麦克风不可用",
                    "无法初始化录音设备。\n"
                    "请检查：\n"
                    "1) Windows 设置 > 隐私和安全性 > 麦克风，已允许桌面应用访问\n"
                    "2) 麦克风未被其他程序独占\n"
                    "3) 系统声音设置中存在可用输入设备"
                )
            except Exception:
                pass

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
        3. 处理引擎状态消息 (ready, listening started) - 记录到日志
        """
        try:
            if msg.startswith("asr_progress:"):
                return
            if msg.startswith("asr: "):
                t = msg[5:].strip()
                self._on_log(f"ASR识别: {t}")
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
            self._on_log(f"浏览器识别: {text}")
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

    def _cleanup_local_server_port(self, port: int) -> bool:
        cleaned = False
        try:
            current_pid = os.getpid()
            for conn in psutil.net_connections(kind="inet"):
                try:
                    if not conn.laddr:
                        continue
                    if int(getattr(conn.laddr, "port", 0) or 0) != int(port):
                        continue
                    pid = getattr(conn, "pid", None)
                    if not pid or pid == current_pid:
                        continue
                    proc = psutil.Process(pid)
                    self._on_log(f"发现端口 {port} 被占用，结束进程 pid={pid} name={proc.name()}")
                    proc.kill()
                    cleaned = True
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
                except Exception as e:
                    self._on_log(f"清理端口 {port} 占用失败: {e}")
        except Exception as e:
            self._on_log(f"扫描端口 {port} 占用失败: {e}")
        return cleaned

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
            try:
                srv = ThreadingHTTPServer(("127.0.0.1", 3400), SimpleHTTPRequestHandler)
                self._on_log("HTTP服务已启动: 127.0.0.1:3400")
                srv.serve_forever()
            except OSError as e:
                self._on_log(f"HTTP服务启动失败: {e}")
        self.http_thread = threading.Thread(target=run, daemon=True)
        if not self.ensure_port_free(3400):
            self._cleanup_local_server_port(3400)
        self.http_thread.start()

    def _get_aliyun_tts_api_key(self) -> str:
        default_asr_key = "sk-28313ea70a8f47d09a6cd1cab51c477e"
        api_key = (self.secrets.get("asr_api_key", self.cfg.get("aliyun_appkey", default_asr_key)) or "").strip()
        if not api_key:
            api_key = default_asr_key
        return api_key

    def _reload_tts_config(self):
        self.tts_model = str(self.cfg.get("aliyun_tts_model", "cosyvoice-v3-flash") or "cosyvoice-v3-flash").strip()
        self.tts_voice = str(self.cfg.get("aliyun_tts_voice", "longxiang") or "longxiang").strip()
        self._on_log(f"TTS 设置已应用: model={self.tts_model} voice={self.tts_voice}")

    def _normalize_aliyun_tts_rate(self, value) -> float:
        try:
            if isinstance(value, str):
                s = value.strip()
                if s.endswith("%"):
                    return max(0.5, min(2.0, 1.0 + float(s[:-1]) / 100.0))
                value = float(s)
            return max(0.5, min(2.0, float(value)))
        except Exception:
            return 1.0

    def _normalize_aliyun_tts_volume(self, value) -> int:
        try:
            if isinstance(value, str):
                s = value.strip()
                if s.endswith("%"):
                    return max(0, min(100, int(round(50 + float(s[:-1]) * 0.5))))
                value = float(s)
            return max(0, min(100, int(round(float(value)))))
        except Exception:
            return 50

    def _normalize_aliyun_tts_pitch(self, value) -> float:
        try:
            if isinstance(value, str):
                s = value.strip().lower()
                if s.endswith("hz"):
                    return max(0.5, min(2.0, 1.0 + float(s[:-2]) / 240.0))
                value = float(s)
            return max(0.5, min(2.0, float(value)))
        except Exception:
            return 1.0

    def _get_aliyun_tts_pool(self) -> SpeechSynthesizerObjectPool:
        api_key = self._get_aliyun_tts_api_key()
        if not api_key:
            raise RuntimeError("未配置阿里云 TTS API Key")
        dashscope.api_key = api_key
        with self.tts_pool_lock:
            if self.tts_pool is None:
                self.tts_pool = SpeechSynthesizerObjectPool(max_size=1)
            return self.tts_pool

    def _get_aliyun_tts_model_candidates(self):
        models = []
        for model in [self.tts_model, "cosyvoice-v1"]:
            name = str(model or "").strip()
            if name and name not in models:
                models.append(name)
        return models or ["cosyvoice-v1"]

    def _get_aliyun_tts_voice_candidates(self):
        voices = []
        for voice in [
            self.tts_voice,
            "longxiang",
            "longyuan",
            "longfei",
            "longtong",
            "longshuo",
            "longshu",
            "longlaotie",
            "longxiaocheng",
            "longxiaochun",
        ]:
            name = str(voice or "").strip()
            if name and name not in voices:
                voices.append(name)
        return voices or ["longxiang", "longxiaochun"]

    def _stream_aliyun_tts_once(self, req_id: str, text: str, cancel_seq: int, model: str, voice: str):
        callback = AliyunTtsStreamCallback()
        synthesizer = None
        sent_any_audio = False
        error_message = None
        pool = self._get_aliyun_tts_pool()
        try:
            synthesizer = pool.borrow_synthesizer(
                model=model,
                voice=voice,
                format=AudioFormat.MP3_22050HZ_MONO_256KBPS,
                volume=self.tts_volume,
                speech_rate=self.tts_rate,
                pitch_rate=self.tts_pitch,
                callback=callback,
                additional_params={"enable_ssml": False},
            )
            synthesizer.call(text, timeout_millis=30000)
            while True:
                if cancel_seq != self.tts_cancel_seq:
                    try:
                        synthesizer.streaming_cancel()
                    except Exception:
                        pass
                    break
                try:
                    chunk_data = callback.audio_queue.get(timeout=0.1)
                    sent_any_audio = True
                    b64 = base64.b64encode(chunk_data).decode("utf-8")
                    self.ws_broadcast({"type": "TTS_STREAM_CHUNK", "req_id": req_id, "chunk": b64})
                except queue.Empty:
                    if callback.done_event.is_set() and callback.audio_queue.empty():
                        break
            error_message = callback.error_message
        finally:
            if synthesizer is not None:
                try:
                    pool.return_synthesizer(synthesizer)
                except Exception:
                    pass
        return sent_any_audio, error_message

    def clear_tts_cache(self) -> tuple[int, int]:
        removed = 0
        failed = 0
        cache_dirs = [
            Path(__file__).resolve().parent / "player" / "tts_cache",
            self.runtime_data_dir / "tts_cache",
        ]
        for cache_dir in cache_dirs:
            try:
                if not cache_dir.exists():
                    continue
                for p in cache_dir.glob("tts_*.mp3"):
                    try:
                        p.unlink()
                        removed += 1
                    except Exception:
                        failed += 1
            except Exception:
                failed += 1
        self._on_log(f"TTS缓存清理: removed={removed} failed={failed}")
        return removed, failed

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

    def _process_streaming_tts(self, req_id: str, text: str, cancel_seq: int):
        self.ws_broadcast({"type": "TTS_STREAM_START", "req_id": req_id, "text": text})
        last_error = None
        try:
            for model in self._get_aliyun_tts_model_candidates():
                for voice in self._get_aliyun_tts_voice_candidates():
                    sent_any_audio, error_message = self._stream_aliyun_tts_once(req_id, text, cancel_seq, model, voice)
                    if cancel_seq != self.tts_cancel_seq:
                        last_error = None
                        break
                    if error_message:
                        last_error = RuntimeError(error_message)
                        if sent_any_audio:
                            break
                        self._on_log(f"阿里云TTS参数回退: model={model} voice={voice} error={error_message}")
                        continue
                    self.tts_model = model
                    self.tts_voice = voice
                    if self.cfg.get("aliyun_tts_model", "") != model:
                        self.cfg.set("aliyun_tts_model", model)
                    if self.cfg.get("aliyun_tts_voice", "") != voice:
                        self.cfg.set("aliyun_tts_voice", voice)
                    try:
                        self.cfg.save()
                    except Exception:
                        pass
                    last_error = None
                    break
                if cancel_seq != self.tts_cancel_seq or last_error is None:
                    break
        except Exception as e:
            last_error = e
        finally:
            if last_error:
                self._on_log(f"TTS Stream Error: {last_error}")
            self.ws_broadcast({"type": "TTS_STREAM_END", "req_id": req_id})

    def _handle_tts_request(self, order_index: int, req_id: str, tts_text: str, cancel_seq: int):
        should_advance = False
        try:
            text = (tts_text or "").strip()
            if not text:
                self.ws_broadcast({"type": "TTS_AUDIO", "req_id": req_id, "url": "", "text": "", "backend_played": False})
                should_advance = True
                return
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
            if cancel_seq != self.tts_cancel_seq:
                self.ws_broadcast({"type": "TTS_AUDIO", "req_id": req_id, "url": "", "text": text, "backend_played": False})
                should_advance = True
                return
            self.tts_start_signal.emit()
            play_event = threading.Event()
            with self.tts_lock:
                self.tts_played_events[req_id] = play_event
            self._process_streaming_tts(req_id, text, cancel_seq)
            wait_start = time.time()
            while time.time() - wait_start < 30.0:
                if cancel_seq != self.tts_cancel_seq:
                    break
                if play_event.is_set():
                    break
                time.sleep(0.1)
                
            with self.tts_lock:
                self.tts_played_events.pop(req_id, None)
                
            self.tts_end_signal.emit()
            self.ws_broadcast({"type": "TTS_PLAYBACK_END", "req_id": req_id, "text": text})
            self.ws_broadcast({"type": "TTS_AUDIO", "req_id": req_id, "url": "", "text": text, "backend_played": True})
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
            self.player_launch_started_ts = 0.0
            self.player_relaunch_count = 0
            self._on_log("前端连接已建立")
            # 新客户端连接，自动切换到 chat 模式
            if self.worker:
                self.worker.set_mode("chat")
                self.worker.set_asr_enabled(True)
                self.worker.set_wake_listener_enabled(False)
                self.worker.set_interrupt_listener_enabled(False)
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
                self._enter_waiting_tts_start("前端连接播报开场白")
                await websocket.send(json.dumps({
                    "type": "CHAT_APPEND",
                    "role": "assistant",
                    "text": intro_msg,
                }))
                self._mark_successful_response(intro_msg)
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
                        try:
                            QTimer.singleShot(0, lambda: self.handle_audio_interrupt_command("frontend"))
                        except Exception:
                            self.handle_audio_interrupt_command("frontend")
                    elif data.get("type") == "LOG":
                        # 处理前端发来的日志消息
                        log_text = data.get("text", "")
                        if log_text:
                            self._on_log(f"前端日志: {log_text}")
                    elif data.get("type") == "TTS_START":
                        self.tts_start_signal.emit()
                    elif data.get("type") == "TTS_END":
                        self.tts_end_signal.emit()
                    elif data.get("type") == "TTS_PLAYED":
                        req_id = data.get("req_id")
                        with self.tts_lock:
                             if req_id in self.tts_played_events:
                                self.tts_played_events[req_id].set()
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
                if self.player_active:
                    self.player_launch_started_ts = time.monotonic()
                    self._on_log("前端连接已断开，等待页面重连")
                else:
                    self.player_launch_started_ts = 0.0
                self._on_log("All clients disconnected, switching to wake mode")
                if self.worker:
                    self._to_wake_mode("前端断开")
                
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
            self._set_single_turn_close_pending(False)
            return
        
        # 纠正用户输入中的同音错别字，避免在对话上下文中出现乱码词
        text = self._normalize_text(text)
        
        async with self.llm_lock:
            self.stop_generation = False # Reset flag
            try:
                self._file_log(f"llm ask (hidden={hidden_input}) text={text}")
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
                self._set_single_turn_close_pending(False)
                intro_text = "你好，请讲！"
                if not hidden_input:
                    self.chat_history.append({"role": "user", "content": text})

                self.awaiting_response = False
                if self.ws_clients:
                    self._enter_waiting_tts_start("纯唤醒回复")
                    self.chat_history.append({"role": "assistant", "content": intro_text})
                    self.ws_broadcast({"type": "CHAT_APPEND", "role": "assistant", "text": intro_text})
                    self._mark_successful_response(intro_text)
                else:
                    self.pending_intro = intro_text
                return
            # ----------------------

            try:
                base_url = self.cfg.get("llm_base_url", "https://api.openai.com/v1")
                api_key = self.secrets.get("llm_api_key", "")
                model = self.cfg.get("llm_model", "gpt-3.5-turbo")
                
                if not base_url or not api_key:
                    self._set_single_turn_close_pending(False)
                    self.awaiting_response = False
                    self._enter_waiting_tts_start("API未配置")
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
                def _flush_tts(force: bool = False, punct_triggered: bool = False):
                    nonlocal buffer, last_tts_flush_ts, sent_any_tts
                    if not buffer:
                        return
                    buf = buffer
                    now = time.monotonic()
                    elapsed = now - last_tts_flush_ts
                    min_len = 26 if not sent_any_tts else 42
                    max_wait = 0.65 if not sent_any_tts else 0.95
                    if not force:
                        if len(buf) < min_len and elapsed < max_wait:
                            return
                    elif punct_triggered:
                        short_hold = 18 if not sent_any_tts else 14
                        punct_wait = 0.90 if not sent_any_tts else 0.55
                        if len(buf) < short_hold and elapsed < punct_wait:
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
                    next_buffer = ""
                    if force:
                        out = buf.strip()
                        next_buffer = ""
                    else:
                        if cut >= 0:
                            out = buf[:cut + 1].strip()
                            next_buffer = buf[cut + 1:]
                        else:
                            out = buf.strip()
                            next_buffer = ""
                    compact_out = re.sub(r"\s+", "", out or "")
                    normalized_out = compact_out.strip("，,。！？!?；;：:、…~")
                    compact_next = re.sub(r"\s+", "", next_buffer or "")
                    if (
                        not sent_any_tts
                        and punct_triggered
                        and normalized_out
                        and len(normalized_out) <= 8
                        and len(normalized_out) + len(compact_next) < 18
                        and elapsed < 1.8
                    ):
                        return
                    buffer = next_buffer
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
                        if first_chunk:
                            self.ws_broadcast({"type": "CHAT_START", "role": "assistant", "text": ""})
                            first_chunk = False
                        full_content += content
                        buffer += content
                        has_strong_punct = any(p in content for p in "。！？；.!?;\n")
                        _flush_tts(force=has_strong_punct, punct_triggered=has_strong_punct)
                        self.ws_broadcast({"type": "CHAT_PARTIAL", "text": content})
                
                if buffer and not self.stop_generation:
                    _flush_tts(force=True, punct_triggered=False)
                
                self.ws_broadcast({"type": "STREAM_END"})
                        
                self.chat_history.append({"role": "assistant", "content": full_content})
                if full_content.strip():
                    self._mark_successful_response(full_content)
                self.awaiting_response = False
                if full_content.strip() and self.ws_clients:
                    self._enter_waiting_tts_start("回答流结束-等待TTS启动")
                else:
                    self._set_single_turn_close_pending(False)
                    self._to_chat_idle_mode("回答流结束")
                self._on_log("会话完成: 已生成回答并同步前后端")
            except Exception as e:
                # 不重置 self.close_after_answer，使得单轮对话失败时也能播报抱歉后退出
                err_msg = f"LLM 请求失败: {str(e)}"
                self._on_error(err_msg)
                self.ws_broadcast({"type": "CHAT_APPEND", "role": "assistant", "text": "抱歉，我遇到了一些问题，请稍后再试。"})
                self.awaiting_response = False
                self._awaiting_tts_start = False
                self.force_exit_watch_started_ts = time.time()
                self._to_chat_idle_mode("LLM失败恢复")

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
                        self._on_log("WebSocket服务已启动: 127.0.0.1:3399")
                        await asyncio.Future()
                loop.run_until_complete(start())
            except Exception as e:
                self._file_log(f"WS Error: {e}")
                try:
                    self._on_log(f"WebSocket服务启动失败: {e}")
                except Exception:
                    pass
                self.ws_loop = None
        if not self.ensure_port_free(3399):
            self._cleanup_local_server_port(3399)
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
                ws_count = len(self.ws_clients)
                if ws_count > 0:
                    self.player_launch_started_ts = 0.0
                    self.player_relaunch_count = 0
                    return
                if self.player_launch_started_ts > 0:
                    launch_elapsed = time.monotonic() - self.player_launch_started_ts
                    if launch_elapsed >= float(self.player_connect_timeout_seconds):
                        if self.player_relaunch_count < int(self.player_max_relaunch_attempts):
                            if self._restart_player_window(f"前端连接超时 {launch_elapsed:.1f}s"):
                                return
                        else:
                            self.player_active = False
                            self.browser_pid = None
                            self.player_launch_started_ts = 0.0
                            try:
                                self._on_log("播放器启动失败: 前端长时间未建立连接，恢复唤醒监听")
                            except Exception:
                                pass
                            try:
                                self.start_listening()
                            except Exception:
                                pass
                            return
                if (not pid_ok) and (not window_ok) and (len(self.ws_clients) == 0):
                    self.player_active = False
                    self.browser_pid = None
                    self.player_launch_started_ts = 0.0
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
                    # PID exists but window is closed (e.g. Edge background process)
                    # We must kill it and relaunch
                    try:
                        self._kill_player_processes()
                    except Exception:
                        pass
                    self.player_active = False
                    should_launch = True
            else:
                should_launch = True
        
        if should_launch:
            self.launch_player()
            self._on_log("唤醒命中：正在启动播放器窗口...")
        else:
            self.process_wake_response()

        try:
            self._on_log("手动唤醒")
            self._on_log(f"状态检查: ws_clients={len(self.ws_clients)}, player_active={self.player_active}")
            # Capture context if we are starting fresh
            if not self.ws_clients:
                 try:
                     self.window_manager.capture_context()
                 except Exception as e:
                     self._on_error(f"Context capture failed: {e}")
            
            return

        except Exception as e:
            self._on_error(str(e))

    def process_wake_response(self):
        """统一处理唤醒后的响应逻辑"""
        self.last_wake_ts = time.monotonic()
        self.last_wake_wallclock = time.strftime("%Y-%m-%d %H:%M:%S")
        self.last_session_start_ts = time.monotonic()
        self._set_single_turn_close_pending(False)
        self._clear_manual_command_state()
        self._force_release_speaking_state()
        self._note_activity()
        self.awaiting_response = False
        self._awaiting_tts_start = False
        self._on_log(f"唤醒完成: wake_time={self.last_wake_wallclock}")
        intro_text = "你好，请讲！"
        
        if self.ws_clients:
            # 如果已连接，直接发送唤醒信号和欢迎语
            self.ws_broadcast({"type": "WAKE_EXISTING"})
            self.chat_history.append({"role": "assistant", "content": intro_text})
            self._enter_waiting_tts_start("唤醒词回复")
            self.ws_broadcast({"type": "CHAT_APPEND", "role": "assistant", "text": intro_text})
            self._mark_successful_response(intro_text)
        else:
            # 如果未连接（正在启动中），设置 pending，等待连接后发送
            # 注意：ws_handler 会在连接建立时将 pending_intro 添加到 chat_history 并发送
            self.pending_intro = intro_text
            self._enter_waiting_tts_start("等待前端连接播报唤醒词")
            
        # 切换 ASR 状态
        self._to_chat_idle_mode("规则一-唤醒后进入会话")

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

    def launch_player(self, trigger_wake_response: bool = True):
        """
        启动数字人播放器
        
        1. 确保服务已运行
        2. 启动 Edge 浏览器
        """
        self.player_active = True
        self.player_launch_started_ts = time.monotonic()
        font_size = self.cfg.get("chat_font_size", 36)
        url = f"http://localhost:3400/player/index.html?fontsize={font_size}"
        exe = self.find_edge()
        if exe:
            try:
                # --app=URL 以应用模式启动（无地址栏等），--kiosk 全屏模式（无最大化按钮，只能 Alt+F4 或 Esc 配合 JS 关闭）
                # 必须使用独立的 user-data-dir，否则如果后台有 Edge 进程，新窗口会合并到现有进程，导致 --kiosk 参数失效
                user_data_dir = str(self.runtime_data_dir / "edge_kiosk_data")
                try:
                    Path(user_data_dir).mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass
                
                args = [
                    exe, 
                    "--new-window", 
                    "--kiosk",
                    "--edge-kiosk-type=fullscreen",
                    "--app=" + url, 
                    f"--user-data-dir={user_data_dir}", 

                    "--autoplay-policy=no-user-gesture-required", 
                    "--use-fake-ui-for-media-stream", 
                    "--no-first-run",
                    "--disable-session-crashed-bubble"
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
        
        if trigger_wake_response:
            self.process_wake_response()

    def _kill_player_processes(self):
        target_url_snippet = "localhost:3400"
        closed_count = 0
        if self.browser_pid:
            try:
                if psutil.pid_exists(self.browser_pid):
                    p = psutil.Process(self.browser_pid)
                    children = p.children(recursive=True)
                    for child in children:
                        try:
                            child.kill()
                        except Exception:
                            pass
                    p.kill()
                    closed_count += 1
            except Exception:
                pass
        try:
            cmd = f"wmic process where \"name='msedge.exe' and commandline like '%{target_url_snippet}%'\" call terminate"
            subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
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
        return closed_count

    def _restart_player_window(self, reason: str) -> bool:
        if self.player_relaunch_count >= int(self.player_max_relaunch_attempts):
            return False
        self.player_relaunch_count += 1
        self._on_log(f"播放器重启: {reason}，第 {self.player_relaunch_count} 次")
        try:
            self._kill_player_processes()
        except Exception as e:
            self._on_log(f"播放器重启前清理失败: {e}")
        self.player_active = False
        self.browser_pid = None
        self.player_launch_started_ts = time.monotonic()
        QTimer.singleShot(600, lambda: self.launch_player(trigger_wake_response=False))
        return True

    def close_player(self):
        """
        手动关闭数字人播放器进程
        
        策略：
        1. [优先] 查找标题包含 "Live2D Player" 的窗口并发送关闭消息 (最优雅)
        2. [后备] 使用系统命令 (wmic/taskkill) 强制关闭包含 localhost:3400 的进程
        """
        self._on_log("正在执行关闭操作...")
        self._set_single_turn_close_pending(False)
        self._clear_manual_command_state()
        self._note_activity()
        
        # 切换 ASR 回到唤醒模式
        if self.worker:
            self._to_wake_mode("规则五/七-关闭窗口回到唤醒")
        
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
        self._kill_player_processes()

        self._on_log("关闭操作已完成")

        # 清理状态
        self.player_active = False
        self.browser_pid = None
        self.player_launch_started_ts = 0.0
        self.player_relaunch_count = 0
        
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
            mode = getattr(self.worker, "_mode", "none") if self.worker else "none"
            asr_on = getattr(self.worker, "_asr_enabled", False) if self.worker else False
            wake_on = getattr(self.worker, "_wake_listener_enabled", False) if self.worker else False
            interrupt_on = getattr(self.worker, "_interrupt_listener_enabled", False) if self.worker else False
            speaking_sec = (time.monotonic() - self.last_speaking_start_ts) if self.is_speaking and self.last_speaking_start_ts > 0 else 0.0
            session_sec = (time.monotonic() - self.last_session_start_ts) if self.last_session_start_ts > 0 else 0.0
            self._file_log(
                f"HEALTH rss={mem} threads={th} ws={len(self.ws_clients)} player={self.player_active} "
                f"mode={mode} asr={asr_on} wake={wake_on} interrupt={interrupt_on} "
                f"awaiting={self.awaiting_response} speaking={self.is_speaking} speaking_sec={speaking_sec:.1f} session_sec={session_sec:.1f}"
            )
        except Exception:
            pass

    def _mark_successful_response(self, text: str = ""):
        self.last_successful_response_ts = time.time()
        self.last_successful_response_text = (text or "").strip()
        self.force_exit_watch_started_ts = 0.0
        try:
            preview = (text or "").strip().replace("\n", " ")
            if len(preview) > 20:
                preview = preview[:20] + "..."
            self._on_log(f"成功回答时间更新: {preview}")
        except Exception:
            pass

    def _force_exit_guard_tick(self):
        try:
            if not self.player_active:
                return
            if len(self.ws_clients) == 0:
                return
            is_chat_idle = False
            if self.worker and getattr(self.worker, "_mode", "") == "chat":
                is_chat_idle = True
                
            if not (self.awaiting_response or self.is_speaking or self._awaiting_tts_start or is_chat_idle):
                return
            watch_started_ts = float(self.force_exit_watch_started_ts or 0.0)
            if watch_started_ts <= 0:
                return
            delta = time.time() - watch_started_ts
            timeout_sec = float(self.force_exit_timeout_seconds)
            if delta <= timeout_sec:
                return
            self._on_log(f"兜底退出触发: 当前会话 {delta:.1f}s 无进展，执行强制关闭")
            self.close_player()
        except Exception as e:
            self._on_log(f"兜底退出检查异常: {e}")

    def _normalize_text(self, s: str) -> str:
        """
        文本标准化 (用于 ASR 结果匹配)
        
        1. 去除标点符号
        2. 修正常见的同音近音写法 (如 "小时" -> "小石")
        """
        t = re.sub(r"[\s,.!?;:，。！？；：、（）()《》〈〉「」『』“”‘’—…·\-]+", "", s or "")
        t = t.lower()
        for v in ["小时","消失","小是","小识","小事","肖石","晓石","晓诗","萧石","小十", "消石", "销石", "小实", "小视", "孝石", "笑死", "笑时", "消逝", "小市"]:
            t = t.replace(v, "小石")
        for v in ["景观","尽管","警管","井关","金冠","尽关","经管","景觀","尽古","頂關","景瓜","金关","里关", "敬官", "静观", "警馆", "井官", "经官", "竞管", "境关", "警棍", "景管", "经关", "近关"]:
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
