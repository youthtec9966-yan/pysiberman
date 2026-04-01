import json
import os
from pathlib import Path
from typing import Optional

import keyring


class ConfigManager:
    """
    配置管理器
    负责加载和保存应用程序的配置信息到 JSON 文件。
    """
    def __init__(self):
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
                "aliyun_appkey": "sk-28313ea70a8f47d09a6cd1cab51c477e",
                "chat_engine": "llm",
                "device_index": -1,
                "llm_base_url": "https://api.deepseek.com/v1",
                "llm_model": "deepseek-chat",
                "wake_engine": "asr",
                "asr_wake_phrases": "小石警官",
                "asr_model_size": "small",
                "asr_compute_type": "int8",
                "asr_device": "cpu",
                "asr_model_dir": "",
                "audio_match_enabled": True,
                "audio_match_threshold": 0.45,
                "kws_enabled": False,
                "porcupine_access_key": "",
                "porcupine_keyword_path": "./sjg_zh_windows_v4_0_0.ppn",
                "porcupine_model_path": "./porcupine_params_zh.pv",
                "porcupine_sensitivity": 0.7,
                "interrupt_audio_match_threshold": 0.45,
                "interrupt_audio_window_ms": 1000,
                "interrupt_audio_check_interval_frames": 8,
                "interrupt_audio_cooldown_ms": 800,
                "interrupt_audio_standby_rms_gate": 0.006,
                "asr_interrupt_enabled": False,
                "audio_command_interrupt_enabled": False,
                "asr_profile_mode": "smart",
                "asr_standby_noise_margin": 0.015,
                "asr_speaking_noise_margin": 0.05,
                "asr_standby_energy_ratio": 1.35,
                "asr_speaking_energy_ratio": 2.2,
                "asr_interrupt_peak": 0.085,
                "asr_interrupt_rms": 0.015,
                "force_exit_check_interval_ms": 10000,
                "force_exit_timeout_seconds": 23.6
            }, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load(self):
        """从文件加载配置"""
        try:
            self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        except Exception:
            # 加载失败时使用默认配置
            self.config = {
                "aliyun_appkey": "sk-28313ea70a8f47d09a6cd1cab51c477e",
                "chat_engine": "llm",
                "device_index": -1,
                "llm_base_url": "https://api.deepseek.com/v1",
                "llm_model": "deepseek-chat",
                "wake_engine": "asr",
                "asr_wake_phrases": "小石警官",
                "asr_model_size": "small",
                "asr_compute_type": "int8",
                "asr_device": "cpu",
                "asr_model_dir": "",
                "audio_match_enabled": True,
                "audio_match_threshold": 0.45,
                "kws_enabled": False,
                "porcupine_access_key": "",
                "porcupine_keyword_path": "./sjg_zh_windows_v4_0_0.ppn",
                "porcupine_model_path": "./porcupine_params_zh.pv",
                "porcupine_sensitivity": 0.7,
                "interrupt_audio_match_threshold": 0.45,
                "interrupt_audio_window_ms": 1000,
                "interrupt_audio_check_interval_frames": 8,
                "interrupt_audio_cooldown_ms": 800,
                "interrupt_audio_standby_rms_gate": 0.006,
                "asr_interrupt_enabled": False,
                "audio_command_interrupt_enabled": False,
                "asr_profile_mode": "smart",
                "asr_standby_noise_margin": 0.015,
                "asr_speaking_noise_margin": 0.05,
                "asr_standby_energy_ratio": 1.35,
                "asr_speaking_energy_ratio": 2.2,
                "asr_interrupt_peak": 0.085,
                "asr_interrupt_rms": 0.015,
                "force_exit_check_interval_ms": 10000,
                "force_exit_timeout_seconds": 23.6
            }

    def save(self):
        """保存当前配置到文件"""
        self.config_path.write_text(json.dumps(self.config, ensure_ascii=False, indent=2), encoding="utf-8")

    def get(self, key: str, default=None):
        """获取配置项，如果存储的值为空字符串且提供了默认值，则返回默认值"""
        val = self.config.get(key, default)
        if val == "" and default is not None:
            return default
        return val

    def set(self, key: str, value):
        """设置配置项（需要手动调用 save 才能持久化）"""
        self.config[key] = value


class SecretStore:
    """
    安全的密钥管理
    使用操作系统的 Keyring 存储 (Windows Credential Manager / macOS Keychain / Linux Secret Service)
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
