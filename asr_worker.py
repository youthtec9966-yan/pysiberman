import difflib
import os
import re
import tempfile
import threading
import time
import wave

import faulthandler
import numpy as np
from dashscope.audio.asr import Recognition, RecognitionCallback
import pvporcupine
from pvrecorder import PvRecorder
from PySide6.QtCore import QObject, Signal

from audio_templates import AudioTemplateMatcher, resolve_template_dir


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

    def __init__(self, phrases: str, device_index: int, api_key: str, audio_match_enabled: bool = True, audio_match_threshold: float = 0.45, kws_config: dict = None):
        super().__init__()
        # 处理唤醒词列表
        self.phrases = [p.strip() for p in (phrases or "").split(",") if p.strip()]
        self.phrases_norm = [self._normalize(p) for p in self.phrases]
        
        self.device_index = device_index
        self.api_key = api_key
        
        self.wake_matcher = AudioTemplateMatcher(str(resolve_template_dir("templates/wake")))
        self.interrupt_matcher = AudioTemplateMatcher(str(resolve_template_dir("templates/interrupt")))
        self.audio_match_enabled = bool(audio_match_enabled)
        self.audio_match_threshold = float(audio_match_threshold)
        self.wake_audio_match_threshold = float(audio_match_threshold)
        self.interrupt_audio_match_threshold = float(audio_match_threshold)
        self.interrupt_audio_window_ms = 1000.0
        self.interrupt_audio_check_interval_frames = 8
        self.interrupt_audio_cooldown_ms = 800.0
        self.interrupt_audio_standby_rms_gate = 0.006
        kws_config = kws_config or {}
        self.kws_enabled = bool(kws_config.get("enabled", False))
        self.kws_access_key = str(kws_config.get("access_key", "") or "").strip()
        self.kws_keyword_path = str(kws_config.get("keyword_path", "") or "").strip()
        self.kws_model_path = str(kws_config.get("model_path", "") or "").strip()
        self.kws_sensitivity = float(kws_config.get("sensitivity", 0.7))
        self.kws_interrupt_fallback_enabled = bool(kws_config.get("interrupt_fallback_enabled", True))
        self.kws_interrupt_cooldown_ms = float(kws_config.get("interrupt_cooldown_ms", 800.0))
        self.kws_interrupt_cooldown_ms = min(5000.0, max(100.0, self.kws_interrupt_cooldown_ms))
        self.mic_recover_error_threshold = 6
        self.mic_recover_base_delay_sec = 0.5
        self.mic_recover_max_delay_sec = 3.0
        
        self._running = False
        self._paused = False
        self._mode = "wake"
        self._thread = None
        self._asr_enabled = True
        self._wake_listener_enabled = True
        self._interrupt_listener_enabled = False
        self._kws_ready = False
        self._kws_fallback_active = False
        
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

    def apply_interrupt_match_config(self, cfg: dict):
        if not cfg:
            return
        self.interrupt_audio_match_threshold = float(cfg.get("interrupt_audio_match_threshold", self.interrupt_audio_match_threshold))
        self.interrupt_audio_window_ms = float(cfg.get("interrupt_audio_window_ms", self.interrupt_audio_window_ms))
        self.interrupt_audio_check_interval_frames = int(cfg.get("interrupt_audio_check_interval_frames", self.interrupt_audio_check_interval_frames))
        self.interrupt_audio_cooldown_ms = float(cfg.get("interrupt_audio_cooldown_ms", self.interrupt_audio_cooldown_ms))
        self.interrupt_audio_standby_rms_gate = float(cfg.get("interrupt_audio_standby_rms_gate", self.interrupt_audio_standby_rms_gate))
        self.interrupt_audio_window_ms = min(3000.0, max(200.0, self.interrupt_audio_window_ms))
        self.interrupt_audio_check_interval_frames = min(30, max(1, self.interrupt_audio_check_interval_frames))
        self.interrupt_audio_cooldown_ms = min(5000.0, max(100.0, self.interrupt_audio_cooldown_ms))
        self.interrupt_audio_standby_rms_gate = min(0.10, max(0.001, self.interrupt_audio_standby_rms_gate))
        self.log.emit(
            f"Interrupt match cfg threshold={self.interrupt_audio_match_threshold:.3f} "
            f"window_ms={self.interrupt_audio_window_ms:.0f} hop={self.interrupt_audio_check_interval_frames} "
            f"cooldown_ms={self.interrupt_audio_cooldown_ms:.0f} standby_rms={self.interrupt_audio_standby_rms_gate:.4f}"
        )

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

    def set_asr_enabled(self, enabled: bool):
        self._asr_enabled = bool(enabled)
        self.log.emit(f"ASR enabled={self._asr_enabled}")

    def set_wake_listener_enabled(self, enabled: bool):
        self._wake_listener_enabled = bool(enabled)
        self.log.emit(f"Wake listener enabled={self._wake_listener_enabled}")

    def set_interrupt_listener_enabled(self, enabled: bool):
        self._interrupt_listener_enabled = bool(enabled)
        self.log.emit(f"Interrupt listener enabled={self._interrupt_listener_enabled}")

    def _activate_asr_wake_fallback(self, reason: str):
        self._kws_ready = False
        if self._mode == "wake" and self._wake_listener_enabled and not self._asr_enabled:
            self._asr_enabled = True
            self.log.emit("ASR enabled=True")
        if not self._kws_fallback_active:
            self._kws_fallback_active = True
            self.log.emit(f"KWS fallback active: {reason}")

    def _mask_kws_key(self) -> str:
        key = str(self.kws_access_key or "")
        if not key:
            return "<empty>"
        if len(key) <= 12:
            return key[0:2] + "***" + key[-2:]
        return key[0:6] + "***" + key[-6:]

    def _kws_init_hint(self, e: Exception) -> str:
        if isinstance(e, pvporcupine.PorcupineActivationLimitError):
            return "AccessKey已达激活上限，请在Picovoice控制台释放旧设备或更换新Key"
        if isinstance(e, pvporcupine.PorcupineActivationRefusedError):
            return "AccessKey被拒绝，请检查Key是否有效且具备Porcupine权限"
        if isinstance(e, pvporcupine.PorcupineActivationThrottledError):
            return "AccessKey触发限流，请稍后重试"
        if isinstance(e, pvporcupine.PorcupineKeyError):
            return "AccessKey格式无效，请检查是否复制完整"
        if isinstance(e, pvporcupine.PorcupineInvalidArgumentError):
            return "模型参数无效，请检查keyword/model文件是否匹配当前SDK"
        if isinstance(e, pvporcupine.PorcupineIOError):
            return "模型文件读取失败，请检查文件路径与访问权限"
        return "请检查AccessKey状态、模型文件与网络环境"
            
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
        t = t.lower()
        for v in ["小时","消失","小是","小识","小事","肖石","晓石","晓诗","萧石","小十", "消石", "销石", "小实", "小视", "孝石", "笑死", "笑时", "消逝", "小市"]:
            t = t.replace(v, "小石")
        for v in ["景观","尽管","警管","井关","金冠","尽关","经管","景觀","尽古","頂關","景瓜","金关","里关", "敬官", "静观", "警馆", "井官", "经官", "竞管", "境关", "警棍", "景管", "经关", "近关"]:
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
        if self.kws_enabled:
            return False
        if not (self.audio_match_enabled and self.wake_matcher):
            return False
        if not (self._mode == "wake" and self._wake_listener_enabled):
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
        if not (self._interrupt_listener_enabled or self._speaking_state):
            return False
        match_name, match_dist = self.interrupt_matcher.match(pcm_window, threshold=self.interrupt_audio_match_threshold)
        if not match_name:
            return False
        if match_dist is not None:
            self.log.emit(f"Interrupt Audio Match: {match_name} ({match_dist:.3f})")
        else:
            self.log.emit(f"Interrupt Audio Match: {match_name}")
        self.interrupt_detected.emit(match_name)
        return True

    def _build_candidate_indices(self, devices):
        candidate_indices = []
        try:
            pref = int(self.device_index)
        except Exception:
            pref = -1
        if pref >= 0:
            candidate_indices.append(pref)
        candidate_indices.append(-1)
        for i in range(len(devices)):
            if i not in candidate_indices:
                candidate_indices.append(i)
        return candidate_indices

    def _describe_device(self, idx: int, devices):
        if idx >= 0 and idx < len(devices):
            return f"{devices[idx]} (index={idx})"
        return "系统默认设备"

    def _release_recorder(self, recorder):
        if recorder is None:
            return
        try:
            recorder.stop()
        except Exception:
            pass
        try:
            recorder.delete()
        except Exception:
            pass

    def _create_recorder_with_fallback(self):
        devices = []
        try:
            devices = PvRecorder.get_available_devices()
        except Exception:
            devices = []
        candidate_indices = self._build_candidate_indices(devices)
        last_error = None
        for idx in candidate_indices:
            recorder = None
            try:
                self.log.emit(f"尝试麦克风: {self._describe_device(idx, devices)}")
                recorder = PvRecorder(device_index=idx, frame_length=512)
                recorder.start()
                self.device_index = idx
                self.log.emit(f"麦克风启动成功: {self._describe_device(idx, devices)}")
                return recorder
            except Exception as e:
                last_error = e
                self.error.emit(f"麦克风启动失败(index={idx}): {e}")
                self._release_recorder(recorder)
        raise RuntimeError(f"Recorder init failed: {last_error}")

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
        if not pcm_data or not self._running or not self._asr_enabled:
            return
        t_begin = time.monotonic()

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
            t_api_begin = time.monotonic()
            result = recognition.call(file=temp_filename)
            t_api_ms = (time.monotonic() - t_api_begin) * 1000.0
            
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
            total_ms = (time.monotonic() - t_begin) * 1000.0
            self.log.emit(f"ASR耗时: total={total_ms:.0f}ms api={t_api_ms:.0f}ms audio={len(pcm_data)/16000:.2f}s")
            self.log.emit(f"asr: {full_text}")

            # --- 模式分支 ---
            if self._mode == "chat":
                if full_text.strip():
                    self.chat_input_detected.emit(full_text)
            else:
                # 唤醒模式
                wake_tokens = ("小石", "警官")
                hit = any(token in norm_text for token in wake_tokens)
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
        porcupine = None
        try:
            faulthandler.enable()
            self.log.emit("asr engine ready")
            if self.kws_enabled and self.kws_access_key and self.kws_keyword_path and self.kws_model_path:
                try:
                    porcupine = pvporcupine.create(
                        access_key=self.kws_access_key,
                        keyword_paths=[self.kws_keyword_path],
                        model_path=self.kws_model_path,
                        sensitivities=[self.kws_sensitivity]
                    )
                    self._kws_ready = True
                    self._kws_fallback_active = False
                    self.log.emit("kws engine ready")
                except Exception as e:
                    porcupine = None
                    keyword_exists = bool(self.kws_keyword_path and os.path.exists(self.kws_keyword_path))
                    model_exists = bool(self.kws_model_path and os.path.exists(self.kws_model_path))
                    hint = self._kws_init_hint(e)
                    self.error.emit(
                        f"KWS init failed[{type(e).__name__}]: {e} | hint={hint} | "
                        f"key={self._mask_kws_key()} keyword_exists={keyword_exists} model_exists={model_exists}"
                    )
                    self._activate_asr_wake_fallback("kws init failed")
            elif self.kws_enabled:
                self.error.emit("KWS init failed: missing access_key/keyword_path/model_path")
                self._activate_asr_wake_fallback("kws config missing")
            else:
                self._kws_ready = False
                self._kws_fallback_active = False
                self.log.emit("KWS disabled, using audio template + ASR wake")

            recorder = self._create_recorder_with_fallback()
            
            self.log.emit("wake listening started")

            # VAD 状态机变量
            speech_frames = []
            silence_counter = 0
            in_speech = False
            interrupt_ring = []
            interrupt_tick = 0
            last_interrupt_ts = 0.0
            interrupt_window_samples = int(16000 * (float(self.interrupt_audio_window_ms) / 1000.0))
            interrupt_window_samples = max(3200, min(32000, interrupt_window_samples))
            interrupt_hop_frames = max(1, int(self.interrupt_audio_check_interval_frames))
            interrupt_cooldown_sec = max(0.1, float(self.interrupt_audio_cooldown_ms) / 1000.0)
            kws_interrupt_cooldown_sec = max(0.1, float(self.kws_interrupt_cooldown_ms) / 1000.0)
            mic_error_count = 0
            mic_recover_attempts = 0
            
            # 参数配置
            MAX_SILENCE_FRAMES_WAKE = 25  # 约 800ms 静音判定为结束
            MAX_SILENCE_FRAMES_CHAT = 22  # 约 700ms 静音判定为结束（降低识别尾延迟）
            MIN_SPEECH_FRAMES = 15   # 约 500ms 最短语音长度
            MIN_MATCH_FRAMES = 6     # 约 200ms 最短匹配长度
            MAX_SPEECH_DURATION_FRAMES = 468 # 约 15秒 最长语音
            
            while self._running:
                try:
                    if self._mode == "wake" and self._wake_listener_enabled and (not self._kws_ready) and (not porcupine) and (not self._asr_enabled):
                        self._activate_asr_wake_fallback("kws unavailable while wake mode")
                    pcm = recorder.read()
                    mic_error_count = 0
                    mic_recover_attempts = 0
                    if self._paused:
                        # 暂停时清空状态
                        speech_frames = []
                        in_speech = False
                        silence_counter = 0
                        continue
                    
                    if not pcm:
                        continue
                    if porcupine:
                        keyword_index = porcupine.process(pcm)
                        if keyword_index >= 0:
                            now_ts = time.monotonic()
                            if self._mode == "wake" and self._wake_listener_enabled:
                                self.log.emit("KWS Wake Match: hit")
                                self.wake_detected.emit()
                                self._wake_listener_enabled = False
                                speech_frames = []
                                in_speech = False
                                silence_counter = 0
                                interrupt_ring = []
                                continue
                            should_kws_interrupt = self.kws_interrupt_fallback_enabled and (self._speaking_state or self._interrupt_listener_enabled)
                            if should_kws_interrupt and (now_ts - last_interrupt_ts) > kws_interrupt_cooldown_sec:
                                self.log.emit("KWS Interrupt Fallback: hit")
                                self.interrupt_detected.emit("kws_fallback")
                                last_interrupt_ts = now_ts
                                speech_frames = []
                                in_speech = False
                                silence_counter = 0
                                interrupt_ring = []
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
                    if len(interrupt_ring) > interrupt_window_samples:
                        interrupt_ring = interrupt_ring[-interrupt_window_samples:]
                    interrupt_tick += 1
                    if interrupt_tick % interrupt_hop_frames == 0:
                        now_ts = time.monotonic()
                        if (now_ts - last_interrupt_ts) > interrupt_cooldown_sec:
                            base_gate = self.speaking_noise_margin if self._speaking_state else self.standby_noise_margin
                            vol_gate = max(float(self.noise_floor) + float(base_gate), 0.02)
                            rms_gate = float(self.speaking_interrupt_rms) if self._speaking_state else float(self.interrupt_audio_standby_rms_gate)
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
                    mic_error_count += 1
                    if mic_error_count < int(self.mic_recover_error_threshold):
                        time.sleep(0.12)
                        continue
                    self.log.emit("检测到连续麦克风读流失败，开始重连")
                    self._release_recorder(recorder)
                    recorder = None
                    speech_frames = []
                    in_speech = False
                    silence_counter = 0
                    interrupt_ring = []
                    delay = min(
                        float(self.mic_recover_max_delay_sec),
                        float(self.mic_recover_base_delay_sec) * (2 ** min(mic_recover_attempts, 4))
                    )
                    time.sleep(delay)
                    try:
                        recorder = self._create_recorder_with_fallback()
                        self.log.emit("麦克风重连成功，继续监听")
                        mic_error_count = 0
                        mic_recover_attempts = 0
                    except Exception as reconnect_error:
                        self.error.emit(f"麦克风重连失败: {reconnect_error}")
                        mic_recover_attempts += 1

        except Exception as e:
            self.error.emit(f"Worker Exception: {str(e)}")
        finally:
            self._release_recorder(recorder)
            if porcupine is not None:
                try:
                    porcupine.delete()
                except Exception:
                    pass
            import gc
            gc.collect()
            self.log.emit("wake listening stopped")
