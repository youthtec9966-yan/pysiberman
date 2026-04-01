import pvporcupine
import pyaudio
import struct

# -------------------------- 配置区（必须改） --------------------------
ACCESS_KEY = "0iyMqZkQqdmPJkiG4erkA6OgQz5rlEYX4dT4jLP6d/cGAXO5HJr2jw=="  # 替换成你的AccessKey
KEYWORD_PATH = "./sjg_zh_windows_v4_0_0.ppn"  # 你的中文唤醒词ppn文件路径
MODEL_PATH = "./porcupine_params_zh.pv"  # 中文语言模型路径
SENSITIVITY = 0.7
# ---------------------------------------------------------------------

def main():
    try:
        # 初始化Porcupine（指定自定义模型路径）
        porcupine = pvporcupine.create(
            access_key=ACCESS_KEY,
            keyword_paths=[KEYWORD_PATH],  # 注意：是keyword_paths，不是keywords
            model_path=MODEL_PATH,  # 必须指定中文模型
            sensitivities=[SENSITIVITY]
        )
    except Exception as e:
        print(f"初始化失败: {e}")
        return

    # 初始化麦克风
    pa = pyaudio.PyAudio()
    audio_stream = pa.open(
        rate=porcupine.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=porcupine.frame_length
    )

    print("正在监听中文唤醒词「豆包」...（按 Ctrl+C 退出）")
    try:
        while True:
            pcm = audio_stream.read(porcupine.frame_length)
            pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
            keyword_index = porcupine.process(pcm)
            if keyword_index >= 0:
                print("✅ 检测到中文唤醒词！")
    except KeyboardInterrupt:
        print("\n退出监听")
    finally:
        audio_stream.stop_stream()
        audio_stream.close()
        pa.terminate()
        porcupine.delete()

if __name__ == "__main__":
    main()