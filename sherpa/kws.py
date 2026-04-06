import numpy as np
import sounddevice as sd
import sherpa_onnx

kws = sherpa_onnx.KeywordSpotter(
    tokens="tokens.txt",
    encoder="encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
    decoder="decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
    joiner="joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
    keywords_file="keywords.txt",
    num_threads=4,
    keywords_score=1.5,
    keywords_threshold=0.1,
    max_active_paths=8,
)

stream = kws.create_stream()
sample_rate = 16000
block_size = 1600

print("=== 已启动，说：小石警官 ===")

with sd.InputStream(
    samplerate=sample_rate,
    blocksize=block_size,
    dtype="int16",
    channels=1,
) as mic:
    while True:
        audio_data, _ = mic.read(block_size)
        samples = audio_data.flatten().astype(np.float32) / 32768.0
        stream.accept_waveform(sample_rate, samples)
        
        while kws.is_ready(stream):
            kws.decode_stream(stream)
            result = kws.get_result(stream)
            if result:
                print(f"✅ 唤醒成功：{result}")
                kws.reset_stream(stream)
