import wave
import numpy as np
import sherpa_onnx

print("sherpa_onnx version:", sherpa_onnx.__version__)

kws = sherpa_onnx.KeywordSpotter(
    tokens="tokens.txt",
    encoder="encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
    decoder="decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
    joiner="joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
    keywords_file="test_wavs/test_keywords.txt",
    num_threads=4,
    keywords_score=1.0,
    keywords_threshold=0.0,
    max_active_paths=4,
    num_trailing_blanks=1,
)

print("Model loaded successfully")

def read_wav(path):
    with wave.open(path, 'rb') as wf:
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        return sample_rate, samples

sample_rate = 16000
chunk_size = 1600

for i in range(7):
    stream = kws.create_stream()
    sr, samples = read_wav(f"test_wavs/{i}.wav")
    print(f"\nAudio {i}: {len(samples)} samples, {len(samples)/sr:.2f}s")
    
    for start in range(0, len(samples), chunk_size):
        chunk = samples[start:start+chunk_size]
        stream.accept_waveform(sample_rate, chunk)
        
        while kws.is_ready(stream):
            kws.decode_stream(stream)
            result = kws.get_result(stream)
            if result:
                print(f"  Detected: '{result}'")
                kws.reset_stream(stream)
    
    stream.input_finished()
    while kws.is_ready(stream):
        kws.decode_stream(stream)
        result = kws.get_result(stream)
        if result:
            print(f"  Detected (final): '{result}'")
