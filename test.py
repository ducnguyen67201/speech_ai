from faster_whisper import WhisperModel
import ffmpeg
import numpy as np

def load_audio_fast(file_path, sr=16000):
    """Efficiently load audio using ffmpeg with direct streaming."""
    out, _ = (
        ffmpeg.input(file_path, threads=0)  # Automatically choose best threading
        .output("pipe:", format="wav", acodec="pcm_s16le", ac=1, ar=sr)  # Convert to WAV PCM 16-bit
        .run(capture_stdout=True, capture_stderr=True)
    )

    # Convert raw bytes to NumPy float32 array
    audio = np.frombuffer(out, np.int16).astype(np.float32) / 32768.0  
    return audio 

# 🔥 Load Whisper Model (Use "cuda" for GPU)
model = WhisperModel("large-v2", device="cpu", compute_type="int8")  # "cuda" for GPU

# 🎙 Load and Process Audio Instantly
audio = load_audio_fast("audio/2.m4a")

# 🚀 Fast Transcription
segments, _ = model.transcribe(audio, beam_size=1)  # beam_size=1 speeds up inference

# 🔊 Output Text
for segment in segments:
    print(segment.text)
