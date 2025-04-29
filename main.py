import os
import time
import numpy as np
import ffmpeg
from faster_whisper import WhisperModel
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
import requests
import tempfile

# 🌟 Initialize FastAPI
app = FastAPI()

# 🔥 Load the Whisper model ONCE at startup
print("🚀 Loading Whisper model...")
model = WhisperModel("small", device="cpu", compute_type="int8")  # "cuda" for GPU
print("✅ Model loaded and ready!")


# 🎙 Fast audio loading function
def load_audio_fast(file_path, sr=16000):
    """Efficiently loads audio using FFmpeg with direct streaming."""
    out, _ = (
        ffmpeg.input(file_path, threads=0)
        .output("pipe:", format="wav", acodec="pcm_s16le", ac=1, ar=sr)
        .run(capture_stdout=True, capture_stderr=True)
    )
    return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0


def load_audio_fast_web(file_path, sr=16000):
    """Decode WebM/Opus from browser and return raw float32 PCM."""
    try:
        out, err = (
            ffmpeg.input(file_path)
            .output("pipe:", format="wav", acodec="pcm_s16le", ac=1, ar=sr)
            .run(capture_stdout=True, capture_stderr=True)
        )
        return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0
    except ffmpeg.Error as e:
        print("⚠️ ffmpeg error output:", e.stderr.decode())
        raise


# 📌 API Endpoint: Transcribe Audio
@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Receives an audio file and returns the transcribed text."""

    # Save file temporarily
    file_path = f"temp_{int(time.time())}.m4a"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # 🎙 Load audio instantly
    start_time = time.time()
    audio = load_audio_fast(file_path)
    print(f"🔹 Audio loaded in {time.time() - start_time:.2f} sec")

    # 🚀 Transcription
    start_time = time.time()
    segments, _ = model.transcribe(audio, beam_size=1)
    print(f"📝 Transcription completed in {time.time() - start_time:.2f} sec")

    # Remove temp file
    os.remove(file_path)

    # Return transcription result
    return {"text": " ".join([seg.text for seg in segments])}


@app.post("/evaluate-with-llm")
async def evaluate_with_llm(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    file_path = f"temp_{int(time.time())}.m4a"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Load audio
    start_time = time.time()
    audio = load_audio_fast(file_path)
    print(f"🔹 Audio loaded in {time.time() - start_time:.2f} sec")

    # Transcribe
    start_time = time.time()
    segments, _ = model.transcribe(audio, beam_size=1)
    print(f"📝 Transcription completed in {time.time() - start_time:.2f} sec")

    # Delete the temp file
    os.remove(file_path)

    # Extract transcription text
    transcribed_text = " ".join([seg.text for seg in segments])

    # For conversation, expectation is optional or can be something generic
    expected_text = "Provide a helpful and natural response to the user's message."

    # Prepare payload for LLM
    payload = {
        "id": "chat_001",
        "input": transcribed_text,
        "expectation": expected_text,
    }

    try:
        response = requests.post("http://localhost:8080/run-test", json=payload)
        response.raise_for_status()
        llm_response = response.json()
    except Exception as e:
        return {"error": str(e)}

    return {"transcription": transcribed_text, "llm_response": llm_response}


@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🔗 WebSocket connection established!")

    try:
        while True:
            data = await websocket.receive_bytes()
            print("📥 Receiving audio chunk...")

            try:
                audio = decode_webm_blob_to_pcm(data)
                segments, _ = model.transcribe(audio, beam_size=1)
                transcription = " ".join([seg.text for seg in segments])
            except Exception as e:
                transcription = f"[Error] {e}"

            payload = {
                "id": "chat_001",
                "input": transcription,
                "expectation": "response naturally as a casual chat",
            }

            try:
                response = requests.post("http://localhost:8080/run-test", json=payload)
                response.raise_for_status()
                llm_response = response.json()
                print("llm_response: ", llm_response)
                message = llm_response.get("message", "[No LLM response]")
                await websocket.send_text(message)

            except Exception as e:
                await websocket.send_text(f"[LLM Error] {str(e)}")

    except WebSocketDisconnect:
        print("🔌 WebSocket connection closed.")


def decode_webm_blob_to_pcm(webm_bytes, sr=16000):
    """Decode a WebM (Opus) audio blob to float32 PCM numpy array."""
    try:
        process = (
            ffmpeg.input("pipe:0")
            .output("pipe:", format="wav", acodec="pcm_s16le", ac=1, ar=sr)
            .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
        )

        out, err = process.communicate(input=webm_bytes)

        if process.returncode != 0:
            print("⚠️ FFmpeg stderr:", err.decode())
            raise RuntimeError("FFmpeg failed")

        return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0

    except Exception as e:
        raise RuntimeError(f"Decoding error: {e}")
