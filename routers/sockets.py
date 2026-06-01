# routers/realtime.py
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import numpy as np
import torch
from moonshine_onnx import MoonshineOnnxModel
import models
import vad

router = APIRouter()

BYTES_PER_SAMPLE = 4 
CHUNK_SIZE_SAMPLES = 512  
CHUNK_SIZE_BYTES = CHUNK_SIZE_SAMPLES * BYTES_PER_SAMPLE

MAX_SPEECH_DURATION_SECS = 15  
MAX_SPEECH_SAMPLES = MAX_SPEECH_DURATION_SECS * vad.SAMPLE_RATE

@router.websocket("/stream")
async def realtime_audio_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time streaming audio transcription.
    Expects raw float32 binary audio data chunks (16kHz, mono).
    
    Yields JSON text transcripts back to the client immediately upon speech termination.
    """
    await websocket.accept()
    print("Real-time audio WebSocket client connected.")
    audio_buffer = []
    speech_started = False
    consecutive_silence_chunks = 0
    
    SILENCE_TIMEOUT_CHUNKS = 25  
    VAD_THRESHOLD = 0.45

    try:
        while True:
            data = await websocket.receive_bytes()
            
            if len(data) < CHUNK_SIZE_BYTES:
                data += b'\x00' * (CHUNK_SIZE_BYTES - len(data))
                
            chunk_np = np.frombuffer(data, dtype=np.float32)

            speech_prob = vad.is_speech(chunk_np)
            is_current_chunk_speech = speech_prob > VAD_THRESHOLD

            if is_current_chunk_speech:
                if not speech_started:
                    speech_started = True
                    print("🗣️ Speech detected. Buffering audio...")
                
                audio_buffer.append(chunk_np)
                consecutive_silence_chunks = 0
            else:
                if speech_started:
                    audio_buffer.append(chunk_np)
                    consecutive_silence_chunks += 1

            reached_silence_timeout = speech_started and (consecutive_silence_chunks >= SILENCE_TIMEOUT_CHUNKS)
            reached_buffer_limit = len(audio_buffer) * CHUNK_SIZE_SAMPLES >= MAX_SPEECH_SAMPLES

            if reached_silence_timeout or reached_buffer_limit:
                print("⏱️ End of phrase detected. Running low-latency transcription...")
                
                full_phrase_audio = np.concatenate(audio_buffer)
                loop = asyncio.get_running_loop()
                
                def _inference_worker():
                    moonshine_model = models.get("moonshine")
                    tokens = moonshine_model.transcribe(full_phrase_audio)
                    with torch.no_grad():
                        text = MoonshineOnnxModel.decode(tokens)
                    return text

                transcript = await loop.run_in_executor(None, _inference_worker)
                if transcript.strip():
                    await websocket.send_json({
                        "event": "transcript",
                        "text": transcript.strip(),
                        "final": True
                    })
                
                audio_buffer.clear()
                speech_started = False
                consecutive_silence_chunks = 0

    except WebSocketDisconnect:
        print("Real-time audio WebSocket client disconnected.")
    except Exception as e:
        print(f"Error in streaming websocket session: {str(e)}")
        await websocket.close(code=1011)
    finally:
        del audio_buffer