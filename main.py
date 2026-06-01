from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import soundfile as sf
import numpy as np
import tempfile
import os
import io
import time
import torch
from moonshine_onnx import MoonshineOnnxModel   
import models
from routers import stt, tts, sockets

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application setup and breakdown cycles cleanly.
    Initializes models on boot and purges memory buffers upon termination.
    """
    models.load_all()
    yield
    models.clear_all()

app = FastAPI(
    title="Agent Orchestration API",
    description="faster-whisper + Moonshine STT | Kokoro ONNX TTS | Live WebSockets",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Transcript", "X-Response", "X-STT-Ms", "X-TTS-Ms"]
)

app.include_router(stt.router, prefix="/stt", tags=["Speech to Text"])
app.include_router(tts.router, prefix="/tts", tags=["Text to Speech"])
app.include_router(sockets.router, prefix="/realtime", tags=["Streaming Real-time"])


@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "models": ["faster_whisper", "moonshine", "kokoro"]}

@app.post("/pipeline", tags=["Full Pipeline"])
@app.get("/pipeline", tags=["Full Pipeline"])  
def full_pipeline(
    audio: UploadFile = File(None), 
    stt_backend: str = Form("moonshine"),
    voice: str = Form("af_heart"),
    speed: float = Form(1.0),
):
    """
    Full voice pipeline orchestration endpoint.
    Audio → STT (Moonshine / Whisper) → Response Transformation → TTS Engine (Kokoro)
    
    This endpoint executes sequentially on individual background threads 
    to preserve complete system throughput and event loop concurrency.
    """
    transcript = ""
    response_text = ""
    timings = {"stt_ms": 0.0}

    if audio is None:
        response_text = "Pipeline diagnostic active. Please send a POST multipart audio file."
        transcript = "[Diagnostic Mode]"
    else:
        data = audio.file.read()
        tmp_path = None
        
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".wav")
            try:
                with os.fdopen(fd, "wb") as tmp:
                    tmp.write(data)
                
                t0 = time.perf_counter()
                
                if stt_backend == "moonshine":
                    audio_np, sr = sf.read(tmp_path, dtype="float32")
                    if sr != 16000:
                        raise HTTPException(422, f"Moonshine requires 16kHz audio. Got {sr}Hz.")
                    
                    if audio_np.ndim > 1:
                        audio_np = audio_np.mean(axis=1)
                        
                    model = models.get("moonshine")
                    tokens = model.transcribe(audio_np)
                    
                    with torch.no_grad():
                        transcript = MoonshineOnnxModel.decode(tokens)

                else:  
                    model = models.get("faster_whisper")
                    segments, _ = model.transcribe(tmp_path, beam_size=5, vad_filter=True)
                    transcript = " ".join(s.text.strip() for s in segments)
                    
                timings["stt_ms"] = round((time.perf_counter() - t0) * 1000, 1)

            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Pipeline STT stage failure: {str(e)}")
            
        response_text = f"You said: {transcript}"  

    try:
        t1 = time.perf_counter()
        kokoro = models.get("kokoro")
        
        if not response_text.strip():
            response_text = "No speech context captured."

        samples, sample_rate = kokoro.create(
            response_text, voice=voice, speed=speed, lang="en-us"
        )
        
        stt_timing = timings.get("stt_ms", 0.0)
        tts_timing = round((time.perf_counter() - t1) * 1000, 1)

        buf = io.BytesIO()
        sf.write(buf, samples, sample_rate, format="WAV")
        buf.seek(0)

        return StreamingResponse(
            buf,
            media_type="audio/wav",
            headers={
                "X-Transcript": transcript.encode('utf-8').decode('latin-1'), 
                "X-Response": response_text.encode('utf-8').decode('latin-1'),
                "X-STT-Ms": str(stt_timing),
                "X-TTS-Ms": str(tts_timing),
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline TTS stage failure: {str(e)}")