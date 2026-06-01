from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import soundfile as sf
import numpy as np
import tempfile, os, io, time
from moonshine_onnx import MoonshineOnnxModel   
import models
from routers import stt, tts

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models at startup."""
    models.load_all()
    yield

app = FastAPI(
    title="Agent Orchestration API",
    description="faster-whisper + Moonshine STT | Kokoro ONNX TTS",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Transcript", "X-Response", "X-STT-Ms", "X-TTS-Ms"] # Browser headers access ke liye
)

app.include_router(stt.router, prefix="/stt", tags=["Speech to Text"])
app.include_router(tts.router, prefix="/tts", tags=["Text to Speech"])


@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "models": ["faster_whisper", "moonshine", "kokoro"]}


@app.post("/pipeline", tags=["Full Pipeline"])
@app.get("/pipeline", tags=["Full Pipeline"])  
async def full_pipeline(
    audio: UploadFile = File(None), 
    stt_backend: str = Form("moonshine"),
    voice: str = Form("af_heart"),
    speed: float = Form(1.0),
):
    """
    Full voice pipeline:
    Audio → STT (moonshine or faster_whisper) → echo TTS (Kokoro) → Audio

    Replace the echo with your LLM call for a full voice agent.
    """
    if audio is None:
        response_text = "Pipeline diagnostic active. Please send a POST multipart audio file."
        transcript = "[Diagnostic Mode]"
        timings = {"stt_ms": 0.0}
    else:
        timings = {}
        data = await audio.read()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            t0 = time.perf_counter()
            if stt_backend == "moonshine":
                audio_np, sr = sf.read(tmp_path, dtype="float32")
                if sr != 16000:
                    raise HTTPException(422, "Moonshine needs 16kHz audio")
                if audio_np.ndim > 1:
                    audio_np = audio_np.mean(axis=1)
                model = models.get("moonshine")
                tokens = model.transcribe(audio_np)
                transcript = MoonshineOnnxModel.decode(tokens)

            else:  
                model = models.get("faster_whisper")
                segments, _ = model.transcribe(tmp_path, beam_size=5, vad_filter=True)
                transcript = " ".join(s.text.strip() for s in segments)
            timings["stt_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        finally:
            os.unlink(tmp_path)
        response_text = f"You said: {transcript}"   # ← replace with LLM call

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