from fastapi import APIRouter, Form, HTTPException, Query
from fastapi.responses import StreamingResponse
import soundfile as sf
import numpy as np
import io, time
import models

router = APIRouter()

AVAILABLE_VOICES = [
    "af_heart", "af_bella", "af_nicole", "af_sky",
    "am_adam", "am_michael",
    "bf_emma", "bf_isabella",
    "bm_george", "bm_lewis"
]

@router.post("/synthesize")
@router.get("/synthesize")
async def synthesize(
    text: str = Query(None, description="Text to convert to speech (For GET request)"),
    voice: str = Query("af_heart", description="Voice ID (For GET request)"),
    speed: float = Query(1.0, description="Speed multiplier (0.5 - 2.0) (For GET request)"),
    
    text_form: str = Form(None, alias="text", description="Text to convert to speech (For POST request)"),
    voice_form: str = Form("af_heart", alias="voice", description="Voice ID (For POST request)"),
    speed_form: float = Form(1.0, alias="speed", description="Speed multiplier (0.5 - 2.0) (For POST request)"),
):
    """
    Convert text to speech using Kokoro ONNX.
    Supports both POST (multipart/form-data) and GET (query params) for Swagger UI audio compatibility.
    Returns a WAV audio stream.
    """
    final_text = text if text is not None else text_form
    final_voice = voice if text is not None else voice_form
    final_speed = speed if text is not None else speed_form

    if not final_text:
        raise HTTPException(
            status_code=400, 
            detail="Text parameter is required."
        )

    if final_voice not in AVAILABLE_VOICES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice '{final_voice}'. Available: {AVAILABLE_VOICES}"
        )

    if not 0.5 <= final_speed <= 2.0:
        raise HTTPException(status_code=400, detail="Speed must be between 0.5 and 2.0")

    start = time.perf_counter()
    kokoro = models.get("kokoro")

    samples, sample_rate = kokoro.create(final_text, voice=final_voice, speed=final_speed, lang="en-us")

    elapsed = time.perf_counter() - start

    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV")
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="audio/wav",
        headers={
            "X-Latency-Ms": str(round(elapsed * 1000, 1)),
            "X-Voice": final_voice,
            "Content-Disposition": "inline; filename=speech.wav"
        }
    )

@router.get("/voices")
def list_voices():
    """List all available Kokoro voices."""
    return {"voices": AVAILABLE_VOICES}