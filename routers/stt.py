from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import soundfile as sf
import numpy as np
import tempfile
import os
import time
import torch
from moonshine_onnx import MoonshineOnnxModel
import models

router = APIRouter()

def _run_silero_vad(audio_np: np.ndarray, sr: int = 16000, threshold: float = 0.5) -> np.ndarray:
    """
    Filters out non-speech regions using Silero VAD.
    Returns concatenated speech-only audio as float32 numpy array.
    """
    vad_model = models.get("vad")
    vad_utils = models.get("vad_utils")
    get_speech_timestamps, collect_chunks, _, _, _ = vad_utils

    audio_tensor = torch.from_numpy(audio_np).float()

    with torch.no_grad():
        speech_timestamps = get_speech_timestamps(
            audio_tensor,
            vad_model,
            threshold=threshold,
            sampling_rate=sr,
            min_speech_duration_ms=250,
            min_silence_duration_ms=100,
        )

        if not speech_timestamps:
            return audio_np  

        speech_audio = collect_chunks(speech_timestamps, audio_tensor)
        return speech_audio.numpy()


@router.post("/transcribe")
def transcribe(
    audio: UploadFile = File(..., description="WAV audio file (16kHz mono for Moonshine)"),
    backend: str = Form(
        "faster_whisper",
        description="STT backend: 'faster_whisper' or 'moonshine'"
    ),
    vad: bool = Form(
        False,
        description="Apply Silero VAD to strip silence before transcription (Moonshine only). "
                    "faster_whisper uses its own built-in vad_filter."
    ),
    vad_threshold: float = Form(
        0.5,
        description="Silero VAD speech probability threshold (0.0–1.0). Default 0.5."
    ),
):
    """
    Transcribe audio using either faster-whisper or Moonshine.
    """
    data = audio.file.read()

    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        try:
            with os.fdopen(fd, "wb") as tmp:
                tmp.write(data)
            
            start = time.perf_counter()

            if backend == "faster_whisper":
                model = models.get("faster_whisper")
                segments, info = model.transcribe(
                    tmp_path,
                    beam_size=5,
                    vad_filter=True,                    
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                text = " ".join(seg.text.strip() for seg in segments)
                elapsed = time.perf_counter() - start
                
                return {
                    "text": text,
                    "language": info.language,
                    "language_probability": round(info.language_probability, 3),
                    "backend": backend,
                    "vad": "built-in",
                    "latency_ms": round(elapsed * 1000, 1)
                }

            elif backend == "moonshine":
                audio_np, sr = sf.read(tmp_path, dtype="float32")

                if sr != 16000:
                    raise HTTPException(
                        status_code=422,
                        detail=f"Moonshine requires 16kHz audio. Got {sr}Hz. "
                               f"Resample first or use faster_whisper backend."
                    )

                if audio_np.ndim > 1:
                    audio_np = audio_np.mean(axis=1)

                vad_applied = False
                if vad:
                    filtered = _run_silero_vad(audio_np, sr=sr, threshold=vad_threshold)
                    if filtered.size == 0 or (filtered is audio_np and len(audio_np) == 0):
                        return {
                            "text": "",
                            "backend": backend,
                            "vad": "silero",
                            "vad_threshold": vad_threshold,
                            "note": "No speech detected by VAD.",
                            "latency_ms": round((time.perf_counter() - start) * 1000, 1)
                        }
                    audio_np = filtered
                    vad_applied = True

                model = models.get("moonshine")
                tokens = model.transcribe(audio_np)
                
                with torch.no_grad():
                    text = MoonshineOnnxModel.decode(tokens)
                    
                elapsed = time.perf_counter() - start

                return {
                    "text": text,
                    "backend": backend,
                    "vad": "silero" if vad_applied else "none",
                    "vad_threshold": vad_threshold if vad_applied else None,
                    "latency_ms": round(elapsed * 1000, 1)
                }

            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown backend '{backend}'. Use 'faster_whisper' or 'moonshine'."
                )
        
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription pipeline failure: {str(e)}")