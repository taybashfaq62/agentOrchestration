# vad.py
import torch
import numpy as np
import models

SAMPLE_RATE = 16000
CHUNK_SIZE = 512  


def is_speech(audio_chunk: np.ndarray) -> float:
    """
    Returns speech probability (0.0 to 1.0) for a single audio chunk.
    audio_chunk: numpy array of float32, shape (512,) at 16kHz
    """
    vad_model = models.get("vad")
    tensor = torch.from_numpy(audio_chunk).float()
    with torch.no_grad():
        prob = vad_model(tensor, SAMPLE_RATE).item()
    return prob


def filter_speech_chunks(
    audio: np.ndarray,
    threshold: float = 0.5,
    min_speech_chunks: int = 3
) -> list[dict]:
    """
    Splits audio into speech segments using Silero VAD.

    Args:
        audio: full audio as float32 numpy array at 16kHz
        threshold: speech probability cutoff (0.5 recommended)
        min_speech_chunks: minimum consecutive speech chunks to count as speech

    Returns:
        List of dicts with 'start' and 'end' sample indices of speech segments
    """
    vad_model = models.get("vad")
    vad_utils = models.get("vad_utils")
    get_speech_timestamps, _, _, _, _ = vad_utils

    audio_tensor = torch.from_numpy(audio).float()

    speech_timestamps = get_speech_timestamps(
        audio_tensor,
        vad_model,
        threshold=threshold,
        sampling_rate=SAMPLE_RATE,
        min_speech_duration_ms=250,
        min_silence_duration_ms=100,
        return_seconds=False,
    )
    return speech_timestamps  


def extract_speech_audio(audio: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Returns only the speech portions of audio concatenated together.
    Useful for feeding clean audio to Whisper/Moonshine.
    """
    vad_utils = models.get("vad_utils")
    _, collect_chunks, _, _, _ = vad_utils

    segments = filter_speech_chunks(audio, threshold=threshold)
    if not segments:
        return np.array([], dtype=np.float32)

    audio_tensor = torch.from_numpy(audio).float()
    speech_audio = collect_chunks(segments, audio_tensor)
    return speech_audio.numpy()