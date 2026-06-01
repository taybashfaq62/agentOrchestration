# models.py

import torch
from faster_whisper import WhisperModel
from moonshine_onnx import MoonshineOnnxModel
from kokoro_onnx import Kokoro

_registry = {}

def load_all():

    print("[1/4] Loading faster-whisper (base.en, CPU int8)...")
    _registry["faster_whisper"] = WhisperModel(
        "base.en",
        device="cpu",
        compute_type="int8"
    )

    print("[2/4] Loading Moonshine ONNX (base)...")
    _registry["moonshine"] = MoonshineOnnxModel(model_name="moonshine/base")

    print("[3/4] Loading Kokoro ONNX TTS...")
    _registry["kokoro"] = Kokoro(
        model_path="kokoro-v1_0.onnx",
        voices_path="voices.bin"
    )

    print("[4/4] Loading Silero VAD...")
    vad_model, vad_utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
        trust_repo=True
    )
    _registry["vad"] = vad_model
    _registry["vad_utils"] = vad_utils

    print("✅ All models loaded and ready.")


def get(name):

    if name not in _registry:
        raise KeyError(
            f"Model '{name}' not loaded. Available: {list(_registry.keys())}"
        )

    return _registry[name]