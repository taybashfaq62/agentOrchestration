import torch
from faster_whisper import WhisperModel
from moonshine_onnx import MoonshineOnnxModel
from kokoro_onnx import Kokoro

_registry = {}

def load_all():
    """
    Initializes and caches all models in the global registry.
    Designed to be invoked exactly once via the FastAPI lifespan startup event.
    """
    if _registry:
        print("Models are already initialized. Skipping redundant loading.")
        return

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

    print("All models loaded and ready inside the global registry.")


def clear_all():
    """
    Clears the global model registry and runs garbage collection.
    Designed to be invoked via the FastAPI lifespan shutdown event.
    """
    global _registry
    if not _registry:
        return
        
    print("🧹 Cleaning up model registry resources...")
    _registry.clear()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Model registry successfully cleared.")


def get(name: str):
    """
    Retrieves a loaded model from the registry by its identifier string.
    """
    if not _registry:
        raise RuntimeError(
            f"The model registry is completely empty. "
            f"Ensure `models.load_all()` was called at application startup."
        )

    if name not in _registry:
        raise KeyError(
            f"Model '{name}' was not found in the registry. "
            f"Available models: {list(_registry.keys())}"
        )

    return _registry[name]