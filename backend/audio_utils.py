from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

MODEL_SAMPLE_RATE = 16000
MODEL_INPUT_SAMPLES = 64000
MODEL_INPUT_SECONDS = MODEL_INPUT_SAMPLES / MODEL_SAMPLE_RATE


def read_audio_bytes(data: bytes) -> tuple[np.ndarray, int]:
    if not data:
        raise ValueError("Uploaded audio file is empty.")
    with io.BytesIO(data) as buffer:
        audio, sample_rate = sf.read(buffer, dtype="float32", always_2d=False)
    if audio.size == 0:
        raise ValueError("Uploaded audio file is empty or unreadable.")
    if not np.isfinite(audio).all():
        raise ValueError("Uploaded audio contains invalid numerical values.")
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def ensure_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32, copy=False)
    if audio.ndim == 2:
        return np.mean(audio, axis=1, dtype=np.float32).astype(np.float32)
    raise ValueError(f"Unsupported audio shape: {audio.shape}")


def resample_to_target(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio.astype(np.float32, copy=False)
    if len(audio) == 0:
        return audio.astype(np.float32, copy=False)
    divisor = math.gcd(source_rate, target_rate)
    numerator = target_rate // divisor
    denominator = source_rate // divisor
    converted = resample_poly(audio, numerator, denominator)
    return np.asarray(converted, dtype=np.float32)


def normalize_peak(audio: np.ndarray) -> tuple[np.ndarray, float]:
    if audio.size == 0:
        raise ValueError("Audio is empty.")
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-8:
        return audio.astype(np.float32, copy=False), 1.0
    return (audio / peak).astype(np.float32), peak


def chunk_audio(audio: np.ndarray, chunk_size: int = MODEL_INPUT_SAMPLES) -> list[np.ndarray]:
    if audio.size == 0:
        raise ValueError("Audio is empty.")
    if audio.size < chunk_size:
        padded = np.pad(audio, (0, chunk_size - audio.size), mode="constant", constant_values=0.0)
        return [padded.astype(np.float32, copy=False)]

    chunks = [audio[index:index + chunk_size] for index in range(0, audio.size, chunk_size)]
    last_chunk = chunks[-1]
    if last_chunk.size < chunk_size:
        chunks[-1] = np.pad(last_chunk, (0, chunk_size - last_chunk.size), mode="constant", constant_values=0.0)
    return [chunk.astype(np.float32, copy=False) for chunk in chunks]


def reconstruct_audio(chunks: list[np.ndarray], original_size: int, chunk_size: int = MODEL_INPUT_SAMPLES) -> np.ndarray:
    if not chunks:
        return np.empty(0, dtype=np.float32)
    merged = np.concatenate(chunks, axis=0)
    trimmed = merged[:original_size]
    return trimmed.astype(np.float32, copy=False)


def save_audio_wav(path: Path, audio: np.ndarray, sample_rate: int = MODEL_SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_audio = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    if safe_audio.size == 0:
        raise ValueError("No denoised audio was produced.")
    sf.write(str(path), safe_audio, sample_rate, subtype="PCM_16")
