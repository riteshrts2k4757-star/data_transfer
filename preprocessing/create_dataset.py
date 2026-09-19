from __future__ import annotations

import csv
import math
import random
import re
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


# Configuration
CLEAN_DIR = "clean"
NOISE_DIR = "noise"
OUTPUT_DIR = "data"

SAMPLE_RATE = 16000
SNR_LEVELS = [0, 5, 10, 15, 20]
MIXTURES_PER_CLEAN = 5

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42

WAV_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".aif"}
EXPECTED_SNR_TOLERANCE_DB = 0.15


def validate_configuration() -> None:
    if not math.isclose(TRAIN_RATIO + VAL_RATIO + TEST_RATIO, 1.0):
        raise ValueError("TRAIN_RATIO, VAL_RATIO, and TEST_RATIO must sum to 1")
    if SAMPLE_RATE <= 0 or MIXTURES_PER_CLEAN <= 0:
        raise ValueError("SAMPLE_RATE and MIXTURES_PER_CLEAN must be positive")
    if not SNR_LEVELS:
        raise ValueError("SNR_LEVELS must contain at least one value")


def list_audio_files(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in WAV_EXTENSIONS
    )


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    try:
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    except Exception as error:
        raise RuntimeError(f"Could not read {path}: {error}") from error

    if audio.size == 0 or not np.isfinite(audio).all():
        raise RuntimeError(f"Audio is empty or contains NaN/Inf values: {path}")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    elif audio.ndim != 1:
        raise RuntimeError(f"Unsupported audio shape {audio.shape}: {path}")
    return audio.astype(np.float32, copy=False), sample_rate


def convert_audio(audio: np.ndarray, source_rate: int) -> np.ndarray:
    if source_rate == SAMPLE_RATE:
        return audio
    divisor = math.gcd(source_rate, SAMPLE_RATE)
    converted = resample_poly(audio, SAMPLE_RATE // divisor, source_rate // divisor)
    return converted.astype(np.float32)


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))


def crop_or_loop_noise(noise: np.ndarray, length: int, rng: random.Random) -> np.ndarray:
    if len(noise) == 0:
        raise ValueError("Noise audio is empty")
    if len(noise) >= length:
        start = rng.randrange(len(noise) - length + 1)
        return noise[start:start + length].copy()
    repeats = (length + len(noise) - 1) // len(noise)
    tiled = np.tile(noise, repeats)
    return tiled[:length].copy()


def category_for_noise(path: Path) -> str:
    match = re.match(r"([A-Za-z]+(?:Fire|Engine)?)_\d+\.wav$", path.name)
    if match:
        return match.group(1)
    return "unknown"


def split_sources(files: list[Path], rng: random.Random) -> dict[str, list[Path]]:
    shuffled = files.copy()
    rng.shuffle(shuffled)
    train_count = int(len(shuffled) * TRAIN_RATIO)
    val_count = int(len(shuffled) * VAL_RATIO)
    return {
        "train": shuffled[:train_count],
        "validation": shuffled[train_count:train_count + val_count],
        "test": shuffled[train_count + val_count:],
    }


def prepare_output() -> None:
    output = Path(OUTPUT_DIR)
    for split in ("train", "validation", "test"):
        (output / split / "noisy").mkdir(parents=True, exist_ok=True)
        (output / split / "clean").mkdir(parents=True, exist_ok=True)


def write_wav(path: Path, audio: np.ndarray) -> None:
    if not np.isfinite(audio).all():
        raise RuntimeError(f"Generated audio contains NaN/Inf values: {path}")
    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        audio = audio / peak * 0.999
    sf.write(path, np.clip(audio, -1.0, 1.0), SAMPLE_RATE, subtype="PCM_16")


def generate_dataset(
    clean_files: list[Path],
    noise_files: list[Path],
    splits: dict[str, list[Path]],
    rng: random.Random,
) -> list[dict[str, object]]:
    noise_audio: list[tuple[Path, np.ndarray]] = []
    invalid_noise: list[tuple[Path, str]] = []
    for path in noise_files:
        try:
            audio, rate = load_audio(path)
            converted = convert_audio(audio, rate)
            if rms(converted) <= 1e-8:
                invalid_noise.append((path, "silent audio"))
                continue
            noise_audio.append((path, converted))
        except RuntimeError as error:
            invalid_noise.append((path, str(error)))
    if not noise_audio:
        raise RuntimeError("No readable noise files were found")

    shuffled_noise = noise_audio.copy()
    rng.shuffle(shuffled_noise)
    metadata: list[dict[str, object]] = []
    sample_number = 1
    noise_index = 0
    snr_values = list(SNR_LEVELS)
    for split_name, source_files in splits.items():
        split_snr_values = snr_values.copy()
        rng.shuffle(split_snr_values)
        for clean_path in source_files:
            clean, rate = load_audio(clean_path)
            clean = convert_audio(clean, rate)
            if len(clean) == 0:
                raise RuntimeError(f"Clean file is empty: {clean_path}")

            for mixture_index in range(MIXTURES_PER_CLEAN):
                snr_db = split_snr_values[mixture_index % len(split_snr_values)]
                clean_rms = rms(clean)
                if clean_rms == 0:
                    raise RuntimeError(f"Silent source audio cannot be mixed: {clean_path}")
                for _ in range(len(shuffled_noise)):
                    noise_path, noise = shuffled_noise[noise_index % len(shuffled_noise)]
                    noise_index += 1
                    cropped_noise = crop_or_loop_noise(noise, len(clean), rng)
                    noise_rms = rms(cropped_noise)
                    if noise_rms > 1e-8:
                        break
                else:
                    raise RuntimeError("No non-silent noise crop was available")
                target_ratio = 10 ** (-snr_db / 20)
                scaled_noise = cropped_noise * (clean_rms * target_ratio / noise_rms)
                noisy = clean + scaled_noise
                pair_peak = max(float(np.max(np.abs(clean))), float(np.max(np.abs(noisy))))
                if pair_peak > 0.999:
                    clean = clean * (0.999 / pair_peak)
                    noisy = noisy * (0.999 / pair_peak)
                sample_id = f"sample_{sample_number:06d}"
                clean_output = Path(OUTPUT_DIR) / split_name / "clean" / f"{sample_id}.wav"
                noisy_output = Path(OUTPUT_DIR) / split_name / "noisy" / f"{sample_id}.wav"
                write_wav(clean_output, clean)
                write_wav(noisy_output, noisy)
                metadata.append({
                    "id": sample_id,
                    "split": split_name,
                    "clean_file": clean_path.name,
                    "noise_file": noise_path.name,
                    "noise_category": category_for_noise(noise_path),
                    "snr_db": snr_db,
                    "sample_rate": SAMPLE_RATE,
                    "duration": len(clean) / SAMPLE_RATE,
                })
                sample_number += 1
    if invalid_noise:
        print(f"Skipped unreadable noise files: {len(invalid_noise)}")
        for path, reason in invalid_noise:
            print(f"  {path.name}: {reason}")
    return metadata


def validate_dataset(metadata: list[dict[str, object]]) -> None:
    expected_pairs = len(metadata)
    actual_snr_errors: list[float] = []
    for row in metadata:
        split = str(row["split"])
        sample_id = str(row["id"])
        clean_path = Path(OUTPUT_DIR) / split / "clean" / f"{sample_id}.wav"
        noisy_path = Path(OUTPUT_DIR) / split / "noisy" / f"{sample_id}.wav"
        clean, clean_rate = load_audio(clean_path)
        noisy, noisy_rate = load_audio(noisy_path)
        if clean_rate != SAMPLE_RATE or noisy_rate != SAMPLE_RATE:
            raise RuntimeError(f"Wrong sample rate for {sample_id}")
        if len(clean) != len(noisy):
            raise RuntimeError(f"Duration mismatch for {sample_id}")
        if rms(clean) == 0:
            raise RuntimeError(f"Silent clean output for {sample_id}")
        clean_power = float(np.mean(np.square(clean), dtype=np.float64))
        residual = noisy - clean
        noise_power = float(np.mean(np.square(residual), dtype=np.float64))
        measured_snr = 10 * math.log10(clean_power / noise_power)
        actual_snr_errors.append(abs(measured_snr - float(row["snr_db"])))
        if np.max(np.abs(noisy)) > 1.0:
            raise RuntimeError(f"Clipping detected in {sample_id}")

    expected_splits = {
        "train": int(expected_pairs * TRAIN_RATIO / (TRAIN_RATIO + VAL_RATIO + TEST_RATIO)),
        "validation": int(expected_pairs * VAL_RATIO / (TRAIN_RATIO + VAL_RATIO + TEST_RATIO)),
    }
    expected_splits["test"] = expected_pairs - sum(expected_splits.values())
    for split, expected in expected_splits.items():
        noisy_count = len(list((Path(OUTPUT_DIR) / split / "noisy").glob("*.wav")))
        clean_count = len(list((Path(OUTPUT_DIR) / split / "clean").glob("*.wav")))
        if noisy_count != expected or clean_count != expected:
            raise RuntimeError(f"{split} pair count mismatch: {noisy_count}/{clean_count}, expected {expected}")
    if max(actual_snr_errors, default=0) > EXPECTED_SNR_TOLERANCE_DB:
        raise RuntimeError(f"Measured SNR error exceeds tolerance: {max(actual_snr_errors):.3f} dB")


def write_metadata(metadata: list[dict[str, object]]) -> None:
    path = Path(OUTPUT_DIR) / "metadata.csv"
    fields = ["id", "split", "clean_file", "noise_file", "noise_category", "snr_db", "sample_rate", "duration"]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metadata)


def main() -> None:
    validate_configuration()
    clean_files = list_audio_files(Path(CLEAN_DIR))
    noise_files = list_audio_files(Path(NOISE_DIR))
    if not clean_files or not noise_files:
        raise RuntimeError("CLEAN_DIR and NOISE_DIR must contain readable audio files")
    rng = random.Random(SEED)
    splits = split_sources(clean_files, rng)
    prepare_output()
    metadata = generate_dataset(clean_files, noise_files, splits, rng)
    write_metadata(metadata)
    validate_dataset(metadata)

    print("====================================")
    print("DATASET GENERATION COMPLETE")
    print("====================================")
    print(f"Clean source files : {len(clean_files)}")
    print(f"Noise source files : {len(noise_files)}")
    for split in ("train", "validation", "test"):
        print(f"{split.capitalize():18}: {sum(row['split'] == split for row in metadata)}")
    print(f"Sample rate        : {SAMPLE_RATE} Hz")
    print("Channels           : Mono")
    print(f"SNR levels         : {', '.join(map(str, SNR_LEVELS))} dB")
    print(f"Output directory   : {Path(OUTPUT_DIR).resolve()}\\")
    print("Dataset validation : PASSED")
    print("====================================")


if __name__ == "__main__":
    main()