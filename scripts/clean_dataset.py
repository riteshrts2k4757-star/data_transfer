"""Validate, standardize, align, and segment paired noisy/clean audio.

The input dataset is never modified. Each pair is loaded, validated, optionally
trimmed using the clean reference, normalized with one shared gain, and written
to OUTPUT_DIR as fixed-length WAV chunks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - keeps the script usable without tqdm
    def tqdm(iterable, **_kwargs):
        return iterable


# ----------------------------- Configuration -----------------------------
INPUT_DIR = "data"
OUTPUT_DIR = "cleaned_data"
LOG_DIR = "logs"

TARGET_SR = 16000
TARGET_CHANNELS = 1
CHUNK_DURATION = 4.0

NORMALIZE = True
TARGET_PEAK = 0.95

SILENCE_THRESHOLD_DB = -40.0
MIN_SPEECH_DURATION = 0.2
MIN_AUDIO_DURATION = 1.0

CLIPPING_THRESHOLD = 0.99
MAX_CLIPPING_RATIO = 0.01
MAX_DURATION_DIFFERENCE = 0.05
HANDLE_LAST_CHUNK = "discard"  # "discard" or "pad"

SPLITS = ("train", "validation", "test")
CHUNK_SAMPLES = int(TARGET_SR * CHUNK_DURATION)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if audio.size == 0:
        raise ValueError("empty audio")
    if not np.isfinite(audio).all():
        raise ValueError("NaN or infinite samples")
    audio = np.mean(audio, axis=1)
    if sample_rate <= 0:
        raise ValueError("invalid sample rate")
    if sample_rate != TARGET_SR:
        gcd = math.gcd(sample_rate, TARGET_SR)
        audio = resample_poly(audio, TARGET_SR // gcd, sample_rate // gcd).astype(np.float32)
    return np.asarray(audio, dtype=np.float32), TARGET_SR


def clipping_ratio(audio: np.ndarray) -> tuple[int, float]:
    clipped = int(np.count_nonzero(np.abs(audio) >= CLIPPING_THRESHOLD))
    return clipped, clipped / max(1, audio.size)


def speech_bounds(clean: np.ndarray) -> tuple[int, int]:
    if not clean.size:
        return 0, 0
    window = max(1, int(MIN_SPEECH_DURATION * TARGET_SR))
    squared = clean.astype(np.float64) ** 2
    rms = np.sqrt(np.convolve(squared, np.ones(window) / window, mode="same"))
    peak = float(np.max(rms))
    threshold = peak * (10 ** (SILENCE_THRESHOLD_DB / 20))
    active = np.flatnonzero(rms >= threshold)
    if active.size == 0:
        return 0, clean.size
    return int(active[0]), int(active[-1] + 1)


def pair_reason(noisy_path: Path, clean_path: Path) -> tuple[str | None, dict]:
    try:
        noisy, _ = load_audio(noisy_path)
    except Exception as exc:
        return f"noisy: {exc}", {}
    try:
        clean, _ = load_audio(clean_path)
    except Exception as exc:
        return f"clean: {exc}", {}

    noisy_duration = noisy.size / TARGET_SR
    clean_duration = clean.size / TARGET_SR
    if noisy_duration < MIN_AUDIO_DURATION or clean_duration < MIN_AUDIO_DURATION:
        return "audio shorter than MIN_AUDIO_DURATION", {}
    if abs(noisy_duration - clean_duration) > MAX_DURATION_DIFFERENCE:
        return f"duration mismatch ({noisy_duration:.4f}s vs {clean_duration:.4f}s)", {}

    noisy_count, noisy_ratio = clipping_ratio(noisy)
    clean_count, clean_ratio = clipping_ratio(clean)
    if max(noisy_ratio, clean_ratio) > MAX_CLIPPING_RATIO:
        return f"clipping ratio too high ({max(noisy_ratio, clean_ratio):.6f})", {}

    return None, {
        "noisy": noisy,
        "clean": clean,
        "noisy_rms": float(np.sqrt(np.mean(noisy.astype(np.float64) ** 2))),
        "clean_rms": float(np.sqrt(np.mean(clean.astype(np.float64) ** 2))),
        "clipping_ratio": max(noisy_ratio, clean_ratio),
        "clipped_samples": noisy_count + clean_count,
    }


def prepare_pair(noisy: np.ndarray, clean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    length = min(noisy.size, clean.size)
    noisy = noisy[:length]
    clean = clean[:length]
    start, end = speech_bounds(clean)
    if end - start >= int(MIN_SPEECH_DURATION * TARGET_SR):
        noisy, clean = noisy[start:end], clean[start:end]
    if NORMALIZE:
        shared_peak = float(max(np.max(np.abs(noisy)), np.max(np.abs(clean)), 1e-12))
        gain = TARGET_PEAK / shared_peak
        noisy = noisy * gain
        clean = clean * gain
    return noisy.astype(np.float32), clean.astype(np.float32)


def discover_pairs(split_dir: Path) -> tuple[list[tuple[str, Path, Path]], list[dict]]:
    noisy_dir = split_dir / "noisy"
    clean_dir = split_dir / "clean"
    noisy_files = {path.stem: path for path in noisy_dir.glob("*") if path.is_file()}
    clean_files = {path.stem: path for path in clean_dir.glob("*") if path.is_file()}
    missing: list[dict] = []
    for stem in sorted(noisy_files.keys() - clean_files.keys()):
        missing.append({"split": split_dir.name, "filename": stem, "missing": "clean"})
    for stem in sorted(clean_files.keys() - noisy_files.keys()):
        missing.append({"split": split_dir.name, "filename": stem, "missing": "noisy"})
    pairs = [(stem, noisy_files[stem], clean_files[stem]) for stem in sorted(noisy_files.keys() & clean_files.keys())]
    return pairs, missing


def make_dirs(output_dir: Path) -> None:
    for split in SPLITS:
        for kind in ("noisy", "clean"):
            (output_dir / split / kind).mkdir(parents=True, exist_ok=True)


def process(input_dir: Path, output_dir: Path, log_dir: Path, dry_run: bool) -> dict:
    corrupted: list[dict] = []
    missing: list[dict] = []
    clipped: list[dict] = []
    alignment: list[dict] = []
    metadata: list[dict] = []
    before_files = 0
    valid_pairs = 0
    rejected_pairs = 0
    chunks_by_split = Counter()
    durations: list[float] = []
    rms_values: list[float] = []
    sample_rates = Counter()
    channels = Counter()

    if not dry_run:
        make_dirs(output_dir)

    for split in SPLITS:
        pairs, split_missing = discover_pairs(input_dir / split)
        missing.extend(split_missing)
        before_files += len(pairs) * 2 + len(split_missing)
        for stem, noisy_path, clean_path in tqdm(pairs, desc=f"Processing {split}"):
            reason, details = pair_reason(noisy_path, clean_path)
            if reason:
                rejected_pairs += 1
                corrupted.append({"split": split, "filename": stem, "reason": reason})
                if "duration mismatch" in reason:
                    alignment.append({"split": split, "filename": stem, "reason": reason})
                if "clipping ratio" in reason:
                    clipped.append({"split": split, "filename": stem, "reason": reason})
                continue

            valid_pairs += 1
            noisy, clean = prepare_pair(details["noisy"], details["clean"])
            sample_rates[TARGET_SR] += 2
            channels[TARGET_CHANNELS] += 2
            durations.append(clean.size / TARGET_SR)
            rms_values.extend([float(np.sqrt(np.mean(noisy ** 2))), float(np.sqrt(np.mean(clean ** 2)))])
            full_chunks, remainder = divmod(clean.size, CHUNK_SAMPLES)
            chunk_count = full_chunks + int(HANDLE_LAST_CHUNK == "pad" and remainder > 0)
            for chunk_index in range(chunk_count):
                start = chunk_index * CHUNK_SAMPLES
                end = start + CHUNK_SAMPLES
                noisy_chunk, clean_chunk = noisy[start:end], clean[start:end]
                if noisy_chunk.size < CHUNK_SAMPLES:
                    if HANDLE_LAST_CHUNK != "pad":
                        continue
                    pad = CHUNK_SAMPLES - noisy_chunk.size
                    noisy_chunk = np.pad(noisy_chunk, (0, pad))
                    clean_chunk = np.pad(clean_chunk, (0, pad))
                item_id = f"{split}_{len(metadata) + 1:06d}"
                filename = f"{item_id}.wav"
                if not dry_run:
                    sf.write(output_dir / split / "noisy" / filename, noisy_chunk, TARGET_SR, subtype="PCM_16")
                    sf.write(output_dir / split / "clean" / filename, clean_chunk, TARGET_SR, subtype="PCM_16")
                metadata.append({
                    "id": item_id,
                    "split": split,
                    "noisy_path": f"{split}/noisy/{filename}",
                    "clean_path": f"{split}/clean/{filename}",
                    "original_file": stem,
                    "duration": f"{CHUNK_DURATION:.3f}",
                    "sample_rate": TARGET_SR,
                    "channels": TARGET_CHANNELS,
                    "num_samples": CHUNK_SAMPLES,
                    "clipping_ratio": f"{details['clipping_ratio']:.8f}",
                    "rms_noisy": f"{np.sqrt(np.mean(noisy_chunk ** 2)):.8f}",
                    "rms_clean": f"{np.sqrt(np.mean(clean_chunk ** 2)):.8f}",
                    "snr": "",
                    "processing_status": "cleaned",
                })
                chunks_by_split[split] += 1

    fields = ["split", "filename", "reason"]
    write_csv(log_dir / "corrupted_files.csv", corrupted, fields)
    write_csv(log_dir / "missing_pairs.csv", missing, ["split", "filename", "missing"])
    write_csv(log_dir / "clipped_files.csv", clipped, fields)
    write_csv(log_dir / "alignment_issues.csv", alignment, fields)
    report = {
        "configuration": {
            "target_sr": TARGET_SR, "target_channels": TARGET_CHANNELS,
            "chunk_duration": CHUNK_DURATION, "chunk_samples": CHUNK_SAMPLES,
            "normalize": NORMALIZE, "target_peak": TARGET_PEAK,
            "silence_threshold_db": SILENCE_THRESHOLD_DB,
            "min_speech_duration": MIN_SPEECH_DURATION,
            "min_audio_duration": MIN_AUDIO_DURATION,
            "max_clipping_ratio": MAX_CLIPPING_RATIO,
            "max_duration_difference": MAX_DURATION_DIFFERENCE,
            "handle_last_chunk": HANDLE_LAST_CHUNK,
        },
        "before": {"files": before_files, "paired_files": before_files - len(missing)},
        "after": {
            "valid_pairs": valid_pairs, "rejected_pairs": rejected_pairs,
            "chunks": sum(chunks_by_split.values()), "chunks_by_split": dict(chunks_by_split),
            "average_duration": float(np.mean(durations)) if durations else 0,
            "minimum_duration": float(np.min(durations)) if durations else 0,
            "maximum_duration": float(np.max(durations)) if durations else 0,
            "average_rms": float(np.mean(rms_values)) if rms_values else 0,
            "average_clipping_ratio": float(np.mean([float(row["clipping_ratio"]) for row in metadata])) if metadata else 0,
            "sample_rate_distribution": dict(sample_rates),
            "channel_distribution": dict(channels),
        },
        "removed": {
            "corrupted_or_rejected_pairs": len(corrupted),
            "missing_pairs": len(missing),
            "clipped_pairs": len(clipped),
            "alignment_issues": len(alignment),
        },
        "dry_run": dry_run,
    }
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_text = [
        "====================================", "DATASET CLEANING COMPLETE", "====================================",
        f"Sample Rate: {TARGET_SR} Hz", f"Channels: {TARGET_CHANNELS}",
        f"Chunk Length: {CHUNK_DURATION:g} seconds", f"Chunk Samples: {CHUNK_SAMPLES}", "",
        f"Original paired files: {report['before']['paired_files']}",
        f"Valid pairs: {valid_pairs}", f"Rejected pairs: {rejected_pairs}",
        f"Missing pairs: {len(missing)}", f"Clipped pairs: {len(clipped)}",
        f"Alignment issues: {len(alignment)}", "",
        f"Train chunks: {chunks_by_split['train']}",
        f"Validation chunks: {chunks_by_split['validation']}",
        f"Test chunks: {chunks_by_split['test']}",
        f"Total chunks: {sum(chunks_by_split.values())}", "",
        f"Output directory: {output_dir}", f"Dataset status: {'ANALYSIS ONLY' if dry_run else 'READY FOR TRAINING'}",
        "====================================",
    ]
    (log_dir / "dataset_report.txt").write_text("\n".join(report_text) + "\n", encoding="utf-8")
    if not dry_run:
        write_csv(output_dir / "metadata.csv", metadata, [
            "id", "split", "noisy_path", "clean_path", "original_file", "duration",
            "sample_rate", "channels", "num_samples", "clipping_ratio",
            "rms_noisy", "rms_clean", "snr", "processing_status",
        ])
    return report


def main() -> None:
    global TARGET_SR, CHUNK_DURATION, CHUNK_SAMPLES, SILENCE_THRESHOLD_DB
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=INPUT_DIR)
    parser.add_argument("--output", default=OUTPUT_DIR)
    parser.add_argument("--sample-rate", type=int, default=TARGET_SR)
    parser.add_argument("--chunk-duration", type=float, default=CHUNK_DURATION)
    parser.add_argument("--silence-threshold", type=float, default=SILENCE_THRESHOLD_DB)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    TARGET_SR = args.sample_rate
    CHUNK_DURATION = args.chunk_duration
    CHUNK_SAMPLES = int(TARGET_SR * CHUNK_DURATION)
    SILENCE_THRESHOLD_DB = args.silence_threshold
    report = process(Path(args.input), Path(args.output), Path(LOG_DIR), args.dry_run)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
