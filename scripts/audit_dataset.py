"""Audit paired speech-denoising audio without modifying the source dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import correlate, resample_poly

SPLITS = ("train", "validation", "test")
AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".aif"}
MODEL_RATE = 16000
SILENCE_DB = -45.0
MIN_DURATION = 0.25
ALIGNMENT_LIMIT_SECONDS = 0.25
ALIGNMENT_REVIEW_SECONDS = 0.02
DURATION_REVIEW_SECONDS = 0.05
SNR_REVIEW_LOW = -10.0
SNR_REVIEW_HIGH = 25.0
CLIP_WARNING = 0.001
CLIP_SEVERE = 0.01
SILENCE_REVIEW = 0.98


def csv_write(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if audio.size else 0.0


def db(value: float) -> float:
    return 20.0 * math.log10(max(value, 1e-12))


def audio_hash(audio: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(audio, dtype=np.float32).tobytes()).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_audio(path: Path) -> tuple[np.ndarray, int, int, str | None]:
    try:
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        if audio.size == 0:
            raise ValueError("empty audio")
        if not np.isfinite(audio).all():
            raise ValueError("NaN or infinite samples")
        channels = int(audio.shape[1])
        return audio.mean(axis=1).astype(np.float32), int(sample_rate), channels, None
    except Exception as exc:
        return np.empty(0, dtype=np.float32), 0, 0, str(exc)


def resample(audio: np.ndarray, source_rate: int) -> np.ndarray:
    if source_rate == MODEL_RATE or not audio.size:
        return audio.astype(np.float32, copy=False)
    divisor = math.gcd(source_rate, MODEL_RATE)
    return resample_poly(audio, MODEL_RATE // divisor, source_rate // divisor).astype(np.float32)


def silence_ratio(audio: np.ndarray, sample_rate: int) -> tuple[float, float, float]:
    if not audio.size:
        return 1.0, 0.0, 0.0
    threshold = max(rms(audio) * 10 ** (SILENCE_DB / 20.0), 1e-5)
    frame = max(1, int(sample_rate * 0.02))
    usable = audio[: len(audio) - len(audio) % frame]
    if not usable.size:
        return float(np.all(np.abs(audio) < threshold)), 0.0, len(audio) / sample_rate
    frame_rms = np.sqrt(np.mean(usable.reshape(-1, frame) ** 2, axis=1))
    silent = frame_rms < threshold
    longest = current = 0
    for value in silent:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return float(np.mean(silent)), float(np.mean(~silent)), longest * frame / sample_rate


def clipping(audio: np.ndarray) -> tuple[float, int, float]:
    if not audio.size:
        return 0.0, 0, 0.0
    clipped = np.abs(audio) >= 0.999
    return float(np.mean(clipped)), int(np.count_nonzero(clipped)), float(np.max(np.abs(audio)))


def estimate_alignment(clean: np.ndarray, noisy: np.ndarray, sample_rate: int) -> tuple[int, float]:
    length = min(len(clean), len(noisy))
    if length < sample_rate // 4:
        return 0, 0.0
    step = max(1, sample_rate // 100)
    clean_env = np.sqrt(np.convolve(clean[:length] ** 2, np.ones(step) / step, mode="valid")[::step])
    noisy_env = np.sqrt(np.convolve(noisy[:length] ** 2, np.ones(step) / step, mode="valid")[::step])
    count = min(len(clean_env), len(noisy_env))
    clean_env = clean_env[:count] - np.mean(clean_env[:count])
    noisy_env = noisy_env[:count] - np.mean(noisy_env[:count])
    denominator = np.linalg.norm(clean_env) * np.linalg.norm(noisy_env)
    if denominator <= 1e-12:
        return 0, 0.0
    max_lag = min(int(ALIGNMENT_LIMIT_SECONDS * sample_rate / step), count - 1)
    values = correlate(noisy_env, clean_env, mode="full")
    center = count - 1
    window = values[center - max_lag:center + max_lag + 1]
    index = int(np.argmax(window)) - max_lag
    return index * step, float(window[index + max_lag] / denominator)


def noise_features(clean: np.ndarray, noisy: np.ndarray) -> dict[str, float]:
    length = min(len(clean), len(noisy))
    clean = clean[:length]
    noisy = noisy[:length]
    noise = noisy - clean
    speech_rms = rms(clean)
    noise_rms = rms(noise)
    snr = db(speech_rms / noise_rms) if noise_rms > 1e-9 else 60.0
    speech_threshold = max(speech_rms * 10 ** (SILENCE_DB / 20.0), 1e-5)
    speech_active = np.abs(clean) >= speech_threshold
    inactive = ~speech_active
    inactive_noise_rms = rms(noise[inactive]) if np.any(inactive) else noise_rms
    inactive_ratio = float(np.mean(np.abs(noise[inactive]) > max(inactive_noise_rms * 2.5, 1e-4))) if np.any(inactive) else 0.0
    return {
        "noise_rms": noise_rms,
        "speech_rms": speech_rms,
        "estimated_snr_db": snr,
        "noise_to_clean_rms": noise_rms / max(speech_rms, 1e-9),
        "noise_in_clean_silence_ratio": inactive_ratio,
    }


def suspicious_noise(features: dict[str, float], clean: np.ndarray, noisy: np.ndarray) -> tuple[str, str]:
    snr = features["estimated_snr_db"]
    similarity = float(np.corrcoef(clean[: min(len(clean), len(noisy))], noisy[: min(len(clean), len(noisy))])[0, 1]) if len(clean) > 1 else 0.0
    if snr >= SNR_REVIEW_HIGH or similarity >= 0.999:
        return "REVIEW", "almost clean / little measurable noise"
    if snr <= SNR_REVIEW_LOW:
        return "REVIEW", "extreme low SNR; speech may be buried"
    if features["noise_in_clean_silence_ratio"] > 0.35:
        return "REVIEW", "possible speech or structured content in noise estimate"
    return "", ""


def discover(input_dir: Path) -> tuple[list[tuple[str, str, Path, Path]], list[dict]]:
    pairs = []
    missing = []
    for split in SPLITS:
        noisy_dir = input_dir / split / "noisy"
        clean_dir = input_dir / split / "clean"
        noisy = {p.stem: p for p in noisy_dir.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS} if noisy_dir.exists() else {}
        clean = {p.stem: p for p in clean_dir.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS} if clean_dir.exists() else {}
        for stem in sorted(noisy.keys() - clean.keys()):
            missing.append({"split": split, "filename": stem, "missing": "clean", "classification": "REJECT"})
        for stem in sorted(clean.keys() - noisy.keys()):
            missing.append({"split": split, "filename": stem, "missing": "noisy", "classification": "REJECT"})
        pairs.extend((split, stem, noisy[stem], clean[stem]) for stem in sorted(noisy.keys() & clean.keys()))
    return pairs, missing


def inventory_row(path: Path, split: str, kind: str) -> dict:
    audio, rate, channels, error = read_audio(path)
    clip_ratio, clipped_count, peak = clipping(audio)
    silent, active, longest = silence_ratio(audio, rate) if error is None else (1.0, 0.0, 0.0)
    return {
        "split": split, "kind": kind, "filename": path.name, "path": str(path),
        "duration_seconds": round(len(audio) / rate, 6) if rate else 0,
        "sample_rate": rate, "channels": channels, "num_samples": len(audio),
        "minimum_amplitude": float(np.min(audio)) if audio.size else 0,
        "maximum_amplitude": float(np.max(audio)) if audio.size else 0,
        "rms": rms(audio), "peak_amplitude": peak, "clipping_percentage": clip_ratio * 100,
        "clipped_samples": clipped_count, "silence_percentage": silent * 100,
        "speech_active_percentage": active * 100, "longest_silence_seconds": longest,
        "file_size_bytes": path.stat().st_size if path.exists() else 0,
        "format": path.suffix.lower().lstrip("."), "sha256": file_hash(path) if error is None else "",
        "error": error or "",
    }


def plot_histogram(values: list[float], path: Path, title: str, xlabel: str) -> None:
    plt.figure(figsize=(9, 5))
    if values:
        plt.hist(values, bins=30, color="#246a73", edgecolor="white")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Pairs")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def write_html(path: Path, summary: dict, rows: list[dict], review_dir: Path) -> None:
    table_rows = []
    for row in rows[:2000]:
        table_rows.append("<tr>" + "".join(f"<td>{html.escape(str(row.get(k, '')))}</td>" for k in ("split", "filename", "classification", "reasons", "estimated_snr_db", "alignment_offset_samples")) + "</tr>")
    path.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>Dataset Audit</title>
<style>body{{font:14px system-ui;margin:2rem;color:#18252b}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccd6d8;padding:6px;text-align:left}}th{{background:#e8f0ef}}.stat{{display:inline-block;margin:0 1rem 1rem 0;padding:1rem;background:#eef5f3}}</style></head><body>
<h1>Dataset Audit</h1><p>Source: <code>{html.escape(summary['input'])}</code>. Original files were read-only.</p>
<div>{''.join(f'<span class="stat"><b>{html.escape(k)}</b><br>{html.escape(str(v))}</span>' for k,v in summary['counts'].items())}</div>
<h2>Before / after</h2><pre>{html.escape(json.dumps(summary['split_summary'], indent=2))}</pre>
<h2>Reports</h2><p><a href="snr_distribution.png">SNR distribution</a> | <a href="duration_distribution.png">Duration distribution</a> | <a href="quality_summary.png">Quality summary</a> | manual review: <code>{html.escape(str(review_dir))}</code></p>
<h2>Pair results (first 2,000)</h2><table><tr><th>Split</th><th>Filename</th><th>Classification</th><th>Reasons</th><th>SNR dB</th><th>Alignment samples</th></tr>{''.join(table_rows)}</table></body></html>""", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data")
    parser.add_argument("--audit-output", default="dataset_audit")
    parser.add_argument("--corrected-output", default="dataset_corrected")
    parser.add_argument("--review-limit", type=int, default=200)
    args = parser.parse_args()
    input_dir = Path(args.input)
    audit_dir = Path(args.audit_output)
    corrected_dir = Path(args.corrected_output)
    audit_dir.mkdir(parents=True, exist_ok=True)
    review_dir = audit_dir / "manual_review"
    review_dir.mkdir(parents=True, exist_ok=True)

    pairs, missing = discover(input_dir)
    inventory = []
    for split, _, noisy_path, clean_path in pairs:
        inventory.extend((inventory_row(noisy_path, split, "noisy"), inventory_row(clean_path, split, "clean")))
    inventory_fields = list(inventory[0].keys()) if inventory else ["path"]
    csv_write(audit_dir / "audio_inventory.csv", inventory, inventory_fields)

    pair_rows = []
    duplicate_groups: dict[str, list[str]] = defaultdict(list)
    leakage_groups: dict[str, list[str]] = defaultdict(list)
    snr_values = []
    duration_values = []
    split_summary = {split: {"before_pairs": 0, "keep": 0, "review": 0, "reject": 0} for split in SPLITS}
    for split, stem, noisy_path, clean_path in pairs:
        split_summary[split]["before_pairs"] += 1
        noisy, noisy_rate, noisy_channels, noisy_error = read_audio(noisy_path)
        clean, clean_rate, clean_channels, clean_error = read_audio(clean_path)
        reasons = []
        classification = "KEEP"
        alignment_offset = 0
        alignment_corr = 0.0
        features = {"noise_rms": 0.0, "speech_rms": 0.0, "estimated_snr_db": 0.0, "noise_to_clean_rms": 0.0, "noise_in_clean_silence_ratio": 0.0}
        if noisy_error:
            reasons.append(f"CORRUPTED_NOISY: {noisy_error}")
        if clean_error:
            reasons.append(f"CORRUPTED_CLEAN: {clean_error}")
        if not reasons:
            noisy_model = resample(noisy, noisy_rate)
            clean_model = resample(clean, clean_rate)
            duration_difference = abs(len(noisy_model) - len(clean_model)) / MODEL_RATE
            alignment_offset, alignment_corr = estimate_alignment(clean_model, noisy_model, MODEL_RATE)
            features = noise_features(clean_model, noisy_model)
            snr_values.append(features["estimated_snr_db"])
            duration_values.append(len(clean_model) / MODEL_RATE)
            if len(clean_model) / MODEL_RATE < MIN_DURATION:
                reasons.append("too short")
            if duration_difference > DURATION_REVIEW_SECONDS:
                reasons.append(f"duration mismatch {duration_difference:.4f}s")
            if max(noisy_channels, clean_channels) != 1:
                reasons.append("multi-channel source")
            if alignment_corr < 0.15:
                reasons.append("weak speech relationship / possible wrong pair")
            if abs(alignment_offset) / MODEL_RATE > ALIGNMENT_REVIEW_SECONDS:
                reasons.append(f"alignment offset {alignment_offset} samples")
            noisy_clip = clipping(noisy)[0]
            clean_clip = clipping(clean)[0]
            if max(noisy_clip, clean_clip) >= CLIP_SEVERE:
                reasons.append("severe clipping")
            elif max(noisy_clip, clean_clip) >= CLIP_WARNING:
                reasons.append("clipping warning")
            noisy_silence = silence_ratio(noisy, noisy_rate)[0]
            clean_silence = silence_ratio(clean, clean_rate)[0]
            if max(noisy_silence, clean_silence) >= SILENCE_REVIEW:
                reasons.append("mostly silent")
            noise_class, noise_reason = suspicious_noise(features, clean_model, noisy_model)
            if noise_reason:
                reasons.append(noise_reason)
            duplicate_groups[audio_hash(clean_model)].append(f"{split}/{stem}/clean")
            duplicate_groups[audio_hash(noisy_model)].append(f"{split}/{stem}/noisy")
            leakage_groups[audio_hash(clean_model)].append(f"{split}/{stem}")
        hard_reasons = any(x.startswith(("CORRUPTED", "severe clipping", "weak speech relationship")) for x in reasons)
        review_reasons = any(x.startswith(("duration mismatch", "alignment offset", "clipping warning", "mostly silent", "almost clean", "extreme low SNR", "possible speech", "multi-channel")) for x in reasons)
        if hard_reasons:
            classification = "REJECT"
        elif review_reasons:
            classification = "REVIEW"
        split_summary[split][classification.lower()] += 1
        if classification == "REVIEW" and sum(1 for r in pair_rows if r["classification"] == "REVIEW") < args.review_limit:
            target = review_dir / split / stem
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(noisy_path, target / f"{stem}_noisy{noisy_path.suffix.lower()}")
            shutil.copy2(clean_path, target / f"{stem}_clean{clean_path.suffix.lower()}")
        if classification == "KEEP":
            for kind, source in (("noisy", noisy_path), ("clean", clean_path)):
                target = corrected_dir / split / kind / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        pair_rows.append({
            "split": split, "filename": stem, "noisy_path": str(noisy_path), "clean_path": str(clean_path),
            "classification": classification, "reasons": "; ".join(reasons) or "none",
            "noisy_sample_rate": noisy_rate, "clean_sample_rate": clean_rate,
            "noisy_channels": noisy_channels, "clean_channels": clean_channels,
            "duration_seconds": round(len(clean) / clean_rate, 6) if clean_rate else 0,
            "duration_difference_seconds": round(abs(len(noisy) / max(noisy_rate, 1) - len(clean) / max(clean_rate, 1)), 6),
            "alignment_offset_samples": alignment_offset, "alignment_offset_seconds": alignment_offset / MODEL_RATE,
            "alignment_correlation": alignment_corr, **features,
            "noisy_sha256": file_hash(noisy_path) if not noisy_error else "", "clean_sha256": file_hash(clean_path) if not clean_error else "",
        })

    for item in missing:
        pair_rows.append({"split": item["split"], "filename": item["filename"], "classification": "REJECT", "reasons": f"missing {item['missing']} pair"})
        split_summary[item["split"]]["before_pairs"] += 1
        split_summary[item["split"]]["reject"] += 1
    pair_fields = sorted({key for row in pair_rows for key in row})
    csv_write(audit_dir / "pair_analysis.csv", pair_rows, pair_fields)
    csv_write(audit_dir / "rejected_files.csv", [r for r in pair_rows if r.get("classification") == "REJECT"], pair_fields)
    csv_write(audit_dir / "review_files.csv", [r for r in pair_rows if r.get("classification") == "REVIEW"], pair_fields)
    csv_write(audit_dir / "kept_files.csv", [r for r in pair_rows if r.get("classification") == "KEEP"], pair_fields)
    duplicates = [{"hash": key, "files": "; ".join(value), "count": len(value)} for key, value in duplicate_groups.items() if len(value) > 1]
    leakage = [{"clean_audio_hash": key, "splits_and_files": "; ".join(value), "count": len(value)} for key, value in leakage_groups.items() if len({v.split("/")[0] for v in value}) > 1]
    csv_write(audit_dir / "duplicates.csv", duplicates, ["hash", "files", "count"])
    csv_write(audit_dir / "data_leakage.csv", leakage, ["clean_audio_hash", "splits_and_files", "count"])

    plot_histogram(snr_values, audit_dir / "snr_distribution.png", "Input SNR distribution", "Estimated input SNR (dB)")
    plot_histogram(duration_values, audit_dir / "duration_distribution.png", "Duration distribution", "Duration (seconds)")
    quality_values = [sum(r["classification"] == value for r in pair_rows) for value in ("KEEP", "REVIEW", "REJECT")]
    plt.figure(figsize=(7, 5)); plt.bar(("KEEP", "REVIEW", "REJECT"), quality_values, color=("#338a70", "#d39a32", "#b94a48")); plt.title("Quality classification"); plt.ylabel("Pairs"); plt.tight_layout(); plt.savefig(audit_dir / "quality_summary.png", dpi=140); plt.close()

    summary = {
        "input": str(input_dir), "outputs": {"audit": str(audit_dir), "corrected": str(corrected_dir)},
        "counts": {"total_pairs": len(pair_rows), "keep": sum(r["classification"] == "KEEP" for r in pair_rows), "review": sum(r["classification"] == "REVIEW" for r in pair_rows), "reject": sum(r["classification"] == "REJECT" for r in pair_rows), "duplicate_groups": len(duplicates), "leakage_groups": len(leakage)},
        "split_summary": split_summary,
        "snr": {"average_db": float(np.mean(snr_values)) if snr_values else None, "median_db": float(np.median(snr_values)) if snr_values else None, "minimum_db": float(np.min(snr_values)) if snr_values else None, "maximum_db": float(np.max(snr_values)) if snr_values else None},
        "configuration": {"silence_db": SILENCE_DB, "snr_review_low": SNR_REVIEW_LOW, "snr_review_high": SNR_REVIEW_HIGH, "alignment_review_seconds": ALIGNMENT_REVIEW_SECONDS, "review_limit": args.review_limit},
        "note": "This audit never denoises input audio and never modifies the original dataset. REVIEW pairs are excluded from dataset_corrected.",
    }
    (audit_dir / "audit_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_html(audit_dir / "audit_report.html", summary, pair_rows, review_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
