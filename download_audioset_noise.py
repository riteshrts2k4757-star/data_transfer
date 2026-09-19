from pathlib import Path
import csv
import io
import subprocess
import sys
import urllib.request

import imageio_ffmpeg

SEGMENTS_URL = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv"
OUTPUT_DIR = Path(__file__).parent / "noise"
DATASETS = {
    "engine": ("/m/02mk9", 10),
    "helicopter": ("/m/09ct_", 15),
    "heavyEngine": ("/t/dd00067", 15),
    "explosion": ("/m/014zdl", 10),
    "artilleryFire": ("/m/0_1c", 10),
    "aircraft": ("/m/0k5j", 10),
}

def load_segments() -> dict[str, list[tuple[str, float, float]]]:
    segments = {name: [] for name in DATASETS}
    labels_to_names = {label: name for name, (label, _) in DATASETS.items()}
    with urllib.request.urlopen(SEGMENTS_URL) as response:
        text = io.TextIOWrapper(response, encoding="utf-8")
        for row in csv.reader(line for line in text if not line.startswith("#")):
            video_id, start, end, *labels = [value.strip().strip('"') for value in row]
            for label in labels:
                name = labels_to_names.get(label)
                if name is not None:
                    segments[name].append((video_id, float(start), float(end)))
    return segments

def download_segment(ffmpeg: str, output_path: Path, video_id: str, start: float, end: float) -> bool:
    command = [
        sys.executable,
    "-m",
    "yt_dlp",
    "--no-playlist",
    "--quiet",
    "--no-warnings",
    "--ffmpeg-location",
    ffmpeg,
    "--download-sections",
    f"*{start}-{end}",
    "--force-keyframes-at-cuts",
    "-x",
    "--audio-format",
    "wav",
    "--audio-quality",
    "0",
    "-o",
    str(output_path.with_suffix(".%(ext)s")),
    f"https://www.youtube.com/watch?v={video_id}",
    ]
    result = subprocess.run(command, check=False)
    if result.returncode != 0 or not output_path.exists():
        output_path.unlink(missing_ok=True)
        return False
    return True

def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    segments = load_segments()

    for name, (_, target) in DATASETS.items():
        existing = sorted(OUTPUT_DIR.glob(f"{name}_*.wav"))
        downloaded = len(existing)
        if downloaded >= target:
            print(f"{name}: already have {downloaded}/{target}")
            continue

        for video_id, start, end in segments[name]:
            if downloaded >= target:
                break
            output_path = OUTPUT_DIR / f"{name}_{downloaded + 1:04d}.wav"
            if download_segment(ffmpeg, output_path, video_id, start, end):
                downloaded += 1
                print(f"{name}: {downloaded}/{target} ({video_id}, {start:.3f}-{end:.3f}s)")
            else:
                print(f"{name}: skipped unavailable video {video_id}")

        if downloaded < target:
            raise RuntimeError(f"{name}: only downloaded {downloaded} of {target} segments")

    print(f"Saved requested AudioSet segments to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
from pathlib import Path
import csv
import io
import subprocess
import sys
import urllib.request

import imageio_ffmpeg


SEGMENTS_URL = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv"
GUNSHOT_LABEL = "/m/032s66"
OUTPUT_DIR = Path(__file__).parent / "noise"
TARGET_COUNT = 10


def load_segments() -> list[tuple[str, float, float]]:
    with urllib.request.urlopen(SEGMENTS_URL) as response:
        text = io.TextIOWrapper(response, encoding="utf-8")
        segments = []
        for row in csv.reader(line for line in text if not line.startswith("#")):
            video_id, start, end, *labels = [value.strip().strip('"') for value in row]
            if GUNSHOT_LABEL in labels:
                segments.append((video_id, float(start), float(end)))
        return segments


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    existing = sorted(OUTPUT_DIR.glob("*.wav"))
    if len(existing) >= TARGET_COUNT:
        print(f"Already have {len(existing)} WAV files in {OUTPUT_DIR}")
        return

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    segments = load_segments()
    downloaded = len(existing)
    for video_id, start, end in segments:
        if downloaded >= TARGET_COUNT:
            break
        output_path = OUTPUT_DIR / f"{downloaded + 1:04d}.wav"
        command = [
            sys.executable,
            "-m",
            "yt_dlp",
            "--no-playlist",
            "--quiet",
            "--no-warnings",
            "--ffmpeg-location",
            ffmpeg,
            "--download-sections",
            f"*{start}-{end}",
            "--force-keyframes-at-cuts",
            "-x",
            "--audio-format",
            "wav",
            "--audio-quality",
            "0",
            "-o",
            str(output_path.with_suffix(".%(ext)s")),
            f"https://www.youtube.com/watch?v={video_id}",
        ]
        result = subprocess.run(command, check=False)
        if result.returncode != 0 or not output_path.exists():
            print(f"Skipped unavailable video {video_id}")
            output_path.unlink(missing_ok=True)
            continue

        downloaded += 1
        print(f"Downloaded {downloaded}/{TARGET_COUNT} ({video_id}, {start:.3f}-{end:.3f}s)")

    if downloaded != TARGET_COUNT:
        raise RuntimeError(f"Only downloaded {downloaded} of {TARGET_COUNT} labeled segments")
    print(f"Saved {downloaded} WAV files to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()