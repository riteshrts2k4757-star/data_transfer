from pathlib import Path
import io
import tarfile
import urllib.request
import wave

import soundfile as sf


ARCHIVE_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
OUTPUT_DIR = Path(__file__).parent / "clean"
TARGET_COUNT = 1000


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    existing = sorted(OUTPUT_DIR.glob("*.wav"))
    if len(existing) >= TARGET_COUNT:
        print(f"Already have {len(existing)} WAV files in {OUTPUT_DIR}")
        return

    downloaded = len(existing)
    print(f"Streaming {ARCHIVE_URL}")
    with urllib.request.urlopen(ARCHIVE_URL) as response:
        with tarfile.open(fileobj=response, mode="r|gz") as archive:
            for member in archive:
                if downloaded >= TARGET_COUNT:
                    break
                if not member.isfile() or not member.name.lower().endswith(".flac"):
                    continue

                source = archive.extractfile(member)
                if source is None:
                    continue

                audio, sample_rate = sf.read(io.BytesIO(source.read()), dtype="int16")
                output_path = OUTPUT_DIR / f"{downloaded + 1:04d}.wav"
                with wave.open(str(output_path), "wb") as wav_file:
                    wav_file.setnchannels(1 if audio.ndim == 1 else audio.shape[1])
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(audio.tobytes())

                downloaded += 1
                if downloaded % 100 == 0 or downloaded == TARGET_COUNT:
                    print(f"Downloaded {downloaded}/{TARGET_COUNT}")

    if downloaded != TARGET_COUNT:
        raise RuntimeError(f"Archive ended after {downloaded} clean utterances")
    print(f"Saved {downloaded} WAV files to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()