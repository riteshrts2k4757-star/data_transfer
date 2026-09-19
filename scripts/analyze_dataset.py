"""Create a small before/after waveform report for cleaned paired audio."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import soundfile as sf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data")
    parser.add_argument("--output", default="cleaned_data")
    parser.add_argument("--plot-dir", default="logs/plots")
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    created = 0
    for split in ("train", "validation", "test"):
        originals = sorted((Path(args.input) / split / "noisy").glob("*.wav"))
        cleaned = sorted((Path(args.output) / split / "noisy").glob("*.wav"))
        for original in originals[:args.count]:
            clean_original = Path(args.input) / split / "clean" / original.name
            if not clean_original.exists():
                continue
            noisy, sr = sf.read(original, always_2d=False)
            clean, _ = sf.read(clean_original, always_2d=False)
            fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=False)
            axes[0].plot(noisy, linewidth=0.35, label="Original noisy")
            axes[0].plot(clean, linewidth=0.35, label="Original clean")
            axes[0].legend()
            axes[0].set_title(f"{split}/{original.stem} before cleaning")
            if cleaned:
                processed, _ = sf.read(cleaned[0], always_2d=False)
                processed_clean_path = Path(args.output) / split / "clean" / cleaned[0].name
                processed_clean, _ = sf.read(processed_clean_path, always_2d=False)
                axes[1].plot(processed, linewidth=0.35, label="Cleaned noisy")
                axes[1].plot(processed_clean, linewidth=0.35, label="Cleaned clean")
                axes[1].legend()
            axes[1].set_title("After cleaning / first generated chunk")
            fig.tight_layout()
            fig.savefig(plot_dir / f"{split}_{original.stem}.png", dpi=140)
            plt.close(fig)
            created += 1
    print(f"Created {created} waveform plots in {plot_dir}")


if __name__ == "__main__":
    main()
