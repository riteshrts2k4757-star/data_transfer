from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from backend.audio_utils import MODEL_INPUT_SAMPLES, MODEL_SAMPLE_RATE, chunk_audio, normalize_peak, reconstruct_audio, resample_to_target


class DenoisingAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.enc4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=15, padding=7),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=15, padding=7),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=16, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=16, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=16, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose1d(16, 16, kernel_size=16, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.final_conv = nn.Conv1d(16, 1, kernel_size=15, padding=7)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = self.enc1(audio)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.bottleneck(x)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.final_conv(x)
        return torch.tanh(x)


def resolve_model_path() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    return project_root / "best_model.pth"


class ModelService:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: DenoisingAutoencoder | None = None
        self.model_path = resolve_model_path()
        self.model_loaded = False
        self.error_message: str | None = None

    def load(self) -> None:
        if self.model is not None:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. Place best_model.pth in the project root directory."
            )

        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
        except Exception as exc:  # pragma: no cover - runtime-specific
            raise RuntimeError(f"Could not load model checkpoint: {exc}") from exc

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            raise TypeError("Model checkpoint format is unsupported; expected a state_dict or a dict with model_state_dict.")

        model = DenoisingAutoencoder().to(self.device)
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "Model architecture is incompatible with best_model.pth. The encoder/decoder structure does not match the checkpoint."
            ) from exc

        model.eval()
        self.model = model
        self.model_loaded = True
        self.error_message = None

    def inference(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        if waveform.size == 0:
            raise ValueError("Input audio is empty.")

        if self.model is None:
            self.load()

        audio = np.asarray(waveform, dtype=np.float32)
        if sample_rate != MODEL_SAMPLE_RATE:
            audio = resample_to_target(audio, sample_rate, MODEL_SAMPLE_RATE)

        audio = np.clip(audio, -1.0, 1.0)
        normalized_audio, peak = normalize_peak(audio)
        chunks = chunk_audio(normalized_audio, MODEL_INPUT_SAMPLES)
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for chunk in chunks:
                tensor = torch.from_numpy(chunk.reshape(1, 1, -1)).to(self.device)
                output = self.model(tensor)
                output = output.detach().cpu().numpy().reshape(-1)
                outputs.append(np.asarray(output, dtype=np.float32))

        combined = np.concatenate(outputs, axis=0)
        combined = combined[: audio.size]
        denoised = combined * peak
        denoised = np.clip(denoised.astype(np.float32), -1.0, 1.0)
        return denoised


def denoise_waveform(waveform: np.ndarray, sample_rate: int, model_service: ModelService | None = None) -> np.ndarray:
    manager = model_service or ModelService()
    return manager.inference(waveform, sample_rate)
