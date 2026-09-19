from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.audio_utils import MODEL_SAMPLE_RATE, ensure_mono, read_audio_bytes, resample_to_target
from backend.inference import ModelService

MODEL_SERVICE = ModelService()
app = FastAPI(title="AI Audio Denoising API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event() -> None:
    try:
        MODEL_SERVICE.load()
    except Exception as exc:  # pragma: no cover - startup validation path
        app.state.model_error = str(exc)
        app.state.model_loaded = False
    else:
        app.state.model_error = None
        app.state.model_loaded = True


@app.get("/api/health")
def health() -> dict[str, object]:
    loaded = getattr(app.state, "model_loaded", False)
    error_text = getattr(app.state, "model_error", None)
    return {
        "status": "ok" if loaded else "error",
        "model_loaded": loaded,
        "device": str(MODEL_SERVICE.device),
        "sample_rate": MODEL_SAMPLE_RATE,
        "model_path": str(MODEL_SERVICE.model_path),
        "error": error_text,
    }


@app.post("/api/denoise")
async def denoise(audio: UploadFile = File(...)) -> FileResponse:
    if not audio or not audio.filename:
        raise HTTPException(status_code=400, detail="No audio file was provided.")

    try:
        file_bytes = await audio.read()
        waveform, sample_rate = read_audio_bytes(file_bytes)
        waveform = ensure_mono(waveform)
        if waveform.size == 0:
            raise ValueError("Empty file after decoding.")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to decode uploaded audio: {exc}") from exc

    try:
        if not getattr(app.state, "model_loaded", False):
            MODEL_SERVICE.load()
        denoised = MODEL_SERVICE.inference(waveform, sample_rate)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unable to process audio with the model: {exc}") from exc

    output_dir = Path(tempfile.gettempdir())
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"denoised_{uuid.uuid4().hex}.wav"
    sf.write(str(output_path), np.clip(denoised.astype(np.float32), -1.0, 1.0), MODEL_SAMPLE_RATE, subtype="PCM_16")
    return FileResponse(
        output_path,
        media_type="audio/wav",
        filename=f"denoised_{audio.filename or 'audio'}.wav",
        headers={
            "X-Model-Loaded": "true",
            "X-Sample-Rate": str(MODEL_SAMPLE_RATE),
            "X-Duration-Seconds": str(max(len(denoised) / MODEL_SAMPLE_RATE, 0.0)),
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
