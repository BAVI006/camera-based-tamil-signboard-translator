"""FastAPI entry point shared by the web and mobile clients."""

import os

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from backend.services.ocr_service import OCRConfigurationError, OCRProcessingError
from backend.services.history_service import HistoryService
from backend.services.preprocessing import ImageValidationError, MAX_UPLOAD_BYTES
from backend.services.sign_pipeline import (
    PipelineOptions,
    SignPipeline,
    UnsupportedTranslationTarget,
)


class SignAnalysisResponse(BaseModel):
    tamil_text: str
    transliteration: str | None
    english_translation: str | None
    category: str
    context: str
    confidence: float | None
    needs_review: bool
    processing_time_ms: int | None
    warnings: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    ocr_model: str = "Tesseract OCR"
    translation_model: str | None = None
    context_model: str = "Rule-based classifier + reviewed phrase dictionary"


app = FastAPI(
    title="SignSense AI API",
    description="Tamil signboard preprocessing, OCR, translation, and context services.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv(
        "SIGN_CORS_ORIGINS",
        "http://localhost:8501,http://127.0.0.1:8501",
    ).split(","),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

pipeline = SignPipeline()
history = HistoryService()


@app.get("/health")
def health():
    translation_backend = os.getenv("SIGN_TRANSLATION_BACKEND", "")
    return {
        "status": "ok",
        "ocr_engine": "tesseract",
        "translation_configured": translation_backend in {"indictrans2", "nllb", "http"},
    }


@app.get("/api/v1/history")
def get_history(limit: int = Query(default=50, ge=1, le=100)):
    return {"items": history.list_recent(limit)}


@app.delete("/api/v1/history")
def delete_history():
    history.clear()
    return {"cleared": True}


@app.post("/api/v1/translate", response_model=SignAnalysisResponse)
async def translate_sign(
    file: UploadFile = File(...),
    translation_language: str = Query(default="eng_Latn"),
    transliteration_enabled: bool = Query(default=True),
):
    if translation_language != "eng_Latn":
        raise HTTPException(
            status_code=422,
            detail="English is currently the only enabled translation target.",
        )

    image_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 15 MB upload limit.")

    try:
        result = pipeline.analyze(
            image_bytes,
            options=PipelineOptions(
                translation_target=translation_language,
                transliteration_enabled=transliteration_enabled,
            ),
        )
    except ImageValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except UnsupportedTranslationTarget as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OCRConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except OCRProcessingError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    await run_in_threadpool(history.save, file.filename or "signboard", result)
    return SignAnalysisResponse(**result)
