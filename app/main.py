"""FastAPI backend for SentimentAI."""
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.preprocessing import clean_text
from app.schemas import HealthResponse, PredictRequest, PredictResponse, RootResponse

logger = logging.getLogger("sentimentai")
BASE_DIR = Path(__file__).resolve().parent.parent


def _artifact_paths():
    """Read paths at startup (not import time) so tests can override them."""
    model = Path(os.getenv("MODEL_PATH", BASE_DIR / "model" / "sentiment_model.joblib"))
    vectorizer = Path(os.getenv("VECTORIZER_PATH", BASE_DIR / "model" / "tfidf_vectorizer.joblib"))
    return model, vectorizer


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    app.state.vectorizer = None
    app.state.load_error = None
    model_path, vectorizer_path = _artifact_paths()
    try:
        app.state.model = joblib.load(model_path)
        app.state.vectorizer = joblib.load(vectorizer_path)
        logger.info("Model and vectorizer loaded.")
    except Exception as exc:  # missing file, version mismatch, corrupt file...
        app.state.model = None
        app.state.vectorizer = None
        app.state.load_error = f"{type(exc).__name__}: {exc}"
        logger.error("Could not load artifacts: %s", app.state.load_error)
    yield


app = FastAPI(
    title="SentimentAI API",
    version="1.0.0",
    description="Predicts whether a piece of text is positive or negative.",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Turn Pydantic's long error list into one readable message."""
    first = exc.errors()[0]
    message = str(first.get("msg", "Invalid input")).replace("Value error, ", "")
    field = ".".join(str(p) for p in first.get("loc", []) if p != "body")
    if field:
        message = f"{field}: {message}"
    return JSONResponse(status_code=422, content={"detail": message})


def _require_model(request: Request):
    if request.app.state.model is None or request.app.state.vectorizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Run model/train_model.py and restart the API.",
        )
    return request.app.state.model, request.app.state.vectorizer


@app.get("/", response_model=RootResponse)
def root():
    return RootResponse(
        name="SentimentAI API",
        version="1.0.0",
        description="Send text to POST /predict to get a positive/negative prediction.",
        endpoints=["GET /", "GET /health", "POST /predict", "GET /docs"],
    )


@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    if request.app.state.model is None or request.app.state.vectorizer is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unavailable",
                "model_loaded": False,
                "detail": request.app.state.load_error or "Model artifacts not loaded.",
            },
        )
    return HealthResponse(status="ok", model_loaded=True, detail="API and model are ready.")


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request):
    model, vectorizer = _require_model(request)

    cleaned = clean_text(payload.text)
    if not cleaned:
        raise HTTPException(
            status_code=422,
            detail="Text must contain at least one letter (numbers/symbols only are not supported).",
        )

    try:
        features = vectorizer.transform([cleaned])
        probabilities = model.predict_proba(features)[0]
        classes = [str(c) for c in model.classes_]
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed due to an internal error.")

    prob_map = {label: round(float(p), 4) for label, p in zip(classes, probabilities)}
    best_label = max(prob_map, key=prob_map.get)
    return PredictResponse(
        text=payload.text,
        sentiment=best_label,
        confidence=prob_map[best_label],
        probabilities=prob_map,
    )
