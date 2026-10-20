"""FastAPI application factory.

    uvicorn fraudguard.api.main:app --reload
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Callable, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from .. import __version__
from ..serving import ModelBundle, load_bundle, predict_records
from .schemas import (
    BatchPredictResponse,
    BatchRequest,
    PredictResponse,
    Transaction,
)

logger = logging.getLogger("fraudguard.api")

_THRESHOLD_QUERY = Query(
    None,
    gt=0,
    le=1,
    description="Optional decision threshold for this call. Defaults to the model's cost-tuned threshold.",
)


def create_app(loader: Optional[Callable[[], ModelBundle]] = None) -> FastAPI:
    """`loader` is injectable so tests can serve a tiny model without any MLflow or files."""
    loader = loader or load_bundle

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.bundle = None
        app.state.load_error = None
        try:
            app.state.bundle = loader()
            logger.info("Loaded model version %s", app.state.bundle.version)
        except Exception as exc:  # keep the process up so /health can explain what is wrong
            logger.exception("Model failed to load")
            app.state.load_error = f"{type(exc).__name__}: {exc}"
        yield

    app = FastAPI(
        title="fraudguard API",
        version=__version__,
        description="Scores card transactions for fraud with the champion model from the registry.",
        lifespan=lifespan,
    )

    origins = [o.strip() for o in os.environ.get("FRAUD_CORS_ORIGINS", "*").split(",") if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["*"])

    def get_bundle(request: Request) -> ModelBundle:
        bundle = request.app.state.bundle
        if bundle is None:
            raise HTTPException(status_code=503, detail=f"Model not loaded: {request.app.state.load_error}")
        return bundle

    @app.get("/")
    def root():
        return {"service": "fraudguard", "version": __version__, "docs": "/docs", "health": "/health"}

    @app.get("/health")
    def health(request: Request):
        bundle = get_bundle(request)
        return {"status": "ok", "model_loaded": True, "model_version": bundle.version}

    @app.get("/model-info")
    def model_info(request: Request):
        bundle = get_bundle(request)
        return {**bundle.info, "threshold": bundle.threshold}

    @app.post("/predict", response_model=PredictResponse)
    def predict(tx: Transaction, request: Request, threshold: Optional[float] = _THRESHOLD_QUERY):
        bundle = get_bundle(request)
        result = predict_records(bundle, [tx.model_dump()], threshold)[0]
        return {**result, "model_version": bundle.version}

    @app.post("/predict/batch", response_model=BatchPredictResponse)
    def predict_batch(payload: BatchRequest, request: Request, threshold: Optional[float] = _THRESHOLD_QUERY):
        bundle = get_bundle(request)
        predictions = predict_records(bundle, [t.model_dump() for t in payload.transactions], threshold)
        return {
            "model_version": bundle.version,
            "threshold": predictions[0]["threshold"],
            "count": len(predictions),
            "flagged": sum(p["is_fraud"] for p in predictions),
            "predictions": predictions,
        }

    return app
