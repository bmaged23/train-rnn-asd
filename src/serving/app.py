"""Local HTTP API: readiness is distinct from a merely running web process."""
import hmac
import logging
import pickle
import threading
import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt

from src.inference import WindowPredictor
from src.serving.settings import Settings

Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    feature_names: list[str] = Field(min_length=1)
    features: list[list[list[Number]]] = Field(min_length=1, max_length=64)
    lengths: list[StrictInt] = Field(min_length=1, max_length=64)


class PredictionResponse(BaseModel):
    probabilities: list[float]
    speaking: list[bool]
    threshold: float
    checkpoint_sha256: str


class BodyLimit:
    """Bound actual received bytes, including requests without Content-Length."""
    def __init__(self, app, limit):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in ("POST", "PUT", "PATCH"):
            return await self.app(scope, receive, send)
        chunks, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > self.limit:
                response = JSONResponse({"detail": "Request body exceeds configured byte limit"}, status_code=413)
                return await response(scope, receive, send)
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        delivered = False

        async def bounded_receive():
            nonlocal delivered
            if delivered:
                return await receive()
            delivered = True
            return {"type": "http.request", "body": b"".join(chunks), "more_body": False}
        await self.app(scope, bounded_receive, send)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    inference_lock, metrics_lock = threading.Lock(), threading.Lock()
    counters = dict(completed=0, rejected_input=0, busy=0, failed=0, inference_seconds=0.)

    @asynccontextmanager
    async def lifespan(app):
        app.state.predictor = None
        if settings.checkpoint and settings.manifest:
            try:
                app.state.predictor = WindowPredictor(settings.checkpoint, settings.manifest, settings.device)
            except (OSError, ValueError, RuntimeError, TypeError, KeyError, pickle.UnpicklingError) as exc:
                logging.getLogger(__name__).error("Checkpoint initialization failed (%s)", type(exc).__name__)
        yield
        app.state.predictor = None

    app = FastAPI(title="Active Speaker Detection — prepared-window API", version="1.0.0", lifespan=lifespan)
    app.add_middleware(BodyLimit, limit=settings.max_body_bytes)
    app.state.predictor = None

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        # Do not echo entire landmark payloads into validation errors.
        return JSONResponse({"detail": [{"loc": error["loc"], "msg": error["msg"], "type": error["type"]}
                                        for error in exc.errors()]}, status_code=422)

    def authorize(authorization):
        if settings.token and not hmac.compare_digest((authorization or "").encode(), ("Bearer " + settings.token).encode()):
            raise HTTPException(401, "A valid bearer token is required", headers={"WWW-Authenticate": "Bearer"})

    def predictor():
        if app.state.predictor is None:
            raise HTTPException(503, "Model unavailable: supply a valid checkpoint and manifest, then restart")
        return app.state.predictor

    @app.get("/health/live")
    async def live():
        return {"status": "alive"}

    @app.get("/health/ready")
    async def ready():
        model = predictor()
        return {"status": "ready", "device": str(model.device),
                "checkpoint_sha256": model.metadata["checkpoint_sha256"]}

    @app.get("/v1/model")
    async def model_metadata(authorization: str | None = Header(default=None)):
        authorize(authorization)
        return predictor().metadata

    @app.get("/metrics")
    async def metrics(authorization: str | None = Header(default=None)):
        authorize(authorization)
        with metrics_lock:
            return {**counters, "scope": "process-local prediction handler; excludes HTTP parsing/validation"}

    @app.post("/v1/predict", response_model=PredictionResponse)
    def predict(body: PredictionRequest, authorization: str | None = Header(default=None)):
        authorize(authorization)
        model = predictor()
        if not inference_lock.acquire(blocking=False):
            with metrics_lock:
                counters["busy"] += 1
            raise HTTPException(503, "Inference busy; retry later", headers={"Retry-After": "1"})
        started = time.perf_counter()
        try:
            result = model.predict(body.model_dump())
            with metrics_lock:
                counters["completed"] += 1
            return result
        except (ValueError, TypeError, OverflowError) as exc:
            with metrics_lock:
                counters["rejected_input"] += 1
            raise HTTPException(422, "Input is incompatible with the model manifest") from exc
        except RuntimeError as exc:
            with metrics_lock:
                counters["failed"] += 1
            raise HTTPException(500, "Model execution failed") from exc
        finally:
            with metrics_lock:
                counters["inference_seconds"] += time.perf_counter() - started
            inference_lock.release()

    return app
