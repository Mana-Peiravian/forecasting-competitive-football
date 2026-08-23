"""FastAPI application for frozen pre-match and in-play inference."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Path, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from api import __version__
from api.config import APISettings
from api.errors import (
    InferenceUnavailableError,
    InvalidRequestError,
    MatchNotFoundError,
    SnapshotNotFoundError,
)
from api.schemas import (
    CombinedPrematchResponse,
    ErrorResponse,
    HealthResponse,
    InPlayRequest,
    InPlayResponse,
    MatchDetail,
    MatchesResponse,
    ModelsResponse,
    PrematchMarginResponse,
    PrematchOutcomeResponse,
    PrematchRequest,
    TimelineResponse,
)


settings = APISettings.from_environment()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Deferred imports keep static OpenAPI generation independent of ML wheels.
    from api.model_registry import ModelRegistry

    registry = ModelRegistry(settings)
    registry.load()
    app.state.registry = registry
    yield
    registry.models.clear()


app = FastAPI(
    title="Forecasting Competitive Football API",
    summary="Frozen-artifact inference for leakage-safe football forecasts",
    description=(
        "Versioned access to the finished project's Model 1 pre-match outcome probabilities, "
        "Model 2 signed goal margin, and Model 3 five-minute historical in-play snapshots. "
        "The service loads frozen artifacts once and never trains on startup. GitHub Pages hosts "
        "the static documentation only; this Python API must run separately."
    ),
    version=__version__,
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    contact={
        "name": "Project repository",
        "url": "https://github.com/Mana-Peiravian/forecasting-competitive-football",
    },
    servers=[
        {
            "url": settings.public_api_base_url,
            "description": "Configured API server (local by default)",
        }
    ],
    openapi_tags=[
        {"name": "service", "description": "Health and frozen-model metadata."},
        {"name": "matches", "description": "Bounded lookup over the public frozen match store."},
        {"name": "predictions", "description": "Model 1, Model 2, and Model 3 inference."},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.allowed_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Accept", "Content-Type"],
)


def get_registry(request: Request) -> Any:
    return request.app.state.registry


Registry = Annotated[Any, Depends(get_registry)]


def _error(status: int, code: str, message: str, details: dict[str, object] | None = None):
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message, "details": details}},
    )


@app.exception_handler(MatchNotFoundError)
async def match_not_found_handler(request: Request, error: MatchNotFoundError):
    return _error(404, "match_not_found", str(error), {"match_id": error.match_id})


@app.exception_handler(InvalidRequestError)
async def invalid_request_handler(request: Request, error: InvalidRequestError):
    return _error(400, error.code, str(error), error.details)


@app.exception_handler(SnapshotNotFoundError)
async def snapshot_not_found_handler(request: Request, error: SnapshotNotFoundError):
    return _error(
        404,
        "snapshot_not_found",
        str(error),
        {"match_id": error.match_id, "minute": error.minute},
    )


@app.exception_handler(InferenceUnavailableError)
async def unavailable_handler(request: Request, error: InferenceUnavailableError):
    return _error(503, "inference_unavailable", str(error))


@app.exception_handler(Exception)
async def safe_exception_handler(request: Request, error: Exception):
    return _error(500, "internal_error", "The service could not complete the request.")


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get(
    "/api/v1/health",
    tags=["service"],
    summary="Check service and artifact readiness",
    response_model=HealthResponse,
)
def health(registry: Registry) -> dict[str, object]:
    return {
        "status": "ok" if registry.ready else "degraded",
        "service": "forecasting-competitive-football",
        "api_version": "v1",
        "models_loaded": registry.ready,
    }


@app.get(
    "/api/v1/models",
    tags=["service"],
    summary="List the public frozen model configurations",
    response_model=ModelsResponse,
)
def models(registry: Registry) -> dict[str, object]:
    return {"class_order": ["away_win", "draw", "home_win"], "models": registry.model_metadata()}


@app.get(
    "/api/v1/matches",
    tags=["matches"],
    summary="Search the bounded public match catalogue",
    response_model=MatchesResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def matches(
    registry: Registry,
    competition: str | None = Query(default=None, description="`La Liga` in this release."),
    season: str | None = Query(default=None, examples=["2016/2017"]),
    team: str | None = Query(default=None, min_length=1, max_length=80),
    date_from: date | None = Query(default=None),
    date_to: date | None = Query(default=None),
    split: Literal["train", "validation", "calibration", "pilot_holdout", "final_test"]
    | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> dict[str, object]:
    return registry.list_matches(
        competition=competition,
        season=season,
        team=team,
        date_from=date_from,
        date_to=date_to,
        split=split,
        limit=limit,
        offset=offset,
    )


@app.get(
    "/api/v1/matches/{match_id}",
    tags=["matches"],
    summary="Get safe metadata for one match",
    response_model=MatchDetail,
    responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def match(
    match_id: Annotated[int, Path(gt=0, description="Known StatsBomb match identifier")],
    registry: Registry,
) -> dict[str, object]:
    return registry.match_detail(match_id)


@app.post(
    "/api/v1/predict/prematch/outcome",
    tags=["predictions"],
    summary="Run Model 1 pre-match outcome inference",
    response_model=PrematchOutcomeResponse,
    responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def predict_prematch_outcome(request: PrematchRequest, registry: Registry) -> dict[str, object]:
    """Return calibrated P(Home), P(Draw), and P(Away) from frozen kickoff data."""

    return registry.predict_prematch_outcome(request.match_id)


@app.post(
    "/api/v1/predict/prematch/margin",
    tags=["predictions"],
    summary="Run Model 2 signed goal-margin inference",
    response_model=PrematchMarginResponse,
    responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def predict_prematch_margin(request: PrematchRequest, registry: Registry) -> dict[str, object]:
    """Return expected clipped home-minus-away final goal margin."""

    return registry.predict_prematch_margin(request.match_id)


@app.post(
    "/api/v1/predict/prematch",
    tags=["predictions"],
    summary="Run the combined Model 1 and Model 2 forecast",
    response_model=CombinedPrematchResponse,
    responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def predict_prematch(request: PrematchRequest, registry: Registry) -> dict[str, object]:
    return registry.predict_prematch(request.match_id)


@app.post(
    "/api/v1/predict/inplay",
    tags=["predictions"],
    summary="Run Model 3 at a frozen five-minute snapshot",
    response_model=InPlayResponse,
    responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def predict_inplay(request: InPlayRequest, registry: Registry) -> dict[str, object]:
    """Use only the saved features whose events are at or before the snapshot boundary."""

    return registry.predict_inplay(request.match_id, request.minute)


@app.get(
    "/api/v1/matches/{match_id}/timeline",
    tags=["predictions"],
    summary="Replay all 19 Model 3 snapshots for one historical match",
    response_model=TimelineResponse,
    responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def timeline(
    match_id: Annotated[int, Path(gt=0, description="Known StatsBomb match identifier")],
    registry: Registry,
) -> dict[str, object]:
    return registry.timeline(match_id)
