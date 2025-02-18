# -*- coding: utf-8 -*-
import gc
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict

from starlette.responses import Response
# Imports OpenTelemetry pour Jaeger
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi_async_sqlalchemy import SQLAlchemyMiddleware
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_limiter import FastAPILimiter
from fastapi_pagination import add_pagination
from jose import jwt
from langchain.cache import RedisCache
from langchain.globals import set_llm_cache
from pydantic import ValidationError
from starlette.middleware.cors import CORSMiddleware

from app.api.deps import get_redis_client, get_redis_client_sync
from app.api.v1.api import api_router as api_router_v1
from app.core.config import settings, yaml_configs
from app.core.fastapi import FastAPIWithInternalModels
from app.utils.config_loader import load_agent_config, load_ingestion_configs
from app.utils.fastapi_globals import GlobalsMiddleware, g
from prometheus_client import start_http_server, Counter, generate_latest, CONTENT_TYPE_LATEST, Histogram
import time

REQUEST_COUNT = Counter("http_request_total", "Total HTTP requests")

REQUEST_LATENCY = Histogram("http_request_latency", "Request latency")

ERROR_COUNT = Counter(
    "http_requests_errors_total", "Total error requests",
)


def process_request():
    REQUEST_COUNT.inc()
    time.sleep(2)


# Configuration du tracing Jaeger
trace.set_tracer_provider(TracerProvider())
otlp_exporter = OTLPSpanExporter(endpoint="jaeger:4317")
trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(otlp_exporter))


async def user_id_identifier(request: Request) -> str:
    """Identify the user from the request."""
    if request.scope["type"] == "http":
        # Retrieve the Authorization header from the request
        auth_header = request.headers.get("Authorization")

        if auth_header is not None:
            # Check that the header is in the correct format
            header_parts = auth_header.split()
            if len(header_parts) == 2 and header_parts[0].lower() == "bearer":
                token = header_parts[1]
                try:
                    payload = jwt.decode(
                        token,
                        settings.SECRET_KEY,
                        algorithms=["HS256"],
                    )
                except (
                        jwt.JWTError,
                        ValidationError,
                ):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Could not validate credentials",
                    )
                user_id = payload["sub"]
                print(
                    "here2",
                    user_id,
                )
                return user_id

    if request.scope["type"] == "websocket":
        return request.scope["path"]

    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0]

    ip = request.client.host if request.client else ""
    return ip + ":" + request.scope["path"]


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Start up and shutdown tasks."""
    # startup
    yaml_configs["agent_config"] = load_agent_config()
    yaml_configs["ingestion_config"] = load_ingestion_configs()

    redis_client = await get_redis_client()

    if settings.ENABLE_LLM_CACHE:
        set_llm_cache(RedisCache(redis_=get_redis_client_sync()))

    FastAPICache.init(
        RedisBackend(redis_client),
        prefix="fastapi-cache",
    )
    await FastAPILimiter.init(
        redis_client,
        identifier=user_id_identifier,
    )

    logging.info("Start up FastAPI [Full dev mode]")
    yield

    # shutdown
    await FastAPICache.clear()
    await FastAPILimiter.close()
    g.cleanup()
    gc.collect()
    yaml_configs.clear()


logging.basicConfig(level=logging.INFO)

# Core Application Instance
app = FastAPIWithInternalModels(
    title=settings.PROJECT_NAME,
    version=settings.API_VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    lifespan=lifespan,
)

# Instrumentation Jaeger pour FastAPI
FastAPIInstrumentor.instrument_app(app)

app.add_middleware(
    SQLAlchemyMiddleware,
    db_url=settings.ASYNC_DATABASE_URI,
    engine_args={
        "echo": False,
        "pool_pre_ping": True,
        "pool_size": settings.POOL_SIZE,
        "max_overflow": 64,
    },
)
app.add_middleware(GlobalsMiddleware)

# Set all CORS origins enabled
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

@app.middleware('http')
async def prometheus_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    method = request.method
    endpoint = request.url.path
    status_code = response.status_code

    # Mise à jour des métriques
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, http_status=status_code).inc()
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(process_time)

    if status_code >= 400:
        ERROR_COUNT.labels(method=method, endpoint=endpoint, http_status=status_code).inc()

    return response

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    exc_str = f"{exc}".replace("\n", " ").replace("   ", " ")
    logging.error(f"{request}: {exc_str}")
    content = {"status_code": 10422, "message": exc_str, "data": None}
    return JSONResponse(content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)


@app.get("/test-jaeger")
async def test_jaeger():
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("test-operation") as span:
        span.set_attribute("test.attribute", "test-value")
        return {"message": "Test trace generated"}


@app.get("/")
async def root() -> Dict[str, str]:
    """An example "Hello world" FastAPI route."""
    process_request()
    return {"message": "FastAPI backend"}


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Add Routers
app.include_router(
    api_router_v1,
    prefix=settings.API_V1_STR,
)
add_pagination(app)
