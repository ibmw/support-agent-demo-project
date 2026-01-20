"""
FastAPI application setup.

Creates the app with middleware, routes, and error handlers.
Includes LangFuse integration for observability.
"""

import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..clients import shutdown_langfuse
from ..config import settings
from ..exceptions import AgentError, InvalidResponseError, SessionNotFoundError
from ..logging import get_logger
from .routes import chat_router, health_router, sessions_router
from .schemas import ErrorResponse

logger = get_logger(__name__, component="api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Runs on startup and shutdown.
    """
    # Startup
    logger.info(
        "API starting",
        host=settings.api_host,
        port=settings.api_port,
    )
    yield
    # Shutdown
    logger.info("API shutting down")
    # Flush any pending LangFuse events
    shutdown_langfuse()


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI app
    """
    app = FastAPI(
        title="Support Agent API",
        description="Customer support AI assistant API with RAG-powered responses",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ========================================================================
    # CORS Middleware
    # ========================================================================
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ========================================================================
    # Request Logging Middleware
    # ========================================================================
    @app.middleware("http")
    async def logging_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Log all requests with timing and request ID."""
        # Generate request ID
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id

        # Log request
        log = logger.bind(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client=request.client.host if request.client else "unknown",
        )
        log.info("Request started")

        # Process request
        start_time = time.time()
        response = await call_next(request)
        latency_ms = (time.time() - start_time) * 1000

        # Log response
        log.info(
            "Request completed",
            status_code=response.status_code,
            latency_ms=round(latency_ms, 2),
        )

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        return response

    # ========================================================================
    # Exception Handlers
    # ========================================================================
    @app.exception_handler(SessionNotFoundError)
    async def session_not_found_handler(
        request: Request, exc: SessionNotFoundError
    ) -> JSONResponse:
        """Handle session not found errors."""
        request_id = getattr(request.state, "request_id", None)
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(
                error="session_not_found",
                message=str(exc),
                request_id=request_id,
            ).model_dump(),
        )

    @app.exception_handler(AgentError)
    async def agent_error_handler(request: Request, exc: AgentError) -> JSONResponse:
        """Handle agent processing errors."""
        request_id = getattr(request.state, "request_id", None)
        logger.error(
            "Agent error",
            error=str(exc),
            request_id=request_id,
        )
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="agent_error",
                message=str(exc),
                request_id=request_id,
            ).model_dump(),
        )

    @app.exception_handler(InvalidResponseError)
    async def invalid_response_handler(
        request: Request, exc: InvalidResponseError
    ) -> JSONResponse:
        """Handle invalid response errors."""
        request_id = getattr(request.state, "request_id", None)
        logger.error(
            "Invalid response error",
            error=str(exc),
            request_id=request_id,
        )
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="invalid_response",
                message=str(exc),
                request_id=request_id,
            ).model_dump(),
        )

    # ========================================================================
    # Register Routers
    # ========================================================================
    app.include_router(health_router)
    app.include_router(chat_router)
    app.include_router(sessions_router)

    logger.info("FastAPI app created", routes=len(app.routes))

    return app


# Module-level app instance for uvicorn
app = create_app()
