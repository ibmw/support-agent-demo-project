"""
API route definitions.

Organizes endpoints into logical routers for health, chat, and sessions.
"""

from uuid import UUID

from fastapi import APIRouter, HTTPException, status

from ..agent.crew import get_support_crew
from ..exceptions import AgentError, InvalidResponseError, SessionNotFoundError
from ..logging import get_logger
from ..memory import get_session_manager
from .schemas import (
    ChatRequest,
    ChatResponse,
    CreateSessionRequest,
    ErrorResponse,
    HealthResponse,
    SendMessageRequest,
    SendMessageResponse,
    SessionHistoryResponse,
    SessionResponse,
    TurnResponse,
)

logger = get_logger(__name__, component="api")


# ============================================================================
# Health Router
# ============================================================================

health_router = APIRouter(tags=["health"])


@health_router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check if the API is running and healthy.",
)
async def health_check() -> HealthResponse:
    """Return health status."""
    return HealthResponse()


# ============================================================================
# Chat Router (Single-turn, stateless)
# ============================================================================

chat_router = APIRouter(prefix="/chat", tags=["chat"])


@chat_router.post(
    "",
    response_model=ChatResponse,
    responses={
        500: {"model": ErrorResponse, "description": "Agent processing error"},
    },
    summary="Single-turn chat",
    description="Process a single query without conversation memory. Stateless.",
)
async def chat(request: ChatRequest) -> ChatResponse:
    """Process a single-turn chat query."""
    logger.info("Processing single-turn chat", message_length=len(request.message))

    try:
        crew = get_support_crew()
        response = crew.process_query(query=request.message, session_id=None)

        return ChatResponse(
            action=response.action,
            message=response.message,
            sources=response.sources,
            confidence=response.confidence,
        )

    except (AgentError, InvalidResponseError) as e:
        logger.error(
            "Chat processing failed", error=str(e), error_type=type(e).__name__
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


# ============================================================================
# Sessions Router (Multi-turn)
# ============================================================================

sessions_router = APIRouter(prefix="/sessions", tags=["sessions"])


@sessions_router.post(
    "",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create session",
    description="Create a new conversation session for multi-turn interactions.",
)
async def create_session(
    request: CreateSessionRequest | None = None,
) -> SessionResponse:
    """Create a new conversation session."""
    metadata = request.metadata if request else {}
    manager = get_session_manager()
    session = manager.create_session(metadata=metadata)

    logger.info("Session created via API", session_id=str(session.id))

    return SessionResponse(
        id=session.id,
        turn_count=session.turn_count,
        created_at=session.created_at,
        updated_at=session.updated_at,
        last_action=session.last_action,
        is_resolved=session.is_resolved(),
    )


@sessions_router.get(
    "/{session_id}",
    response_model=SessionHistoryResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Session not found"},
    },
    summary="Get session history",
    description="Retrieve full conversation history for a session.",
)
async def get_session(session_id: UUID) -> SessionHistoryResponse:
    """Get session with full conversation history."""
    manager = get_session_manager()

    try:
        session = manager.get_session(session_id)
    except SessionNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        ) from e

    turns = [
        TurnResponse(
            role=turn.role,
            content=turn.content,
            timestamp=turn.timestamp,
            action=turn.action,
            sources=turn.sources,
        )
        for turn in session.turns
    ]

    return SessionHistoryResponse(
        id=session.id,
        turns=turns,
        created_at=session.created_at,
        updated_at=session.updated_at,
        metadata=session.metadata,
    )


@sessions_router.post(
    "/{session_id}/messages",
    response_model=SendMessageResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Session not found"},
        500: {"model": ErrorResponse, "description": "Agent processing error"},
    },
    summary="Send message",
    description="Send a message in an existing session (multi-turn conversation).",
)
async def send_message(
    session_id: UUID, request: SendMessageRequest
) -> SendMessageResponse:
    """Process a message within a conversation session."""
    manager = get_session_manager()

    # Verify session exists
    try:
        manager.get_session(session_id)
    except SessionNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        ) from e

    logger.info(
        "Processing message in session",
        session_id=str(session_id),
        message_length=len(request.message),
    )

    try:
        crew = get_support_crew()
        response = crew.process_query(
            query=request.message,
            session_id=session_id,
            store_in_session=True,
        )

        # Get updated session for turn count
        session = manager.get_session(session_id)

        return SendMessageResponse(
            session_id=session_id,
            action=response.action,
            message=response.message,
            sources=response.sources,
            confidence=response.confidence,
            turn_count=session.turn_count,
        )

    except (AgentError, InvalidResponseError) as e:
        logger.error(
            "Message processing failed",
            session_id=str(session_id),
            error=str(e),
            error_type=type(e).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


@sessions_router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        404: {"model": ErrorResponse, "description": "Session not found"},
    },
    summary="Delete session",
    description="End and delete a conversation session.",
)
async def delete_session(session_id: UUID) -> None:
    """Delete a conversation session."""
    manager = get_session_manager()
    deleted = manager.delete_session(session_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    logger.info("Session deleted via API", session_id=str(session_id))
