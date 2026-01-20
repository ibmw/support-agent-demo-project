"""
CrewAI agent, task, and crew orchestration for the support agent.

This module sets up the Support Specialist agent with knowledge base tools
and provides methods to process customer queries with optional conversation memory.
"""

import hashlib
import time
from typing import Any
from uuid import UUID

from crewai import Agent, Crew, Process, Task
from pydantic import ValidationError

from ..config import settings
from ..exceptions import AgentError, InvalidResponseError
from ..logging import get_logger
from ..memory import get_session_manager
from ..rag.tools import search_knowledge_base
from .prompts import (
    SUPPORT_SPECIALIST_BACKSTORY,
    SUPPORT_SPECIALIST_GOAL,
    SUPPORT_SPECIALIST_ROLE,
    SUPPORT_TASK_DESCRIPTION,
    SUPPORT_TASK_DESCRIPTION_WITH_HISTORY,
    SUPPORT_TASK_EXPECTED_OUTPUT,
)
from .schemas import AgentResponse

logger = get_logger(__name__, component="agent")


class SupportCrew:
    """
    CrewAI-based support agent crew.

    Orchestrates the support specialist agent to process customer queries
    using the knowledge base retrieval tools.
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.7,
        verbose: bool = False,
        max_execution_time: int = 120,
    ):
        """
        Initialize the support crew.

        Args:
            model: LLM model to use (defaults to settings.llm_model)
            temperature: LLM temperature for response generation
            verbose: Enable verbose output from CrewAI
            max_execution_time: Maximum seconds for task execution (default 120)
        """
        self.model = model or settings.llm_model
        self.temperature = temperature
        self.verbose = verbose
        self.max_execution_time = max_execution_time

        # Create the support specialist agent
        self._agent = self._create_agent()

        logger.info(
            "SupportCrew initialized",
            model=self.model,
            temperature=self.temperature,
            verbose=self.verbose,
            max_execution_time=self.max_execution_time,
        )

    def _create_agent(self) -> Agent:
        """Create the support specialist agent."""
        return Agent(
            role=SUPPORT_SPECIALIST_ROLE,
            goal=SUPPORT_SPECIALIST_GOAL.strip(),
            backstory=SUPPORT_SPECIALIST_BACKSTORY.strip(),
            tools=[search_knowledge_base],
            llm=self.model,
            verbose=self.verbose,
            allow_delegation=False,  # Single agent, no delegation needed
        )

    def _create_task(self, query: str, conversation_history: str | None = None) -> Task:
        """
        Create a task for processing a customer query.

        Args:
            query: The customer's query
            conversation_history: Optional formatted conversation history for context

        Returns:
            Configured Task instance
        """
        if conversation_history:
            description = SUPPORT_TASK_DESCRIPTION_WITH_HISTORY.format(
                query=query,
                conversation_history=conversation_history,
            )
        else:
            description = SUPPORT_TASK_DESCRIPTION.format(query=query)

        return Task(
            description=description,
            expected_output=SUPPORT_TASK_EXPECTED_OUTPUT.strip(),
            agent=self._agent,
            output_pydantic=AgentResponse,
            max_execution_time=self.max_execution_time,
        )

    def _create_crew(self, task: Task) -> Crew:
        """
        Create a crew to execute the task.

        Args:
            task: The task to execute

        Returns:
            Configured Crew instance
        """
        return Crew(
            agents=[self._agent],
            tasks=[task],
            process=Process.sequential,
            verbose=self.verbose,
        )

    def _extract_token_usage(self, result: Any) -> dict[str, int]:
        """
        Extract token usage from crew result.

        Args:
            result: The CrewOutput from kickoff()

        Returns:
            Dict with token usage stats, empty if not available
        """
        token_usage = getattr(result, "token_usage", None)
        if not token_usage:
            return {}

        return {
            "total_tokens": getattr(token_usage, "total_tokens", 0),
            "prompt_tokens": getattr(token_usage, "prompt_tokens", 0),
            "completion_tokens": getattr(token_usage, "completion_tokens", 0),
            "successful_requests": getattr(token_usage, "successful_requests", 0),
        }

    def process_query(
        self,
        query: str,
        session_id: UUID | None = None,
        store_in_session: bool = True,
    ) -> AgentResponse:
        """
        Process a customer support query with optional conversation memory.

        Args:
            query: The customer's query
            session_id: Optional session ID for multi-turn conversations.
                        If provided, conversation history will be included in context.
            store_in_session: Whether to store the query and response in the session.
                              Only applies if session_id is provided. Defaults to True.

        Returns:
            Structured AgentResponse

        Raises:
            AgentError: If the agent fails to process the query
            InvalidResponseError: If the response cannot be parsed
            SessionNotFoundError: If session_id is provided but session doesn't exist
        """
        start_time = time.time()
        # Create query hash for log correlation across services
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        log = logger.bind(
            query_preview=query[:100],
            query_hash=query_hash,
            query_length=len(query),
            session_id=str(session_id) if session_id else None,
        )

        log.info("Processing support query")

        # Get conversation history if session exists
        conversation_history: str | None = None
        session = None
        if session_id:
            manager = get_session_manager()
            session = manager.get_session(session_id)
            conversation_history = manager.get_context_window(session_id)

            # Store the user query in session
            if store_in_session:
                session.add_user_message(query)

            log.debug(
                "Using conversation context",
                turn_count=session.turn_count,
                context_length=len(conversation_history) if conversation_history else 0,
            )

        try:
            # Create task and crew (with conversation history if available)
            task = self._create_task(query, conversation_history)
            crew = self._create_crew(task)

            # Execute the crew
            result = crew.kickoff()

            # Get the pydantic output directly from CrewAI
            if hasattr(result, "pydantic") and result.pydantic is not None:
                response = result.pydantic
            else:
                # Fallback: try to get from tasks_output
                task_output = result.tasks_output[0] if result.tasks_output else None
                if (
                    task_output
                    and hasattr(task_output, "pydantic")
                    and task_output.pydantic
                ):
                    response = task_output.pydantic
                else:
                    raise InvalidResponseError(
                        "CrewAI did not return a valid pydantic response"
                    )

            # Validate response type
            if not isinstance(response, AgentResponse):
                raise InvalidResponseError(
                    f"Expected AgentResponse, got {type(response).__name__}"
                )

            # Store response in session if applicable
            if session and store_in_session:
                session.add_agent_response(response)
                log.debug(
                    "Stored response in session",
                    session_id=str(session_id),
                    turn_count=session.turn_count,
                )

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            # Extract token usage
            token_usage = self._extract_token_usage(result)

            log.info(
                "Query processed successfully",
                action=response.action,
                confidence=response.confidence,
                sources_count=len(response.sources),
                latency_ms=round(latency_ms, 2),
                **token_usage,
            )

            return response

        except ValidationError as e:
            latency_ms = (time.time() - start_time) * 1000
            log.error(
                "Agent response failed validation",
                error=str(e),
                latency_ms=round(latency_ms, 2),
            )
            raise InvalidResponseError(f"Invalid agent response format: {e}") from e

        except (InvalidResponseError, AgentError):
            # Re-raise known errors
            raise

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            log.error(
                "Agent execution failed",
                error=str(e),
                error_type=type(e).__name__,
                latency_ms=round(latency_ms, 2),
            )
            raise AgentError(f"Failed to process query: {e}") from e

    def process_conversation(
        self,
        query: str,
        session_id: UUID | None = None,
        metadata: dict[str, str] | None = None,
    ) -> tuple[AgentResponse, UUID]:
        """
        Process a query within a conversation session.

        This is a convenience method that handles session creation/retrieval
        and returns both the response and session ID for continued conversations.

        Args:
            query: The customer's query
            session_id: Optional existing session ID. If None, creates a new session.
            metadata: Optional metadata for new session (ignored if session exists)

        Returns:
            Tuple of (AgentResponse, session_id) for use in subsequent calls

        Example:
            # First turn - creates session
            response, session_id = crew.process_conversation("How do I reset my password?")

            # Follow-up turn - continues conversation
            response, _ = crew.process_conversation("What if that doesn't work?", session_id)
        """
        manager = get_session_manager()
        session = manager.get_or_create_session(
            session_id=session_id, metadata=metadata
        )

        response = self.process_query(
            query=query,
            session_id=session.id,
            store_in_session=True,
        )

        return response, session.id


# Module-level singleton for convenience
_crew: SupportCrew | None = None


def get_support_crew(**kwargs: Any) -> SupportCrew:
    """
    Get or create the singleton SupportCrew instance.

    Args:
        **kwargs: Arguments passed to SupportCrew constructor (only used on first call)

    Returns:
        SupportCrew instance
    """
    global _crew
    if _crew is None:
        _crew = SupportCrew(**kwargs)
    return _crew


def reset_crew() -> None:
    """Reset the singleton crew (useful for testing)."""
    global _crew
    _crew = None
