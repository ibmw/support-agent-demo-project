"""
CrewAI agent, task, and crew orchestration for the support agent.

This module sets up the Support Specialist agent with knowledge base tools
and provides methods to process customer queries.
"""

import hashlib
import time
from typing import Any

from crewai import Agent, Crew, Process, Task
from pydantic import ValidationError

from ..config import settings
from ..exceptions import AgentError, InvalidResponseError
from ..logging import get_logger
from ..rag.tools import search_knowledge_base
from .prompts import (
    SUPPORT_SPECIALIST_BACKSTORY,
    SUPPORT_SPECIALIST_GOAL,
    SUPPORT_SPECIALIST_ROLE,
    SUPPORT_TASK_DESCRIPTION,
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

    def _create_task(self, query: str) -> Task:
        """
        Create a task for processing a customer query.

        Args:
            query: The customer's query

        Returns:
            Configured Task instance
        """
        return Task(
            description=SUPPORT_TASK_DESCRIPTION.format(query=query),
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

    def process_query(self, query: str) -> AgentResponse:
        """
        Process a customer support query.

        Args:
            query: The customer's query

        Returns:
            Structured AgentResponse

        Raises:
            AgentError: If the agent fails to process the query
            InvalidResponseError: If the response cannot be parsed

        Note:
            Conversation memory will be added in Phase 3.2 (Memory Module).
        """
        start_time = time.time()
        # Create query hash for log correlation across services
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        log = logger.bind(
            query_preview=query[:100],
            query_hash=query_hash,
            query_length=len(query),
        )

        log.info("Processing support query")

        try:
            # Create task and crew
            task = self._create_task(query)
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
