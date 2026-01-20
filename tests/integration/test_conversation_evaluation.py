"""Integration tests for conversation evaluation runner."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from support_agent.agent.crew import SupportCrew, reset_crew
from support_agent.agent.schemas import AgentResponse
from support_agent.evaluation import (
    ConversationBatchConfig,
    ConversationEvaluationRunner,
    ConversationResult,
    ConversationScenario,
    evaluate_conversation,
    export_conversation_results,
    export_conversation_results_csv,
    export_conversation_results_json,
    filter_scenarios,
    load_scenarios,
    run_conversation_evaluation,
)
from support_agent.memory import reset_session_manager


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singletons before each test."""
    reset_crew()
    reset_session_manager()
    yield
    reset_crew()
    reset_session_manager()


@pytest.fixture
def mock_crew_result():
    """Factory fixture for creating mock crew results."""

    def _create_result(
        action: str = "CLOSE",
        message: str = "Here's the answer.",
        sources: list[str] | None = None,
        confidence: float = 0.9,
    ) -> MagicMock:
        response = AgentResponse(
            action=action,
            message=message,
            sources=sources or ["Source"],
            confidence=confidence,
        )
        mock_result = MagicMock()
        mock_result.pydantic = response
        mock_result.tasks_output = []
        mock_result.token_usage = None
        return mock_result

    return _create_result


@pytest.fixture
def mock_crewai(mock_crew_result):
    """Mock CrewAI components."""
    with (
        patch("support_agent.agent.crew.Agent") as mock_agent,
        patch("support_agent.agent.crew.Crew") as mock_crew_cls,
        patch("support_agent.agent.crew.Task") as mock_task,
    ):
        mock_crew_cls.return_value.kickoff.return_value = mock_crew_result()
        yield {
            "agent": mock_agent,
            "crew_cls": mock_crew_cls,
            "task": mock_task,
            "set_result": lambda r: setattr(
                mock_crew_cls.return_value.kickoff, "return_value", r
            ),
        }


@pytest.fixture
def sample_scenarios_json(tmp_path) -> Path:
    """Create a temporary scenarios JSON file."""
    scenarios = {
        "scenarios": [
            {
                "id": "simple-question",
                "name": "Simple Question",
                "description": "Single turn question",
                "category": "general",
                "user_messages": ["How do I reset my password?"],
                "expected_final_action": "CLOSE",
            },
            {
                "id": "multi-turn",
                "name": "Multi-turn Conversation",
                "description": "Requires clarification",
                "category": "support",
                "user_messages": ["I need help", "With my account"],
                "expected_final_action": "CLOSE",
            },
            {
                "id": "handover-request",
                "name": "Handover Request",
                "description": "Should escalate to human",
                "category": "billing",
                "user_messages": ["I want a refund"],
                "expected_final_action": "HANDOVER",
            },
        ]
    }
    json_path = tmp_path / "scenarios.json"
    with open(json_path, "w") as f:
        json.dump(scenarios, f)
    return json_path


@pytest.fixture
def sample_scenarios() -> list[ConversationScenario]:
    """Sample scenarios for testing."""
    return [
        ConversationScenario(
            id="simple-question",
            name="Simple Question",
            category="general",
            user_messages=["How do I reset my password?"],
            expected_final_action="CLOSE",
        ),
        ConversationScenario(
            id="multi-turn",
            name="Multi-turn Conversation",
            category="support",
            user_messages=["I need help", "With my account"],
            expected_final_action="CLOSE",
        ),
        ConversationScenario(
            id="handover-request",
            name="Handover Request",
            category="billing",
            user_messages=["I want a refund"],
            expected_final_action="HANDOVER",
        ),
    ]


@pytest.fixture
def sample_results() -> list[ConversationResult]:
    """Sample conversation results for export tests."""
    from support_agent.evaluation import TurnResult

    return [
        ConversationResult(
            scenario_id="simple-question",
            scenario_name="Simple Question",
            turns=[
                TurnResult(
                    turn_index=0,
                    user_message="How do I reset my password?",
                    response=AgentResponse(
                        action="CLOSE",
                        message="Go to Settings",
                        sources=["Password Guide"],
                    ),
                    latency_ms=150.0,
                ),
            ],
            total_latency_ms=150.0,
        ),
        ConversationResult(
            scenario_id="multi-turn",
            scenario_name="Multi-turn Conversation",
            turns=[
                TurnResult(
                    turn_index=0,
                    user_message="I need help",
                    response=AgentResponse(
                        action="WAIT",
                        message="What do you need help with?",
                        sources=[],
                    ),
                    latency_ms=100.0,
                ),
                TurnResult(
                    turn_index=1,
                    user_message="With my account",
                    response=AgentResponse(
                        action="CLOSE",
                        message="Here's how to manage your account",
                        sources=["Account Guide"],
                    ),
                    latency_ms=200.0,
                ),
            ],
            total_latency_ms=300.0,
        ),
    ]


# ============================================================================
# load_scenarios Tests
# ============================================================================


class TestLoadScenarios:
    """Tests for load_scenarios function."""

    def test_loads_scenarios_from_json(self, sample_scenarios_json):
        """Loads scenarios from JSON file."""
        scenarios = load_scenarios(sample_scenarios_json)

        assert len(scenarios) == 3
        assert scenarios[0].id == "simple-question"
        assert scenarios[0].name == "Simple Question"
        assert scenarios[0].category == "general"

    def test_handles_list_format(self, tmp_path):
        """Handles JSON files with list format (no wrapper object)."""
        scenarios_data = [
            {
                "id": "test",
                "name": "Test",
                "user_messages": ["Hello"],
            }
        ]
        json_path = tmp_path / "scenarios.json"
        with open(json_path, "w") as f:
            json.dump(scenarios_data, f)

        scenarios = load_scenarios(json_path)

        assert len(scenarios) == 1
        assert scenarios[0].id == "test"


# ============================================================================
# filter_scenarios Tests
# ============================================================================


class TestFilterScenarios:
    """Tests for filter_scenarios function."""

    def test_returns_all_when_no_filters(self, sample_scenarios):
        """Returns all scenarios when no filters applied."""
        filtered = filter_scenarios(sample_scenarios)
        assert len(filtered) == 3

    def test_filters_by_scenario_ids(self, sample_scenarios):
        """Filters by specific scenario IDs."""
        filtered = filter_scenarios(
            sample_scenarios, scenario_ids=["simple-question", "handover-request"]
        )

        assert len(filtered) == 2
        assert filtered[0].id == "simple-question"
        assert filtered[1].id == "handover-request"

    def test_filters_by_categories(self, sample_scenarios):
        """Filters by categories."""
        filtered = filter_scenarios(sample_scenarios, categories=["general", "billing"])

        assert len(filtered) == 2
        ids = [s.id for s in filtered]
        assert "simple-question" in ids
        assert "handover-request" in ids

    def test_combines_filters(self, sample_scenarios):
        """Combines ID and category filters."""
        filtered = filter_scenarios(
            sample_scenarios,
            scenario_ids=["simple-question", "multi-turn"],
            categories=["general"],
        )

        assert len(filtered) == 1
        assert filtered[0].id == "simple-question"


# ============================================================================
# evaluate_conversation Tests
# ============================================================================


class TestEvaluateConversation:
    """Tests for evaluate_conversation function."""

    def test_evaluates_single_turn_scenario(self, mock_crewai, mock_crew_result):
        """Evaluates single turn scenario correctly."""
        scenario = ConversationScenario(
            id="test",
            name="Test",
            user_messages=["How do I reset my password?"],
        )

        crew = SupportCrew()
        result = evaluate_conversation(scenario, crew)

        assert result.scenario_id == "test"
        assert result.success is True
        assert result.turn_count == 1
        assert result.final_action == "CLOSE"

    def test_evaluates_multi_turn_scenario(self, mock_crewai, mock_crew_result):
        """Evaluates multi-turn scenario correctly."""
        # First call returns WAIT, second returns CLOSE
        results = [
            mock_crew_result(action="WAIT", message="Which integration?"),
            mock_crew_result(action="CLOSE", message="Here's the guide"),
        ]
        call_count = [0]

        def side_effect():
            result = results[call_count[0]]
            call_count[0] += 1
            return result

        mock_crewai["crew_cls"].return_value.kickoff.side_effect = side_effect

        scenario = ConversationScenario(
            id="multi",
            name="Multi-turn",
            user_messages=["I need help", "With Shopify integration"],
        )

        crew = SupportCrew()
        result = evaluate_conversation(scenario, crew)

        assert result.turn_count == 2
        assert result.turns[0].action == "WAIT"
        assert result.turns[1].action == "CLOSE"
        assert result.final_action == "CLOSE"

    def test_handles_errors(self, mock_crewai):
        """Handles errors during evaluation."""
        mock_crewai["crew_cls"].return_value.kickoff.side_effect = Exception(
            "LLM failed"
        )

        scenario = ConversationScenario(
            id="error",
            name="Error Scenario",
            user_messages=["Test"],
        )

        crew = SupportCrew()
        result = evaluate_conversation(scenario, crew)

        assert result.success is False
        assert result.turns[0].error is not None
        assert "LLM failed" in result.turns[0].error

    def test_stops_early_on_close(self, mock_crewai, mock_crew_result):
        """Stops evaluation early when CLOSE is received."""
        mock_crewai["crew_cls"].return_value.kickoff.return_value = mock_crew_result(
            action="CLOSE"
        )

        scenario = ConversationScenario(
            id="early-close",
            name="Early Close",
            user_messages=["Q1", "Q2", "Q3"],  # 3 messages
        )

        crew = SupportCrew()
        result = evaluate_conversation(scenario, crew)

        # Should stop after first turn since it returned CLOSE
        assert result.turn_count == 1
        assert result.final_action == "CLOSE"

    def test_tracks_latency(self, mock_crewai, mock_crew_result):
        """Tracks latency for each turn."""
        scenario = ConversationScenario(
            id="latency",
            name="Latency Test",
            user_messages=["Q1"],
        )

        crew = SupportCrew()
        result = evaluate_conversation(scenario, crew)

        assert result.turns[0].latency_ms > 0
        assert result.total_latency_ms > 0


# ============================================================================
# run_conversation_evaluation Tests
# ============================================================================


class TestRunConversationEvaluation:
    """Tests for run_conversation_evaluation function."""

    def test_evaluates_all_scenarios(self, mock_crewai, sample_scenarios):
        """Evaluates all provided scenarios."""
        results, summary = run_conversation_evaluation(sample_scenarios)

        assert len(results) == 3
        assert summary.total_scenarios == 3

    def test_respects_scenario_filter(self, mock_crewai, sample_scenarios):
        """Respects scenario ID filter."""
        config = ConversationBatchConfig(scenario_ids=["simple-question"])

        results, summary = run_conversation_evaluation(sample_scenarios, config=config)

        assert len(results) == 1
        assert results[0].scenario_id == "simple-question"

    def test_calls_progress_callback(self, mock_crewai, sample_scenarios):
        """Calls progress callback for each scenario."""
        progress_calls = []

        def callback(current, total, result):
            progress_calls.append((current, total, result.scenario_id))

        run_conversation_evaluation(sample_scenarios, progress_callback=callback)

        assert len(progress_calls) == 3
        assert progress_calls[0][0] == 1
        assert progress_calls[2][0] == 3

    def test_calculates_summary(self, mock_crewai, sample_scenarios):
        """Calculates correct summary statistics."""
        results, summary = run_conversation_evaluation(sample_scenarios)

        assert summary.successful >= 0
        assert summary.failed >= 0
        assert summary.duration_seconds >= 0
        assert summary.metrics.total_scenarios == 3

    def test_includes_eval_tag(self, mock_crewai, sample_scenarios):
        """Includes eval_tag in summary."""
        config = ConversationBatchConfig(eval_tag="test-experiment")
        results, summary = run_conversation_evaluation(sample_scenarios, config=config)

        assert summary.eval_tag == "test-experiment"


# ============================================================================
# Export Tests
# ============================================================================


class TestExportConversationResultsJson:
    """Tests for JSON export."""

    def test_exports_to_json(self, sample_results, tmp_path):
        """Exports results to JSON file."""
        output_path = tmp_path / "results.json"
        export_conversation_results_json(sample_results, output_path)

        assert output_path.exists()

        with open(output_path) as f:
            data = json.load(f)

        assert "results" in data
        assert len(data["results"]) == 2

    def test_includes_summary(self, sample_results, sample_scenarios, tmp_path):
        """Includes summary when provided."""
        from datetime import UTC, datetime

        from support_agent.evaluation import calculate_conversation_summary

        start = datetime.now(UTC)
        end = datetime.now(UTC)
        summary = calculate_conversation_summary(
            sample_results, sample_scenarios, start, end, eval_tag="test"
        )

        output_path = tmp_path / "results.json"
        export_conversation_results_json(sample_results, output_path, summary=summary)

        with open(output_path) as f:
            data = json.load(f)

        assert "summary" in data
        assert data["summary"]["eval_tag"] == "test"

    def test_includes_golden_comparisons(self, sample_results, sample_scenarios, tmp_path):
        """Includes golden comparisons when scenarios provided."""
        output_path = tmp_path / "results.json"
        export_conversation_results_json(
            sample_results, output_path, scenarios=sample_scenarios
        )

        with open(output_path) as f:
            data = json.load(f)

        assert "golden_comparisons" in data

    def test_creates_parent_directories(self, sample_results, tmp_path):
        """Creates parent directories if needed."""
        output_path = tmp_path / "subdir" / "nested" / "results.json"
        export_conversation_results_json(sample_results, output_path)

        assert output_path.exists()


class TestExportConversationResultsCsv:
    """Tests for CSV export."""

    def test_exports_to_csv(self, sample_results, tmp_path):
        """Exports results to CSV file."""
        import csv

        output_path = tmp_path / "results.csv"
        export_conversation_results_csv(sample_results, output_path)

        assert output_path.exists()

        with open(output_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # 1 turn + 2 turns = 3 rows
        assert len(rows) == 3
        assert rows[0]["scenario_id"] == "simple-question"
        assert rows[1]["scenario_id"] == "multi-turn"
        assert rows[2]["scenario_id"] == "multi-turn"

    def test_csv_contains_expected_columns(self, sample_results, tmp_path):
        """CSV contains expected columns."""
        import csv

        output_path = tmp_path / "results.csv"
        export_conversation_results_csv(sample_results, output_path)

        with open(output_path) as f:
            reader = csv.reader(f)
            header = next(reader)

        expected_columns = [
            "scenario_id",
            "scenario_name",
            "turn_index",
            "user_message",
            "action",
            "message",
            "sources",
            "latency_ms",
            "success",
            "error",
            "final_action",
            "total_turns",
            "total_latency_ms",
            "timestamp",
        ]
        assert header == expected_columns


class TestExportConversationResults:
    """Tests for export_conversation_results dispatcher."""

    def test_exports_json_by_default(self, sample_results, tmp_path):
        """Exports JSON by default."""
        output_path = tmp_path / "results.json"
        export_conversation_results(sample_results, output_path)

        assert output_path.exists()
        with open(output_path) as f:
            data = json.load(f)
        assert "results" in data

    def test_exports_csv_when_specified(self, sample_results, tmp_path):
        """Exports CSV when format='csv'."""
        output_path = tmp_path / "results.csv"
        export_conversation_results(sample_results, output_path, format="csv")

        assert output_path.exists()
        with open(output_path) as f:
            assert "scenario_id" in f.readline()  # CSV header


# ============================================================================
# ConversationEvaluationRunner Tests
# ============================================================================


class TestConversationEvaluationRunner:
    """Tests for ConversationEvaluationRunner class."""

    def test_initializes_with_config(self, sample_scenarios_json):
        """Initializes with configuration."""
        runner = ConversationEvaluationRunner(
            scenarios_path=sample_scenarios_json,
            scenario_ids=["simple-question"],
            categories=["general"],
            eval_tag="test",
            output_format="json",
        )

        assert runner.config.scenario_ids == ["simple-question"]
        assert runner.config.categories == ["general"]
        assert runner.config.eval_tag == "test"
        assert runner.config.output_format == "json"

    def test_lazy_loads_scenarios(self, sample_scenarios_json):
        """Loads scenarios lazily."""
        runner = ConversationEvaluationRunner(scenarios_path=sample_scenarios_json)

        # Scenarios not loaded yet
        assert runner._scenarios is None

        # Access triggers load
        scenarios = runner.scenarios
        assert len(scenarios) == 3
        assert runner._scenarios is not None

    def test_run_executes_evaluation(self, mock_crewai, sample_scenarios_json):
        """Run executes conversation evaluation."""
        runner = ConversationEvaluationRunner(
            scenarios_path=sample_scenarios_json,
            scenario_ids=["simple-question"],
        )

        results, summary = runner.run()

        assert len(results) == 1
        assert summary.total_scenarios == 1

    def test_export_writes_file(self, mock_crewai, sample_scenarios_json, tmp_path):
        """Export writes results to file."""
        runner = ConversationEvaluationRunner(
            scenarios_path=sample_scenarios_json,
            scenario_ids=["simple-question"],
        )

        results, summary = runner.run()
        output_path = tmp_path / "results.json"
        runner.export(results, summary, output_path)

        assert output_path.exists()

    def test_format_report_returns_string(self, mock_crewai, sample_scenarios_json):
        """Format report returns readable string."""
        runner = ConversationEvaluationRunner(
            scenarios_path=sample_scenarios_json,
            scenario_ids=["simple-question"],
        )

        results, summary = runner.run()
        report = runner.format_report(summary)

        assert "CONVERSATION EVALUATION SUMMARY" in report
        assert "Total Scenarios:" in report
