"""
Tests for the OrchestratorAgent class.
"""

import pytest
from agents.orchestrator_agent import OrchestratorAgent, TaskPriority, TaskStatus


@pytest.fixture
def orchestrator():
    """Create an OrchestratorAgent instance for testing."""
    return OrchestratorAgent()


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    class MockAgent:
        def process(self, data):
            return {"processed": True, "data": data}

        def predict_water_quality(self, **kwargs):
            return {"quality_score": 85}

    return MockAgent()


class TestOrchestratorInit:
    """Tests for OrchestratorAgent initialization."""

    def test_init_creates_empty_state(self):
        """Test that initialization creates empty state."""
        orchestrator = OrchestratorAgent()
        assert len(orchestrator._tasks) == 0
        assert len(orchestrator._agents) == 0
        assert len(orchestrator._task_queue) == 0


class TestAgentRegistration:
    """Tests for agent registration."""

    def test_register_agent(self, orchestrator, mock_agent):
        """Test registering an agent."""
        result = orchestrator.register_agent(
            agent_id="model_1",
            agent_type="model",
            agent_instance=mock_agent,
            capabilities=["predict", "analyze"],
        )
        assert result["success"] is True
        assert result["agent_id"] == "model_1"

    def test_register_agent_without_id(self, orchestrator, mock_agent):
        """Test registering without ID returns error."""
        result = orchestrator.register_agent(
            agent_id="",
            agent_type="model",
            agent_instance=mock_agent,
        )
        assert result["success"] is False

    def test_register_duplicate_agent(self, orchestrator, mock_agent):
        """Test registering duplicate agent returns error."""
        orchestrator.register_agent("agent_1", "model", mock_agent)
        result = orchestrator.register_agent("agent_1", "model", mock_agent)
        assert result["success"] is False

    def test_unregister_agent(self, orchestrator, mock_agent):
        """Test unregistering an agent."""
        orchestrator.register_agent("agent_1", "model", mock_agent)
        result = orchestrator.unregister_agent("agent_1")
        assert result["success"] is True

    def test_unregister_nonexistent(self, orchestrator):
        """Test unregistering nonexistent agent returns error."""
        result = orchestrator.unregister_agent("nonexistent")
        assert result["success"] is False


class TestTaskCreation:
    """Tests for task creation."""

    def test_create_task(self, orchestrator):
        """Test creating a task."""
        result = orchestrator.create_task(
            task_type="predict",
            payload={"data": "test"},
        )
        assert result["success"] is True
        assert "task_id" in result
        assert result["status"] == "queued"

    def test_create_task_without_type(self, orchestrator):
        """Test creating task without type returns error."""
        result = orchestrator.create_task(
            task_type="",
            payload={},
        )
        assert result["success"] is False

    def test_create_task_with_priority(self, orchestrator):
        """Test creating task with priority."""
        result = orchestrator.create_task(
            task_type="critical_task",
            payload={},
            priority=TaskPriority.CRITICAL,
        )
        assert result["success"] is True

    def test_create_task_with_dependencies(self, orchestrator):
        """Test creating task with dependencies."""
        task1 = orchestrator.create_task("task1", {"step": 1})
        task2 = orchestrator.create_task(
            "task2",
            {"step": 2},
            depends_on=[task1["task_id"]],
        )
        assert task2["success"] is True

    def test_get_task(self, orchestrator):
        """Test getting task details."""
        create_result = orchestrator.create_task("test_task", {"data": "value"})
        task_id = create_result["task_id"]

        get_result = orchestrator.get_task(task_id)
        assert get_result["success"] is True
        assert get_result["task"]["type"] == "test_task"

    def test_get_nonexistent_task(self, orchestrator):
        """Test getting nonexistent task returns error."""
        result = orchestrator.get_task("nonexistent")
        assert result["success"] is False


class TestTaskExecution:
    """Tests for task execution."""

    def test_execute_task(self, orchestrator, mock_agent):
        """Test executing a task."""
        # Register agent
        orchestrator.register_agent(
            "model_agent",
            "predict",
            mock_agent,
            ["predict"],
        )

        # Create and execute task
        create_result = orchestrator.create_task("predict", {"ph": 7.0})
        task_id = create_result["task_id"]

        exec_result = orchestrator.execute_task(task_id)
        assert exec_result["success"] is True
        assert exec_result["status"] == "completed"

    def test_execute_nonexistent_task(self, orchestrator):
        """Test executing nonexistent task returns error."""
        result = orchestrator.execute_task("nonexistent")
        assert result["success"] is False

    def test_execute_without_agent(self, orchestrator):
        """Test executing without suitable agent returns error."""
        create_result = orchestrator.create_task("special_task", {})
        result = orchestrator.execute_task(create_result["task_id"])
        assert result["success"] is False
        assert "No suitable agent" in result["error"]


class TestTaskQueue:
    """Tests for task queue management."""

    def test_process_queue(self, orchestrator, mock_agent):
        """Test processing task queue."""
        orchestrator.register_agent("agent", "predict", mock_agent, ["predict"])

        orchestrator.create_task("predict", {"test": 1})
        orchestrator.create_task("predict", {"test": 2})

        result = orchestrator.process_queue()
        assert result["success"] is True
        assert result["processed"] == 2

    def test_process_queue_with_limit(self, orchestrator, mock_agent):
        """Test processing queue with limit."""
        orchestrator.register_agent("agent", "predict", mock_agent, ["predict"])

        for i in range(5):
            orchestrator.create_task("predict", {"test": i})

        result = orchestrator.process_queue(max_tasks=2)
        assert result["processed"] == 2
        assert result["remaining_in_queue"] == 3

    def test_get_queue_status(self, orchestrator):
        """Test getting queue status."""
        orchestrator.create_task("task1", {})
        orchestrator.create_task("task2", {}, priority=TaskPriority.HIGH)

        result = orchestrator.get_queue_status()
        assert result["success"] is True
        assert result["queue_length"] == 2


class TestTaskHandoff:
    """Tests for task handoff functionality."""

    def test_handoff_task(self, orchestrator, mock_agent):
        """Test handing off task to specific agent."""
        orchestrator.register_agent("target_agent", "model", mock_agent)

        create_result = orchestrator.create_task("task", {})
        task_id = create_result["task_id"]

        handoff_result = orchestrator.handoff_task(
            task_id,
            "target_agent",
            additional_context={"priority": "high"},
        )
        assert handoff_result["success"] is True
        assert handoff_result["target_agent"] == "target_agent"

    def test_handoff_nonexistent_task(self, orchestrator, mock_agent):
        """Test handing off nonexistent task returns error."""
        orchestrator.register_agent("agent", "model", mock_agent)
        result = orchestrator.handoff_task("nonexistent", "agent")
        assert result["success"] is False

    def test_handoff_to_nonexistent_agent(self, orchestrator):
        """Test handing off to nonexistent agent returns error."""
        create_result = orchestrator.create_task("task", {})
        result = orchestrator.handoff_task(create_result["task_id"], "nonexistent")
        assert result["success"] is False


class TestAgentStatus:
    """Tests for agent status monitoring."""

    def test_get_agent_status(self, orchestrator, mock_agent):
        """Test getting specific agent status."""
        orchestrator.register_agent("agent_1", "model", mock_agent, ["predict"])

        result = orchestrator.get_agent_status("agent_1")
        assert result["success"] is True
        assert result["agent"]["status"] == "available"

    def test_get_all_agents_status(self, orchestrator, mock_agent):
        """Test getting all agents status."""
        orchestrator.register_agent("agent_1", "model", mock_agent)
        orchestrator.register_agent("agent_2", "image", mock_agent)

        result = orchestrator.get_agent_status()
        assert result["success"] is True
        assert len(result["agents"]) == 2

    def test_get_nonexistent_agent_status(self, orchestrator):
        """Test getting nonexistent agent status returns error."""
        result = orchestrator.get_agent_status("nonexistent")
        assert result["success"] is False


class TestTaskCancellation:
    """Tests for task cancellation."""

    def test_cancel_task(self, orchestrator):
        """Test cancelling a pending task."""
        create_result = orchestrator.create_task("task", {})
        task_id = create_result["task_id"]

        cancel_result = orchestrator.cancel_task(task_id)
        assert cancel_result["success"] is True

        # Verify task is cancelled
        task = orchestrator.get_task(task_id)
        assert task["task"]["status"] == "cancelled"

    def test_cancel_nonexistent_task(self, orchestrator):
        """Test cancelling nonexistent task returns error."""
        result = orchestrator.cancel_task("nonexistent")
        assert result["success"] is False


class TestWorkloadAnalysis:
    """Tests for workload analysis."""

    def test_analyze_workload(self, orchestrator, mock_agent):
        """Test workload analysis."""
        orchestrator.register_agent("agent_1", "model", mock_agent)
        orchestrator.create_task("task1", {})
        orchestrator.create_task("task2", {}, priority=TaskPriority.HIGH)

        result = orchestrator.analyze_workload()
        assert result["success"] is True
        assert "analysis" in result
        assert "recommendations" in result
        assert result["analysis"]["pending_tasks"] == 2

    def test_analyze_empty_workload(self, orchestrator):
        """Test workload analysis with no tasks or agents."""
        result = orchestrator.analyze_workload()
        assert result["success"] is True
        assert result["analysis"]["utilization_percentage"] == 0


class TestHandlerRegistration:
    """Tests for handler registration."""

    def test_register_handler(self, orchestrator):
        """Test registering a task handler."""

        def my_handler(payload, context):
            return {"handled": True}

        result = orchestrator.register_handler("custom_task", my_handler)
        assert result["success"] is True

    def test_register_handler_without_type(self, orchestrator):
        """Test registering handler without type returns error."""
        result = orchestrator.register_handler("", lambda x, y: None)
        assert result["success"] is False
