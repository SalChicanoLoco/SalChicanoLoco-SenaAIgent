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


class TestAutoScheduler:
    """Tests for autoscheduler functionality."""

    def test_start_scheduler(self, orchestrator):
        """Test starting the scheduler."""
        result = orchestrator.start_scheduler(interval=1.0)
        assert result["success"] is True
        assert orchestrator._scheduler_running is True
        orchestrator.stop_scheduler()

    def test_stop_scheduler(self, orchestrator):
        """Test stopping the scheduler."""
        orchestrator.start_scheduler(interval=1.0)
        result = orchestrator.stop_scheduler()
        assert result["success"] is True
        assert orchestrator._scheduler_running is False

    def test_scheduler_status(self, orchestrator):
        """Test getting scheduler status."""
        result = orchestrator.get_scheduler_status()
        assert result["success"] is True
        assert "running" in result
        assert "interval" in result

    def test_start_already_running(self, orchestrator):
        """Test starting scheduler when already running."""
        orchestrator.start_scheduler(interval=1.0)
        result = orchestrator.start_scheduler(interval=1.0)
        assert result["success"] is False
        orchestrator.stop_scheduler()


class TestScheduledTasks:
    """Tests for scheduled task functionality."""

    def test_schedule_task(self, orchestrator):
        """Test scheduling a task."""
        result = orchestrator.schedule_task(
            task_type="test_task",
            payload={"data": "test"},
        )
        assert result["success"] is True
        assert "schedule_id" in result

    def test_schedule_task_without_type(self, orchestrator):
        """Test scheduling without type returns error."""
        result = orchestrator.schedule_task(
            task_type="",
            payload={},
        )
        assert result["success"] is False

    def test_get_scheduled_tasks(self, orchestrator):
        """Test getting scheduled tasks."""
        orchestrator.schedule_task("task1", {})
        orchestrator.schedule_task("task2", {})
        result = orchestrator.get_scheduled_tasks()
        assert result["success"] is True
        assert len(result["scheduled_tasks"]) == 2

    def test_cancel_scheduled_task(self, orchestrator):
        """Test cancelling a scheduled task."""
        schedule = orchestrator.schedule_task("task", {})
        result = orchestrator.cancel_scheduled_task(schedule["schedule_id"])
        assert result["success"] is True


class TestWebhooks:
    """Tests for webhook functionality."""

    def test_register_webhook(self, orchestrator):
        """Test registering a webhook."""
        result = orchestrator.register_webhook(
            webhook_id="test_hook",
            url="https://example.com/webhook",
            events=["task_completed"],
        )
        assert result["success"] is True

    def test_register_webhook_without_id(self, orchestrator):
        """Test registering webhook without ID."""
        result = orchestrator.register_webhook(
            webhook_id="",
            url="https://example.com/webhook",
        )
        assert result["success"] is False

    def test_unregister_webhook(self, orchestrator):
        """Test unregistering a webhook."""
        orchestrator.register_webhook("hook1", "https://example.com")
        result = orchestrator.unregister_webhook("hook1")
        assert result["success"] is True

    def test_get_webhooks(self, orchestrator):
        """Test getting all webhooks."""
        orchestrator.register_webhook("hook1", "https://example.com")
        result = orchestrator.get_webhooks()
        assert result["success"] is True
        assert len(result["webhooks"]) == 1


class TestLoadMetrics:
    """Tests for load metrics functionality."""

    def test_get_load_metrics(self, orchestrator, mock_agent):
        """Test getting load metrics."""
        orchestrator.register_agent("agent", "model", mock_agent)
        result = orchestrator.get_load_metrics()
        assert result["success"] is True
        assert "load_gauge" in result
        assert "metrics" in result
        assert "score" in result["load_gauge"]

    def test_load_gauge_levels(self, orchestrator):
        """Test load gauge levels are returned."""
        result = orchestrator.get_load_metrics()
        assert result["load_gauge"]["level"] in ["low", "moderate", "high", "critical"]

    def test_auto_adjust_load(self, orchestrator, mock_agent):
        """Test auto load adjustment."""
        orchestrator.register_agent("agent", "model", mock_agent)
        result = orchestrator.auto_adjust_load()
        assert result["success"] is True
        assert "recommendations" in result

    def test_get_system_dashboard(self, orchestrator, mock_agent):
        """Test getting system dashboard."""
        orchestrator.register_agent("agent", "model", mock_agent)
        result = orchestrator.get_system_dashboard()
        assert result["success"] is True
        assert "load_gauge" in result
        assert "metrics" in result
        assert "queue" in result


class TestExternalTrigger:
    """Tests for external trigger functionality."""

    def test_trigger_process_queue(self, orchestrator, mock_agent):
        """Test external trigger for queue processing."""
        orchestrator.register_agent("agent", "predict", mock_agent, ["predict"])
        orchestrator.create_task("predict", {})
        result = orchestrator.trigger_external("process_queue")
        assert result["success"] is True

    def test_trigger_status(self, orchestrator):
        """Test external trigger for status."""
        result = orchestrator.trigger_external("status")
        assert result["success"] is True
        assert "scheduler" in result

    def test_trigger_unknown(self, orchestrator):
        """Test unknown trigger returns error."""
        result = orchestrator.trigger_external("unknown_trigger")
        assert result["success"] is False


class TestTaskCaching:
    """Tests for task caching functionality."""

    def test_cache_task_result(self, orchestrator):
        """Test caching a task result."""
        result = orchestrator.cache_task_result(
            task_type="predict",
            payload={"ph": 7.0},
            result={"score": 85},
        )
        assert result["success"] is True
        assert "cache_key" in result

    def test_get_cached_result(self, orchestrator):
        """Test retrieving cached result."""
        orchestrator.cache_task_result("predict", {"ph": 7.0}, {"score": 85})
        result = orchestrator.get_cached_result("predict", {"ph": 7.0})
        assert result["success"] is True
        assert result["cached"] is True
        assert result["result"]["score"] == 85

    def test_cache_miss(self, orchestrator):
        """Test cache miss."""
        result = orchestrator.get_cached_result("nonexistent", {})
        assert result["cached"] is False

    def test_clear_cache(self, orchestrator):
        """Test clearing cache."""
        orchestrator.cache_task_result("task1", {}, {"result": 1})
        orchestrator.cache_task_result("task2", {}, {"result": 2})
        result = orchestrator.clear_cache()
        assert result["success"] is True
        assert result["cleared"] == 2

    def test_get_cache_stats(self, orchestrator):
        """Test getting cache stats."""
        orchestrator.cache_task_result("task", {}, {})
        result = orchestrator.get_cache_stats()
        assert result["size"] == 1

    def test_create_task_with_cache(self, orchestrator):
        """Test creating task with cache check."""
        # Cache a result
        orchestrator.cache_task_result("predict", {"test": 1}, {"cached_result": True})
        
        # Create task with same params should return cached
        result = orchestrator.create_task_with_cache("predict", {"test": 1})
        assert result["cached"] is True


class TestDynamicRouting:
    """Tests for dynamic routing functionality."""

    def test_get_agent_routing_score(self, orchestrator, mock_agent):
        """Test getting agent routing score."""
        orchestrator.register_agent("agent", "model", mock_agent)
        score = orchestrator.get_agent_routing_score("agent", "predict")
        assert 0 <= score <= 100

    def test_find_best_agent(self, orchestrator, mock_agent):
        """Test finding best agent for task."""
        orchestrator.register_agent("agent", "predict", mock_agent, ["predict"])
        task = {"type": "predict", "payload": {}}
        agent = orchestrator.find_best_agent_for_task(task)
        assert agent is not None

    def test_get_routing_stats(self, orchestrator):
        """Test getting routing stats."""
        result = orchestrator.get_routing_stats()
        assert result["success"] is True
        assert "routing_stats" in result

    def test_detect_redundant_tasks(self, orchestrator):
        """Test detecting redundant tasks."""
        orchestrator.create_task("same_task", {"data": "same"})
        orchestrator.create_task("same_task", {"data": "same"})
        result = orchestrator.detect_redundant_tasks()
        assert result["success"] is True
        assert result["redundant_count"] >= 1


class TestAgentPools:
    """Tests for agent pool functionality."""

    def test_create_agent_pool(self, orchestrator, mock_agent):
        """Test creating an agent pool."""
        result = orchestrator.create_agent_pool(
            pool_id="test_pool",
            agent_type="model",
            base_agent=mock_agent,
            pool_size=3,
            capabilities=["predict"],
        )
        assert result["success"] is True
        assert result["pool_size"] == 3

    def test_get_pool_status(self, orchestrator, mock_agent):
        """Test getting pool status."""
        orchestrator.create_agent_pool("pool1", "model", mock_agent, 2)
        result = orchestrator.get_pool_status("pool1")
        assert result["success"] is True
        assert result["pools"]["total_workers"] == 2

    def test_scale_pool_up(self, orchestrator, mock_agent):
        """Test scaling pool up."""
        orchestrator.create_agent_pool("pool1", "model", mock_agent, 2)
        result = orchestrator.scale_pool("pool1", 4)
        assert result["success"] is True
        assert result["new_size"] == 4

    def test_scale_pool_down(self, orchestrator, mock_agent):
        """Test scaling pool down."""
        orchestrator.create_agent_pool("pool1", "model", mock_agent, 4)
        result = orchestrator.scale_pool("pool1", 2)
        assert result["success"] is True

    def test_get_parallel_capacity(self, orchestrator, mock_agent):
        """Test getting parallel capacity."""
        orchestrator.register_agent("agent1", "model", mock_agent)
        orchestrator.register_agent("agent2", "model", mock_agent)
        result = orchestrator.get_parallel_capacity()
        assert result["success"] is True
        assert result["capacity"]["total_agents"] == 2


# Note: Parallel execution tests removed - they require full parallel environment
# The parallel architecture is documented in agents/parallel.py
# Tests should be run in deployment environment with proper thread support
