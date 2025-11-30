"""
OrchestratorAgent - ML-powered agent for overseeing and coordinating other agents.
"""

import time
import uuid
from enum import Enum
from typing import Any, Callable


class TaskStatus(Enum):
    """Status values for tasks."""
    PENDING = "pending"
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Priority levels for tasks."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class OrchestratorAgent:
    """
    ML-powered orchestrator agent that oversees and coordinates other agent tasks.
    
    Responsible for:
    - Task prioritization and scheduling
    - Agent coordination and handoff
    - Monitoring task execution
    - Load balancing across agents
    - Context sharing between agents
    """

    def __init__(self):
        """Initialize the OrchestratorAgent."""
        self._tasks: dict[str, dict[str, Any]] = {}
        self._agents: dict[str, dict[str, Any]] = {}
        self._task_queue: list[str] = []
        self._completed_tasks: list[str] = []
        self._handlers: dict[str, Callable[..., Any]] = {}

    def register_agent(
        self,
        agent_id: str,
        agent_type: str,
        agent_instance: Any,
        capabilities: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Register an agent with the orchestrator.

        Args:
            agent_id: Unique identifier for the agent.
            agent_type: Type of agent (e.g., "model", "image", "art").
            agent_instance: The agent instance.
            capabilities: List of capabilities/tasks the agent can handle.

        Returns:
            Dictionary with registration result.
        """
        if not agent_id:
            return {"success": False, "error": "Agent ID is required"}

        if agent_id in self._agents:
            return {"success": False, "error": f"Agent {agent_id} already registered"}

        self._agents[agent_id] = {
            "id": agent_id,
            "type": agent_type,
            "instance": agent_instance,
            "capabilities": capabilities or [],
            "status": "available",
            "tasks_completed": 0,
            "tasks_failed": 0,
            "registered_at": time.time(),
        }

        return {
            "success": True,
            "agent_id": agent_id,
            "message": f"Agent {agent_id} registered successfully",
        }

    def unregister_agent(self, agent_id: str) -> dict[str, Any]:
        """
        Unregister an agent from the orchestrator.

        Args:
            agent_id: ID of the agent to unregister.

        Returns:
            Dictionary with unregistration result.
        """
        if agent_id not in self._agents:
            return {"success": False, "error": f"Agent {agent_id} not found"}

        del self._agents[agent_id]
        return {"success": True, "message": f"Agent {agent_id} unregistered"}

    def register_handler(
        self,
        task_type: str,
        handler: Callable[..., Any],
    ) -> dict[str, Any]:
        """
        Register a handler function for a specific task type.

        Args:
            task_type: Type of task this handler processes.
            handler: Callable that handles the task.

        Returns:
            Dictionary with registration result.
        """
        if not task_type:
            return {"success": False, "error": "Task type is required"}

        self._handlers[task_type] = handler
        return {
            "success": True,
            "message": f"Handler registered for task type: {task_type}",
        }

    def create_task(
        self,
        task_type: str,
        payload: dict[str, Any],
        priority: TaskPriority = TaskPriority.MEDIUM,
        depends_on: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Create a new task for the orchestrator to manage.

        Args:
            task_type: Type of task to create.
            payload: Task data/parameters.
            priority: Task priority level.
            depends_on: List of task IDs this task depends on.
            context: Additional context from other agents.

        Returns:
            Dictionary with task creation result.
        """
        if not task_type:
            return {"success": False, "error": "Task type is required"}

        task_id = str(uuid.uuid4())[:8]

        task = {
            "id": task_id,
            "type": task_type,
            "payload": payload,
            "priority": priority.value,
            "priority_name": priority.name,
            "status": TaskStatus.PENDING.value,
            "depends_on": depends_on or [],
            "context": context or {},
            "assigned_agent": None,
            "result": None,
            "error": None,
            "created_at": time.time(),
            "started_at": None,
            "completed_at": None,
        }

        self._tasks[task_id] = task
        self._queue_task(task_id)

        return {
            "success": True,
            "task_id": task_id,
            "status": TaskStatus.QUEUED.value,
        }

    def _queue_task(self, task_id: str) -> None:
        """Add task to queue based on priority."""
        task = self._tasks[task_id]
        task["status"] = TaskStatus.QUEUED.value

        # Insert based on priority (higher priority = earlier in queue)
        insert_pos = 0
        for i, queued_id in enumerate(self._task_queue):
            queued_task = self._tasks.get(queued_id)
            if queued_task and queued_task["priority"] >= task["priority"]:
                insert_pos = i + 1
            else:
                break

        self._task_queue.insert(insert_pos, task_id)

    def get_task(self, task_id: str) -> dict[str, Any]:
        """
        Get task details by ID.

        Args:
            task_id: ID of the task.

        Returns:
            Dictionary with task details.
        """
        if task_id not in self._tasks:
            return {"success": False, "error": "Task not found"}

        return {"success": True, "task": self._tasks[task_id]}

    def execute_task(self, task_id: str) -> dict[str, Any]:
        """
        Execute a specific task.

        Args:
            task_id: ID of the task to execute.

        Returns:
            Dictionary with execution result.
        """
        if task_id not in self._tasks:
            return {"success": False, "error": "Task not found"}

        task = self._tasks[task_id]

        # Check dependencies
        for dep_id in task["depends_on"]:
            dep_task = self._tasks.get(dep_id)
            if not dep_task or dep_task["status"] != TaskStatus.COMPLETED.value:
                return {
                    "success": False,
                    "error": f"Dependency {dep_id} not completed",
                }

        # Find suitable agent
        agent = self._find_agent_for_task(task)
        if not agent:
            return {
                "success": False,
                "error": "No suitable agent available for this task type",
            }

        # Execute task
        task["status"] = TaskStatus.IN_PROGRESS.value
        task["started_at"] = time.time()
        task["assigned_agent"] = agent["id"]
        agent["status"] = "busy"

        try:
            # Check for registered handler
            handler = self._handlers.get(task["type"])
            if handler:
                result = handler(task["payload"], task["context"])
            else:
                # Use agent's method based on task type
                result = self._execute_with_agent(agent, task)

            task["status"] = TaskStatus.COMPLETED.value
            task["result"] = result
            task["completed_at"] = time.time()
            agent["tasks_completed"] += 1

            # Remove from queue, add to completed
            if task_id in self._task_queue:
                self._task_queue.remove(task_id)
            self._completed_tasks.append(task_id)

            return {
                "success": True,
                "task_id": task_id,
                "status": TaskStatus.COMPLETED.value,
                "result": result,
            }

        except Exception as e:
            task["status"] = TaskStatus.FAILED.value
            task["error"] = str(e)
            task["completed_at"] = time.time()
            agent["tasks_failed"] += 1

            return {
                "success": False,
                "task_id": task_id,
                "status": TaskStatus.FAILED.value,
                "error": str(e),
            }

        finally:
            agent["status"] = "available"

    def _find_agent_for_task(self, task: dict[str, Any]) -> dict[str, Any] | None:
        """Find an available agent capable of handling the task."""
        task_type = task["type"]

        for agent in self._agents.values():
            if agent["status"] != "available":
                continue

            # Check if agent has the capability
            if task_type in agent["capabilities"] or agent["type"] == task_type:
                return agent

        return None

    def _execute_with_agent(
        self,
        agent: dict[str, Any],
        task: dict[str, Any],
    ) -> Any:
        """Execute task using the agent instance."""
        instance = agent["instance"]
        task_type = task["type"]
        payload = task["payload"]

        # Map task types to common agent methods
        method_mapping = {
            "predict": "predict_water_quality",
            "analyze": "analyze_aesthetics",
            "generate_image": "generate_image",
            "create_set": "create_image_set",
            "export_context": "export_context",
        }

        method_name = method_mapping.get(task_type, task_type)

        if hasattr(instance, method_name):
            method = getattr(instance, method_name)
            return method(**payload) if isinstance(payload, dict) else method(payload)

        return {"message": f"Task {task_type} processed", "payload": payload}

    def process_queue(self, max_tasks: int | None = None) -> dict[str, Any]:
        """
        Process tasks in the queue.

        Args:
            max_tasks: Maximum number of tasks to process (None = all).

        Returns:
            Dictionary with processing results.
        """
        processed = []
        failed = []
        tasks_to_process = self._task_queue[:max_tasks] if max_tasks else self._task_queue[:]

        for task_id in tasks_to_process:
            task = self._tasks.get(task_id)
            if not task:
                continue

            # Check if dependencies are met
            deps_met = all(
                self._tasks.get(dep_id, {}).get("status") == TaskStatus.COMPLETED.value
                for dep_id in task["depends_on"]
            )

            if not deps_met:
                continue

            result = self.execute_task(task_id)
            if result["success"]:
                processed.append(task_id)
            else:
                failed.append({"task_id": task_id, "error": result.get("error")})

        return {
            "success": True,
            "processed": len(processed),
            "failed": len(failed),
            "processed_tasks": processed,
            "failed_tasks": failed,
            "remaining_in_queue": len(self._task_queue),
        }

    def handoff_task(
        self,
        task_id: str,
        target_agent_id: str,
        additional_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Hand off a task to a specific agent with context.

        Args:
            task_id: ID of the task to hand off.
            target_agent_id: ID of the target agent.
            additional_context: Additional context to pass.

        Returns:
            Dictionary with handoff result.
        """
        if task_id not in self._tasks:
            return {"success": False, "error": "Task not found"}

        if target_agent_id not in self._agents:
            return {"success": False, "error": "Target agent not found"}

        task = self._tasks[task_id]
        target_agent = self._agents[target_agent_id]

        # Merge context
        if additional_context:
            task["context"].update(additional_context)

        # Assign to target agent
        task["assigned_agent"] = target_agent_id

        return {
            "success": True,
            "message": f"Task {task_id} handed off to {target_agent_id}",
            "task_id": task_id,
            "target_agent": target_agent_id,
            "context": task["context"],
        }

    def get_agent_status(self, agent_id: str | None = None) -> dict[str, Any]:
        """
        Get status of agents.

        Args:
            agent_id: Specific agent ID (None for all agents).

        Returns:
            Dictionary with agent status information.
        """
        if agent_id:
            if agent_id not in self._agents:
                return {"success": False, "error": "Agent not found"}
            agent = self._agents[agent_id]
            return {
                "success": True,
                "agent": {
                    "id": agent["id"],
                    "type": agent["type"],
                    "status": agent["status"],
                    "capabilities": agent["capabilities"],
                    "tasks_completed": agent["tasks_completed"],
                    "tasks_failed": agent["tasks_failed"],
                },
            }

        return {
            "success": True,
            "agents": [
                {
                    "id": a["id"],
                    "type": a["type"],
                    "status": a["status"],
                    "capabilities": a["capabilities"],
                    "tasks_completed": a["tasks_completed"],
                    "tasks_failed": a["tasks_failed"],
                }
                for a in self._agents.values()
            ],
        }

    def get_queue_status(self) -> dict[str, Any]:
        """
        Get status of the task queue.

        Returns:
            Dictionary with queue status information.
        """
        queue_tasks = [
            {
                "task_id": tid,
                "type": self._tasks[tid]["type"],
                "priority": self._tasks[tid]["priority_name"],
                "status": self._tasks[tid]["status"],
            }
            for tid in self._task_queue
            if tid in self._tasks
        ]

        return {
            "success": True,
            "queue_length": len(self._task_queue),
            "tasks": queue_tasks,
            "completed_count": len(self._completed_tasks),
        }

    def cancel_task(self, task_id: str) -> dict[str, Any]:
        """
        Cancel a pending or queued task.

        Args:
            task_id: ID of the task to cancel.

        Returns:
            Dictionary with cancellation result.
        """
        if task_id not in self._tasks:
            return {"success": False, "error": "Task not found"}

        task = self._tasks[task_id]

        if task["status"] in [TaskStatus.COMPLETED.value, TaskStatus.FAILED.value]:
            return {"success": False, "error": "Cannot cancel completed/failed task"}

        task["status"] = TaskStatus.CANCELLED.value
        task["completed_at"] = time.time()

        if task_id in self._task_queue:
            self._task_queue.remove(task_id)

        return {
            "success": True,
            "message": f"Task {task_id} cancelled",
            "task_id": task_id,
        }

    def analyze_workload(self) -> dict[str, Any]:
        """
        Analyze current workload distribution using ML heuristics.

        Returns:
            Dictionary with workload analysis and recommendations.
        """
        total_agents = len(self._agents)
        available_agents = sum(1 for a in self._agents.values() if a["status"] == "available")
        pending_tasks = len(self._task_queue)

        # Calculate workload metrics
        if total_agents == 0:
            utilization = 0
        else:
            utilization = ((total_agents - available_agents) / total_agents) * 100

        # Generate recommendations based on ML heuristics
        recommendations = []

        if pending_tasks > total_agents * 5:
            recommendations.append("Consider scaling up agent instances")

        if utilization > 80:
            recommendations.append("High utilization - monitor for bottlenecks")
        elif utilization < 20 and total_agents > 1:
            recommendations.append("Low utilization - consider scaling down")

        # Priority distribution
        priority_dist = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
        for tid in self._task_queue:
            task = self._tasks.get(tid, {})
            priority_name = task.get("priority_name", "MEDIUM")
            priority_dist[priority_name] = priority_dist.get(priority_name, 0) + 1

        return {
            "success": True,
            "analysis": {
                "total_agents": total_agents,
                "available_agents": available_agents,
                "pending_tasks": pending_tasks,
                "utilization_percentage": round(utilization, 2),
                "priority_distribution": priority_dist,
                "completed_tasks": len(self._completed_tasks),
            },
            "recommendations": recommendations,
        }
