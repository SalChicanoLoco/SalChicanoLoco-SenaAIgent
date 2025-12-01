"""
SenaAIgent - Modular Python agent classes for ML analytics, image generation,
aesthetic analysis, and task orchestration.

ARCHITECTURE DOCUMENTATION
==========================

SenaAIgent is built on a true parallelized architecture designed for
maximum throughput and scalability. Key components:

1. AGENTS
---------
- ModelAgent: ML analytics (water quality prediction)
- ImageAgent: AI image generation
- ArtAgent: Aesthetic analysis and computer vision
- OrchestratorAgent: Task coordination and parallel execution

2. PARALLEL EXECUTION
---------------------
The system uses Python's concurrent.futures for true parallelism:
- ThreadPoolExecutor for I/O-bound tasks (API calls, file I/O)
- Agent pools for horizontal scaling
- Dynamic routing based on agent performance

3. TASK MANAGEMENT
------------------
- Priority-based queue (CRITICAL > HIGH > MEDIUM > LOW)
- Dependency resolution with DAG
- Result caching to avoid redundant computation
- Redundancy detection

4. PERFORMANCE
--------------
- Auto-adjusting scheduler intervals based on load
- Load gauge metrics (0-100 score)
- Parallel capacity monitoring
- Speedup factor reporting

See agents/parallel.py for detailed parallel architecture documentation.
"""

from .model_agent import ModelAgent
from .image_agent import ImageAgent
from .art_agent import ArtAgent
from .orchestrator_agent import OrchestratorAgent, TaskPriority, TaskStatus, RecurrencePattern
from .parallel import (
    ParallelEngine,
    ParallelConfig,
    Priority,
    ExecutionResult,
    PerformanceMetrics,
    parallel_map,
    parallel_execute,
    get_engine,
)

__all__ = [
    # Agents
    "ModelAgent",
    "ImageAgent",
    "ArtAgent",
    "OrchestratorAgent",
    # Orchestrator enums
    "TaskPriority",
    "TaskStatus",
    "RecurrencePattern",
    # Parallel execution
    "ParallelEngine",
    "ParallelConfig",
    "Priority",
    "ExecutionResult",
    "PerformanceMetrics",
    "parallel_map",
    "parallel_execute",
    "get_engine",
]
