"""In-memory реестр запусков ГА.

Хранит состояние каждого job: статус, прогресс по поколениям, текущий лучший
результат. Используется FastAPI-эндпоинтами /optimize/{module}/status/{job_id}.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import uuid4


@dataclass
class JobState:
    job_id: str
    module: str
    status: str = "queued"  # queued | running | done | error
    progress: float = 0.0  # 0..1
    generation: int = 0
    total_generations: int = 0
    best_fitness: Optional[float] = None
    best_params: Optional[Dict[str, Any]] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None  # полный GAResult.dict()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "module": self.module,
            "status": self.status,
            "progress": self.progress,
            "generation": self.generation,
            "total_generations": self.total_generations,
            "best_fitness": self.best_fitness,
            "best_params": self.best_params,
            "history": self.history,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "result": self.result,
        }


class JobRegistry:
    def __init__(self) -> None:
        self._jobs: Dict[str, JobState] = {}
        self._lock = threading.Lock()

    def create(self, module: str, total_generations: int) -> JobState:
        job = JobState(
            job_id=uuid4().hex,
            module=module,
            total_generations=total_generations,
        )
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Optional[JobState]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_by_module(self, module: str) -> List[JobState]:
        with self._lock:
            return [j for j in self._jobs.values() if j.module == module]

    def update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            for key, value in fields.items():
                setattr(job, key, value)


registry = JobRegistry()
