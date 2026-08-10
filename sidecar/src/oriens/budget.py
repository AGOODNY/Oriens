"""模型用量与费用估算。"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

from .config import ModelRoleSettings
from .modeling import ModelUsage


@dataclass(frozen=True, slots=True)
class CostInfo:
    model: str
    input_tokens: int
    output_tokens: int
    estimated_cost_cny: float
    run_total_cny: float
    currency: str = "CNY"
    estimated: bool = True


class BudgetTracker:
    def __init__(self, run_limit_cny: float) -> None:
        self.run_limit_cny = run_limit_cny
        self._run_id: str | None = None
        self._run_total = 0.0
        self._lock = Lock()

    @property
    def run_total_cny(self) -> float:
        with self._lock:
            return self._run_total

    def set_run(self, run_id: str | None) -> None:
        with self._lock:
            if run_id != self._run_id:
                self._run_id = run_id
                self._run_total = 0.0

    def can_call_online(self) -> bool:
        with self._lock:
            return self._run_total < self.run_limit_cny

    def record(
        self,
        display_name: str,
        usage: ModelUsage,
        model: ModelRoleSettings | None,
    ) -> CostInfo:
        if model is None:
            cost = 0.0
        else:
            cost = (
                usage.input_tokens * model.input_price_per_million_cny
                + usage.output_tokens * model.output_price_per_million_cny
            ) / 1_000_000
        with self._lock:
            self._run_total += cost
            total = self._run_total
        return CostInfo(
            model=display_name,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            estimated_cost_cny=cost,
            run_total_cny=total,
        )
