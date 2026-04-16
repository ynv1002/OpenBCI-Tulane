from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Deque, Dict

from .event_utils import EVENT_ACTIVE, EVENT_INACTIVE, EVENT_OFFSET, EVENT_ONSET


SUPPORTED_TRIGGER_STRATEGIES = (
    "binary_clench_threshold",
    "onset_threshold",
    "hybrid_transition",
)


@dataclass(frozen=True)
class JawClickTriggerConfig:
    strategy_name: str = "hybrid_transition"
    clench_probability_threshold: float = 0.70
    onset_probability_threshold: float = 0.45
    active_probability_threshold: float = 0.55
    rearm_clench_probability_threshold: float = 0.35
    cooldown_ms: int = 450
    minimum_separation_ms: int = 450
    smoothing_windows: int = 3
    hold_suppression: bool = True
    require_transition_from_inactive: bool = True
    minimum_clench_rise: float = 0.06
    minimum_envelope_uv: float = 0.0

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)

    def compact_name(self) -> str:
        parts = [
            self.strategy_name,
            f"cl{self.clench_probability_threshold:.2f}",
            f"on{self.onset_probability_threshold:.2f}",
            f"ac{self.active_probability_threshold:.2f}",
            f"re{self.rearm_clench_probability_threshold:.2f}",
            f"sm{self.smoothing_windows:d}",
            f"cd{self.cooldown_ms:d}",
            f"rise{self.minimum_clench_rise:.2f}",
        ]
        return "_".join(parts).replace(".", "p")


@dataclass(frozen=True)
class TriggerDecision:
    emitted_click: bool
    reason: str
    smoothed_scores: Dict[str, float]
    armed: bool


class JawClickTrigger:
    def __init__(self, config: JawClickTriggerConfig) -> None:
        if config.strategy_name not in SUPPORTED_TRIGGER_STRATEGIES:
            raise ValueError(f"Unsupported trigger strategy: {config.strategy_name}")
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._score_history: Deque[Dict[str, float]] = deque(maxlen=max(1, self.config.smoothing_windows))
        self._last_smoothed_scores: Dict[str, float] = {}
        self._last_event_label = EVENT_INACTIVE
        self._last_click_time_sec = float("-inf")
        self._armed = True

    def step(self, timestamp_sec: float, scores: Dict[str, float], event_label: str) -> TriggerDecision:
        self._score_history.append(scores)
        smoothed_scores = self._smooth_scores()
        current_label = str(event_label)

        block_ms = max(int(self.config.cooldown_ms), int(self.config.minimum_separation_ms))
        in_cooldown = (timestamp_sec - self._last_click_time_sec) * 1000.0 < float(block_ms)
        current_clench = float(smoothed_scores.get("clench_probability", 0.0))
        current_onset = float(smoothed_scores.get("onset_probability", 0.0))
        current_active = float(smoothed_scores.get("active_probability", 0.0))
        current_inactive = float(smoothed_scores.get("inactive_probability", 0.0))
        current_envelope = float(smoothed_scores.get("envelope_uv", 0.0))

        if self.config.hold_suppression and not self._armed:
            rearmed = current_clench <= self.config.rearm_clench_probability_threshold
            if self.config.require_transition_from_inactive:
                rearmed = rearmed and current_label in (EVENT_INACTIVE, EVENT_OFFSET)
            if rearmed:
                self._armed = True

        reason = ""
        emitted_click = False
        prev_clench = float(self._last_smoothed_scores.get("clench_probability", 0.0))
        prev_onset = float(self._last_smoothed_scores.get("onset_probability", 0.0))

        if current_envelope < self.config.minimum_envelope_uv:
            reason = "below_envelope_floor"
        elif in_cooldown:
            reason = "cooldown"
        elif self.config.hold_suppression and not self._armed:
            reason = "hold_suppressed"
        else:
            if self.config.strategy_name == "binary_clench_threshold":
                crossed = prev_clench < self.config.clench_probability_threshold <= current_clench
                if crossed:
                    emitted_click = True
                    reason = "binary_crossing"
            elif self.config.strategy_name == "onset_threshold":
                crossed = prev_onset < self.config.onset_probability_threshold <= current_onset
                onset_like = current_label in (EVENT_ONSET, EVENT_ACTIVE)
                if crossed and onset_like:
                    emitted_click = True
                    reason = "onset_crossing"
            else:
                onset_crossed = prev_onset < self.config.onset_probability_threshold <= current_onset
                transition_gate = (
                    current_clench >= self.config.clench_probability_threshold
                    and current_active >= self.config.active_probability_threshold
                    and (current_clench - prev_clench) >= self.config.minimum_clench_rise
                )
                if self.config.require_transition_from_inactive:
                    transition_gate = transition_gate and self._last_event_label in (
                        EVENT_INACTIVE,
                        EVENT_OFFSET,
                    )
                onset_like = current_label in (EVENT_ONSET, EVENT_ACTIVE)
                if onset_crossed and onset_like:
                    emitted_click = True
                    reason = "hybrid_onset_crossing"
                elif transition_gate and onset_like and current_inactive < 0.60:
                    emitted_click = True
                    reason = "hybrid_inactive_to_active"

        if emitted_click:
            self._last_click_time_sec = timestamp_sec
            if self.config.hold_suppression:
                self._armed = False

        self._last_smoothed_scores = smoothed_scores
        self._last_event_label = current_label
        return TriggerDecision(
            emitted_click=emitted_click,
            reason=reason,
            smoothed_scores=smoothed_scores,
            armed=self._armed,
        )

    def _smooth_scores(self) -> Dict[str, float]:
        if not self._score_history:
            return {}
        keys = self._score_history[0].keys()
        return {
            key: float(sum(frame[key] for frame in self._score_history) / len(self._score_history))
            for key in keys
        }
