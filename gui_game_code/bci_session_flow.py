from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Sequence

import numpy as np


SESSION_PHASE_PARTICIPANT = "Participant Prompt"
SESSION_PHASE_BASELINE = "Baseline"
SESSION_PHASE_GUIDED = "Guided Collection"
SESSION_PHASE_ADAPTING = "Adapting Session"
SESSION_PHASE_READY = "Session Ready"
SESSION_PHASE_GAMEPLAY = "Gameplay"
SESSION_PHASE_REVIEW = "Session Review"

PROTOCOL_LABEL_LEFT = "LEFT"
PROTOCOL_LABEL_RIGHT = "RIGHT"
PROTOCOL_LABEL_JAW_TAP = "JAW_TAP"
PROTOCOL_LABEL_HOLD = "HOLD"

NOTE_LEFT = "LEFT"
NOTE_RIGHT = "RIGHT"
NOTE_JAW_TAP = "JAW_TAP"
NOTE_JAW_HOLD = "JAW_HOLD"

MARKER_CODE_BY_LABEL = {
    PROTOCOL_LABEL_LEFT: 1,
    PROTOCOL_LABEL_RIGHT: 2,
    PROTOCOL_LABEL_JAW_TAP: 3,
    PROTOCOL_LABEL_HOLD: 4,
}

FEEDBACK_FIELDS = (
    "click_control",
    "hold_control",
    "left_control",
    "right_control",
    "overall_control",
)


@dataclass(frozen=True)
class GuidedProtocolStep:
    step_id: str
    action_label: str
    display_text: str
    instruction_text: str
    marker_code: int
    expected_count: int | None
    prep_sec: float
    active_sec: float
    rest_sec: float
    hold_duration_sec: float | None = None


@dataclass(frozen=True)
class RhythmNote:
    note_id: int
    note_type: str
    lane_index: int
    start_time_sec: float
    hit_time_sec: float
    end_time_sec: float
    hold_required_sec: float


@dataclass(frozen=True)
class RhythmGameConfig:
    lane_labels: tuple[str, ...] = (NOTE_LEFT, NOTE_RIGHT, NOTE_JAW_TAP, NOTE_JAW_HOLD)
    canvas_width: int = 920
    canvas_height: int = 460
    lane_padding_px: int = 26
    target_line_y_px: int = 390
    note_spawn_y_px: int = 46
    note_radius_px: int = 24
    travel_time_sec: float = 1.85
    hit_window_sec: float = 0.40
    hold_hit_window_sec: float = 0.45
    note_gap_sec: float = 0.95
    note_base_sec: float = 1.10
    hold_note_sec: float = 1.50
    chart_seed: int = 17


class RhythmGame:
    def __init__(self, notes: Sequence[RhythmNote], config: RhythmGameConfig | None = None) -> None:
        self.config = config or RhythmGameConfig()
        self.notes = list(notes)
        self.reset()

    def reset(self) -> None:
        self.score = 0
        self.hits = 0
        self.misses = 0
        self.completed = False
        self.active_hold_note_id: int | None = None
        self.active_hold_started_sec: float | None = None
        self.current_hold_active = False
        self.note_states = {
            note.note_id: {
                "status": "pending",
                "hit_time_sec": None,
                "release_time_sec": None,
                "hold_started": False,
            }
            for note in self.notes
        }

    def _lane_width(self) -> float:
        return (self.config.canvas_width - 2 * self.config.lane_padding_px) / len(self.config.lane_labels)

    def lane_center_x(self, lane_index: int) -> float:
        lane_width = self._lane_width()
        return self.config.lane_padding_px + lane_width * (lane_index + 0.5)

    def note_y_px(self, note: RhythmNote, run_time_sec: float) -> float:
        progress = (run_time_sec - note.start_time_sec) / max(self.config.travel_time_sec, 1e-6)
        progress = min(max(progress, 0.0), 1.0)
        return self.config.note_spawn_y_px + (
            self.config.target_line_y_px - self.config.note_spawn_y_px
        ) * progress

    def _expected_action(self, note_type: str) -> str:
        if note_type == NOTE_LEFT:
            return "left"
        if note_type == NOTE_RIGHT:
            return "right"
        if note_type == NOTE_JAW_TAP:
            return "click"
        return "hold"

    def _mark_missed(self, note: RhythmNote, run_time_sec: float, reason: str) -> dict[str, Any]:
        state = self.note_states[note.note_id]
        if state["status"] in {"hit", "missed"}:
            return {}
        state["status"] = "missed"
        self.misses += 1
        return {
            "run_time_sec": float(run_time_sec),
            "event_type": "note_miss",
            "note_id": int(note.note_id),
            "note_type": note.note_type,
            "lane": int(note.lane_index),
            "score": int(self.score),
            "details": reason,
        }

    def _mark_hit(self, note: RhythmNote, run_time_sec: float, event_type: str, score_delta: int, details: str = "") -> dict[str, Any]:
        state = self.note_states[note.note_id]
        if state["status"] == "hit":
            return {}
        state["status"] = "hit"
        state["hit_time_sec"] = float(run_time_sec)
        self.score += int(score_delta)
        self.hits += 1
        return {
            "run_time_sec": float(run_time_sec),
            "event_type": event_type,
            "note_id": int(note.note_id),
            "note_type": note.note_type,
            "lane": int(note.lane_index),
            "score": int(self.score),
            "details": details,
        }

    def _visible_notes(self, run_time_sec: float) -> list[RhythmNote]:
        return [
            note
            for note in self.notes
            if note.start_time_sec - 0.05 <= run_time_sec <= note.end_time_sec + self.config.hit_window_sec
            and self.note_states[note.note_id]["status"] != "missed"
        ]

    def update(self, snapshot: Any) -> list[dict[str, Any]]:
        run_time_sec = float(snapshot.run_time_sec)
        events: list[dict[str, Any]] = []
        self.current_hold_active = bool(snapshot.jaw_hold_active)

        for note in self.notes:
            state = self.note_states[note.note_id]
            if state["status"] in {"hit", "missed"}:
                continue
            if note.note_type == NOTE_JAW_HOLD:
                if run_time_sec > note.end_time_sec + self.config.hold_hit_window_sec:
                    reason = "hold_never_completed" if state["hold_started"] else "hold_not_started"
                    missed = self._mark_missed(note, run_time_sec, reason)
                    if missed:
                        events.append(missed)
                elif state["hold_started"] and not snapshot.jaw_hold_active and run_time_sec >= note.hit_time_sec:
                    missed = self._mark_missed(note, run_time_sec, "hold_released_early")
                    if missed:
                        events.append(missed)
            elif run_time_sec > note.hit_time_sec + self.config.hit_window_sec:
                missed = self._mark_missed(note, run_time_sec, "hit_window_elapsed")
                if missed:
                    events.append(missed)

        for detection in snapshot.detections:
            if not bool(detection.get("accepted", False)):
                continue
            action = str(detection.get("action", "")).lower()
            action_time_sec = float(detection.get("timestamp_sec", run_time_sec))
            matching_notes = []
            for note in self._visible_notes(run_time_sec):
                state = self.note_states[note.note_id]
                if state["status"] != "pending":
                    continue
                expected_action = self._expected_action(note.note_type)
                if expected_action == "hold" and action not in {"hold_start", "hold_end"}:
                    continue
                if expected_action != "hold" and action != expected_action:
                    continue
                matching_notes.append(note)
            if not matching_notes:
                continue
            note = min(matching_notes, key=lambda candidate: abs(candidate.hit_time_sec - action_time_sec))
            state = self.note_states[note.note_id]
            if note.note_type == NOTE_JAW_HOLD:
                if action == "hold_start":
                    if abs(note.hit_time_sec - action_time_sec) <= self.config.hold_hit_window_sec:
                        state["hold_started"] = True
                        state["hit_time_sec"] = float(action_time_sec)
                        self.active_hold_note_id = int(note.note_id)
                        self.active_hold_started_sec = float(action_time_sec)
                        events.append(
                            {
                                "run_time_sec": float(run_time_sec),
                                "event_type": "hold_note_started",
                                "note_id": int(note.note_id),
                                "note_type": note.note_type,
                                "lane": int(note.lane_index),
                                "score": int(self.score),
                                "details": "",
                            }
                        )
                elif action == "hold_end" and self.active_hold_note_id == int(note.note_id):
                    if action_time_sec < note.end_time_sec - 0.15:
                        missed = self._mark_missed(note, run_time_sec, "hold_released_early")
                        if missed:
                            events.append(missed)
                    self.active_hold_note_id = None
                    self.active_hold_started_sec = None
            else:
                if abs(note.hit_time_sec - action_time_sec) <= self.config.hit_window_sec:
                    hit = self._mark_hit(note, run_time_sec, "note_hit", 2)
                    if hit:
                        events.append(hit)

        for note in self.notes:
            state = self.note_states[note.note_id]
            if note.note_type != NOTE_JAW_HOLD or state["status"] != "pending" or not state["hold_started"]:
                continue
            if snapshot.jaw_hold_active and run_time_sec >= note.end_time_sec:
                hit = self._mark_hit(note, run_time_sec, "hold_note_complete", 3)
                if hit:
                    events.append(hit)
                    self.active_hold_note_id = None
                    self.active_hold_started_sec = None

        self.completed = all(state["status"] in {"hit", "missed"} for state in self.note_states.values())
        return [event for event in events if event]

    def render(self, canvas: Any, run_time_sec: float, title: str, subtitle: str) -> None:
        canvas.delete("all")
        width = self.config.canvas_width
        height = self.config.canvas_height
        canvas.create_rectangle(0, 0, width, height, fill="#08101f", outline="")
        lane_width = self._lane_width()
        colors = {
            NOTE_LEFT: "#60a5fa",
            NOTE_RIGHT: "#f87171",
            NOTE_JAW_TAP: "#f59e0b",
            NOTE_JAW_HOLD: "#14b8a6",
        }

        for lane_index, lane_label in enumerate(self.config.lane_labels):
            left = self.config.lane_padding_px + lane_width * lane_index
            right = left + lane_width
            canvas.create_rectangle(left, 24, right, height - 30, fill="#0f172a", outline="#1f2937", width=2)
            canvas.create_text(
                (left + right) / 2,
                42,
                text=lane_label.replace("_", " "),
                fill=colors[lane_label],
                font=("Helvetica", 12, "bold"),
            )

        canvas.create_line(
            self.config.lane_padding_px,
            self.config.target_line_y_px,
            width - self.config.lane_padding_px,
            self.config.target_line_y_px,
            fill="#e5e7eb",
            width=3,
        )

        for note in self.notes:
            state = self.note_states[note.note_id]
            if state["status"] == "missed":
                continue
            if run_time_sec < note.start_time_sec - 0.25 or run_time_sec > note.end_time_sec + self.config.hit_window_sec:
                continue
            x_pos = self.lane_center_x(note.lane_index)
            color = colors[note.note_type]
            y_pos = self.note_y_px(note, run_time_sec)
            if note.note_type == NOTE_JAW_HOLD:
                hold_end_y = self.note_y_px(
                    RhythmNote(
                        note_id=note.note_id,
                        note_type=note.note_type,
                        lane_index=note.lane_index,
                        start_time_sec=note.start_time_sec,
                        hit_time_sec=note.end_time_sec,
                        end_time_sec=note.end_time_sec,
                        hold_required_sec=note.hold_required_sec,
                    ),
                    run_time_sec,
                )
                canvas.create_rectangle(
                    x_pos - 18,
                    y_pos,
                    x_pos + 18,
                    hold_end_y,
                    fill=color,
                    outline=color,
                    stipple="gray25",
                )
            canvas.create_oval(
                x_pos - self.config.note_radius_px,
                y_pos - self.config.note_radius_px,
                x_pos + self.config.note_radius_px,
                y_pos + self.config.note_radius_px,
                fill=color,
                outline="#e5e7eb",
                width=2,
            )

        canvas.create_text(24, 18, anchor="nw", fill="#e5e7eb", font=("Helvetica", 16, "bold"), text=title)
        canvas.create_text(24, height - 18, anchor="sw", fill="#94a3b8", font=("Helvetica", 11), text=subtitle)
        canvas.create_text(
            width - 24,
            18,
            anchor="ne",
            fill="#e5e7eb",
            font=("Helvetica", 14, "bold"),
            text=f"Score {self.score}  |  Hits {self.hits}  |  Misses {self.misses}",
        )


def build_guided_protocol(
    *,
    prep_sec: float,
    rest_sec: float,
    tap_base_sec: float,
    tap_per_count_sec: float,
    hold_duration_sec: float,
    hold_trials: int,
) -> list[GuidedProtocolStep]:
    steps: list[GuidedProtocolStep] = []
    step_number = 1

    for action_label, display_prefix in (
        (PROTOCOL_LABEL_LEFT, "LEFT"),
        (PROTOCOL_LABEL_RIGHT, "RIGHT"),
        (PROTOCOL_LABEL_JAW_TAP, "JAW"),
    ):
        marker_code = MARKER_CODE_BY_LABEL[action_label]
        for expected_count in range(1, 7):
            active_sec = float(tap_base_sec + tap_per_count_sec * expected_count)
            if action_label == PROTOCOL_LABEL_JAW_TAP:
                instruction = f"Jaw clench {expected_count} time{'s' if expected_count != 1 else ''} during the block."
            else:
                instruction = f"{display_prefix} hand clench {expected_count} time{'s' if expected_count != 1 else ''} during the block."
            steps.append(
                GuidedProtocolStep(
                    step_id=f"step_{step_number:02d}_{action_label.lower()}_{expected_count}",
                    action_label=action_label,
                    display_text=f"{display_prefix} x {expected_count}",
                    instruction_text=instruction,
                    marker_code=marker_code,
                    expected_count=expected_count,
                    prep_sec=float(prep_sec),
                    active_sec=active_sec,
                    rest_sec=float(rest_sec),
                )
            )
            step_number += 1

    for hold_index in range(1, hold_trials + 1):
        steps.append(
            GuidedProtocolStep(
                step_id=f"step_{step_number:02d}_hold_{hold_index}",
                action_label=PROTOCOL_LABEL_HOLD,
                display_text=f"HOLD trial {hold_index}",
                instruction_text=f"Jaw clench and hold for {hold_duration_sec:.1f} seconds.",
                marker_code=MARKER_CODE_BY_LABEL[PROTOCOL_LABEL_HOLD],
                expected_count=None,
                prep_sec=float(prep_sec),
                active_sec=float(hold_duration_sec),
                rest_sec=float(rest_sec),
                hold_duration_sec=float(hold_duration_sec),
            )
        )
        step_number += 1
    return steps


def build_simple_rhythm_chart(
    *,
    config: RhythmGameConfig | None = None,
    cycles: int = 3,
) -> list[RhythmNote]:
    game_config = config or RhythmGameConfig()
    rng = random.Random(game_config.chart_seed)
    base_pattern = [
        NOTE_LEFT,
        NOTE_RIGHT,
        NOTE_JAW_TAP,
        NOTE_LEFT,
        NOTE_JAW_HOLD,
        NOTE_RIGHT,
        NOTE_JAW_TAP,
        NOTE_LEFT,
        NOTE_RIGHT,
        NOTE_JAW_HOLD,
        NOTE_JAW_TAP,
        NOTE_RIGHT,
    ]
    note_types = base_pattern * max(1, cycles)
    notes: list[RhythmNote] = []
    current_hit_time = 2.0
    for note_id, note_type in enumerate(note_types, start=1):
        hold_required_sec = game_config.hold_note_sec if note_type == NOTE_JAW_HOLD else 0.0
        duration_sec = hold_required_sec if hold_required_sec > 0 else 0.0
        lane_index = game_config.lane_labels.index(note_type)
        notes.append(
            RhythmNote(
                note_id=note_id,
                note_type=note_type,
                lane_index=lane_index,
                start_time_sec=current_hit_time - game_config.travel_time_sec,
                hit_time_sec=current_hit_time,
                end_time_sec=current_hit_time + duration_sec,
                hold_required_sec=hold_required_sec,
            )
        )
        gap_sec = game_config.note_gap_sec + rng.uniform(-0.15, 0.18)
        current_hit_time += game_config.note_base_sec + duration_sec + max(0.55, gap_sec)
    return notes


def _window_rows(rows: Sequence[dict[str, Any]], start_sec: float, end_sec: float) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if start_sec <= float(row.get("run_time_sec", -math.inf)) <= end_sec
    ]


def _quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=float), q))


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def compute_session_adaptation(
    *,
    control_trace_rows: Sequence[dict[str, Any]],
    protocol_rows: Sequence[dict[str, Any]],
    base_hand: dict[str, Any],
    base_jaw: dict[str, Any],
) -> dict[str, Any]:
    baseline_rows = [row for row in protocol_rows if str(row.get("phase_name", "")) == SESSION_PHASE_BASELINE]
    baseline_window = baseline_rows[0] if baseline_rows else None
    baseline_trace = (
        _window_rows(
            control_trace_rows,
            float(baseline_window.get("start_run_time_sec", 0.0)),
            float(baseline_window.get("end_run_time_sec", 0.0)),
        )
        if baseline_window is not None
        else []
    )

    baseline_jaw = [float(row.get("jaw_confidence", 0.0)) for row in baseline_trace]
    baseline_jaw_active = [float(row.get("jaw_active_confidence", 0.0)) for row in baseline_trace]
    baseline_hand_left = [float(row.get("hand_left_confidence", 0.0)) for row in baseline_trace]
    baseline_hand_right = [float(row.get("hand_right_confidence", 0.0)) for row in baseline_trace]

    jaw_tap_rows = [row for row in protocol_rows if str(row.get("protocol_label", "")) == PROTOCOL_LABEL_JAW_TAP]
    hold_rows = [row for row in protocol_rows if str(row.get("protocol_label", "")) == PROTOCOL_LABEL_HOLD]
    left_rows = [row for row in protocol_rows if str(row.get("protocol_label", "")) == PROTOCOL_LABEL_LEFT]
    right_rows = [row for row in protocol_rows if str(row.get("protocol_label", "")) == PROTOCOL_LABEL_RIGHT]

    jaw_tap_trace = [
        sample
        for row in jaw_tap_rows
        for sample in _window_rows(control_trace_rows, float(row["start_run_time_sec"]), float(row["end_run_time_sec"]))
    ]
    hold_trace = [
        sample
        for row in hold_rows
        for sample in _window_rows(control_trace_rows, float(row["start_run_time_sec"]), float(row["end_run_time_sec"]))
    ]

    left_trace = [
        sample
        for row in left_rows
        for sample in _window_rows(control_trace_rows, float(row["start_run_time_sec"]), float(row["end_run_time_sec"]))
    ]
    right_trace = [
        sample
        for row in right_rows
        for sample in _window_rows(control_trace_rows, float(row["start_run_time_sec"]), float(row["end_run_time_sec"]))
    ]

    jaw_click_threshold = float(base_jaw.get("click_threshold", 0.80))
    jaw_rearm_threshold = float(base_jaw.get("rearm_threshold", 0.50))
    jaw_hold_probability_threshold = float(base_jaw.get("hold_probability_threshold", 0.70))
    jaw_hold_onset_sec = float(base_jaw.get("hold_onset_sec", 0.45))
    jaw_hold_release_sec = float(base_jaw.get("hold_release_sec", 0.18))
    jaw_reason = "fallback_to_base"
    jaw_adapted = False

    tap_q40 = _quantile([float(row.get("jaw_confidence", 0.0)) for row in jaw_tap_trace], 0.40)
    tap_q65 = _quantile([float(row.get("jaw_confidence", 0.0)) for row in jaw_tap_trace], 0.65)
    hold_active_q50 = _quantile([float(row.get("jaw_active_confidence", 0.0)) for row in hold_trace], 0.50)
    hold_active_q75 = _quantile([float(row.get("jaw_active_confidence", 0.0)) for row in hold_trace], 0.75)
    base_jaw_q98 = _quantile(baseline_jaw, 0.98)
    base_jaw_active_q98 = _quantile(baseline_jaw_active, 0.98)

    if tap_q40 is not None and base_jaw_q98 is not None and tap_q40 > base_jaw_q98 + 0.04:
        jaw_click_threshold = _clamp((tap_q40 + base_jaw_q98) / 2.0, 0.58, 0.88)
        jaw_rearm_threshold = _clamp(jaw_click_threshold - 0.20, 0.25, jaw_click_threshold - 0.06)
        jaw_adapted = True
        jaw_reason = "session_quantile_shift"
    if hold_active_q50 is not None and base_jaw_active_q98 is not None and hold_active_q50 > base_jaw_active_q98 + 0.03:
        jaw_hold_probability_threshold = _clamp((hold_active_q50 + base_jaw_active_q98) / 2.0, 0.55, 0.88)
        jaw_hold_onset_sec = 0.40 if (hold_active_q75 is not None and hold_active_q75 >= 0.78) else 0.48
        jaw_adapted = True
        jaw_reason = "session_quantile_shift"

    direction_min_confidence = float(base_hand.get("direction_min_confidence", 0.60))
    direction_margin = float(base_hand.get("direction_margin", 0.10))
    hand_action_latch_sec = float(base_hand.get("hand_action_latch_sec", 0.45))
    hand_switch_cooldown_sec = float(base_hand.get("hand_switch_cooldown_sec", 0.20))
    hand_reason = "fallback_to_base"
    hand_adapted = False

    left_target_probs = [float(row.get("hand_left_confidence", 0.0)) for row in left_trace]
    left_margins = [
        float(row.get("hand_left_confidence", 0.0)) - float(row.get("hand_right_confidence", 0.0))
        for row in left_trace
    ]
    right_target_probs = [float(row.get("hand_right_confidence", 0.0)) for row in right_trace]
    right_margins = [
        float(row.get("hand_right_confidence", 0.0)) - float(row.get("hand_left_confidence", 0.0))
        for row in right_trace
    ]
    combined_target_probs = left_target_probs + right_target_probs
    combined_margins = left_margins + right_margins
    target_q35 = _quantile(combined_target_probs, 0.35)
    target_q65 = _quantile(combined_target_probs, 0.65)
    margin_q35 = _quantile(combined_margins, 0.35)
    base_lr_q98 = _quantile(baseline_hand_left + baseline_hand_right, 0.98)

    if (
        target_q35 is not None
        and margin_q35 is not None
        and base_lr_q98 is not None
        and target_q35 > base_lr_q98 + 0.03
        and margin_q35 > 0.05
    ):
        direction_min_confidence = _clamp(max(0.50, min(target_q35 - 0.02, 0.78)), 0.50, 0.78)
        direction_margin = _clamp(margin_q35 * 0.85, 0.05, 0.22)
        hand_action_latch_sec = 0.55 if target_q65 is not None and target_q65 < 0.72 else 0.45
        hand_switch_cooldown_sec = 0.26 if margin_q35 < 0.10 else 0.20
        hand_adapted = True
        hand_reason = "session_confidence_separation"

    return {
        "jaw": {
            "used_adapted_settings": bool(jaw_adapted),
            "decision_style": "jaw_trigger_plus_hold_interpreter",
            "settings": {
                "click_threshold": float(jaw_click_threshold),
                "rearm_threshold": float(jaw_rearm_threshold),
                "hold_probability_threshold": float(jaw_hold_probability_threshold),
                "hold_onset_sec": float(jaw_hold_onset_sec),
                "hold_release_sec": float(jaw_hold_release_sec),
            },
            "reason": jaw_reason,
            "stats": {
                "baseline_jaw_q98": base_jaw_q98,
                "baseline_jaw_active_q98": base_jaw_active_q98,
                "tap_jaw_q40": tap_q40,
                "tap_jaw_q65": tap_q65,
                "hold_active_q50": hold_active_q50,
                "hold_active_q75": hold_active_q75,
            },
        },
        "hand": {
            "used_adapted_settings": bool(hand_adapted),
            "decision_style": "event_gated_direction_vote",
            "settings": {
                "direction_min_confidence": float(direction_min_confidence),
                "direction_margin": float(direction_margin),
                "hand_action_latch_sec": float(hand_action_latch_sec),
                "hand_switch_cooldown_sec": float(hand_switch_cooldown_sec),
            },
            "reason": hand_reason,
            "stats": {
                "baseline_lr_q98": base_lr_q98,
                "target_q35": target_q35,
                "target_q65": target_q65,
                "margin_q35": margin_q35,
                "left_segment_count": len(left_rows),
                "right_segment_count": len(right_rows),
            },
        },
    }
