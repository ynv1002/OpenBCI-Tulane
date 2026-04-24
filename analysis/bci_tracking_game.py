from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import sys
import threading
import time
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from analysis.bci_game_runtime import (
        ACTION_CLICK,
        ACTION_HOLD_END,
        ACTION_HOLD_START,
        ACTION_LEFT,
        ACTION_RIGHT,
        BCITrackingGameConfig,
        HAND_NEUTRAL,
        LIVE,
        REPLAY,
        ReplayGameController,
        TrackingGameLogger,
        TrackingSnapshot,
        LiveGameController,
    )
    from analysis.bci_session_flow import (
        FEEDBACK_FIELDS,
        MARKER_CODE_BY_LABEL,
        NOTE_JAW_HOLD,
        NOTE_JAW_TAP,
        PROTOCOL_LABEL_HOLD,
        PROTOCOL_LABEL_JAW_TAP,
        PROTOCOL_LABEL_LEFT,
        PROTOCOL_LABEL_RIGHT,
        SESSION_PHASE_ADAPTING,
        SESSION_PHASE_BASELINE,
        SESSION_PHASE_GAMEPLAY,
        SESSION_PHASE_GUIDED,
        SESSION_PHASE_PARTICIPANT,
        SESSION_PHASE_READY,
        SESSION_PHASE_REVIEW,
        GuidedProtocolStep,
        RhythmGame,
        RhythmGameConfig,
        RhythmNote,
        build_guided_protocol,
        build_simple_rhythm_chart,
        compute_session_adaptation,
    )
else:
    from .bci_game_runtime import (
        ACTION_CLICK,
        ACTION_HOLD_END,
        ACTION_HOLD_START,
        ACTION_LEFT,
        ACTION_RIGHT,
        BCITrackingGameConfig,
        HAND_NEUTRAL,
        LIVE,
        REPLAY,
        ReplayGameController,
        TrackingGameLogger,
        TrackingSnapshot,
        LiveGameController,
    )
    from .bci_session_flow import (
        FEEDBACK_FIELDS,
        MARKER_CODE_BY_LABEL,
        NOTE_JAW_HOLD,
        NOTE_JAW_TAP,
        PROTOCOL_LABEL_HOLD,
        PROTOCOL_LABEL_JAW_TAP,
        PROTOCOL_LABEL_LEFT,
        PROTOCOL_LABEL_RIGHT,
        SESSION_PHASE_ADAPTING,
        SESSION_PHASE_BASELINE,
        SESSION_PHASE_GAMEPLAY,
        SESSION_PHASE_GUIDED,
        SESSION_PHASE_PARTICIPANT,
        SESSION_PHASE_READY,
        SESSION_PHASE_REVIEW,
        GuidedProtocolStep,
        RhythmGame,
        RhythmGameConfig,
        RhythmNote,
        build_guided_protocol,
        build_simple_rhythm_chart,
        compute_session_adaptation,
    )


GAME_BG = "#0b1220"
GAME_PANEL = "#111827"
GAME_TEXT = "#e5e7eb"
GAME_ACCENT = "#22d3ee"
GAME_CLICK = "#f59e0b"
GAME_HOLD = "#14b8a6"
GAME_LEFT = "#60a5fa"
GAME_RIGHT = "#f87171"
GAME_TARGET = "#fbbf24"
INFO_HISTORY_LIMIT = 240


@dataclass(frozen=True)
class LaneGameConfig:
    lane_count: int = 7
    canvas_width: int = 860
    canvas_height: int = 420
    spawn_interval_sec: float = 1.20
    spawn_jitter_sec: float = 0.45
    target_speed_px_sec: float = 108.0
    click_flash_sec: float = 0.25
    hold_capture_y_px: float = 140.0
    track_capture_y_px: float = 360.0
    player_y_px: float = 378.0
    target_radius_px: float = 18.0
    seed: int = 42


@dataclass
class LaneTarget:
    target_id: int
    lane: int
    y_px: float
    speed_px_sec: float
    spawned_sec: float


class LaneTrackingGame:
    def __init__(self, config: LaneGameConfig | None = None) -> None:
        self.config = config or LaneGameConfig()
        self.rng = random.Random(self.config.seed)
        self.reset()

    def reset(self) -> None:
        self.player_lane = self.config.lane_count // 2
        self.targets: list[LaneTarget] = []
        self.next_target_id = 1
        self.next_spawn_time_sec = 1.0
        self.last_run_time_sec: float | None = None
        self.click_flash_until_sec = float("-inf")
        self.hold_active = False
        self.score = 0
        self.hits = 0
        self.misses = 0
        self.hand_enabled = True
        self.jaw_enabled = True
        self.spawn_current_lane_bias = 0.0

    def set_branch_availability(self, hand_enabled: bool, jaw_enabled: bool) -> None:
        self.hand_enabled = bool(hand_enabled)
        self.jaw_enabled = bool(jaw_enabled)
        self.spawn_current_lane_bias = 0.70 if self.jaw_enabled and not self.hand_enabled else 0.15

    def _lane_center_x(self, lane: int) -> float:
        lane_width = self.config.canvas_width / self.config.lane_count
        return lane_width * (lane + 0.5)

    def _spawn_target(self, run_time_sec: float) -> dict[str, Any]:
        lane = self.player_lane if self.rng.random() < self.spawn_current_lane_bias else self.rng.randrange(self.config.lane_count)
        target = LaneTarget(
            target_id=self.next_target_id,
            lane=lane,
            y_px=40.0,
            speed_px_sec=self.config.target_speed_px_sec + self.rng.uniform(-12.0, 12.0),
            spawned_sec=run_time_sec,
        )
        self.next_target_id += 1
        self.targets.append(target)
        return {
            "run_time_sec": float(run_time_sec),
            "event_type": "target_spawn",
            "target_id": int(target.target_id),
            "lane": int(target.lane),
            "score": int(self.score),
            "details": "",
        }

    def _capture_target(self, target: LaneTarget, run_time_sec: float, event_type: str, score_delta: int) -> dict[str, Any]:
        self.targets = [candidate for candidate in self.targets if candidate.target_id != target.target_id]
        self.score += int(score_delta)
        self.hits += 1
        return {
            "run_time_sec": float(run_time_sec),
            "event_type": event_type,
            "target_id": int(target.target_id),
            "lane": int(target.lane),
            "score": int(self.score),
            "details": "",
        }

    def _miss_target(self, target: LaneTarget, run_time_sec: float) -> dict[str, Any]:
        self.targets = [candidate for candidate in self.targets if candidate.target_id != target.target_id]
        self.misses += 1
        return {
            "run_time_sec": float(run_time_sec),
            "event_type": "target_miss",
            "target_id": int(target.target_id),
            "lane": int(target.lane),
            "score": int(self.score),
            "details": "",
        }

    def _handle_control_events(self, snapshot: TrackingSnapshot) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        self.hold_active = bool(snapshot.jaw_hold_active)
        for detection in snapshot.detections:
            if not bool(detection.get("accepted", False)):
                continue
            action = str(detection.get("action", ""))
            if action == ACTION_LEFT:
                previous_lane = self.player_lane
                self.player_lane = max(0, self.player_lane - 1)
                if self.player_lane != previous_lane:
                    events.append(
                        {
                            "run_time_sec": float(snapshot.run_time_sec),
                            "event_type": "move_left",
                            "target_id": "",
                            "lane": int(self.player_lane),
                            "score": int(self.score),
                            "details": "",
                        }
                    )
            elif action == ACTION_RIGHT:
                previous_lane = self.player_lane
                self.player_lane = min(self.config.lane_count - 1, self.player_lane + 1)
                if self.player_lane != previous_lane:
                    events.append(
                        {
                            "run_time_sec": float(snapshot.run_time_sec),
                            "event_type": "move_right",
                            "target_id": "",
                            "lane": int(self.player_lane),
                            "score": int(self.score),
                            "details": "",
                        }
                    )
            elif action == ACTION_CLICK:
                self.click_flash_until_sec = float(snapshot.run_time_sec) + self.config.click_flash_sec
                lane_targets = sorted(
                    [target for target in self.targets if target.lane == self.player_lane],
                    key=lambda target: target.y_px,
                    reverse=True,
                )
                if lane_targets:
                    events.append(self._capture_target(lane_targets[0], float(snapshot.run_time_sec), "click_hit", 2))
                else:
                    events.append(
                        {
                            "run_time_sec": float(snapshot.run_time_sec),
                            "event_type": "click_miss",
                            "target_id": "",
                            "lane": int(self.player_lane),
                            "score": int(self.score),
                            "details": "",
                        }
                    )
            elif action == ACTION_HOLD_START:
                events.append(
                    {
                        "run_time_sec": float(snapshot.run_time_sec),
                        "event_type": "hold_start",
                        "target_id": "",
                        "lane": int(self.player_lane),
                        "score": int(self.score),
                        "details": "",
                    }
                )
            elif action == ACTION_HOLD_END:
                events.append(
                    {
                        "run_time_sec": float(snapshot.run_time_sec),
                        "event_type": "hold_end",
                        "target_id": "",
                        "lane": int(self.player_lane),
                        "score": int(self.score),
                        "details": "",
                    }
                )
        return events

    def update(self, snapshot: TrackingSnapshot) -> list[dict[str, Any]]:
        events = self._handle_control_events(snapshot)
        if self.last_run_time_sec is None:
            self.last_run_time_sec = float(snapshot.run_time_sec)
            return events

        dt_sec = max(0.0, float(snapshot.run_time_sec) - self.last_run_time_sec)
        self.last_run_time_sec = float(snapshot.run_time_sec)
        if dt_sec <= 0.0:
            return events

        while float(snapshot.run_time_sec) >= self.next_spawn_time_sec:
            events.append(self._spawn_target(float(snapshot.run_time_sec)))
            interval = self.config.spawn_interval_sec + self.rng.uniform(-self.config.spawn_jitter_sec, self.config.spawn_jitter_sec)
            self.next_spawn_time_sec += max(0.55, interval)

        for target in self.targets:
            target.y_px += target.speed_px_sec * dt_sec

        if self.hold_active:
            hold_targets = [
                target
                for target in self.targets
                if target.lane == self.player_lane and target.y_px >= self.config.hold_capture_y_px
            ]
            for target in hold_targets:
                events.append(self._capture_target(target, float(snapshot.run_time_sec), "hold_capture", 1))

        bottom_targets = sorted(self.targets, key=lambda target: target.y_px, reverse=True)
        for target in bottom_targets:
            if target.y_px < self.config.track_capture_y_px:
                continue
            if target.lane == self.player_lane:
                events.append(self._capture_target(target, float(snapshot.run_time_sec), "tracked_catch", 1))
            elif target.y_px >= self.config.canvas_height - 24.0:
                events.append(self._miss_target(target, float(snapshot.run_time_sec)))
        return events

    def render(self, canvas: Any, run_time_sec: float) -> None:
        canvas.delete("all")
        canvas.create_rectangle(0, 0, self.config.canvas_width, self.config.canvas_height, fill=GAME_BG, outline="")

        lane_width = self.config.canvas_width / self.config.lane_count
        for lane in range(self.config.lane_count + 1):
            x_pos = lane * lane_width
            canvas.create_line(x_pos, 0, x_pos, self.config.canvas_height, fill="#243244", width=2)
        canvas.create_line(0, self.config.track_capture_y_px, self.config.canvas_width, self.config.track_capture_y_px, fill="#1f2937", dash=(6, 6))

        if self.hold_active:
            left = self.player_lane * lane_width + 8
            right = (self.player_lane + 1) * lane_width - 8
            canvas.create_rectangle(left, 20, right, self.config.canvas_height - 60, fill=GAME_HOLD, stipple="gray25", outline="")

        if run_time_sec <= self.click_flash_until_sec:
            x_pos = self._lane_center_x(self.player_lane)
            canvas.create_oval(x_pos - 44, self.config.player_y_px - 44, x_pos + 44, self.config.player_y_px + 44, outline=GAME_CLICK, width=4)

        for target in self.targets:
            x_pos = self._lane_center_x(target.lane)
            radius = self.config.target_radius_px
            canvas.create_oval(x_pos - radius, target.y_px - radius, x_pos + radius, target.y_px + radius, fill=GAME_TARGET, outline="#f97316", width=2)

        player_x = self._lane_center_x(self.player_lane)
        canvas.create_rectangle(player_x - 34, self.config.player_y_px - 18, player_x + 34, self.config.player_y_px + 18, fill=GAME_ACCENT, outline="#67e8f9", width=2)
        canvas.create_text(player_x, self.config.player_y_px, text="TRACKER", fill=GAME_BG, font=("Helvetica", 10, "bold"))

        canvas.create_text(18, 18, anchor="nw", fill=GAME_TEXT, font=("Helvetica", 15, "bold"), text=f"Score {self.score}")
        canvas.create_text(18, 42, anchor="nw", fill=GAME_TEXT, font=("Helvetica", 11), text=f"Hits {self.hits}   Misses {self.misses}   Targets {len(self.targets)}")
        legend = "LEFT/RIGHT shift lanes  |  CLICK pops a target in-lane  |  CLICK + HOLD opens a capture beam"
        canvas.create_text(18, self.config.canvas_height - 20, anchor="sw", fill="#9ca3af", font=("Helvetica", 10), text=legend)


def _active_note_count(game: RhythmGame) -> int:
    return sum(1 for state in game.note_states.values() if state["status"] == "pending")


def _shift_chart(notes: list[RhythmNote], offset_sec: float) -> list[RhythmNote]:
    shifted: list[RhythmNote] = []
    for note in notes:
        shifted.append(
            RhythmNote(
                note_id=note.note_id,
                note_type=note.note_type,
                lane_index=note.lane_index,
                start_time_sec=float(note.start_time_sec + offset_sec),
                hit_time_sec=float(note.hit_time_sec + offset_sec),
                end_time_sec=float(note.end_time_sec + offset_sec),
                hold_required_sec=float(note.hold_required_sec),
            )
        )
    return shifted


def _render_session_screen(canvas: Any, width: int, height: int, title: str, subtitle: str, footer: str = "") -> None:
    canvas.delete("all")
    canvas.create_rectangle(0, 0, width, height, fill=GAME_BG, outline="")
    canvas.create_oval(-120, -80, 260, 240, fill="#0f1d35", outline="")
    canvas.create_oval(width - 280, height - 220, width + 40, height + 60, fill="#10263c", outline="")
    canvas.create_text(width / 2, 120, text=title, fill="#f8fafc", font=("Helvetica", 28, "bold"))
    canvas.create_text(width / 2, 200, text=subtitle, fill="#cbd5e1", font=("Helvetica", 16), width=width - 160)
    if footer:
        canvas.create_text(width / 2, height - 56, text=footer, fill="#67e8f9", font=("Helvetica", 13, "bold"), width=width - 120)


def _log_game_tick(
    *,
    logger: TrackingGameLogger,
    game: RhythmGame,
    snapshot: TrackingSnapshot,
    game_events: list[dict[str, Any]],
    last_logged_game_sample: int,
) -> int:
    if snapshot.sample_count != last_logged_game_sample or game_events:
        logger.log_game_trace(
            {
                "run_time_sec": float(snapshot.run_time_sec),
                "mode": snapshot.mode,
                "decoded_state": snapshot.decoded_state,
                "score": int(game.score),
                "hits": int(game.hits),
                "misses": int(game.misses),
                "pending_notes": int(_active_note_count(game)),
                "jaw_hold_active": bool(snapshot.jaw_hold_active),
                "jaw_enabled": bool(snapshot.jaw_enabled),
                "hand_enabled": bool(snapshot.hand_enabled),
                "game_completed": bool(game.completed),
            }
        )
        for event in game_events:
            logger.log_game_event(event)
        return snapshot.sample_count
    return last_logged_game_sample


def run_headless_replay_smoke(
    *,
    config: BCITrackingGameConfig,
    replay_csv: Path,
    replay_speed: float,
    calibration_sec: float | None = None,
) -> dict[str, Any]:
    runtime_config = config if calibration_sec is None else BCITrackingGameConfig(
        jaw_artifact_path=config.jaw_artifact_path,
        direction_artifact_path=config.direction_artifact_path,
        output_root=config.output_root,
        replay_files=config.replay_files,
        calibration_sec=float(calibration_sec),
        timer_interval_ms=config.timer_interval_ms,
        replay_step_sec=config.replay_step_sec,
        hand_event_confirmation_sec=config.hand_event_confirmation_sec,
        hand_event_lookback_sec=config.hand_event_lookback_sec,
        hand_action_latch_sec=config.hand_action_latch_sec,
        hand_switch_cooldown_sec=config.hand_switch_cooldown_sec,
        jaw_hold_probability_threshold=config.jaw_hold_probability_threshold,
        jaw_hold_onset_sec=config.jaw_hold_onset_sec,
        jaw_hold_release_sec=config.jaw_hold_release_sec,
        stale_stream_warning_sec=config.stale_stream_warning_sec,
        baseline_sec=config.baseline_sec,
        guided_prep_sec=config.guided_prep_sec,
        guided_rest_sec=config.guided_rest_sec,
        guided_tap_base_sec=config.guided_tap_base_sec,
        guided_tap_per_count_sec=config.guided_tap_per_count_sec,
        guided_hold_sec=config.guided_hold_sec,
        guided_hold_trials=config.guided_hold_trials,
        game_note_cycles=config.game_note_cycles,
    )
    controller = ReplayGameController(config=runtime_config, replay_csv=replay_csv.resolve(), speed=float(replay_speed))
    controller.set_playing(True)
    game_config = RhythmGameConfig()
    game = RhythmGame(_shift_chart(build_simple_rhythm_chart(config=game_config, cycles=runtime_config.game_note_cycles), 0.0), game_config)
    last_logged_game_sample = -1
    last_snapshot: TrackingSnapshot | None = None

    while not controller.finished():
        snapshot = controller.tick()
        game_events = game.update(snapshot)
        last_logged_game_sample = _log_game_tick(
            logger=controller.logger,
            game=game,
            snapshot=snapshot,
            game_events=game_events,
            last_logged_game_sample=last_logged_game_sample,
        )
        last_snapshot = snapshot

    if last_snapshot is None:
        last_snapshot = controller.tick()
        game_events = game.update(last_snapshot)
        last_logged_game_sample = _log_game_tick(
            logger=controller.logger,
            game=game,
            snapshot=last_snapshot,
            game_events=game_events,
            last_logged_game_sample=last_logged_game_sample,
        )

    controller.logger.metadata["headless_smoke"] = True
    controller.logger.metadata["final_score"] = int(game.score)
    controller.logger.metadata["final_hits"] = int(game.hits)
    controller.logger.metadata["final_misses"] = int(game.misses)
    paths = controller.save()
    return {
        "paths": paths,
        "final_score": int(game.score),
        "final_hits": int(game.hits),
        "final_misses": int(game.misses),
        "final_state": last_snapshot.decoded_state,
    }


class BCITrackingGameApp:
    def __init__(
        self,
        *,
        config: BCITrackingGameConfig,
        initial_mode: str,
        replay_csv: Path | None,
        replay_speed: float,
        board: str,
        serial_port: str,
        playback_file: Path | None,
        max_live_sec: float | None,
        autostart: bool = False,
        autoplay_replay: bool = False,
        auto_close_sec: float | None = None,
    ) -> None:
        try:
            import tkinter as tk
            from tkinter import filedialog, ttk
        except ModuleNotFoundError as exc:
            raise RuntimeError("Tkinter is required for the BCI tracking game.") from exc

        self.tk = tk
        self.ttk = ttk
        self.filedialog = filedialog
        self.config = config
        self.board = board
        self.serial_port = serial_port
        self.playback_file = playback_file
        self.max_live_sec = max_live_sec
        self.controller: ReplayGameController | LiveGameController | None = None
        self.last_saved_paths: dict[str, Path] | None = None
        self.game_config = RhythmGameConfig()
        self.game = RhythmGame(_shift_chart(build_simple_rhythm_chart(config=self.game_config, cycles=config.game_note_cycles), 0.0), self.game_config)
        self.last_logged_game_sample = -1
        self.autostart = bool(autostart)
        self.autoplay_replay = bool(autoplay_replay)
        self.auto_close_deadline = time.monotonic() + float(auto_close_sec) if auto_close_sec is not None else None
        self._closing = False
        self._feedback_prompt_open = False
        self.guided_review_recorded = False
        self.gameplay_review_recorded = False
        self.participant_info: dict[str, Any] = {}
        self.session_feedback: dict[str, Any] = {
            "guided_session_review": {},
            "gameplay_review": {},
        }
        self.session_phase = SESSION_PHASE_PARTICIPANT if initial_mode == LIVE else SESSION_PHASE_GAMEPLAY
        self.phase_started_run_time_sec: float | None = None
        self.protocol_steps: list[GuidedProtocolStep] = []
        self.protocol_step_index = 0
        self.protocol_subphase = "idle"
        self.protocol_subphase_started_sec: float | None = None
        self.active_protocol_row: dict[str, Any] | None = None
        self.ready_for_game = False
        self.live_game_started = False
        self.adaptation_summary: dict[str, Any] = {}
        self.busy_reason = ""
        self._pending_live_thread: threading.Thread | None = None
        self._pending_live_result: tuple[str, Any] | None = None

        self.root = tk.Tk()
        self.root.title("BCI Tracking Rhythm Game")
        self.root.geometry("1320x980")
        self.root.configure(bg=GAME_BG)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        replay_default = (replay_csv or (config.replay_files[0] if config.replay_files else Path())).resolve()
        self.mode_var = tk.StringVar(value=initial_mode)
        self.replay_file_var = tk.StringVar(value=str(replay_default))
        self.speed_var = tk.StringVar(value=f"{replay_speed:g}x")
        self.mode_banner_var = tk.StringVar(value=f"Mode: {initial_mode}")
        self.phase_var = tk.StringVar(value=f"Phase: {self.session_phase}")
        self.decoded_state_var = tk.StringVar(value=HAND_NEUTRAL)
        self.status_var = tk.StringVar(value="ready")
        self.instruction_var = tk.StringVar(value="Choose a mode and start.")
        self.truth_var = tk.StringVar(value="Replay truth: n/a")
        self.source_var = tk.StringVar(value="Source: none")
        self.runtime_var = tk.StringVar(value="t=0.0s | samples=0")
        self.readiness_var = tk.StringVar(value="Jaw ready | Hand waiting")
        self.score_var = tk.StringVar(value="Score 0 | Hits 0 | Misses 0")
        self.countdown_var = tk.StringVar(value="")
        self.busy_var = tk.StringVar(value="")
        self.jaw_status_var = tk.StringVar(value="Jaw: idle")
        self.hand_status_var = tk.StringVar(value="Hand: neutral")

        self.left_conf_var = tk.DoubleVar(value=0.0)
        self.right_conf_var = tk.DoubleVar(value=0.0)
        self.jaw_conf_var = tk.DoubleVar(value=0.0)
        self.jaw_onset_var = tk.DoubleVar(value=0.0)
        self.jaw_active_var = tk.DoubleVar(value=0.0)

        self._build_ui()
        self._refresh_controls()
        if self.autostart:
            self.root.after(150, self._handle_primary_action)
        self.root.after(config.timer_interval_ms, self._tick)

    def _build_ui(self) -> None:
        header = self.ttk.Frame(self.root, padding=14)
        header.pack(fill="x")

        self.tk.Label(header, textvariable=self.mode_banner_var, bg=GAME_BG, fg="#67e8f9", font=("Helvetica", 14, "bold")).pack(anchor="w")
        self.tk.Label(header, textvariable=self.phase_var, bg=GAME_BG, fg="#fbbf24", font=("Helvetica", 13, "bold")).pack(anchor="w", pady=(3, 0))
        self.tk.Label(header, textvariable=self.decoded_state_var, bg=GAME_BG, fg="white", font=("Helvetica", 28, "bold")).pack(anchor="w", pady=(4, 0))
        self.tk.Label(header, textvariable=self.status_var, bg=GAME_BG, fg="#d1d5db", font=("Helvetica", 13)).pack(anchor="w", pady=(6, 0))

        controls = self.ttk.LabelFrame(self.root, text="Session Controls", padding=12)
        controls.pack(fill="x", padx=12, pady=(0, 10))

        self.ttk.Label(controls, text="Mode").grid(row=0, column=0, sticky="w")
        self.mode_combo = self.ttk.Combobox(controls, textvariable=self.mode_var, state="readonly", values=[LIVE, REPLAY], width=10)
        self.mode_combo.grid(row=0, column=1, sticky="w", padx=(6, 14))
        self.mode_combo.bind("<<ComboboxSelected>>", lambda _event: self._switch_mode())

        self.replay_label = self.ttk.Label(controls, text="Replay CSV")
        self.replay_label.grid(row=0, column=2, sticky="w")
        replay_values = [str(path.resolve()) for path in self.config.replay_files]
        self.replay_combo = self.ttk.Combobox(controls, textvariable=self.replay_file_var, state="readonly", values=replay_values, width=56)
        self.replay_combo.grid(row=0, column=3, sticky="we", padx=(6, 8))
        self.browse_button = self.ttk.Button(controls, text="Browse", command=self._browse_replay_file)
        self.browse_button.grid(row=0, column=4, padx=(0, 16))

        self.speed_label = self.ttk.Label(controls, text="Replay speed")
        self.speed_label.grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.speed_combo = self.ttk.Combobox(controls, textvariable=self.speed_var, state="readonly", values=["0.5x", "1x", "2x", "4x"], width=10)
        self.speed_combo.grid(row=1, column=1, sticky="w", padx=(6, 14), pady=(8, 0))
        self.speed_combo.bind("<<ComboboxSelected>>", lambda _event: self._apply_speed())

        self.start_button = self.ttk.Button(controls, text="Start Session", command=self._handle_primary_action)
        self.start_button.grid(row=1, column=2, padx=(0, 6), pady=(8, 0))
        self.play_pause_button = self.ttk.Button(controls, text="Play", command=self._toggle_replay_play)
        self.play_pause_button.grid(row=1, column=3, sticky="w", pady=(8, 0))
        self.restart_button = self.ttk.Button(controls, text="Restart", command=self._restart_replay)
        self.restart_button.grid(row=1, column=4, padx=(8, 8), pady=(8, 0), sticky="w")
        self.start_game_button = self.ttk.Button(controls, text="Start Game", command=self._start_gameplay, state="disabled")
        self.start_game_button.grid(row=1, column=5, padx=(0, 8), pady=(8, 0), sticky="w")
        self.stop_button = self.ttk.Button(controls, text="Stop", command=self._stop_controller)
        self.stop_button.grid(row=1, column=6, pady=(8, 0), sticky="w")
        self.start_game_button.grid_remove()

        self.busy_bar = self.ttk.Progressbar(controls, mode="indeterminate", length=220)
        self.busy_bar.grid(row=2, column=0, columnspan=3, sticky="we", pady=(10, 0))
        self.busy_label = self.ttk.Label(controls, textvariable=self.busy_var)
        self.busy_label.grid(row=2, column=3, columnspan=4, sticky="w", pady=(10, 0))
        self.busy_bar.grid_remove()
        self.busy_label.grid_remove()
        controls.columnconfigure(3, weight=1)

        body = self.ttk.Frame(self.root, padding=(12, 0, 12, 0))
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        sidebar = self.ttk.Frame(body)
        sidebar.grid(row=0, column=0, sticky="nsw", padx=(0, 12))

        session_frame = self.ttk.LabelFrame(sidebar, text="Session", padding=12)
        session_frame.pack(fill="x", pady=(0, 10))
        self.ttk.Label(session_frame, textvariable=self.source_var, wraplength=320).pack(anchor="w")
        self.ttk.Label(session_frame, textvariable=self.truth_var, wraplength=320).pack(anchor="w", pady=(6, 0))
        self.ttk.Label(session_frame, textvariable=self.runtime_var).pack(anchor="w", pady=(6, 0))
        self.ttk.Label(session_frame, textvariable=self.readiness_var, wraplength=320).pack(anchor="w", pady=(6, 0))
        self.ttk.Label(session_frame, textvariable=self.score_var).pack(anchor="w", pady=(6, 0))
        self.ttk.Label(session_frame, textvariable=self.instruction_var, wraplength=320).pack(anchor="w", pady=(10, 0))
        self.ttk.Label(session_frame, textvariable=self.countdown_var, wraplength=320).pack(anchor="w", pady=(6, 0))

        jaw_frame = self.ttk.LabelFrame(sidebar, text="Jaw Branch", padding=12)
        jaw_frame.pack(fill="x", pady=(0, 10))
        self.ttk.Label(jaw_frame, textvariable=self.jaw_status_var, wraplength=320).pack(anchor="w")
        self._build_progress_block(jaw_frame, "Clench", self.jaw_conf_var)
        self._build_progress_block(jaw_frame, "Onset", self.jaw_onset_var)
        self._build_progress_block(jaw_frame, "Active", self.jaw_active_var)

        hand_frame = self.ttk.LabelFrame(sidebar, text="Hand Branch", padding=12)
        hand_frame.pack(fill="x", pady=(0, 10))
        self.ttk.Label(hand_frame, textvariable=self.hand_status_var, wraplength=320).pack(anchor="w")
        self._build_progress_block(hand_frame, "LEFT", self.left_conf_var)
        self._build_progress_block(hand_frame, "RIGHT", self.right_conf_var)

        game_frame = self.ttk.LabelFrame(body, text="Closed-Loop Game", padding=12)
        game_frame.grid(row=0, column=1, sticky="nsew")
        self.game_canvas = self.tk.Canvas(
            game_frame,
            width=self.game_config.canvas_width,
            height=self.game_config.canvas_height,
            bg=GAME_BG,
            highlightthickness=0,
        )
        self.game_canvas.pack(fill="both", expand=True)

        log_frame = self.ttk.LabelFrame(self.root, text="Event History", padding=12)
        log_frame.pack(fill="both", expand=False, padx=12, pady=(10, 12))
        self.info_text = self.tk.Text(log_frame, height=10, wrap="word", bg="#0f172a", fg="#e2e8f0", insertbackground="white")
        self.info_text.pack(fill="both", expand=True)
        self.info_text.insert("1.0", "Closed-loop BCI rhythm game ready. Logs will be written under analysis/outputs/bci_tracking_game/\n")
        self.info_text.configure(state="disabled")

        _render_session_screen(
            self.game_canvas,
            self.game_config.canvas_width,
            self.game_config.canvas_height,
            "BCI Rhythm Game",
            "Start a replay to debug the decoder, or start a live session to run baseline, guided collection, adaptation, and gameplay.",
            "LIVE sessions collect data with markers, protocol truth, runtime decisions, and game outcomes.",
        )

    def _build_progress_block(self, parent: Any, label: str, variable: Any) -> None:
        row = self.ttk.Frame(parent)
        row.pack(fill="x", pady=(8, 0))
        self.ttk.Label(row, text=label, width=8).pack(side="left")
        bar = self.ttk.Progressbar(row, orient="horizontal", mode="determinate", maximum=1.0, variable=variable, length=180)
        bar.pack(side="left", fill="x", expand=True, padx=(8, 8))
        value_label = self.ttk.Label(row, textvariable=self.tk.StringVar(value="0%"))
        value_label.pack(side="left")
        variable.trace_add("write", lambda *_args, var=variable, label_var=value_label: label_var.configure(text=f"{var.get() * 100:.0f}%"))

    def _append_info(self, message: str) -> None:
        self.info_text.configure(state="normal")
        self.info_text.insert("end", message.rstrip() + "\n")
        line_count = int(self.info_text.index("end-1c").split(".")[0])
        if line_count > INFO_HISTORY_LIMIT:
            trim_lines = line_count - INFO_HISTORY_LIMIT
            self.info_text.delete("1.0", f"{trim_lines + 1}.0")
        self.info_text.see("end")
        self.info_text.configure(state="disabled")

    def _set_busy(self, reason: str = "") -> None:
        self.busy_reason = reason.strip()
        if self.busy_reason:
            self.busy_var.set(self.busy_reason)
            self.busy_bar.grid()
            self.busy_label.grid()
            self.busy_bar.start(10)
        else:
            self.busy_var.set("")
            self.busy_bar.stop()
            self.busy_bar.grid_remove()
            self.busy_label.grid_remove()
        self._refresh_controls()

    def _refresh_controls(self) -> None:
        is_replay = self.mode_var.get() == REPLAY
        replay_widgets = (
            self.replay_label,
            self.replay_combo,
            self.browse_button,
            self.speed_label,
            self.speed_combo,
            self.play_pause_button,
            self.restart_button,
        )
        for widget in replay_widgets:
            if is_replay:
                widget.grid()
            else:
                widget.grid_remove()

        live_running = (not is_replay) and self.controller is not None
        primary_text = "Load Replay" if is_replay else "Begin Guided Session"
        if not is_replay and self.ready_for_game and self.controller is not None:
            primary_text = "Review + Start Game"
        elif self.busy_reason:
            primary_text = "Working..."
        elif live_running:
            primary_text = "Session Running"
        self.start_button.configure(text=primary_text)

        can_start = not self.busy_reason and (
            is_replay or self.controller is None or (self.ready_for_game and self.controller is not None)
        )
        self.start_button.configure(state="normal" if can_start else "disabled")
        self.mode_combo.configure(state="readonly" if not self.busy_reason and self.controller is None else "disabled")
        self.play_pause_button.configure(state="normal" if is_replay and isinstance(self.controller, ReplayGameController) else "disabled")
        self.restart_button.configure(state="normal" if is_replay and not self.busy_reason else "disabled")
        self.stop_button.configure(state="normal" if (self.controller is not None and not self.busy_reason) else "disabled")
        self.start_game_button.configure(state="disabled")

    def _switch_mode(self) -> None:
        if self.busy_reason:
            return
        self._stop_controller(prompt_feedback=False)
        self.session_phase = SESSION_PHASE_PARTICIPANT if self.mode_var.get() == LIVE else SESSION_PHASE_GAMEPLAY
        self.phase_var.set(f"Phase: {self.session_phase}")
        self.ready_for_game = False
        self._refresh_controls()
        self.mode_banner_var.set(f"Mode: {self.mode_var.get()}")

    def _browse_replay_file(self) -> None:
        selected = self.filedialog.askopenfilename(
            title="Select replay CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=str(self.config.replay_files[0].resolve().parent if self.config.replay_files else Path.cwd()),
        )
        if selected:
            self.replay_file_var.set(str(Path(selected).resolve()))

    def _apply_speed(self) -> None:
        if isinstance(self.controller, ReplayGameController):
            self.controller.set_speed(float(self.speed_var.get().rstrip("x")))

    def _prompt_participant_info(self) -> dict[str, Any] | None:
        dialog = self.tk.Toplevel(self.root)
        dialog.title("Participant Prompt")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)
        # `dialog.lift()` and `attributes("-topmost")` can cause an Objective-C
        # NSInvalidArgumentException on macOS Sonoma/Sequoia.
        # dialog.lift()
        # try:
        #     dialog.attributes("-topmost", True)
        #     dialog.after(250, lambda: dialog.attributes("-topmost", False))
        # except self.tk.TclError:
        #     pass

        result: dict[str, Any] = {}
        session_label_var = self.tk.StringVar()

        self.ttk.Label(dialog, text="Session label").grid(row=0, column=0, sticky="w", padx=12, pady=(12, 4))
        session_entry = self.ttk.Entry(dialog, textvariable=session_label_var, width=36)
        session_entry.grid(row=1, column=0, padx=12, pady=(0, 8))

        def submit() -> None:
            label = session_label_var.get().strip()
            if not label:
                return
            result.update(
                {
                    "session_label": label,
                    "entered_at_wall_time": time.time(),
                }
            )
            dialog.destroy()

        def cancel() -> None:
            dialog.destroy()

        button_row = self.ttk.Frame(dialog)
        button_row.grid(row=2, column=0, sticky="e", padx=12, pady=(0, 12))
        self.ttk.Button(button_row, text="Cancel", command=cancel).pack(side="right", padx=(8, 0))
        self.ttk.Button(button_row, text="Continue", command=submit).pack(side="right")

        session_entry.focus_set()
        dialog.wait_window()
        return result or None

    def _prompt_feedback(
        self,
        *,
        title: str,
        intro_text: str,
        submit_text: str,
    ) -> dict[str, Any] | None:
        if self._feedback_prompt_open:
            return None
        self._feedback_prompt_open = True
        dialog = self.tk.Toplevel(self.root)
        dialog.title(title)
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)
        # `dialog.lift()` and `attributes("-topmost")` can cause an Objective-C
        # NSInvalidArgumentException on macOS Sonoma/Sequoia.
        # dialog.lift()
        # try:
        #     dialog.attributes("-topmost", True)
        #     dialog.after(250, lambda: dialog.attributes("-topmost", False))
        # except self.tk.TclError:
        #     pass
        result: dict[str, Any] = {}
        rating_vars = {field: self.tk.StringVar(value="3") for field in FEEDBACK_FIELDS}
        notes_text = self.tk.Text(dialog, width=46, height=6)

        self.ttk.Label(dialog, text=intro_text, wraplength=420, justify="left").grid(
            row=0, column=0, columnspan=2, sticky="w", padx=12, pady=(12, 6)
        )

        labels = {
            "click_control": "Click control",
            "hold_control": "Hold control",
            "left_control": "Left control",
            "right_control": "Right control",
            "overall_control": "Overall control",
        }
        for row_index, field in enumerate(FEEDBACK_FIELDS):
            dialog_row = row_index + 1
            self.ttk.Label(dialog, text=labels[field]).grid(row=dialog_row, column=0, sticky="w", padx=12, pady=(10 if row_index == 0 else 4, 0))
            self.ttk.Combobox(dialog, textvariable=rating_vars[field], state="readonly", values=["1", "2", "3", "4", "5"], width=8).grid(row=dialog_row, column=1, sticky="w", padx=(10, 12), pady=(10 if row_index == 0 else 4, 0))

        notes_row = len(FEEDBACK_FIELDS) + 1
        self.ttk.Label(dialog, text="Free-text notes").grid(row=notes_row, column=0, sticky="w", padx=12, pady=(10, 4))
        notes_text.grid(row=notes_row + 1, column=0, columnspan=2, padx=12, pady=(0, 10))

        def submit() -> None:
            result.update({field: int(var.get()) for field, var in rating_vars.items()})
            result["free_text_notes"] = notes_text.get("1.0", "end").strip()
            result["submitted_at_wall_time"] = time.time()
            dialog.destroy()

        def skip() -> None:
            dialog.destroy()

        button_row = self.ttk.Frame(dialog)
        button_row.grid(row=notes_row + 2, column=0, columnspan=2, sticky="e", padx=12, pady=(0, 12))
        self.ttk.Button(button_row, text="Skip", command=skip).pack(side="right", padx=(8, 0))
        self.ttk.Button(button_row, text=submit_text, command=submit).pack(side="right")

        dialog.wait_window()
        self._feedback_prompt_open = False
        return result or None

    def _build_live_protocol(self) -> list[GuidedProtocolStep]:
        return build_guided_protocol(
            prep_sec=self.config.guided_prep_sec,
            rest_sec=self.config.guided_rest_sec,
            tap_base_sec=self.config.guided_tap_base_sec,
            tap_per_count_sec=self.config.guided_tap_per_count_sec,
            hold_duration_sec=self.config.guided_hold_sec,
            hold_trials=self.config.guided_hold_trials,
        )

    def _base_chart(self) -> list[RhythmNote]:
        return build_simple_rhythm_chart(config=self.game_config, cycles=self.config.game_note_cycles)

    def _handle_primary_action(self) -> None:
        if self.busy_reason:
            return
        if self.mode_var.get() == LIVE and self.ready_for_game and self.controller is not None:
            self._finalize_guided_review()
            self._start_gameplay()
            return
        self._start_selected_mode()

    def _start_selected_mode(self) -> None:
        if self.busy_reason:
            return
        self._stop_controller(prompt_feedback=False)
        self.last_logged_game_sample = -1
        self.guided_review_recorded = False
        self.gameplay_review_recorded = False
        self.participant_info = {}
        self.session_feedback = {
            "guided_session_review": {},
            "gameplay_review": {},
        }
        self.ready_for_game = False
        mode = self.mode_var.get()
        if mode == REPLAY:
            self._set_busy("Loading replay session...")
            replay_csv = Path(self.replay_file_var.get()).resolve()
            self.controller = ReplayGameController(
                config=self.config,
                replay_csv=replay_csv,
                speed=float(self.speed_var.get().rstrip("x")),
            )
            self.controller.set_playing(bool(self.autoplay_replay))
            self.session_phase = SESSION_PHASE_GAMEPLAY
            self.phase_var.set(f"Phase: {self.session_phase}")
            self.game = RhythmGame(_shift_chart(self._base_chart(), 0.0), self.game_config)
            self.play_pause_button.configure(text="Pause" if self.autoplay_replay else "Play", state="normal")
            self.instruction_var.set("Replay mode drives the rhythm chart directly from the selected CSV.")
            self.countdown_var.set("")
            self._append_info(f"Loaded replay: {replay_csv.name}")
            self._set_busy("")
        else:
            participant_info = self._prompt_participant_info()
            if participant_info is None:
                self._append_info("Live session canceled before baseline.")
                return
            self.participant_info = participant_info
            session_tag = f"{self.board}_{participant_info['session_label']}"
            self._set_busy("Connecting to live board...")
            self.instruction_var.set("Connecting to the live board. Please wait and avoid clicking again.")
            self.countdown_var.set("Opening BrainFlow session and starting stream...")
            self._pending_live_result = None
            self._pending_live_thread = threading.Thread(
                target=self._create_live_controller_async,
                args=(participant_info, session_tag),
                daemon=True,
            )
            self._pending_live_thread.start()

        self.mode_banner_var.set(f"Mode: {mode}")
        if self.controller is not None:
            branch_text = (
                f"Branches active: hand={'yes' if self.controller.decoder.hand_enabled else 'no'} | "
                f"jaw={'yes' if self.controller.decoder.jaw_enabled else 'no'}"
            )
            self._append_info(branch_text)
        self._refresh_controls()

    def _create_live_controller_async(self, participant_info: dict[str, Any], session_tag: str) -> None:
        try:
            controller = LiveGameController(
                config=self.config,
                board=self.board,
                serial_port=self.serial_port,
                playback_file=self.playback_file,
                max_live_sec=self.max_live_sec,
                run_tag=session_tag,
                extra_metadata={
                    "session_mode": "guided_closed_loop",
                    "session_label": participant_info["session_label"],
                    "marker_semantics": "1/1=LEFT, 2/2=RIGHT, 3/3=JAW tap, 4/4=HOLD",
                },
            )
        except Exception as exc:
            self._pending_live_result = ("error", str(exc))
            return
        self._pending_live_result = ("ok", controller, participant_info)

    def _finish_live_start(self, controller: LiveGameController, participant_info: dict[str, Any]) -> None:
        self.controller = controller
        self.controller.logger.set_participant(participant_info)
        self.session_phase = SESSION_PHASE_BASELINE
        self.phase_var.set(f"Phase: {self.session_phase}")
        self.protocol_steps = self._build_live_protocol()
        self.protocol_step_index = 0
        self.protocol_subphase = "idle"
        self.protocol_subphase_started_sec = None
        self.phase_started_run_time_sec = None
        self.active_protocol_row = None
        self.adaptation_summary = {}
        self.game = RhythmGame(_shift_chart(self._base_chart(), 0.0), self.game_config)
        self.instruction_var.set("Baseline in progress. Stay relaxed and still.")
        self.countdown_var.set("")
        self._append_info(f"Live session started for {participant_info['session_label']}.")
        branch_text = (
            f"Branches active: hand={'yes' if self.controller.decoder.hand_enabled else 'no'} | "
            f"jaw={'yes' if self.controller.decoder.jaw_enabled else 'no'}"
        )
        self._append_info(branch_text)
        self._set_busy("")
        self._refresh_controls()

    def _toggle_replay_play(self) -> None:
        if not isinstance(self.controller, ReplayGameController):
            return
        self.controller.set_playing(not self.controller.playing)
        self.play_pause_button.configure(text="Pause" if self.controller.playing else "Play")

    def _restart_replay(self) -> None:
        if self.mode_var.get() != REPLAY:
            return
        was_playing = isinstance(self.controller, ReplayGameController) and self.controller.playing
        self.autoplay_replay = was_playing or self.autoplay_replay
        self._start_selected_mode()

    def _store_session_feedback(self) -> None:
        if self.controller is None:
            return
        self.controller.logger.set_feedback(dict(self.session_feedback))

    def _finalize_guided_review(self) -> None:
        if self.guided_review_recorded or self.controller is None:
            return
        feedback = self._prompt_feedback(
            title="Guided Session Review",
            intro_text="The scripted testing/calibration phase is complete. Rate how the controls felt during the guided collection before gameplay starts.",
            submit_text="Save Testing Review",
        )
        payload = feedback or {field: None for field in FEEDBACK_FIELDS} | {"free_text_notes": ""}
        payload["submitted"] = bool(feedback)
        self.session_feedback["guided_session_review"] = payload
        self.guided_review_recorded = True
        self._store_session_feedback()
        if feedback:
            self._append_info("Saved guided-session review feedback.")

    def _finalize_gameplay_feedback(self) -> None:
        if self.gameplay_review_recorded or self.controller is None:
            return
        feedback = self._prompt_feedback(
            title="Gameplay Review",
            intro_text="Gameplay is complete. Rate how the controls felt during the game itself.",
            submit_text="Save Gameplay Review",
        )
        payload = feedback or {field: None for field in FEEDBACK_FIELDS} | {"free_text_notes": ""}
        payload["submitted"] = bool(feedback)
        self.session_feedback["gameplay_review"] = payload
        self.gameplay_review_recorded = True
        self._store_session_feedback()
        if feedback:
            self._append_info("Saved gameplay review feedback.")

    def _save_controller_logs(self) -> None:
        if self.controller is None:
            return
        self.last_saved_paths = self.controller.save()
        if self.last_saved_paths is not None:
            self._append_info(f"Saved logs to {self.last_saved_paths['metadata_path'].parent}")

    def _stop_controller(self, prompt_feedback: bool = True) -> None:
        if self.controller is None:
            return
        if prompt_feedback and self.mode_var.get() == LIVE:
            if self.session_phase in {SESSION_PHASE_READY, SESSION_PHASE_GAMEPLAY, SESSION_PHASE_REVIEW}:
                self._finalize_guided_review()
            if self.session_phase in {SESSION_PHASE_GAMEPLAY, SESSION_PHASE_REVIEW}:
                self._finalize_gameplay_feedback()
        self._save_controller_logs()
        self.controller = None
        self.ready_for_game = False
        self.play_pause_button.configure(text="Play", state="disabled")
        self.start_game_button.configure(state="disabled")
        self._refresh_controls()

    def _log_game_tick(self, snapshot: TrackingSnapshot, logger: TrackingGameLogger, game_events: list[dict[str, Any]]) -> None:
        self.last_logged_game_sample = _log_game_tick(
            logger=logger,
            game=self.game,
            snapshot=snapshot,
            game_events=game_events,
            last_logged_game_sample=self.last_logged_game_sample,
        )

    def _format_readiness(self, snapshot: TrackingSnapshot) -> str:
        hand_state = "ready" if snapshot.hand_ready else ("disabled" if not snapshot.hand_enabled else f"calibrating {snapshot.calibration_remaining_sec:.1f}s")
        jaw_state = "ready" if snapshot.jaw_enabled else "disabled"
        stale = " | stream stale" if snapshot.stale_stream else ""
        return f"Jaw {jaw_state} | Hand {hand_state}{stale}"

    def _render_phase_canvas(self, snapshot: TrackingSnapshot | None) -> None:
        subtitle = self.instruction_var.get()
        footer = self.countdown_var.get()
        if self.session_phase == SESSION_PHASE_GAMEPLAY and snapshot is not None:
            self.game.render(self.game_canvas, snapshot.run_time_sec, "Closed-Loop Rhythm Game", "Single-note chart. Jaw hold is the only sustained note.")
        elif self.session_phase == SESSION_PHASE_READY:
            _render_session_screen(
                self.game_canvas,
                self.game_config.canvas_width,
                self.game_config.canvas_height,
                "Session Ready",
                subtitle,
                "Review the phase summary, then press Start Game when you want to begin the chart.",
            )
        else:
            _render_session_screen(
                self.game_canvas,
                self.game_config.canvas_width,
                self.game_config.canvas_height,
                self.session_phase,
                subtitle,
                footer,
            )

    def _apply_snapshot(self, snapshot: TrackingSnapshot, game_events: list[dict[str, Any]]) -> None:
        self.decoded_state_var.set(snapshot.decoded_state)
        self.status_var.set(snapshot.status_text)
        self.source_var.set(f"Source: {snapshot.source_name}")
        if snapshot.mode == REPLAY:
            self.truth_var.set(f"Replay truth: {snapshot.ground_truth_state}")
        else:
            self.truth_var.set(self.instruction_var.get())
        self.runtime_var.set(f"t={snapshot.run_time_sec:0.2f}s | samples={snapshot.sample_count}")
        self.readiness_var.set(self._format_readiness(snapshot))
        self.score_var.set(f"Score {self.game.score} | Hits {self.game.hits} | Misses {self.game.misses}")
        self.jaw_status_var.set(
            f"Jaw: {'disabled' if not snapshot.jaw_enabled else snapshot.jaw_event_label} | "
            f"{'hold active' if snapshot.jaw_hold_active else 'hold idle'}"
        )
        self.hand_status_var.set(
            f"Hand: {'disabled' if not snapshot.hand_enabled else snapshot.hand_state} | "
            f"pred={snapshot.hand_prediction} | margin={snapshot.hand_margin:.2f}"
        )
        self.left_conf_var.set(snapshot.hand_left_confidence)
        self.right_conf_var.set(snapshot.hand_right_confidence)
        self.jaw_conf_var.set(snapshot.jaw_confidence)
        self.jaw_onset_var.set(snapshot.jaw_onset_confidence)
        self.jaw_active_var.set(snapshot.jaw_active_confidence)

        for message in snapshot.messages:
            self._append_info(f"[control] {message}")
        for event in snapshot.detections:
            if bool(event.get("accepted", False)):
                self._append_info(f"[decoded] {event['action']} @ {float(event['timestamp_sec']):.2f}s")
        for event in game_events:
            self._append_info(f"[game] {str(event['event_type']).replace('_', ' ')} | score {event['score']}")

        self._render_phase_canvas(snapshot)

    def _finalize_protocol_row(self, snapshot: TrackingSnapshot) -> None:
        if self.active_protocol_row is None or self.controller is None:
            return
        self.active_protocol_row["end_run_time_sec"] = float(snapshot.run_time_sec)
        self.active_protocol_row["end_sample"] = int(snapshot.sample_count)
        self.active_protocol_row["actual_duration_sec"] = float(snapshot.run_time_sec - float(self.active_protocol_row["start_run_time_sec"]))
        self.controller.logger.log_protocol_row(self.active_protocol_row)
        self.active_protocol_row = None

    def _enter_phase(self, phase_name: str, snapshot: TrackingSnapshot | None) -> None:
        self.session_phase = phase_name
        self.phase_var.set(f"Phase: {phase_name}")
        if snapshot is not None:
            self.phase_started_run_time_sec = float(snapshot.run_time_sec)
        else:
            self.phase_started_run_time_sec = None
        if self.controller is not None:
            self.controller.logger.log_phase_row(
                {
                    "mode": self.mode_var.get(),
                    "phase_name": phase_name,
                    "run_time_sec": float(snapshot.run_time_sec) if snapshot is not None else "",
                    "sample_count": int(snapshot.sample_count) if snapshot is not None else "",
                    "wall_time_utc": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "session_label": self.participant_info.get("session_label", ""),
                }
            )

    def _advance_live_protocol(self, snapshot: TrackingSnapshot) -> None:
        if self.controller is None or not isinstance(self.controller, LiveGameController):
            return

        if self.session_phase == SESSION_PHASE_BASELINE:
            if self.phase_started_run_time_sec is None:
                self._enter_phase(SESSION_PHASE_BASELINE, snapshot)
                self.active_protocol_row = {
                    "phase_name": SESSION_PHASE_BASELINE,
                    "protocol_label": "BASELINE",
                    "display_text": "BASELINE",
                    "instruction_text": "Relax and stay still.",
                    "marker_code": "",
                    "expected_count": "",
                    "start_run_time_sec": float(snapshot.run_time_sec),
                    "start_sample": int(snapshot.sample_count),
                }
            remaining = max(0.0, self.config.baseline_sec - (snapshot.run_time_sec - float(self.phase_started_run_time_sec)))
            self.instruction_var.set("Baseline: stay relaxed and still while the hand branch calibrates.")
            self.countdown_var.set(f"{remaining:0.1f}s remaining")
            if remaining <= 0.0:
                self._finalize_protocol_row(snapshot)
                self._enter_phase(SESSION_PHASE_GUIDED, snapshot)
                self.protocol_step_index = 0
                self.protocol_subphase = "prep"
                self.protocol_subphase_started_sec = float(snapshot.run_time_sec)
                self._append_info("Baseline complete. Starting guided collection.")
            return

        if self.session_phase == SESSION_PHASE_GUIDED:
            if self.protocol_step_index >= len(self.protocol_steps):
                self._enter_phase(SESSION_PHASE_ADAPTING, snapshot)
                return

            step = self.protocol_steps[self.protocol_step_index]
            if self.protocol_subphase == "prep":
                if self.protocol_subphase_started_sec is None:
                    self.protocol_subphase_started_sec = float(snapshot.run_time_sec)
                remaining = max(0.0, step.prep_sec - (snapshot.run_time_sec - float(self.protocol_subphase_started_sec)))
                self.instruction_var.set(f"{step.display_text}: {step.instruction_text}")
                self.countdown_var.set(f"Starts in {remaining:0.1f}s")
                if remaining <= 0.0:
                    self.protocol_subphase = "active"
                    self.protocol_subphase_started_sec = float(snapshot.run_time_sec)
                    self.controller.insert_marker(
                        step.marker_code,
                        marker_role="start",
                        phase_name=SESSION_PHASE_GUIDED,
                        protocol_label=step.action_label,
                        expected_count=step.expected_count,
                        notes=step.display_text,
                        run_time_sec=float(snapshot.run_time_sec),
                    )
                    self.active_protocol_row = {
                        "phase_name": SESSION_PHASE_GUIDED,
                        "step_id": step.step_id,
                        "protocol_label": step.action_label,
                        "display_text": step.display_text,
                        "instruction_text": step.instruction_text,
                        "marker_code": int(step.marker_code),
                        "expected_count": step.expected_count if step.expected_count is not None else "",
                        "hold_duration_sec": step.hold_duration_sec if step.hold_duration_sec is not None else "",
                        "start_run_time_sec": float(snapshot.run_time_sec),
                        "start_sample": int(snapshot.sample_count),
                        "prep_sec": float(step.prep_sec),
                        "planned_active_sec": float(step.active_sec),
                    }
                    self._append_info(f"[protocol] start {step.display_text}")
                return

            if self.protocol_subphase == "active":
                remaining = max(0.0, step.active_sec - (snapshot.run_time_sec - float(self.protocol_subphase_started_sec)))
                self.instruction_var.set(f"{step.display_text}: {step.instruction_text}")
                self.countdown_var.set(f"{remaining:0.1f}s active")
                if remaining <= 0.0:
                    self.controller.insert_marker(
                        step.marker_code,
                        marker_role="end",
                        phase_name=SESSION_PHASE_GUIDED,
                        protocol_label=step.action_label,
                        expected_count=step.expected_count,
                        notes=step.display_text,
                        run_time_sec=float(snapshot.run_time_sec),
                    )
                    self._finalize_protocol_row(snapshot)
                    self.protocol_subphase = "rest"
                    self.protocol_subphase_started_sec = float(snapshot.run_time_sec)
                    self._append_info(f"[protocol] end {step.display_text}")
                return

            remaining = max(0.0, step.rest_sec - (snapshot.run_time_sec - float(self.protocol_subphase_started_sec)))
            self.instruction_var.set("Rest briefly before the next scripted block.")
            self.countdown_var.set(f"Next block in {remaining:0.1f}s")
            if remaining <= 0.0:
                self.protocol_step_index += 1
                self.protocol_subphase = "prep"
                self.protocol_subphase_started_sec = float(snapshot.run_time_sec)
            return

        if self.session_phase == SESSION_PHASE_ADAPTING:
            self._set_busy("Adapting session thresholds...")
            provenance = self.controller.decoder.classifier_provenance()
            base_hand = dict(provenance.get("hand_branch", {}).get("runtime_thresholds", {}))
            base_jaw = dict(provenance.get("jaw_branch", {}).get("hold_runtime", {}))
            base_jaw["click_threshold"] = provenance.get("jaw_branch", {}).get("trigger_config", {}).get("clench_probability_threshold", 0.80)
            base_jaw["rearm_threshold"] = provenance.get("jaw_branch", {}).get("trigger_config", {}).get("rearm_clench_probability_threshold", 0.50)
            self.adaptation_summary = compute_session_adaptation(
                control_trace_rows=self.controller.logger.control_trace_rows,
                protocol_rows=self.controller.logger.protocol_rows,
                base_hand=base_hand,
                base_jaw=base_jaw,
            )
            self.controller.logger.set_adaptation_summary(self.adaptation_summary)
            updated_provenance = self.controller.decoder.apply_session_adaptation(self.adaptation_summary)
            self.controller.logger.set_classifier_provenance(updated_provenance)
            self.ready_for_game = True
            hand_reason = self.adaptation_summary["hand"]["reason"]
            jaw_reason = self.adaptation_summary["jaw"]["reason"]
            self.instruction_var.set(
                f"Adaptation finished. Hand: {hand_reason}. Jaw: {jaw_reason}. Click Review + Start Game when you are ready."
            )
            self.countdown_var.set("The testing review prompt will appear before the chart starts.")
            self._append_info("Session adaptation finished. Review + Start Game is ready.")
            self._enter_phase(SESSION_PHASE_READY, snapshot)
            self._set_busy("")
            self._refresh_controls()
            self.start_button.focus_set()
            return

        if self.session_phase == SESSION_PHASE_READY:
            self.instruction_var.set("Session adaptation is complete. Press Review + Start Game when you are ready.")
            self.countdown_var.set("The testing review comes first, then the single-note chart begins.")

    def _start_gameplay(self) -> None:
        if self.controller is None or self.mode_var.get() != LIVE or not self.ready_for_game:
            return
        offset_sec = float(self.controller.decoder._current_run_time_sec())
        self.game = RhythmGame(_shift_chart(self._base_chart(), offset_sec), self.game_config)
        self.live_game_started = True
        self.ready_for_game = False
        self.session_phase = SESSION_PHASE_GAMEPLAY
        self.phase_var.set(f"Phase: {self.session_phase}")
        self.phase_started_run_time_sec = offset_sec
        self.controller.logger.log_phase_row(
            {
                "mode": self.mode_var.get(),
                "phase_name": SESSION_PHASE_GAMEPLAY,
                "run_time_sec": float(offset_sec),
                "sample_count": int(self.controller.decoder.sample_store.total_samples),
                "wall_time_utc": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "session_label": self.participant_info.get("session_label", ""),
            }
        )
        self.instruction_var.set("Match the incoming notes. Jaw hold is the only sustained note.")
        self.countdown_var.set("No overlapping notes. Focus on one action at a time.")
        self._append_info("Gameplay started.")
        self._refresh_controls()

    def _tick(self) -> None:
        if self._pending_live_thread is not None and not self._pending_live_thread.is_alive():
            result = self._pending_live_result
            self._pending_live_thread = None
            self._pending_live_result = None
            if result is None:
                self._set_busy("")
                self._append_info("Live session startup ended without a controller result.")
            elif result[0] == "error":
                self._set_busy("")
                self.status_var.set("live startup failed")
                self.countdown_var.set("")
                self._append_info(f"Live startup failed: {result[1]}")
            else:
                _tag, controller, participant_info = result
                self._finish_live_start(controller, participant_info)

        if self.controller is not None:
            snapshot = self.controller.tick()

            if self.mode_var.get() == LIVE:
                self._advance_live_protocol(snapshot)

            game_events: list[dict[str, Any]] = []
            if self.mode_var.get() == REPLAY or self.session_phase == SESSION_PHASE_GAMEPLAY:
                game_events = self.game.update(snapshot)
                self._log_game_tick(snapshot, self.controller.logger, game_events)
                if self.mode_var.get() == LIVE and self.session_phase == SESSION_PHASE_GAMEPLAY and self.game.completed and not self.gameplay_review_recorded:
                    self._enter_phase(SESSION_PHASE_REVIEW, snapshot)
                    self.instruction_var.set("Chart complete. Please rate how the controls felt.")
                    self.countdown_var.set("A review prompt will appear now.")
                    self.root.after(150, self._finalize_gameplay_feedback)
            self._apply_snapshot(snapshot, game_events)

            if isinstance(self.controller, ReplayGameController) and self.controller.finished():
                self.play_pause_button.configure(text="Play")
            if isinstance(self.controller, LiveGameController) and self.controller.stopped:
                self._append_info("Live run reached the configured stop condition.")

        if self.auto_close_deadline is not None and time.monotonic() >= self.auto_close_deadline:
            self._on_close()
            return
        self.root.after(self.config.timer_interval_ms, self._tick)

    def _on_close(self) -> None:
        if self._closing:
            return
        self._closing = True
        self._stop_controller(prompt_feedback=True)
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()
