from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, welch
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DEFAULT_FS = 250.0
DEFAULT_DROP_CHANNELS = ("Channel_2", "Channel_5")


def load_openbci_tsv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, sep="\t")
    return format_openbci_df(df)


def format_openbci_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.shape[1] < 10:
        raise ValueError("Expected at least 10 columns (index + 8 channels + marker).")
    col_names = [f"col_{i}" for i in range(df.shape[1])]
    col_names[0] = "Sample_Index"
    for i in range(1, 9):
        col_names[i] = f"Channel_{i}"
    col_names[-1] = "Marker"
    out = df.copy()
    out.columns = col_names
    keep = ["Sample_Index"] + [f"Channel_{i}" for i in range(1, 9)] + ["Marker"]
    return out[keep]


def channel_quality_report(df: pd.DataFrame, rail_limit: float = 187000) -> pd.DataFrame:
    rows = []
    for ch in [c for c in df.columns if c.startswith("Channel_")]:
        data = df[ch].astype(float)
        std = float(data.std())
        mn = float(data.min())
        mx = float(data.max())
        rows.append(
            {
                "channel": ch,
                "min": mn,
                "max": mx,
                "std": std,
                "is_flatlined": std == 0.0,
                "is_railed": mx >= rail_limit or mn <= -rail_limit,
            }
        )
    return pd.DataFrame(rows)


def _find_intervals(df: pd.DataFrame, start_marker: float, end_marker: float) -> List[Tuple[int, int]]:
    intervals: List[Tuple[int, int]] = []
    in_trial = False
    start_idx = -1
    markers = df["Marker"].to_numpy()
    for idx, marker in enumerate(markers):
        if marker == start_marker and not in_trial:
            in_trial = True
            start_idx = idx
        elif marker == end_marker and in_trial:
            intervals.append((start_idx, idx))
            in_trial = False
    return intervals


def _complement_intervals(
    total_len: int,
    occupied_intervals: Sequence[Tuple[int, int]],
    buffer_samples: int,
) -> List[Tuple[int, int]]:
    if not occupied_intervals:
        return [(0, total_len - 1)] if total_len > 0 else []
    sorted_intervals = sorted(occupied_intervals, key=lambda x: x[0])
    out: List[Tuple[int, int]] = []
    cursor = 0
    for start, end in sorted_intervals:
        left = max(0, start - buffer_samples)
        if left > cursor:
            out.append((cursor, left - 1))
        cursor = min(total_len - 1, end + buffer_samples) + 1
    if cursor < total_len:
        out.append((cursor, total_len - 1))
    return out


def _slice_intervals(df: pd.DataFrame, intervals: Sequence[Tuple[int, int]]) -> List[pd.DataFrame]:
    return [df.iloc[start : end + 1].reset_index(drop=True) for start, end in intervals]


def build_trial_segments(
    lr_df: pd.DataFrame,
    hr_df: pd.DataFrame,
    drop_channels: Sequence[str] = DEFAULT_DROP_CHANNELS,
    baseline_buffer_samples: int = 500,
) -> Dict[str, List[pd.DataFrame]]:
    lr_clean = lr_df.drop(columns=list(drop_channels), errors="ignore").reset_index(drop=True)
    hr_clean = hr_df.drop(columns=list(drop_channels), errors="ignore").reset_index(drop=True)

    lr_left = _find_intervals(lr_clean, 1.0, 2.0)
    lr_right = _find_intervals(lr_clean, 3.0, 4.0)
    hr_hold = _find_intervals(hr_clean, 1.0, 2.0)
    hr_repeated = _find_intervals(hr_clean, 3.0, 4.0)

    lr_baseline_intervals = _complement_intervals(
        total_len=len(lr_clean),
        occupied_intervals=lr_left + lr_right,
        buffer_samples=baseline_buffer_samples,
    )
    hr_baseline_intervals = _complement_intervals(
        total_len=len(hr_clean),
        occupied_intervals=hr_hold + hr_repeated,
        buffer_samples=baseline_buffer_samples,
    )

    trials = {
        "Left": _slice_intervals(lr_clean, lr_left),
        "Right": _slice_intervals(lr_clean, lr_right),
        "Repeated": _slice_intervals(hr_clean, hr_repeated),
        "Norm": _slice_intervals(lr_clean, lr_baseline_intervals)
        + _slice_intervals(hr_clean, hr_baseline_intervals),
    }
    return trials


def _apply_filters_df(df: pd.DataFrame, fs: float = DEFAULT_FS) -> pd.DataFrame:
    nyq = 0.5 * fs
    b_notch, a_notch = iirnotch(60.0, 30.0, fs)
    b_band, a_band = butter(N=2, Wn=[0.5 / nyq, 50.0 / nyq], btype="band")
    out = df.copy()
    channels = [c for c in df.columns if c.startswith("Channel_")]
    for col in channels:
        sig = out[col].to_numpy(dtype=float)
        sig = filtfilt(b_notch, a_notch, sig)
        sig = filtfilt(b_band, a_band, sig)
        out[col] = sig
    return out


def _extract_features(window_df: pd.DataFrame, fs: float = DEFAULT_FS) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    channels = [c for c in window_df.columns if c.startswith("Channel_")]
    for ch in channels:
        data = window_df[ch].to_numpy(dtype=float)
        feats[f"{ch}_Var"] = float(np.var(data))
        feats[f"{ch}_RMS"] = float(np.sqrt(np.mean(data**2)))
        feats[f"{ch}_WL"] = float(np.sum(np.abs(np.diff(data))))
        freqs, psd = welch(data, fs=fs, nperseg=min(len(data), int(fs)))
        mu_band = (freqs >= 8) & (freqs <= 12)
        beta_band = (freqs >= 13) & (freqs <= 30)
        feats[f"{ch}_Mu_Power"] = float(np.sum(psd[mu_band]))
        feats[f"{ch}_Beta_Power"] = float(np.sum(psd[beta_band]))
    return feats


def add_asymmetry_features(df_features: pd.DataFrame) -> pd.DataFrame:
    out = df_features.copy()
    out["C3_C4_Mu_Diff"] = out["Channel_1_Mu_Power"] - out["Channel_3_Mu_Power"]
    out["C4_C3_Mu_Diff"] = out["Channel_3_Mu_Power"] - out["Channel_1_Mu_Power"]
    out["Mu_Asymmetry_Ratio"] = (
        out["Channel_1_Mu_Power"] - out["Channel_3_Mu_Power"]
    ) / (out["Channel_1_Mu_Power"] + out["Channel_3_Mu_Power"] + 1e-6)
    out["C3_C4_Beta_Diff"] = out["Channel_1_Beta_Power"] - out["Channel_3_Beta_Power"]
    out["C4_C3_Beta_Diff"] = out["Channel_3_Beta_Power"] - out["Channel_1_Beta_Power"]
    return out


def build_feature_dataset(
    trials_by_class: Dict[str, Sequence[pd.DataFrame]],
    fs: float = DEFAULT_FS,
    window_sec: float = 1.0,
    hop_sec: float | None = None,
    include_asymmetry: bool = True,
) -> pd.DataFrame:
    window_size = int(round(window_sec * fs))
    hop_size = int(round((hop_sec if hop_sec is not None else window_sec) * fs))
    if hop_size <= 0:
        raise ValueError("hop size must be > 0")

    rows: List[Dict[str, float]] = []
    group_id = 0
    for label in ("Norm", "Left", "Right", "Repeated"):
        for trial_idx, trial in enumerate(trials_by_class.get(label, [])):
            trial_filtered = _apply_filters_df(trial, fs=fs)
            trial_len = len(trial_filtered)
            for start in range(0, trial_len - window_size + 1, hop_size):
                window = trial_filtered.iloc[start : start + window_size]
                feats = _extract_features(window, fs=fs)
                feats["Label"] = label
                feats["IntentLabel"] = "Norm" if label == "Norm" else "Active"
                feats["GroupID"] = group_id
                feats["TrialIndexWithinClass"] = trial_idx
                rows.append(feats)
            group_id += 1

    df = pd.DataFrame(rows)
    if include_asymmetry and len(df) > 0:
        meta_cols = ["Label", "IntentLabel", "GroupID", "TrialIndexWithinClass"]
        feature_cols = [c for c in df.columns if c not in meta_cols]
        df_features = add_asymmetry_features(df[feature_cols])
        df = pd.concat([df_features, df[meta_cols].reset_index(drop=True)], axis=1)
    return df


def build_model_bank(random_state: int = 42) -> Dict[str, object]:
    return {
        "LDA": LinearDiscriminantAnalysis(),
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "SVM_RBF": SVC(kernel="rbf", probability=True, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, random_state=random_state, class_weight="balanced"
        ),
    }


def evaluate_models_grouped(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    models: Dict[str, object],
    n_splits: int = 5,
) -> pd.DataFrame:
    cv = StratifiedGroupKFold(n_splits=n_splits)
    rows = []
    for name, model in models.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("classifier", clone(model))])
        scores = cross_validate(
            pipe,
            X,
            y,
            groups=groups,
            cv=cv,
            scoring=("accuracy", "f1_macro"),
            return_train_score=False,
        )
        rows.append(
            {
                "Model": name,
                "Accuracy_Mean": float(np.mean(scores["test_accuracy"])),
                "Accuracy_Std": float(np.std(scores["test_accuracy"])),
                "F1_Macro_Mean": float(np.mean(scores["test_f1_macro"])),
                "F1_Macro_Std": float(np.std(scores["test_f1_macro"])),
            }
        )
    return pd.DataFrame(rows).sort_values(
        by=["F1_Macro_Mean", "Accuracy_Mean"], ascending=False
    ).reset_index(drop=True)


def split_grouped_holdout(
    X: np.ndarray,
    y_intent: np.ndarray,
    groups: np.ndarray,
    test_size: float = 0.25,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y_intent, groups))
    return train_idx, test_idx


def split_grouped_holdout_stratified(
    groups: np.ndarray,
    y_group_label: np.ndarray,
    test_size: float = 0.25,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    unique_groups = np.unique(groups)
    group_to_label = {}
    for gid in unique_groups:
        labels = np.unique(y_group_label[groups == gid])
        if len(labels) != 1:
            raise ValueError("Each GroupID must map to exactly one class label.")
        group_to_label[gid] = labels[0]

    group_labels = np.array([group_to_label[g] for g in unique_groups])
    n_groups = len(unique_groups)
    n_test = int(round(n_groups * test_size))
    n_test = max(1, min(n_groups - 1, n_test))

    rng = np.random.default_rng(random_state)
    test_group_set = set()
    for label in np.unique(group_labels):
        label_groups = unique_groups[group_labels == label]
        k = int(round(len(label_groups) * test_size))
        if k == 0 and len(label_groups) > 1:
            k = 1
        k = min(k, len(label_groups))
        if k > 0:
            chosen = rng.choice(label_groups, size=k, replace=False)
            test_group_set.update(int(g) for g in chosen)

    if len(test_group_set) > n_test:
        keep = rng.choice(np.array(sorted(test_group_set)), size=n_test, replace=False)
        test_group_set = set(int(g) for g in keep)
    elif len(test_group_set) < n_test:
        remaining = np.array([g for g in unique_groups if g not in test_group_set])
        if len(remaining) > 0:
            add_n = min(n_test - len(test_group_set), len(remaining))
            add = rng.choice(remaining, size=add_n, replace=False)
            test_group_set.update(int(g) for g in add)

    test_mask = np.array([g in test_group_set for g in groups], dtype=bool)
    test_idx = np.flatnonzero(test_mask)
    train_idx = np.flatnonzero(~test_mask)
    return train_idx, test_idx


def train_two_stage_models(
    X_train: np.ndarray,
    y_train_intent: np.ndarray,
    y_train_label: np.ndarray,
    intent_model: object,
    direction_model: object,
    direction_mode: str = "lr",
) -> Tuple[Pipeline, Pipeline]:
    intent_pipe = Pipeline(
        [("scaler", StandardScaler()), ("classifier", clone(intent_model))]
    )
    intent_pipe.fit(X_train, y_train_intent)

    if direction_mode not in ("lr", "lr_other"):
        raise ValueError("direction_mode must be 'lr' or 'lr_other'.")

    if direction_mode == "lr":
        direction_mask = np.isin(y_train_label, ["Left", "Right"])
        y_dir_train = y_train_label[direction_mask]
    else:
        direction_mask = np.isin(y_train_label, ["Left", "Right", "Repeated"])
        y_dir_train = np.array(
            ["Other" if v == "Repeated" else v for v in y_train_label[direction_mask]]
        )

    direction_pipe = Pipeline(
        [("scaler", StandardScaler()), ("classifier", clone(direction_model))]
    )
    direction_pipe.fit(X_train[direction_mask], y_dir_train)
    return intent_pipe, direction_pipe


@dataclass(frozen=True)
class DecoderConfig:
    enter_active_threshold: float = 0.7
    exit_active_threshold: float = 0.55
    k_consecutive: int = 3
    cooldown_windows: int = 4
    majority_windows: int = 3
    min_direction_votes: int = 2
    direction_min_confidence: float = 0.6
    direction_margin: float = 0.1
    direction_k_consecutive: int = 1


def conservative_decoder_config() -> DecoderConfig:
    return DecoderConfig()


@dataclass
class DiscreteDecoder:
    intent_model: Pipeline
    direction_model: Pipeline
    enter_active_threshold: float = 0.7
    exit_active_threshold: float = 0.55
    k_consecutive: int = 3
    cooldown_windows: int = 4
    majority_windows: int = 3
    min_direction_votes: int = 2
    direction_min_confidence: float = 0.6
    direction_margin: float = 0.1
    direction_k_consecutive: int = 1
    command_labels: Tuple[str, ...] = ("Left", "Right")

    def __post_init__(self) -> None:
        self._active_hist: List[bool] = []
        self._dir_hist: List[str | None] = []
        self._cooldown = 0
        self._in_active_state = False
        self._active_class_index = list(self.intent_model.classes_).index("Active")
        self._dir_classes = list(self.direction_model.classes_)
        self._last_dir: str | None = None
        self._dir_streak = 0

    def _active_probability(self, x: np.ndarray) -> float:
        proba = self.intent_model.predict_proba(x.reshape(1, -1))[0]
        return float(proba[self._active_class_index])

    def _direction_prediction(self, x: np.ndarray) -> str | None:
        proba = self.direction_model.predict_proba(x.reshape(1, -1))[0]
        best_idx = int(np.argmax(proba))
        best_p = float(proba[best_idx])
        second_p = float(np.partition(proba, -2)[-2]) if len(proba) > 1 else 0.0
        if best_p < self.direction_min_confidence:
            return None
        if (best_p - second_p) < self.direction_margin:
            return None
        return str(self._dir_classes[best_idx])

    def _update_active_state(self, p_active: float) -> bool:
        if self._in_active_state:
            self._in_active_state = p_active >= self.exit_active_threshold
        else:
            self._in_active_state = p_active >= self.enter_active_threshold
        return self._in_active_state

    def process_window(self, features: np.ndarray) -> str:
        if self._cooldown > 0:
            self._cooldown -= 1
            self._active_hist.append(False)
            self._dir_hist.append(None)
            self._in_active_state = False
            self._last_dir = None
            self._dir_streak = 0
            return "NONE"

        p_active = self._active_probability(features)
        is_active = self._update_active_state(p_active)
        self._active_hist.append(is_active)
        if is_active:
            pred = self._direction_prediction(features)
            if pred in self.command_labels:
                if pred == self._last_dir:
                    self._dir_streak += 1
                else:
                    self._last_dir = pred
                    self._dir_streak = 1
                if self._dir_streak >= self.direction_k_consecutive:
                    self._dir_hist.append(pred)
                else:
                    self._dir_hist.append(None)
            else:
                self._last_dir = None
                self._dir_streak = 0
                self._dir_hist.append(None)
        else:
            self._last_dir = None
            self._dir_streak = 0
            self._dir_hist.append(None)

        if len(self._active_hist) < self.k_consecutive:
            return "NONE"
        if not all(self._active_hist[-self.k_consecutive :]):
            return "NONE"

        recent_dirs = [d for d in self._dir_hist[-self.majority_windows :] if d is not None]
        if len(recent_dirs) < self.min_direction_votes:
            return "NONE"
        cmd = max(set(recent_dirs), key=recent_dirs.count)
        self._cooldown = self.cooldown_windows
        self._in_active_state = False
        self._last_dir = None
        self._dir_streak = 0
        return cmd


def run_holdout_simulation(
    X_test: np.ndarray,
    y_test_label: np.ndarray,
    groups_test: np.ndarray,
    intent_model: Pipeline,
    direction_model: Pipeline,
    window_sec: float = 1.0,
    hop_sec: float = 1.0,
    enter_active_threshold: float = 0.7,
    exit_active_threshold: float = 0.55,
    k_consecutive: int = 3,
    cooldown_windows: int = 4,
    majority_windows: int = 3,
    min_direction_votes: int = 2,
    direction_min_confidence: float = 0.6,
    direction_margin: float = 0.1,
    direction_k_consecutive: int = 1,
    command_labels: Tuple[str, ...] = ("Left", "Right"),
) -> Dict[str, object]:
    unique_groups = np.unique(groups_test)
    true_cmd: List[str] = []
    pred_cmd: List[str] = []
    latencies: List[float] = []
    false_triggers = 0
    baseline_duration_s = 0.0
    repeated_triggers = 0
    repeated_duration_s = 0.0

    for gid in unique_groups:
        mask = groups_test == gid
        X_group = X_test[mask]
        y_group = y_test_label[mask]
        if len(y_group) == 0:
            continue
        true_label = str(y_group[0])

        decoder = DiscreteDecoder(
            intent_model=intent_model,
            direction_model=direction_model,
            enter_active_threshold=enter_active_threshold,
            exit_active_threshold=exit_active_threshold,
            k_consecutive=k_consecutive,
            cooldown_windows=cooldown_windows,
            majority_windows=majority_windows,
            min_direction_votes=min_direction_votes,
            direction_min_confidence=direction_min_confidence,
            direction_margin=direction_margin,
            direction_k_consecutive=direction_k_consecutive,
            command_labels=command_labels,
        )

        fired_cmds: List[str] = []
        fired_idx: List[int] = []
        for i, window in enumerate(X_group):
            cmd = decoder.process_window(window)
            if cmd != "NONE":
                fired_cmds.append(cmd)
                fired_idx.append(i)

        if true_label in ("Left", "Right"):
            true_cmd.append(true_label)
            if fired_cmds:
                pred_cmd.append(fired_cmds[0])
                latencies.append(fired_idx[0] * hop_sec + window_sec)
            else:
                pred_cmd.append("MISSED")
        elif true_label == "Norm":
            false_triggers += len(fired_cmds)
            baseline_duration_s += len(X_group) * hop_sec
        elif true_label == "Repeated":
            repeated_triggers += len(fired_cmds)
            repeated_duration_s += len(X_group) * hop_sec

    total_direction_trials = len(true_cmd)
    correct = sum(1 for t, p in zip(true_cmd, pred_cmd) if t == p)
    missed = sum(1 for p in pred_cmd if p == "MISSED")
    wrong = total_direction_trials - correct - missed
    fpr_per_min = 0.0
    if baseline_duration_s > 0:
        fpr_per_min = false_triggers / (baseline_duration_s / 60.0)
    repeated_trigger_rate_per_min = 0.0
    if repeated_duration_s > 0:
        repeated_trigger_rate_per_min = repeated_triggers / (repeated_duration_s / 60.0)

    return {
        "total_direction_trials": total_direction_trials,
        "correct": correct,
        "wrong": wrong,
        "missed": missed,
        "accuracy_on_direction_trials": (correct / total_direction_trials) if total_direction_trials else 0.0,
        "avg_latency_s": float(np.mean(latencies)) if latencies else None,
        "std_latency_s": float(np.std(latencies)) if latencies else None,
        "false_triggers": int(false_triggers),
        "baseline_duration_s": float(baseline_duration_s),
        "false_triggers_per_min": float(fpr_per_min),
        "baseline_false_triggers_per_min": float(fpr_per_min),
        "repeated_triggers": int(repeated_triggers),
        "repeated_duration_s": float(repeated_duration_s),
        "repeated_triggers_per_min": float(repeated_trigger_rate_per_min),
        "wrong_rate": (wrong / total_direction_trials) if total_direction_trials else 0.0,
        "miss_rate": (missed / total_direction_trials) if total_direction_trials else 0.0,
        "true_direction_labels": true_cmd,
        "predicted_direction_commands": pred_cmd,
    }


def evaluate_two_stage_over_seeds(
    X: np.ndarray,
    y_intent: np.ndarray,
    y_label: np.ndarray,
    groups: np.ndarray,
    intent_model: object,
    direction_model: object,
    seeds: Sequence[int],
    test_size: float = 0.25,
    window_sec: float = 1.0,
    hop_sec: float = 1.0,
    decoder_config: DecoderConfig | None = None,
    stratified_group_split: bool = True,
    direction_mode: str = "lr",
    command_labels: Tuple[str, ...] = ("Left", "Right"),
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    cfg = decoder_config or conservative_decoder_config()
    rows = []
    for seed in seeds:
        if stratified_group_split:
            train_idx, test_idx = split_grouped_holdout_stratified(
                groups=groups,
                y_group_label=y_label,
                test_size=test_size,
                random_state=int(seed),
            )
        else:
            train_idx, test_idx = split_grouped_holdout(
                X=X, y_intent=y_intent, groups=groups, test_size=test_size, random_state=int(seed)
            )

        X_train, X_test = X[train_idx], X[test_idx]
        y_train_intent = y_intent[train_idx]
        y_train_label, y_test_label = y_label[train_idx], y_label[test_idx]
        groups_test = groups[test_idx]

        intent_pipe, direction_pipe = train_two_stage_models(
            X_train=X_train,
            y_train_intent=y_train_intent,
            y_train_label=y_train_label,
            intent_model=intent_model,
            direction_model=direction_model,
            direction_mode=direction_mode,
        )
        sim = run_holdout_simulation(
            X_test=X_test,
            y_test_label=y_test_label,
            groups_test=groups_test,
            intent_model=intent_pipe,
            direction_model=direction_pipe,
            window_sec=window_sec,
            hop_sec=hop_sec,
            enter_active_threshold=cfg.enter_active_threshold,
            exit_active_threshold=cfg.exit_active_threshold,
            k_consecutive=cfg.k_consecutive,
            cooldown_windows=cfg.cooldown_windows,
            majority_windows=cfg.majority_windows,
            min_direction_votes=cfg.min_direction_votes,
            direction_min_confidence=cfg.direction_min_confidence,
            direction_margin=cfg.direction_margin,
            direction_k_consecutive=cfg.direction_k_consecutive,
            command_labels=command_labels,
        )
        rows.append(
            {
                "seed": int(seed),
                "direction_accuracy": float(sim["accuracy_on_direction_trials"]),
                "avg_latency_s": float(sim["avg_latency_s"]) if sim["avg_latency_s"] is not None else np.nan,
                "false_triggers_per_min": float(sim["false_triggers_per_min"]),
                "repeated_triggers_per_min": float(sim["repeated_triggers_per_min"]),
                "wrong_rate": float(sim["wrong_rate"]),
                "miss_rate": float(sim["miss_rate"]),
                "wrong": int(sim["wrong"]),
                "missed": int(sim["missed"]),
                "direction_trials": int(sim["total_direction_trials"]),
            }
        )

    df = pd.DataFrame(rows)
    summary = {
        "baseline_false_triggers_per_min_mean": float(df["false_triggers_per_min"].mean()),
        "baseline_false_triggers_per_min_worst": float(df["false_triggers_per_min"].max()),
        "repeated_triggers_per_min_mean": float(df["repeated_triggers_per_min"].mean()),
        "repeated_triggers_per_min_worst": float(df["repeated_triggers_per_min"].max()),
        "direction_accuracy_mean": float(df["direction_accuracy"].mean()),
        "direction_accuracy_std": float(df["direction_accuracy"].std(ddof=0)),
        "wrong_rate_mean": float(df["wrong_rate"].mean()),
        "wrong_rate_worst": float(df["wrong_rate"].max()),
        "miss_rate_mean": float(df["miss_rate"].mean()),
        "miss_rate_worst": float(df["miss_rate"].max()),
        "avg_latency_s_mean": float(df["avg_latency_s"].mean(skipna=True)),
        "avg_latency_s_std": float(df["avg_latency_s"].std(skipna=True, ddof=0)),
        "wrong_total": int(df["wrong"].sum()),
        "missed_total": int(df["missed"].sum()),
        "direction_trials_total": int(df["direction_trials"].sum()),
    }
    return df, summary


def tune_decoder_controls(
    X: np.ndarray,
    y_intent: np.ndarray,
    y_label: np.ndarray,
    groups: np.ndarray,
    intent_model: object,
    direction_model: object,
    configs: Sequence[DecoderConfig],
    seeds: Sequence[int],
    test_size: float = 0.25,
    window_sec: float = 1.0,
    hop_sec: float = 1.0,
    stratified_group_split: bool = True,
    direction_mode: str = "lr",
    command_labels: Tuple[str, ...] = ("Left", "Right"),
) -> pd.DataFrame:
    rows = []
    for cfg in configs:
        _, summary = evaluate_two_stage_over_seeds(
            X=X,
            y_intent=y_intent,
            y_label=y_label,
            groups=groups,
            intent_model=intent_model,
            direction_model=direction_model,
            seeds=seeds,
            test_size=test_size,
            window_sec=window_sec,
            hop_sec=hop_sec,
            decoder_config=cfg,
            stratified_group_split=stratified_group_split,
            direction_mode=direction_mode,
            command_labels=command_labels,
        )
        row = {
            "enter_active_threshold": cfg.enter_active_threshold,
            "exit_active_threshold": cfg.exit_active_threshold,
            "k_consecutive": cfg.k_consecutive,
            "cooldown_windows": cfg.cooldown_windows,
            "majority_windows": cfg.majority_windows,
            "min_direction_votes": cfg.min_direction_votes,
            "direction_min_confidence": cfg.direction_min_confidence,
            "direction_margin": cfg.direction_margin,
            "direction_k_consecutive": cfg.direction_k_consecutive,
            **summary,
        }
        rows.append(row)

    # Safety-first sort: fewer false triggers first, then fewer wrong commands, then latency, then accuracy.
    out = pd.DataFrame(rows).sort_values(
        by=[
            "baseline_false_triggers_per_min_mean",
            "baseline_false_triggers_per_min_worst",
            "wrong_total",
            "avg_latency_s_mean",
            "direction_accuracy_mean",
        ],
        ascending=[True, True, True, True, False],
    )
    return out.reset_index(drop=True)


def tune_decoder_controls_balanced(
    X: np.ndarray,
    y_intent: np.ndarray,
    y_label: np.ndarray,
    groups: np.ndarray,
    intent_model: object,
    direction_model: object,
    configs: Sequence[DecoderConfig],
    seeds: Sequence[int],
    test_size: float = 0.25,
    window_sec: float = 1.0,
    hop_sec: float = 1.0,
    stratified_group_split: bool = True,
    direction_mode: str = "lr",
    command_labels: Tuple[str, ...] = ("Left", "Right"),
    baseline_false_triggers_per_min_worst_max: float = 0.5,
    repeated_triggers_per_min_worst_max: float = 0.5,
) -> pd.DataFrame:
    rows = []
    for cfg in configs:
        _, summary = evaluate_two_stage_over_seeds(
            X=X,
            y_intent=y_intent,
            y_label=y_label,
            groups=groups,
            intent_model=intent_model,
            direction_model=direction_model,
            seeds=seeds,
            test_size=test_size,
            window_sec=window_sec,
            hop_sec=hop_sec,
            decoder_config=cfg,
            stratified_group_split=stratified_group_split,
            direction_mode=direction_mode,
            command_labels=command_labels,
        )
        meets_constraints = (
            summary["baseline_false_triggers_per_min_worst"]
            <= baseline_false_triggers_per_min_worst_max
            and summary["repeated_triggers_per_min_worst"]
            <= repeated_triggers_per_min_worst_max
        )
        row = {
            "enter_active_threshold": cfg.enter_active_threshold,
            "exit_active_threshold": cfg.exit_active_threshold,
            "k_consecutive": cfg.k_consecutive,
            "cooldown_windows": cfg.cooldown_windows,
            "majority_windows": cfg.majority_windows,
            "min_direction_votes": cfg.min_direction_votes,
            "direction_min_confidence": cfg.direction_min_confidence,
            "direction_margin": cfg.direction_margin,
            "direction_k_consecutive": cfg.direction_k_consecutive,
            "meets_constraints": bool(meets_constraints),
            **summary,
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    constrained = out[out["meets_constraints"]].copy()
    if len(constrained) == 0:
        constrained = out.copy()
    constrained = constrained.sort_values(
        by=[
            "miss_rate_mean",
            "wrong_rate_mean",
            "avg_latency_s_mean",
            "direction_accuracy_mean",
        ],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)
    return constrained


def build_decoder_config_grid(
    enter_active_thresholds: Sequence[float],
    exit_active_thresholds: Sequence[float],
    k_consecutives: Sequence[int],
    cooldown_windows: Sequence[int],
    majority_windows: Sequence[int],
    min_direction_votes: Sequence[int],
    direction_min_confidences: Sequence[float],
    direction_margins: Sequence[float],
    direction_k_consecutives: Sequence[int],
) -> List[DecoderConfig]:
    configs = []
    for vals in product(
        enter_active_thresholds,
        exit_active_thresholds,
        k_consecutives,
        cooldown_windows,
        majority_windows,
        min_direction_votes,
        direction_min_confidences,
        direction_margins,
        direction_k_consecutives,
    ):
        cfg = DecoderConfig(
            enter_active_threshold=float(vals[0]),
            exit_active_threshold=float(vals[1]),
            k_consecutive=int(vals[2]),
            cooldown_windows=int(vals[3]),
            majority_windows=int(vals[4]),
            min_direction_votes=int(vals[5]),
            direction_min_confidence=float(vals[6]),
            direction_margin=float(vals[7]),
            direction_k_consecutive=int(vals[8]),
        )
        if cfg.exit_active_threshold > cfg.enter_active_threshold:
            continue
        if cfg.min_direction_votes > cfg.majority_windows:
            continue
        configs.append(cfg)
    return configs
