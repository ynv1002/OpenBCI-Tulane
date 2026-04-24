from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .utils import PROJECT_ROOT


OPENBCI_RUNS_ROOT = PROJECT_ROOT.parent / "OPENBCI_runs"


@dataclass(frozen=True)
class FileContract:
    key: str
    subject: str
    csv_path: Path
    protocol_family: str
    trust_tier: str
    processing_family: str
    marker_semantics: str
    supported_targets: tuple[str, ...]

    @property
    def filename(self) -> str:
        return self.csv_path.name


MANUAL_FILE_CONTRACTS: tuple[FileContract, ...] = (
    FileContract(
        key="ben_lrj",
        subject="Ben",
        csv_path=OPENBCI_RUNS_ROOT / "Ben" / "Ben-LRJ(1-6)-4:9.csv",
        protocol_family="LRJ",
        trust_tier="main",
        processing_family="lrj",
        marker_semantics="1=LEFT, 2=RIGHT, 3=JAW; same-code start/end pairs; expected within-label counts 1..6",
        supported_targets=("interval", "event", "side", "jaw_event", "exact_count"),
    ),
    FileContract(
        key="yaniv_lrj",
        subject="Yaniv",
        csv_path=OPENBCI_RUNS_ROOT / "Yaniv" / "Yaniv-LRJ(6)-4:7.csv",
        protocol_family="LRJ",
        trust_tier="main",
        processing_family="lrj",
        marker_semantics="1=LEFT, 2=RIGHT, 3=JAW; same-code start/end pairs; expected within-label counts 1..6",
        supported_targets=("interval", "event", "side", "jaw_event", "exact_count"),
    ),
    FileContract(
        key="yaniv_hr_2026_02_27",
        subject="Yaniv",
        csv_path=OPENBCI_RUNS_ROOT / "Yaniv" / "HR-2-27-26-(01).csv",
        protocol_family="HR",
        trust_tier="main",
        processing_family="jaw",
        marker_semantics="Legacy 1->2 movement1 and 3->4 movement2 markers; movement labels map to HOLD vs REPEATED at the coarse-segment level",
        supported_targets=("jaw_event", "jaw_type"),
    ),
    FileContract(
        key="yaniv_hr_2026_03_08",
        subject="Yaniv",
        csv_path=OPENBCI_RUNS_ROOT / "Yaniv" / "HR-3-8-26-(02).csv",
        protocol_family="HR",
        trust_tier="main",
        processing_family="jaw",
        marker_semantics="Legacy 1->2 movement1 and 3->4 movement2 markers; movement labels map to HOLD vs REPEATED at the coarse-segment level",
        supported_targets=("jaw_event", "jaw_type"),
    ),
    FileContract(
        key="yaniv_hr_2026_03_15",
        subject="Yaniv",
        csv_path=OPENBCI_RUNS_ROOT / "Yaniv" / "HR-3-15-26-(03).csv",
        protocol_family="HR",
        trust_tier="main",
        processing_family="jaw",
        marker_semantics="Legacy 1->2 movement1 and 3->4 movement2 markers; movement labels map to HOLD vs REPEATED at the coarse-segment level",
        supported_targets=("jaw_event", "jaw_type"),
    ),
    FileContract(
        key="yaniv_lr_2026_02_27",
        subject="Yaniv",
        csv_path=OPENBCI_RUNS_ROOT / "Yaniv" / "LR-2-27-26-(01).csv",
        protocol_family="LR",
        trust_tier="main",
        processing_family="left_right",
        marker_semantics="1->2=LEFT block, 3->4=RIGHT block",
        supported_targets=("block", "event", "side"),
    ),
    FileContract(
        key="yaniv_lr_2026_03_15",
        subject="Yaniv",
        csv_path=OPENBCI_RUNS_ROOT / "Yaniv" / "LR-3-15-26-(04).csv",
        protocol_family="LR",
        trust_tier="main",
        processing_family="left_right",
        marker_semantics="1->2=LEFT block, 3->4=RIGHT block",
        supported_targets=("block", "event", "side"),
    ),
    FileContract(
        key="yaniv_lr_2026_03_08_a",
        subject="Yaniv",
        csv_path=OPENBCI_RUNS_ROOT / "Yaniv" / "LR-3-8-26-(02).csv",
        protocol_family="LR",
        trust_tier="stress",
        processing_family="left_right",
        marker_semantics="1->2=LEFT block, 3->4=RIGHT block",
        supported_targets=("block", "event", "side"),
    ),
    FileContract(
        key="yaniv_lr_2026_03_08_b",
        subject="Yaniv",
        csv_path=OPENBCI_RUNS_ROOT / "Yaniv" / "LR-3-8-26-(03).csv",
        protocol_family="LR",
        trust_tier="stress",
        processing_family="left_right",
        marker_semantics="1->2=LEFT block, 3->4=RIGHT block",
        supported_targets=("block", "event", "side"),
    ),
    FileContract(
        key="dalin_lrj_diagnostic",
        subject="Dalin",
        csv_path=OPENBCI_RUNS_ROOT / "Dalin" / "Dalin-LRJ(1,2)-4:7.csv",
        protocol_family="diagnostic",
        trust_tier="diagnostic",
        processing_family="diagnostic",
        marker_semantics="Diagnostic-only LRJ one-off; do not promote into the first benchmark scorecard",
        supported_targets=("diagnostic",),
    ),
    FileContract(
        key="dalin_lr6_diagnostic",
        subject="Dalin",
        csv_path=OPENBCI_RUNS_ROOT / "Dalin" / "Dalin-LR(6)-4:7.csv",
        protocol_family="diagnostic",
        trust_tier="diagnostic",
        processing_family="diagnostic",
        marker_semantics="Diagnostic-only LR(6) benchmark; useful for frozen counter sanity checks only",
        supported_targets=("diagnostic",),
    ),
)


def all_file_contracts() -> list[FileContract]:
    return list(MANUAL_FILE_CONTRACTS)


def contracts_for_protocol(protocol_family: str) -> list[FileContract]:
    return [contract for contract in MANUAL_FILE_CONTRACTS if contract.protocol_family == protocol_family]


def contracts_for_trust_tier(trust_tier: str) -> list[FileContract]:
    return [contract for contract in MANUAL_FILE_CONTRACTS if contract.trust_tier == trust_tier]


def jaw_hr_contracts() -> list[FileContract]:
    return contracts_for_protocol("HR")


def lrj_main_contracts() -> list[FileContract]:
    return [contract for contract in MANUAL_FILE_CONTRACTS if contract.protocol_family == "LRJ"]


def eeg_lr_main_contracts() -> list[FileContract]:
    return [
        contract
        for contract in MANUAL_FILE_CONTRACTS
        if contract.protocol_family == "LR" and contract.trust_tier == "main"
    ]


def eeg_lr_stress_contracts() -> list[FileContract]:
    return [
        contract
        for contract in MANUAL_FILE_CONTRACTS
        if contract.protocol_family == "LR" and contract.trust_tier == "stress"
    ]


def diagnostic_contracts() -> list[FileContract]:
    return contracts_for_trust_tier("diagnostic")


def registry_dataframe(contracts: Iterable[FileContract] | None = None) -> pd.DataFrame:
    rows = []
    for contract in contracts or MANUAL_FILE_CONTRACTS:
        rows.append(
            {
                "key": contract.key,
                "subject": contract.subject,
                "filename": contract.filename,
                "csv_path": str(contract.csv_path.resolve()),
                "protocol_family": contract.protocol_family,
                "trust_tier": contract.trust_tier,
                "processing_family": contract.processing_family,
                "marker_semantics": contract.marker_semantics,
                "supported_targets": ", ".join(contract.supported_targets),
            }
        )
    return pd.DataFrame(rows)
