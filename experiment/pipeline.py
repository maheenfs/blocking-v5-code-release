"""Core experiment logic for Blocking V5.

This file holds the data preparation path, the training loop, the evaluation
code, and the stage runners used by the outer CLI.
"""

from __future__ import annotations

import csv
import json
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .checkpoints import (
    BL_AG_STAGE_CHECKPOINT,
    BL_AG_STORED_CHECKPOINT,
    bl_ag_checkpoint_candidates,
    find_bl_ag_checkpoint,
)
from .names import slugify
from .preflight import PREPARED_DATA_FILES, validate_config_for_release, validate_run_dependencies
from .system import configure_torch_runtime, load_system_settings


RESULTS_SCHEMA_VERSION = "v4.1"
PIPELINE_VERSION = "blocking_v5.0"
BLOCKAGE_PROTOCOL_NAME = "underlying_tx_consistent"
SETB_FEATURE_MODE = "max_over_rx"

StageCallback = Callable[[str, Dict[str, object]], None]


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------
def ensure_dir(path: Path) -> None:
    """Create a directory tree if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def temporary_sibling_path(path: Path) -> Path:
    """Return a collision-resistant temp path beside the target file."""
    return path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write text via a temporary file so stage outputs do not appear half-written."""
    ensure_dir(path.parent)
    tmp = temporary_sibling_path(path)
    tmp.write_text(text, encoding=encoding)
    tmp.replace(path)


def atomic_write_json(path: Path, payload: object) -> None:
    """Serialize JSON using the same atomic file-write pattern as text outputs."""
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=False) + "\n")


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    """Write NumPy arrays atomically so interrupted prepare stages cannot leave corrupt arrays behind."""
    ensure_dir(path.parent)
    tmp = temporary_sibling_path(path)
    with tmp.open("wb") as handle:
        np.save(handle, array)
    tmp.replace(path)


def atomic_torch_save(path: Path, payload: object) -> None:
    """Write Torch checkpoints atomically so interrupted training never leaves a partial checkpoint."""
    ensure_dir(path.parent)
    tmp = temporary_sibling_path(path)
    with tmp.open("wb") as handle:
        torch.save(payload, handle)
    tmp.replace(path)


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    """Write a CSV table atomically so researchers always see a complete file."""
    ensure_dir(path.parent)
    tmp = temporary_sibling_path(path)
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp.replace(path)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    """Read a CSV table into dictionaries keyed by the header row."""
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_npy(path: Path, **kwargs: Any) -> np.ndarray:
    """Load a NumPy array without enabling object-pickle payloads."""
    return np.load(path, allow_pickle=False, **kwargs)


def load_torch_state_dict(path: Path) -> Dict[str, Any]:
    """Load a local model state dict while asking new PyTorch versions to avoid pickle objects."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # PyTorch versions before weights_only still need to load these local
        # release checkpoints. The release never accepts checkpoint paths from
        # untrusted network input.
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a Torch state dict in {path}, got {type(payload).__name__}")
    return payload


def files_exist(directory: Path, names: Sequence[str]) -> bool:
    """Check whether a stage already wrote the full set of files it is expected to produce."""
    return all((directory / name).exists() for name in names)


def clamp_int(value: int, low: int, high: int) -> int:
    """Clamp an integer into an inclusive range."""
    return int(max(low, min(high, int(value))))


def format_metric(value: object, *, ndigits: int = 6) -> str:
    """Format a numeric metric for CSV output while keeping NaN values explicit."""
    try:
        number = float(value)
    except Exception:
        return "nan"
    if not np.isfinite(number):
        return "nan"
    return f"{number:.{int(ndigits)}f}"


def format_seconds(seconds: float) -> str:
    """Format elapsed seconds as m:ss or h:mm:ss for progress displays."""
    total_seconds = int(round(max(0.0, float(seconds))))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds_int = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{seconds_int:02d}"
    return f"{minutes:d}:{seconds_int:02d}"


def set_global_seeds(seed: int) -> None:
    """Use strict reproducible settings on CPU."""

    seed_value = int(seed)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Configuration model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SplitConfig:
    train_frac: float
    val_frac: float
    test_frac: float


@dataclass(frozen=True)
class TrainingConfig:
    seed: int
    seeds: Tuple[int, ...]
    batch_size: int
    lr: float
    weight_decay: float
    kpi_loss_weight: float
    pretrain_epochs: int
    aware_epochs: int
    ft_epoch_sweep: Tuple[int, ...]
    ft_freeze_modes: Tuple[str, ...]
    train_fracs_sweep: Tuple[float, ...]
    ft_sampling: str


@dataclass(frozen=True)
class KpiConfig:
    topks: Tuple[int, ...]
    margins_db: Tuple[int, ...]
    primary_topk: int
    primary_margin_db: int

    def primary_key(self) -> str:
        return f"top{self.primary_topk}_m{self.primary_margin_db}db_%"


@dataclass(frozen=True)
class BatchJob:
    pattern: int
    blocked_beam_index: int
    description: str


@dataclass(frozen=True)
class ExperimentConfig:
    values_cube_path: Path
    outdir: Path
    run_name_prefix: str
    blockage_model: str
    setb_pattern: int
    blocked_setb_indices: Tuple[int, ...]
    stages: Tuple[str, ...]
    experiments: Tuple[str, ...]
    blockage_levels: Tuple[int, ...]
    split: SplitConfig
    training: TrainingConfig
    kpi: KpiConfig
    plotting: Dict[str, object]
    overwrite: bool
    batch_jobs: Tuple[BatchJob, ...]
    representative_beams: Dict[str, int]
    scenario_provenance: Dict[str, object]
    config_path: Path

    @classmethod
    def from_dict(cls, payload: Dict[str, object], *, config_path: Path) -> "ExperimentConfig":
        """Build the typed experiment config and resolve relative paths."""
        base_dir = config_path.parent.resolve()

        def resolve_path(raw: object) -> Path:
            # Relative paths in the package config stay portable by resolving them
            # from the config file location instead of the current shell directory.
            path = Path(str(raw)).expanduser()
            if path.is_absolute():
                return path.resolve()
            return (base_dir / path).resolve()

        split_raw = dict(payload.get("split", {}))
        training_raw = dict(payload.get("training", {}))
        kpi_raw = dict(payload.get("kpi", {}))
        plotting_raw = dict(payload.get("plotting", {}))

        batch_jobs_payload = payload.get("batch_jobs") or []
        batch_jobs = tuple(
            BatchJob(
                pattern=int(item["pattern"]),
                blocked_beam_index=int(item["blocked_beam_index"]),
                description=str(item.get("description", f"P{item['pattern']} block B{item['blocked_beam_index']}")),
            )
            for item in batch_jobs_payload
        )

        seed = int(training_raw["seed"])
        seeds_raw = training_raw.get("seeds", [seed])
        if not seeds_raw:
            seeds_raw = [seed]

        plotting_cfg: Dict[str, object] = dict(plotting_raw)
        plotting_cfg["baseline_ft_epoch_for_plots"] = int(plotting_cfg.get("baseline_ft_epoch_for_plots", 10))

        return cls(
            values_cube_path=resolve_path(payload["values_cube_path"]),
            outdir=resolve_path(payload["outdir"]),
            run_name_prefix=str(payload["run_name_prefix"]),
            blockage_model=str(payload.get("blockage_model", BLOCKAGE_PROTOCOL_NAME)),
            setb_pattern=int(payload.get("setb_pattern", 1)),
            blocked_setb_indices=tuple(int(x) for x in payload.get("blocked_setb_indices", [])),
            stages=tuple(str(x) for x in payload.get("stages", [])),
            experiments=tuple(str(x) for x in payload.get("experiments", [])),
            blockage_levels=tuple(int(x) for x in payload.get("blockage_levels", [])),
            split=SplitConfig(
                train_frac=float(split_raw["train_frac"]),
                val_frac=float(split_raw["val_frac"]),
                test_frac=float(split_raw["test_frac"]),
            ),
            training=TrainingConfig(
                seed=seed,
                seeds=tuple(int(x) for x in seeds_raw),
                batch_size=int(training_raw["batch_size"]),
                lr=float(training_raw["lr"]),
                weight_decay=float(training_raw["weight_decay"]),
                kpi_loss_weight=float(training_raw.get("kpi_loss_weight", 0.0)),
                pretrain_epochs=int(training_raw["pretrain_epochs"]),
                aware_epochs=int(training_raw["aware_epochs"]),
                ft_epoch_sweep=tuple(int(x) for x in training_raw["ft_epoch_sweep"]),
                ft_freeze_modes=tuple(str(x) for x in training_raw["ft_freeze_modes"]),
                train_fracs_sweep=tuple(float(x) for x in training_raw["train_fracs_sweep"]),
                ft_sampling=str(training_raw["ft_sampling"]),
            ),
            kpi=KpiConfig(
                topks=tuple(int(x) for x in kpi_raw["topks"]),
                margins_db=tuple(int(x) for x in kpi_raw["margins_db"]),
                primary_topk=int(kpi_raw["primary_topk"]),
                primary_margin_db=int(kpi_raw["primary_margin_db"]),
            ),
            plotting=plotting_cfg,
            overwrite=bool(payload.get("overwrite", False)),
            batch_jobs=batch_jobs,
            representative_beams={str(k): int(v) for k, v in dict(payload.get("representative_beams", {})).items()},
            scenario_provenance=dict(payload.get("scenario_provenance", {})),
            config_path=config_path.resolve(),
        )

    def run_name(self, *, pattern: int, blocked_beam_index: int) -> str:
        return f"{self.run_name_prefix}_P{int(pattern)}_blockB{int(blocked_beam_index)}"

    def run_dir(self, job: BatchJob) -> Path:
        return self.outdir / self.run_name(pattern=job.pattern, blocked_beam_index=job.blocked_beam_index)

    def seed_values(self) -> Tuple[int, ...]:
        values: List[int] = [int(self.training.seed)]
        values.extend(int(seed) for seed in self.training.seeds)
        return tuple(dict.fromkeys(values))

    def for_seed(self, seed: int) -> "ExperimentConfig":
        seed_value = int(seed)
        prefix = self.run_name_prefix
        for marker in ("_seed", "_s"):
            head, sep, tail = prefix.rpartition(marker)
            if sep and tail.isdigit():
                prefix = head
                break
        prefix = f"{prefix}_seed{seed_value}"
        return replace(
            self,
            run_name_prefix=prefix,
            training=replace(self.training, seed=seed_value, seeds=(seed_value,)),
        )


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate the experiment config file."""
    config_path = Path(path).expanduser().resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    config = ExperimentConfig.from_dict(payload, config_path=config_path)
    if config.blockage_model != BLOCKAGE_PROTOCOL_NAME:
        raise ValueError(
            f"Only blockage_model={BLOCKAGE_PROTOCOL_NAME!r} is supported. "
            f"Got {config.blockage_model!r}."
        )
    validate_config_for_release(config)
    return config


# ---------------------------------------------------------------------------
# Beam layout and feature mapping
# ---------------------------------------------------------------------------


def seta_grid_angles() -> Tuple[List[float], List[float]]:
    """Angle grid for the 32 Set-A transmitter beams."""
    azimuths = [-52.5, -37.5, -22.5, -7.5, 7.5, 22.5, 37.5, 52.5]
    elevations = [106.25, 118.75, 131.25, 143.75]
    return azimuths, elevations


def setb_pattern1_angles() -> List[Tuple[float, float]]:
    """Pattern-1 Set-B beams in the order used by the experiment."""
    return [
        (-52.5, 143.75),
        (-37.5, 131.25),
        (-22.5, 118.75),
        (-7.5, 106.25),
        (7.5, 143.75),
        (22.5, 131.25),
        (37.5, 118.75),
        (52.5, 106.25),
    ]


def setb_pattern2_angles() -> List[Tuple[float, float]]:
    """Pattern-2 Set-B beams in the order used by the experiment."""
    return [
        (-7.5, 143.75),
        (-22.5, 131.25),
        (-37.5, 118.75),
        (-52.5, 106.25),
        (52.5, 143.75),
        (37.5, 131.25),
        (22.5, 118.75),
        (7.5, 106.25),
    ]


def setb_pattern_angles(pattern: int) -> List[Tuple[float, float]]:
    """Set-B beam layout for the requested pattern."""
    if int(pattern) == 1:
        return setb_pattern1_angles()
    if int(pattern) == 2:
        return setb_pattern2_angles()
    raise ValueError(f"Unknown Set-B pattern: {pattern}")


def angle_to_tx_index(angle: Tuple[float, float], az_grid: Sequence[float], el_grid: Sequence[float]) -> int:
    """Map one beam angle pair onto the flattened Set-A index."""
    azimuth, elevation = float(angle[0]), float(angle[1])
    if azimuth not in az_grid or elevation not in el_grid:
        raise ValueError(f"AoD {angle} is not present in the Set-A grid.")
    return int(el_grid.index(elevation) * len(az_grid) + az_grid.index(azimuth))


def build_setb_tx_indices(pattern: int) -> List[int]:
    """Convert one Set-B layout into Set-A transmitter indices."""
    azimuths, elevations = seta_grid_angles()
    return [angle_to_tx_index(angle, azimuths, elevations) for angle in setb_pattern_angles(pattern)]


def build_setb_mapping_rows(pattern: int) -> List[Dict[str, object]]:
    """Rows for the run-local Set-B mapping table."""
    azimuths, elevations = seta_grid_angles()
    rows: List[Dict[str, object]] = []
    for beam_index, (angle, tx_index) in enumerate(zip(setb_pattern_angles(pattern), build_setb_tx_indices(pattern))):
        azimuth, elevation = float(angle[0]), float(angle[1])
        rows.append(
            {
                "setb_order": int(beam_index),
                "beam_name": f"B{beam_index}",
                "az_deg": azimuth,
                "el_deg": elevation,
                "az_idx": int(azimuths.index(azimuth)),
                "el_idx": int(elevations.index(elevation)),
                "tx_index": int(tx_index),
                "setb_pattern": int(pattern),
            }
        )
    return rows


def load_values_cube(path: Path) -> np.ndarray:
    """Load the input values cube from either `.npy` or `.npz` packaging."""
    loaded = load_npy(path)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        with loaded:
            if "values_cube" not in loaded.files:
                raise ValueError(f"{path} is missing 'values_cube'. Available keys: {loaded.files}")
            array = loaded["values_cube"]
    else:
        array = loaded
    if array.ndim != 3:
        raise ValueError(f"Expected [N, 32, Rx], got {array.shape}")
    if int(array.shape[1]) != 32:
        raise ValueError(f"Expected 32 transmitter beams, got {array.shape}")
    if int(array.shape[2]) < 1:
        raise ValueError(f"Expected at least one receiver slice, got {array.shape}")

    # Silent NaN or Inf values would leak into labels, blockage, and reported KPIs.
    finite_mask = np.isfinite(array)
    if not bool(finite_mask.all()):
        invalid_count = int(np.count_nonzero(~finite_mask))
        raise ValueError(f"Input values cube contains {invalid_count} non-finite values.")
    return np.asarray(array)


def best_tx_labels(values_cube: np.ndarray) -> np.ndarray:
    """Choose the best transmitter beam label from the full 32-beam world."""
    # Labels come from the full 32-beam world, not from the Set-B subset seen by the model.
    return values_cube.max(axis=2).argmax(axis=1).astype(np.int64)


def build_setb_features(values_cube: np.ndarray, setb_tx: Sequence[int]) -> np.ndarray:
    """Build Set-B features by taking the best Rx value for each selected transmitter beam."""
    setb_tx_array = np.asarray(list(setb_tx), dtype=np.int64)
    # Each Set-B feature is one Tx beam collapsed over the available Rx dimension.
    features = values_cube[:, setb_tx_array, :].max(axis=2)
    return features.astype(np.float32, copy=False)


def build_setb_features_from_rsrp(rsrp_tx: np.ndarray, setb_tx: Sequence[int]) -> np.ndarray:
    """Rebuild Set-B features from an already-collapsed [N, 32] RSRP matrix."""
    setb_tx_array = np.asarray(list(setb_tx), dtype=np.int64)
    return rsrp_tx[:, setb_tx_array].astype(np.float32, copy=False)


def split_indices(
    sample_count: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create deterministic train/val/test splits from one shuffled index order."""
    if sample_count < 3:
        raise ValueError("Need at least three samples for train/val/test splits.")
    fractions = [float(train_frac), float(val_frac), float(test_frac)]
    if any(value <= 0.0 for value in fractions):
        raise ValueError("train_frac, val_frac, and test_frac must all be positive.")
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0.")

    rng = np.random.default_rng(int(seed))
    indices = np.arange(int(sample_count), dtype=np.int64)
    # One global shuffle keeps every downstream stage aligned to the same split.
    rng.shuffle(indices)

    train_count = int(round(train_frac * sample_count))
    val_count = int(round(val_frac * sample_count))

    # The rounding pass above can leave one split empty on small datasets, so the
    # counts are corrected here before the index slices are taken.
    train_count = max(1, min(sample_count - 2, train_count))
    val_count = max(1, min(sample_count - train_count - 1, val_count))
    test_count = sample_count - train_count - val_count
    if test_count <= 0:
        test_count = 1
        if val_count > 1:
            val_count -= 1
        else:
            train_count -= 1

    train_idx = indices[:train_count]
    val_idx = indices[train_count : train_count + val_count]
    test_idx = indices[train_count + val_count :]
    return train_idx, val_idx, test_idx


def apply_blockage_to_tx_column(
    rsrp_tx: np.ndarray,
    *,
    tx_col: int,
    global_min_db: float,
    blockage_pct: int,
) -> np.ndarray:
    """Attenuate one transmitter column toward the global minimum according to blockage level."""
    if blockage_pct <= 0:
        return rsrp_tx.astype(np.float32, copy=False)
    blocked = np.asarray(rsrp_tx, dtype=np.float32).copy()
    keep_fraction = 1.0 - (float(clamp_int(blockage_pct, 0, 100)) / 100.0)
    column = blocked[:, int(tx_col)]
    blocked[:, int(tx_col)] = float(global_min_db) + keep_fraction * (column - float(global_min_db))
    return blocked.astype(np.float32, copy=False)


def build_blocked_dataset_views(
    *,
    clean_features: np.ndarray,
    clean_labels: np.ndarray,
    clean_rsrp: np.ndarray,
    setb_pattern: int,
    blocked_setb_col: int,
    global_min_db: float,
    blockage_pct: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply blockage in the 32-beam space, then rebuild Set-B features and labels from it."""
    if int(blockage_pct) == 0:
        return clean_features, clean_labels, clean_rsrp

    setb_tx = build_setb_tx_indices(int(setb_pattern))
    blocked_tx = int(setb_tx[int(blocked_setb_col)])
    # Blockage is applied before feature rebuilding so labels and Set-B inputs stay consistent.
    rsrp_blocked = apply_blockage_to_tx_column(
        clean_rsrp,
        tx_col=blocked_tx,
        global_min_db=float(global_min_db),
        blockage_pct=int(blockage_pct),
    )
    features_blocked = build_setb_features_from_rsrp(rsrp_blocked, setb_tx)
    labels_blocked = rsrp_blocked.argmax(axis=1).astype(np.int64, copy=False)
    return features_blocked, labels_blocked, rsrp_blocked


class RsrpDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, rsrp_tx: np.ndarray) -> None:
        """Wrap the NumPy arrays used by training and evaluation in a Torch dataset."""
        if features.ndim != 2:
            raise ValueError(f"Expected features [N, |SetB|], got {features.shape}")
        if labels.ndim != 1:
            raise ValueError(f"Expected labels [N], got {labels.shape}")
        if rsrp_tx.ndim != 2 or int(rsrp_tx.shape[1]) != 32:
            raise ValueError(f"Expected rsrp_tx [N, 32], got {rsrp_tx.shape}")
        if int(features.shape[0]) != int(labels.shape[0]) or int(features.shape[0]) != int(rsrp_tx.shape[0]):
            raise ValueError("Features, labels, and rsrp_tx must have matching first dimensions.")

        self.features = torch.tensor(features, dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.rsrp_tx = torch.tensor(rsrp_tx, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index], self.rsrp_tx[index]


def make_loader(
    features: np.ndarray,
    labels: np.ndarray,
    rsrp_tx: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Build a deterministic DataLoader for one set of features, labels, and RSRP targets."""
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        RsrpDataset(features, labels, rsrp_tx),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=0,
        pin_memory=False,
        generator=generator,
    )


# ---------------------------------------------------------------------------
# Model and training
# ---------------------------------------------------------------------------
class CNNBeamPredictor(nn.Module):
    def __init__(self, *, setb_size: int, num_classes: int = 32) -> None:
        """Small 1D CNN used throughout the experiment for beam prediction."""
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(32, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * int(setb_size), 480)
        self.fc2 = nn.Linear(480, int(num_classes))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(1)
        if inputs.dim() != 3 or int(inputs.shape[1]) != 1:
            raise ValueError(f"Expected [batch, 1, setb] or [batch, setb], got {tuple(inputs.shape)}")
        outputs = F.relu(self.conv1(inputs))
        outputs = F.relu(self.conv2(outputs))
        outputs = F.relu(self.conv3(outputs))
        outputs = outputs.flatten(1)
        outputs = F.relu(self.fc1(outputs))
        return self.fc2(outputs)

    def freeze_cnn(self) -> None:
        """Freeze the convolution stack for FT runs that only update the head."""
        for module in (self.conv1, self.conv2, self.conv3):
            for parameter in module.parameters():
                parameter.requires_grad = False

    def freeze_head(self) -> None:
        """Freeze the fully connected head for FT runs that only update the CNN."""
        for module in (self.fc1, self.fc2):
            for parameter in module.parameters():
                parameter.requires_grad = False


def apply_freeze_mode(model: CNNBeamPredictor, freeze_mode: str) -> None:
    """Apply one configured freeze label to a model before counting or training."""
    mode = str(freeze_mode)
    if mode in {"none", "no_freeze"}:
        return
    if mode == "freeze_cnn":
        model.freeze_cnn()
        return
    if mode == "freeze_head":
        model.freeze_head()
        return
    if mode == "freeze_all":
        for parameter in model.parameters():
            parameter.requires_grad = False
        return
    raise ValueError(f"Unknown freeze mode: {freeze_mode}")


def count_model_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return total and trainable parameter counts for one model."""
    total = int(sum(parameter.numel() for parameter in model.parameters()))
    trainable = int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    return total, trainable


def parameter_counts_for_freeze(*, setb_size: int, freeze_mode: str) -> Tuple[int, int]:
    """Compute parameter counts without loading a checkpoint."""
    model = CNNBeamPredictor(setb_size=int(setb_size), num_classes=32)
    apply_freeze_mode(model, str(freeze_mode))
    return count_model_parameters(model)


@dataclass
class EpochRecord:
    epoch: int
    val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    train_loss_total: float
    train_loss_ce: float
    train_loss_kpi: float
    epoch_time_s: float


@dataclass
class TrainResult:
    best_epoch: int
    best_val_primary_pct: float
    test_primary_at_best_pct: float
    total_params: int
    trainable_params: int
    train_time_s: float
    best_val_metrics: Dict[str, float]
    best_test_metrics: Dict[str, float]
    epoch_records: List[EpochRecord]
    best_state_dict: Dict[str, torch.Tensor]


@dataclass
class PreparedData:
    features: np.ndarray
    labels: np.ndarray
    rsrp_tx: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    global_min_db: float


def build_metric_keys(kpi: KpiConfig) -> List[str]:
    """List every metric column that should appear in histories and aggregated results."""
    topks = sorted({int(x) for x in kpi.topks if int(x) >= 1})
    margins = sorted({int(x) for x in kpi.margins_db if int(x) >= 0})
    keys = ["top1_%"]
    keys.extend(f"top{k}_incl_%" for k in topks if k > 1)
    for topk in topks:
        for margin in margins:
            keys.append(f"top{topk}_m{margin}db_%")
    keys.extend(["avg_rsrp_loss_db", "p95_rsrp_loss_db"])
    return keys


def metrics_to_prefixed_row(metrics: Dict[str, float], *, prefix: str, metric_keys: Sequence[str]) -> Dict[str, str]:
    """Attach a prefix such as `val_` or `test_` to one metric dictionary for CSV output."""
    return {f"{prefix}{key}": format_metric(metrics.get(key, float("nan"))) for key in metric_keys}


def kpi_aligned_surrogate_loss(
    logits: torch.Tensor,
    rsrp_tx: torch.Tensor,
    *,
    primary_topk: int,
    primary_margin_db: float,
) -> torch.Tensor:
    """Optional loss term that rewards probability mass on KPI-acceptable beams."""
    topk = int(max(1, min(int(primary_topk), int(logits.shape[1]))))
    margin_db = float(max(0.0, primary_margin_db))
    probabilities = torch.softmax(logits, dim=1)
    genie = rsrp_tx.max(dim=1, keepdim=True).values
    # Any beam inside the primary dB margin counts as acceptable for this soft KPI objective.
    acceptable = (rsrp_tx >= (genie - margin_db)).to(dtype=probabilities.dtype)
    acceptable_mass = probabilities * acceptable
    topk_mass = torch.topk(acceptable_mass, k=topk, dim=1).values.sum(dim=1)
    return -torch.log(torch.clamp(topk_mass, min=1e-8)).mean()


@torch.no_grad()
def evaluate_beam_kpis(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    rsrp_tx: np.ndarray,
    *,
    kpi: KpiConfig,
    device: torch.device,
    batch_size: int,
) -> Dict[str, float]:
    """Evaluate the configured Top-K and RSRP-loss metrics on one dataset split."""
    model.eval()
    metric_keys = build_metric_keys(kpi)
    topks = sorted({int(x) for x in kpi.topks if int(x) >= 1})
    margins = sorted({int(x) for x in kpi.margins_db if int(x) >= 0})
    max_topk = int(max(topks))

    loader = make_loader(features, labels, rsrp_tx, batch_size=batch_size, shuffle=False, seed=0)

    total = 0
    top1_correct = 0
    inclusion_counts = {k: 0 for k in topks if k > 1}
    margin_counts = {(k, m): 0 for k in topks for m in margins}
    rsrp_losses: List[float] = []

    for batch_features, batch_labels, batch_rsrp in loader:
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)
        batch_rsrp = batch_rsrp.to(device)

        logits = model(batch_features)
        top_indices = torch.topk(logits, k=max_topk, dim=1).indices
        top1 = top_indices[:, 0]
        top1_correct += int((top1 == batch_labels).sum().item())

        for topk in inclusion_counts:
            inclusion = (top_indices[:, :topk] == batch_labels.unsqueeze(1)).any(dim=1)
            inclusion_counts[topk] += int(inclusion.sum().item())

        predicted_rsrp = batch_rsrp.gather(1, top_indices)
        genie_rsrp = batch_rsrp.max(dim=1).values
        for topk in topks:
            best_inside_topk = predicted_rsrp[:, :topk].max(dim=1).values
            for margin in margins:
                success = best_inside_topk >= (genie_rsrp - float(margin))
                margin_counts[(topk, margin)] += int(success.sum().item())

        top1_rsrp = batch_rsrp.gather(1, top1.unsqueeze(1)).squeeze(1)
        rsrp_losses.extend((genie_rsrp - top1_rsrp).detach().cpu().numpy().astype(float).tolist())
        total += int(batch_labels.numel())

    if total == 0:
        return {key: float("nan") for key in metric_keys}

    metrics: Dict[str, float] = {
        "top1_%": 100.0 * float(top1_correct) / float(total),
    }
    for topk, count in inclusion_counts.items():
        metrics[f"top{topk}_incl_%"] = 100.0 * float(count) / float(total)
    for (topk, margin), count in margin_counts.items():
        metrics[f"top{topk}_m{margin}db_%"] = 100.0 * float(count) / float(total)

    loss_array = np.asarray(rsrp_losses, dtype=np.float64)
    metrics["avg_rsrp_loss_db"] = float(np.mean(loss_array)) if loss_array.size else float("nan")
    metrics["p95_rsrp_loss_db"] = float(np.percentile(loss_array, 95.0)) if loss_array.size else float("nan")
    for key in metric_keys:
        metrics.setdefault(key, float("nan"))
    return metrics


def evaluate_ranked_beams(rankings: np.ndarray, rsrp_tx: np.ndarray, *, kpi: KpiConfig) -> Dict[str, float]:
    """Evaluate a non-ML beam ranking against the same KPI definitions used by the model."""
    labels = np.asarray(rsrp_tx, dtype=np.float64).argmax(axis=1).astype(np.int64, copy=False)
    metric_keys = build_metric_keys(kpi)
    topks = sorted({int(x) for x in kpi.topks if int(x) >= 1})
    margins = sorted({int(x) for x in kpi.margins_db if int(x) >= 0})
    max_topk = int(max(topks))
    pred = np.asarray(rankings[:, :max_topk], dtype=np.int64)
    total = int(pred.shape[0])
    if total == 0:
        return {key: float("nan") for key in metric_keys}

    pred_rsrp = np.take_along_axis(np.asarray(rsrp_tx, dtype=np.float64), pred, axis=1)
    genie = np.asarray(rsrp_tx, dtype=np.float64).max(axis=1)
    metrics: Dict[str, float] = {"top1_%": 100.0 * float(np.mean(pred[:, 0] == labels))}
    for topk in topks:
        if topk > 1:
            metrics[f"top{topk}_incl_%"] = 100.0 * float(np.mean(np.any(pred[:, :topk] == labels[:, None], axis=1)))
        best_inside = pred_rsrp[:, :topk].max(axis=1)
        for margin in margins:
            metrics[f"top{topk}_m{margin}db_%"] = 100.0 * float(np.mean(best_inside >= (genie - float(margin))))
    top1_rsrp = pred_rsrp[:, 0]
    loss = genie - top1_rsrp
    metrics["avg_rsrp_loss_db"] = float(np.mean(loss))
    metrics["p95_rsrp_loss_db"] = float(np.percentile(loss, 95.0))
    for key in metric_keys:
        metrics.setdefault(key, float("nan"))
    return metrics


def all_seta_tx_angles() -> np.ndarray:
    """Full Set-A angle table in flattened transmitter-index order."""
    azimuths, elevations = seta_grid_angles()
    return np.asarray([(az, el) for el in elevations for az in azimuths], dtype=np.float64)


def max_setb_rankings(features_b: np.ndarray, pattern: int) -> np.ndarray:
    """Rank the measured Set-B beams by signal strength and map them into Set-A IDs."""
    setb_tx = np.asarray(build_setb_tx_indices(int(pattern)), dtype=np.int64)
    order = np.argsort(-np.asarray(features_b, dtype=np.float64), axis=1)
    return setb_tx[order]


def nn_angle_space_rankings(features_b: np.ndarray, pattern: int) -> np.ndarray:
    """Use the strongest Set-B beam as an anchor and rank all Set-A beams by angle."""
    strongest_setb_col = np.argmax(np.asarray(features_b, dtype=np.float64), axis=1)
    setb_angles = np.asarray(setb_pattern_angles(int(pattern)), dtype=np.float64)
    selected_angles = setb_angles[strongest_setb_col]
    tx_angles = all_seta_tx_angles()
    dist2 = np.sum((selected_angles[:, None, :] - tx_angles[None, :, :]) ** 2, axis=2)
    return np.argsort(dist2, axis=1).astype(np.int64, copy=False)


def random_setb_rankings(features_b: np.ndarray, pattern: int, *, seed: int) -> np.ndarray:
    """Deterministic random Set-B ranking used only as an optional sanity floor."""
    rng = np.random.default_rng(int(seed))
    setb_tx = np.asarray(build_setb_tx_indices(int(pattern)), dtype=np.int64)
    scores = rng.random((int(features_b.shape[0]), len(setb_tx)))
    return setb_tx[np.argsort(scores, axis=1)]


def non_ml_rankings(method: str, features_b: np.ndarray, pattern: int, *, seed: int) -> np.ndarray:
    name = str(method).upper()
    if name == "MAX-SETB":
        return max_setb_rankings(features_b, int(pattern))
    if name == "NN-ANGLE":
        return nn_angle_space_rankings(features_b, int(pattern))
    if name == "RANDOM-SETB":
        return random_setb_rankings(features_b, int(pattern), seed=int(seed))
    raise ValueError(f"Unknown non-ML baseline method: {method}")


def train_model(
    *,
    init_state_dict: Optional[Dict[str, torch.Tensor]],
    freeze_policy: str,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_rsrp: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    val_rsrp: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    test_rsrp: np.ndarray,
    training: TrainingConfig,
    epochs: int,
    kpi: KpiConfig,
    device: torch.device,
    seed: int,
    epoch_callback: Optional[Callable[[Dict[str, object]], None]] = None,
    job_name: str = "",
) -> TrainResult:
    """Train one model instance and keep the epoch with the best validation primary KPI."""
    if freeze_policy not in {"none", "freeze_cnn", "freeze_head"}:
        raise ValueError(f"Unknown freeze_policy: {freeze_policy}")

    set_global_seeds(seed)

    model = CNNBeamPredictor(setb_size=int(train_features.shape[1]), num_classes=32)
    if init_state_dict is not None:
        model.load_state_dict(init_state_dict, strict=True)

    apply_freeze_mode(model, freeze_policy)

    model.to(device)

    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters left after freezing.")
    total_params, trainable_param_count = count_model_parameters(model)

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(training.lr),
        weight_decay=float(training.weight_decay),
    )
    criterion = nn.CrossEntropyLoss()
    # Training stays deterministic because the loader order is driven by the same
    # explicit seed that was already applied to NumPy and Torch.
    loader = make_loader(
        train_features,
        train_labels,
        train_rsrp,
        batch_size=int(training.batch_size),
        shuffle=True,
        seed=int(seed),
    )

    best_val_primary = float("-inf")
    best_epoch = 0
    best_test_primary = float("-inf")
    best_state: Dict[str, torch.Tensor] = {}
    best_val_metrics: Dict[str, float] = {}
    best_test_metrics: Dict[str, float] = {}
    epoch_records: List[EpochRecord] = []
    primary_key = kpi.primary_key()
    train_start = time.perf_counter()

    for epoch in range(1, int(epochs) + 1):
        epoch_start = time.perf_counter()
        model.train()
        loss_total_sum = 0.0
        loss_ce_sum = 0.0
        loss_kpi_sum = 0.0
        sample_count = 0
        for batch_features, batch_labels, batch_rsrp in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            batch_rsrp = batch_rsrp.to(device)
            batch_size = int(batch_labels.shape[0])

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)

            # Cross-entropy remains the base objective for exact best-beam prediction.
            ce_loss = criterion(logits, batch_labels)
            kpi_loss_term = torch.tensor(0.0, device=device)
            if float(training.kpi_loss_weight) > 0.0:
                # The optional KPI term shifts probability toward any beam that would satisfy
                # the primary Top-K-within-margin target.
                kpi_loss_term = float(training.kpi_loss_weight) * kpi_aligned_surrogate_loss(
                    logits,
                    batch_rsrp,
                    primary_topk=int(kpi.primary_topk),
                    primary_margin_db=float(kpi.primary_margin_db),
                )
            loss = ce_loss + kpi_loss_term
            loss.backward()
            optimizer.step()

            loss_total_sum += float(loss.detach().item()) * batch_size
            loss_ce_sum += float(ce_loss.detach().item()) * batch_size
            loss_kpi_sum += float(kpi_loss_term.detach().item()) * batch_size
            sample_count += batch_size

        # Validation chooses the best checkpoint. Test metrics are tracked every
        # epoch only so the selected epoch can report the matching test result.
        val_metrics = evaluate_beam_kpis(
            model,
            val_features,
            val_labels,
            val_rsrp,
            kpi=kpi,
            device=device,
            batch_size=int(training.batch_size),
        )
        test_metrics = evaluate_beam_kpis(
            model,
            test_features,
            test_labels,
            test_rsrp,
            kpi=kpi,
            device=device,
            batch_size=int(training.batch_size),
        )
        epoch_time_s = time.perf_counter() - epoch_start
        epoch_records.append(
            EpochRecord(
                epoch=int(epoch),
                val_metrics=dict(val_metrics),
                test_metrics=dict(test_metrics),
                train_loss_total=(loss_total_sum / sample_count) if sample_count else float("nan"),
                train_loss_ce=(loss_ce_sum / sample_count) if sample_count else float("nan"),
                train_loss_kpi=(loss_kpi_sum / sample_count) if sample_count else float("nan"),
                epoch_time_s=float(epoch_time_s),
            )
        )

        val_primary = float(val_metrics.get(primary_key, float("nan")))
        test_primary = float(test_metrics.get(primary_key, float("nan")))
        # Model selection follows the validation primary KPI exactly, then reports
        # the matching test metrics from that selected epoch.
        if np.isfinite(val_primary) and val_primary > best_val_primary:
            best_val_primary = float(val_primary)
            best_test_primary = float(test_primary) if np.isfinite(test_primary) else float("nan")
            best_epoch = int(epoch)
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            best_val_metrics = dict(val_metrics)
            best_test_metrics = dict(test_metrics)

        if epoch_callback is not None:
            record = epoch_records[-1]
            epoch_callback(
                {
                    "job_name": job_name,
                    "epoch": int(epoch),
                    "epochs": int(epochs),
                    "train_loss_total": float(record.train_loss_total),
                    "train_loss_ce": float(record.train_loss_ce),
                    "train_loss_kpi": float(record.train_loss_kpi),
                    "val_primary_pct": float(val_primary),
                    "test_primary_pct": float(test_primary),
                    "epoch_time_s": float(epoch_time_s),
                    "best_epoch": int(best_epoch),
                    "best_val_primary_pct": float(best_val_primary) if np.isfinite(best_val_primary) else None,
                    "best_test_primary_pct": float(best_test_primary) if np.isfinite(best_test_primary) else None,
                }
            )

    if not best_state and epoch_records:
        # Training should still produce a usable checkpoint even if every
        # validation primary value becomes non-finite. In that case the final
        # epoch is kept so later stages never try to load an empty state dict.
        last_record = epoch_records[-1]
        best_epoch = int(last_record.epoch)
        best_val_metrics = dict(last_record.val_metrics)
        best_test_metrics = dict(last_record.test_metrics)
        best_val_primary = float(best_val_metrics.get(primary_key, float("nan")))
        best_test_primary = float(best_test_metrics.get(primary_key, float("nan")))
        best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    train_time_s = time.perf_counter() - train_start
    return TrainResult(
        best_epoch=int(best_epoch),
        best_val_primary_pct=float(best_val_primary),
        test_primary_at_best_pct=float(best_test_primary),
        total_params=int(total_params),
        trainable_params=int(trainable_param_count),
        train_time_s=float(train_time_s),
        best_val_metrics=best_val_metrics,
        best_test_metrics=best_test_metrics,
        epoch_records=epoch_records,
        best_state_dict=best_state,
    )


def compute_budget_metrics(
    epoch_records: Sequence[EpochRecord],
    budget: int,
    *,
    primary_key: str,
) -> Tuple[int, Dict[str, float], Dict[str, float]]:
    """Choose the best epoch within a budget using validation primary KPI as the selector."""
    usable_records = list(epoch_records[: max(1, int(budget))])
    if not usable_records:
        raise ValueError("epoch_records is empty")

    best_index = 0
    best_value = float("-inf")
    for index, record in enumerate(usable_records):
        value = float(record.val_metrics.get(primary_key, float("nan")))
        if np.isfinite(value) and value > best_value:
            best_index = index
            best_value = value

    selected = usable_records[best_index]
    return int(selected.epoch), dict(selected.val_metrics), dict(selected.test_metrics)


def save_epoch_history(
    path: Path,
    epoch_records: Sequence[EpochRecord],
    *,
    primary_key: str,
    metric_keys: Sequence[str],
) -> None:
    """Write the full per-epoch metric history for one training job."""
    fieldnames = [
        "epoch",
        "train_loss_total",
        "train_loss_ce",
        "train_loss_kpi",
        "val_acc_%",
        "test_acc_%",
        "epoch_time_s",
    ]
    fieldnames += [f"val_{key}" for key in metric_keys]
    fieldnames += [f"test_{key}" for key in metric_keys]

    rows: List[Dict[str, object]] = []
    for record in epoch_records:
        row: Dict[str, object] = {
            "epoch": int(record.epoch),
            "train_loss_total": format_metric(record.train_loss_total),
            "train_loss_ce": format_metric(record.train_loss_ce),
            "train_loss_kpi": format_metric(record.train_loss_kpi),
            "val_acc_%": format_metric(record.val_metrics.get(primary_key, float("nan"))),
            "test_acc_%": format_metric(record.test_metrics.get(primary_key, float("nan"))),
            "epoch_time_s": format_metric(record.epoch_time_s),
        }
        row.update(metrics_to_prefixed_row(record.val_metrics, prefix="val_", metric_keys=metric_keys))
        row.update(metrics_to_prefixed_row(record.test_metrics, prefix="test_", metric_keys=metric_keys))
        rows.append(row)
    write_csv(path, rows, fieldnames)


def save_train_result(path: Path, result: TrainResult, *, primary_key: str) -> None:
    """Write the compact summary for one training job."""
    atomic_write_json(
        path,
        {
            "primary_kpi": primary_key,
            "best_epoch": int(result.best_epoch),
            "best_val_primary_pct": float(result.best_val_primary_pct),
            "test_primary_at_best_pct": float(result.test_primary_at_best_pct),
            "total_params": int(result.total_params),
            "trainable_params": int(result.trainable_params),
            "train_time_s": float(result.train_time_s),
            "mean_epoch_time_s": float(np.mean([record.epoch_time_s for record in result.epoch_records]))
            if result.epoch_records
            else float("nan"),
            "best_val_metrics": result.best_val_metrics,
            "best_test_metrics": result.best_test_metrics,
        },
    )


def optional_float(value: object) -> Optional[float]:
    try:
        number = float(value)
    except Exception:
        return None
    return float(number) if np.isfinite(number) else None


def optional_int(value: object) -> Optional[int]:
    number = optional_float(value)
    return int(number) if number is not None else None


def sum_history_epoch_time(history_path: Path) -> Optional[float]:
    if not history_path.exists():
        return None
    total = 0.0
    found = False
    for row in read_csv_rows(history_path):
        value = optional_float(row.get("epoch_time_s"))
        if value is None:
            continue
        total += float(value)
        found = True
    return float(total) if found else None


def read_train_metadata(artifact_dir: Path) -> Dict[str, object]:
    """Read parameter counts and train timing from a train artifact folder."""
    metadata: Dict[str, object] = {}
    result_path = artifact_dir / "train_result.json"
    if result_path.exists():
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        for key in ("total_params", "trainable_params"):
            value = optional_int(payload.get(key))
            if value is not None:
                metadata[key] = int(value)
        value = optional_float(payload.get("train_time_s"))
        if value is not None:
            metadata["train_time_s"] = float(value)

    if "train_time_s" not in metadata:
        value = sum_history_epoch_time(artifact_dir / "history.csv")
        if value is not None:
            metadata["train_time_s"] = float(value)
    return metadata


def prepare_stage(
    run_dir: Path,
    config: ExperimentConfig,
    job: BatchJob,
    *,
    overwrite: bool,
) -> PreparedData:
    """Prepare the clean Set-B features, labels, and deterministic splits for one run."""
    data_dir = run_dir / "data"
    required = list(PREPARED_DATA_FILES)

    if not overwrite and files_exist(data_dir, required):
        return load_prepared_data(run_dir)

    ensure_dir(data_dir)
    values_cube = load_values_cube(config.values_cube_path)
    setb_tx = build_setb_tx_indices(int(job.pattern))
    # Everything downstream starts from the clean 32-beam world. Blockage is
    # applied later so every stage uses the same saved baseline arrays.
    rsrp_tx = values_cube.max(axis=2).astype(np.float32, copy=False)
    labels = best_tx_labels(values_cube)
    features = build_setb_features(values_cube, setb_tx)
    train_idx, val_idx, test_idx = split_indices(
        int(features.shape[0]),
        train_frac=float(config.split.train_frac),
        val_frac=float(config.split.val_frac),
        test_frac=float(config.split.test_frac),
        seed=int(config.training.seed),
    )

    atomic_save_npy(data_dir / "X_setb.npy", features.astype(np.float32, copy=False))
    atomic_save_npy(data_dir / "y_tx.npy", labels.astype(np.int64, copy=False))
    atomic_save_npy(data_dir / "rsrp_tx.npy", rsrp_tx.astype(np.float32, copy=False))
    atomic_save_npy(data_dir / "train_idx.npy", train_idx.astype(np.int64, copy=False))
    atomic_save_npy(data_dir / "val_idx.npy", val_idx.astype(np.int64, copy=False))
    atomic_save_npy(data_dir / "test_idx.npy", test_idx.astype(np.int64, copy=False))

    atomic_write_json(
        data_dir / "meta.json",
        {
            "pipeline_version": PIPELINE_VERSION,
            "values_cube_path": str(config.values_cube_path),
            "values_cube_shape": [int(x) for x in values_cube.shape],
            "setb_pattern": int(job.pattern),
            "blocked_setb_col": int(job.blocked_beam_index),
            "blockage_model": BLOCKAGE_PROTOCOL_NAME,
            "setb_feature_mode": SETB_FEATURE_MODE,
            "global_min_db": float(values_cube.min()),
            "split_fracs": asdict(config.split),
            "split_sizes": {
                "train": int(len(train_idx)),
                "val": int(len(val_idx)),
                "test": int(len(test_idx)),
            },
            "seed": int(config.training.seed),
        },
    )
    return load_prepared_data(run_dir)


def load_prepared_data(run_dir: Path) -> PreparedData:
    """Reload the saved arrays from the prepare stage."""
    data_dir = run_dir / "data"
    missing = [data_dir / name for name in PREPARED_DATA_FILES if not (data_dir / name).exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Prepared data is missing. This is required when stages omit 'prepare'. "
            "Add 'prepare' to stages, or restore the saved data/ folder before rerunning. "
            f"Missing: {missing_text}"
        )
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    return PreparedData(
        features=load_npy(data_dir / "X_setb.npy"),
        labels=load_npy(data_dir / "y_tx.npy"),
        rsrp_tx=load_npy(data_dir / "rsrp_tx.npy"),
        train_idx=load_npy(data_dir / "train_idx.npy"),
        val_idx=load_npy(data_dir / "val_idx.npy"),
        test_idx=load_npy(data_dir / "test_idx.npy"),
        global_min_db=float(meta["global_min_db"]),
    )


def get_ft_train_indices(
    full_train_idx: np.ndarray,
    train_frac: float,
    *,
    seed: int,
    sampling: str,
) -> np.ndarray:
    """Choose the FT subset indices deterministically from the full training split."""
    fraction = float(train_frac)
    if fraction <= 0.0:
        return np.zeros((0,), dtype=np.int64)
    if fraction >= 1.0:
        return np.asarray(full_train_idx, dtype=np.int64)

    full_train_idx = np.asarray(full_train_idx, dtype=np.int64)
    sample_count = max(1, min(int(full_train_idx.shape[0]), int(round(fraction * int(full_train_idx.shape[0])))))
    sampling_mode = str(sampling).strip().lower()
    if sampling_mode not in {"with_replacement", "without_replacement"}:
        raise ValueError(f"Unknown FT sampling mode: {sampling}")

    # The FT subset has to be stable across reruns and independent of the order in
    # which FT stages happen to execute. The SeedSequence keeps that mapping fixed.
    seed_sequence = np.random.SeedSequence(
        [int(seed), int(round(fraction * 1_000_000)), 1 if sampling_mode == "with_replacement" else 0, 2025]
    )
    rng = np.random.default_rng(seed_sequence)
    return rng.choice(
        full_train_idx,
        size=int(sample_count),
        replace=(sampling_mode == "with_replacement"),
    ).astype(np.int64, copy=False)


def results_row_fieldnames(metric_keys: Sequence[str]) -> List[str]:
    """Common CSV schema used by baseline, finetune, and aggregate outputs."""
    fieldnames = [
        "seed",
        "method",
        "freeze",
        "blockage_%",
        "train_frac",
        "train_samples",
        "ft_epochs",
        "selected_epoch",
        "total_params",
        "trainable_params",
        "train_time_s",
        "eval_time_s",
        "val_acc_%",
        "test_acc_%",
    ]
    fieldnames += [f"val_{key}" for key in metric_keys]
    fieldnames += [f"test_{key}" for key in metric_keys]
    return fieldnames


# ---------------------------------------------------------------------------
# Split helpers used by the baseline and finetune stages
# ---------------------------------------------------------------------------
def split_prepared_arrays(prepared: PreparedData) -> Dict[str, np.ndarray]:
    """Break the prepared arrays into explicit train/val/test dictionaries."""
    return {
        "X_train": prepared.features[prepared.train_idx],
        "y_train": prepared.labels[prepared.train_idx],
        "rsrp_train": prepared.rsrp_tx[prepared.train_idx],
        "X_val": prepared.features[prepared.val_idx],
        "y_val": prepared.labels[prepared.val_idx],
        "rsrp_val": prepared.rsrp_tx[prepared.val_idx],
        "X_test": prepared.features[prepared.test_idx],
        "y_test": prepared.labels[prepared.test_idx],
        "rsrp_test": prepared.rsrp_tx[prepared.test_idx],
    }


def apply_blockage_to_split(
    features: np.ndarray,
    labels: np.ndarray,
    rsrp_tx: np.ndarray,
    *,
    job: BatchJob,
    global_min_db: float,
    blockage_pct: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply one blockage level to one feature, label, and RSRP split."""
    return build_blocked_dataset_views(
        clean_features=features,
        clean_labels=labels,
        clean_rsrp=rsrp_tx,
        setb_pattern=int(job.pattern),
        blocked_setb_col=int(job.blocked_beam_index),
        global_min_db=float(global_min_db),
        blockage_pct=int(blockage_pct),
    )


def blocked_named_split(
    arrays: Dict[str, np.ndarray],
    split_name: str,
    *,
    job: BatchJob,
    global_min_db: float,
    blockage_pct: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rebuild one named split under a given blockage level."""
    return apply_blockage_to_split(
        arrays[f"X_{split_name}"],
        arrays[f"y_{split_name}"],
        arrays[f"rsrp_{split_name}"],
        job=job,
        global_min_db=global_min_db,
        blockage_pct=blockage_pct,
    )


def blocked_named_splits(
    arrays: Dict[str, np.ndarray],
    split_names: Sequence[str],
    *,
    job: BatchJob,
    global_min_db: float,
    blockage_pct: int,
) -> Dict[str, np.ndarray]:
    """Rebuild several named splits under the same blockage level."""
    blocked: Dict[str, np.ndarray] = {}
    for split_name in split_names:
        features, labels, rsrp_tx = blocked_named_split(
            arrays,
            split_name,
            job=job,
            global_min_db=global_min_db,
            blockage_pct=blockage_pct,
        )
        blocked[f"X_{split_name}"] = features
        blocked[f"y_{split_name}"] = labels
        blocked[f"rsrp_{split_name}"] = rsrp_tx
    return blocked


def evaluate_model_on_val_test(
    model: nn.Module,
    *,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    val_rsrp: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    test_rsrp: np.ndarray,
    kpi: KpiConfig,
    batch_size: int,
    device: torch.device,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Run the validation and test evaluation block used by baseline and FT."""
    val_metrics = evaluate_beam_kpis(
        model,
        val_features,
        val_labels,
        val_rsrp,
        kpi=kpi,
        device=device,
        batch_size=int(batch_size),
    )
    test_metrics = evaluate_beam_kpis(
        model,
        test_features,
        test_labels,
        test_rsrp,
        kpi=kpi,
        device=device,
        batch_size=int(batch_size),
    )
    return val_metrics, test_metrics


def load_model_from_weights(weights_path: Path, *, setb_size: int, device: torch.device) -> CNNBeamPredictor:
    """Restore a saved model checkpoint onto the requested device."""
    model = CNNBeamPredictor(setb_size=int(setb_size), num_classes=32)
    model.load_state_dict(load_torch_state_dict(weights_path), strict=True)
    model.to(device)
    return model


def make_results_row(
    *,
    seed: int,
    method: str,
    freeze: str,
    blockage_pct: int,
    train_frac: float,
    train_samples: int,
    ft_epochs: int,
    selected_epoch: int,
    val_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    primary_key: str,
    metric_keys: Sequence[str],
    total_params: Optional[int] = None,
    trainable_params: Optional[int] = None,
    train_time_s: Optional[float] = None,
    eval_time_s: Optional[float] = None,
) -> Dict[str, object]:
    """Create one row in the run-level results schema."""
    row: Dict[str, object] = {
        "seed": int(seed),
        "method": str(method),
        "freeze": str(freeze),
        "blockage_%": int(blockage_pct),
        "train_frac": f"{float(train_frac):.6f}",
        "train_samples": int(train_samples),
        "ft_epochs": int(ft_epochs),
        "selected_epoch": int(selected_epoch),
        "total_params": "" if total_params is None else int(total_params),
        "trainable_params": "" if trainable_params is None else int(trainable_params),
        "train_time_s": "" if train_time_s is None else format_metric(float(train_time_s)),
        "eval_time_s": "" if eval_time_s is None else format_metric(float(eval_time_s)),
        "val_acc_%": format_metric(val_metrics.get(primary_key, float("nan"))),
        "test_acc_%": format_metric(test_metrics.get(primary_key, float("nan"))),
    }
    row.update(metrics_to_prefixed_row(val_metrics, prefix="val_", metric_keys=metric_keys))
    row.update(metrics_to_prefixed_row(test_metrics, prefix="test_", metric_keys=metric_keys))
    return row



# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

NON_ML_BASELINE_METHODS = ("MAX-SETB", "NN-ANGLE", "RANDOM-SETB")


def requested_non_ml_methods(config: ExperimentConfig) -> Tuple[str, ...]:
    return tuple(method for method in NON_ML_BASELINE_METHODS if method in config.experiments)


def run_non_ml_baseline_stage(
    jobs_dir: Path,
    config: ExperimentConfig,
    job: BatchJob,
    prepared: PreparedData,
    arrays: Dict[str, np.ndarray],
    *,
    fieldnames: Sequence[str],
    metric_keys: Sequence[str],
    primary_key: str,
    overwrite: bool,
) -> None:
    """Evaluate non-ML baselines from prepared arrays without training."""
    methods = requested_non_ml_methods(config)
    if not methods:
        return

    baseline_dir = jobs_dir / "non_ml_baselines"
    if not overwrite and files_exist(baseline_dir, ["rows.csv"]):
        return

    ensure_dir(baseline_dir)
    rows: List[Dict[str, object]] = []
    for blockage_pct in config.blockage_levels:
        blocked_eval = blocked_named_splits(
            arrays,
            ("val", "test"),
            job=job,
            global_min_db=float(prepared.global_min_db),
            blockage_pct=int(blockage_pct),
        )
        for method in methods:
            eval_start = time.perf_counter()
            val_metrics = evaluate_ranked_beams(
                non_ml_rankings(
                    method,
                    blocked_eval["X_val"],
                    int(job.pattern),
                    seed=int(config.training.seed) + int(blockage_pct),
                ),
                blocked_eval["rsrp_val"],
                kpi=config.kpi,
            )
            test_metrics = evaluate_ranked_beams(
                non_ml_rankings(
                    method,
                    blocked_eval["X_test"],
                    int(job.pattern),
                    seed=int(config.training.seed) + int(blockage_pct) + 10_000,
                ),
                blocked_eval["rsrp_test"],
                kpi=config.kpi,
            )
            rows.append(
                make_results_row(
                    seed=int(config.training.seed),
                    method=method,
                    freeze="none",
                    blockage_pct=int(blockage_pct),
                    train_frac=0.0,
                    train_samples=0,
                    ft_epochs=0,
                    selected_epoch=0,
                    val_metrics=val_metrics,
                    test_metrics=test_metrics,
                    primary_key=primary_key,
                    metric_keys=metric_keys,
                    total_params=0,
                    trainable_params=0,
                    train_time_s=0.0,
                    eval_time_s=time.perf_counter() - eval_start,
                )
            )
    write_csv(baseline_dir / "rows.csv", rows, list(fieldnames))


def _copy_checkpoint(source: Path, target: Path, *, overwrite: bool) -> bool:
    if not source.exists():
        return False
    if target.exists() and not overwrite:
        return True
    ensure_dir(target.parent)
    tmp = temporary_sibling_path(target)
    shutil.copy2(source, tmp)
    tmp.replace(target)
    return True


def export_baseline_checkpoints(
    run_dir: Path,
    config: ExperimentConfig,
    job: BatchJob,
    *,
    overwrite: bool,
) -> None:
    """Create a clear researcher-facing index of BL-AG and BL-AW checkpoints."""
    export_root = run_dir / "stored_weights" / "baseline_pretrained"
    rows: list[dict[str, object]] = []

    bl_ag_source = run_dir / BL_AG_STAGE_CHECKPOINT
    bl_ag_target = run_dir / BL_AG_STORED_CHECKPOINT
    if _copy_checkpoint(bl_ag_source, bl_ag_target, overwrite=overwrite):
        rows.append(
            {
                "checkpoint_id": "BL-AG_clean_pretrain",
                "method": "BL-AG",
                "role": "clean_pretrained_ft_start",
                "source_path": str(bl_ag_source.relative_to(run_dir)),
                "exported_path": str(bl_ag_target.relative_to(run_dir)),
                "seed": int(config.training.seed),
                "pattern": int(job.pattern),
                "blocked_beam_index": int(job.blocked_beam_index),
                "blockage_pct": "",
                "epochs": int(config.training.pretrain_epochs),
                "use_for_ft_start": "yes",
                "notes": "Use this clean BL-AG checkpoint for FT-only sweeps.",
            }
        )

    if "BL-AW" in config.experiments:
        for blockage_pct in config.blockage_levels:
            source = run_dir / "jobs" / "baseline" / f"bl_aw_blockage_{int(blockage_pct):03d}" / "weights_best.pt"
            target = export_root / f"BL-AW_blockage_{int(blockage_pct):03d}_aware_weights.pt"
            if not _copy_checkpoint(source, target, overwrite=overwrite):
                continue
            rows.append(
                {
                    "checkpoint_id": f"BL-AW_blockage_{int(blockage_pct):03d}_aware",
                    "method": "BL-AW",
                    "role": "blockage_aware_baseline",
                    "source_path": str(source.relative_to(run_dir)),
                    "exported_path": str(target.relative_to(run_dir)),
                    "seed": int(config.training.seed),
                    "pattern": int(job.pattern),
                    "blocked_beam_index": int(job.blocked_beam_index),
                    "blockage_pct": int(blockage_pct),
                    "epochs": int(config.training.aware_epochs),
                    "use_for_ft_start": "no",
                    "notes": "Paper BL-AW baseline endpoint; FT uses BL-AG by default.",
                }
            )

    if not rows:
        return

    guide = """# Stored Baseline Weights

This folder is written after the baseline stage.

- `BL-AG_clean_pretrain_weights.pt` is the clean BL-AG pretrained checkpoint.
  Use this file as the starting checkpoint for FT-only sweeps.
- `BL-AW_blockage_XXX_aware_weights.pt` files are blockage-aware baseline
  endpoints for the paper comparison. They are saved for audit/reuse, but the
  release FT workflow starts from BL-AG unless a researcher intentionally changes
  the method.

The original training artifacts remain under `jobs/baseline/`; this folder is
the researcher-facing index with explicit names and `manifest.csv`.
"""
    atomic_write_text(export_root / "STORED_WEIGHTS_GUIDE.md", guide)
    write_csv(
        export_root / "manifest.csv",
        rows,
        [
            "checkpoint_id",
            "method",
            "role",
            "source_path",
            "exported_path",
            "seed",
            "pattern",
            "blocked_beam_index",
            "blockage_pct",
            "epochs",
            "use_for_ft_start",
            "notes",
        ],
    )


def run_baseline_stage(
    run_dir: Path,
    config: ExperimentConfig,
    job: BatchJob,
    prepared: PreparedData,
    *,
    device: torch.device,
    overwrite: bool,
    epoch_callback: Optional[Callable[[Dict[str, object]], None]] = None,
) -> None:
    """Run BL-AG and BL-AW for one blocked-beam experiment directory."""
    jobs_dir = run_dir / "jobs" / "baseline"
    ensure_dir(jobs_dir)

    arrays = split_prepared_arrays(prepared)
    metric_keys = build_metric_keys(config.kpi)
    primary_key = config.kpi.primary_key()
    fieldnames = results_row_fieldnames(metric_keys)
    run_non_ml_baseline_stage(
        jobs_dir,
        config,
        job,
        prepared,
        arrays,
        fieldnames=fieldnames,
        metric_keys=metric_keys,
        primary_key=primary_key,
        overwrite=overwrite,
    )

    bl_ag_train_dir = jobs_dir / "bl_ag_train"
    bl_ag_eval_dir = jobs_dir / "bl_ag_eval"
    bl_ag_required = ["weights_best.pt", "history.csv", "train_result.json"]
    bl_ag_eval_required = ["rows.csv"]

    need_bl_ag = ("BL-AG" in config.experiments) or ("FT" in config.experiments)

    # BL-AG is trained once on the clean world. The resulting checkpoint is then
    # reused for clean evaluation, blocked evaluation, and every FT branch.
    if need_bl_ag and (overwrite or not files_exist(bl_ag_train_dir, bl_ag_required)):
        ensure_dir(bl_ag_train_dir)
        result = train_model(
            init_state_dict=None,
            freeze_policy="none",
            train_features=arrays["X_train"],
            train_labels=arrays["y_train"],
            train_rsrp=arrays["rsrp_train"],
            val_features=arrays["X_val"],
            val_labels=arrays["y_val"],
            val_rsrp=arrays["rsrp_val"],
            test_features=arrays["X_test"],
            test_labels=arrays["y_test"],
            test_rsrp=arrays["rsrp_test"],
            training=config.training,
            epochs=int(config.training.pretrain_epochs),
            kpi=config.kpi,
            device=device,
            seed=int(config.training.seed),
            epoch_callback=epoch_callback,
            job_name=f"BL-AG clean train",
        )
        atomic_torch_save(bl_ag_train_dir / "weights_best.pt", result.best_state_dict)
        save_epoch_history(
            bl_ag_train_dir / "history.csv",
            result.epoch_records,
            primary_key=primary_key,
            metric_keys=metric_keys,
        )
        save_train_result(bl_ag_train_dir / "train_result.json", result, primary_key=primary_key)

    if need_bl_ag and (overwrite or not files_exist(bl_ag_eval_dir, bl_ag_eval_required)):
        ensure_dir(bl_ag_eval_dir)
        model = load_model_from_weights(
            bl_ag_train_dir / "weights_best.pt",
            setb_size=int(arrays["X_train"].shape[1]),
            device=device,
        )

        rows: List[Dict[str, object]] = []
        train_metadata = read_train_metadata(bl_ag_train_dir)
        for blockage_pct in config.blockage_levels:
            # BL-AG evaluation rebuilds val/test under each blockage level without
            # changing the clean checkpoint itself.
            blocked_eval = blocked_named_splits(
                arrays,
                ("val", "test"),
                job=job,
                global_min_db=float(prepared.global_min_db),
                blockage_pct=int(blockage_pct),
            )
            eval_start = time.perf_counter()
            val_metrics, test_metrics = evaluate_model_on_val_test(
                model,
                val_features=blocked_eval["X_val"],
                val_labels=blocked_eval["y_val"],
                val_rsrp=blocked_eval["rsrp_val"],
                test_features=blocked_eval["X_test"],
                test_labels=blocked_eval["y_test"],
                test_rsrp=blocked_eval["rsrp_test"],
                kpi=config.kpi,
                batch_size=int(config.training.batch_size),
                device=device,
            )
            eval_time_s = time.perf_counter() - eval_start
            rows.append(
                make_results_row(
                    seed=int(config.training.seed),
                    method="BL-AG",
                    freeze="none",
                    blockage_pct=int(blockage_pct),
                    train_frac=1.0,
                    train_samples=int(len(prepared.train_idx)),
                    ft_epochs=0,
                    selected_epoch=0,
                    val_metrics=val_metrics,
                    test_metrics=test_metrics,
                    primary_key=primary_key,
                    metric_keys=metric_keys,
                    total_params=train_metadata.get("total_params"),
                    trainable_params=train_metadata.get("trainable_params"),
                    train_time_s=train_metadata.get("train_time_s"),
                    eval_time_s=eval_time_s,
                )
            )
        write_csv(bl_ag_eval_dir / "rows.csv", rows, fieldnames)

    if "BL-AW" in config.experiments:
        for blockage_pct in config.blockage_levels:
            # BL-AW repeats training inside each blocked world instead of reusing the
            # clean checkpoint. That is the experiment's fully blockage-aware baseline.
            aware_dir = jobs_dir / f"bl_aw_blockage_{int(blockage_pct):03d}"
            required = ["weights_best.pt", "history.csv", "train_result.json", "rows.csv"]
            if not overwrite and files_exist(aware_dir, required):
                continue

            ensure_dir(aware_dir)
            blocked = blocked_named_splits(
                arrays,
                ("train", "val", "test"),
                job=job,
                global_min_db=float(prepared.global_min_db),
                blockage_pct=int(blockage_pct),
            )
            result = train_model(
                init_state_dict=None,
                freeze_policy="none",
                train_features=blocked["X_train"],
                train_labels=blocked["y_train"],
                train_rsrp=blocked["rsrp_train"],
                val_features=blocked["X_val"],
                val_labels=blocked["y_val"],
                val_rsrp=blocked["rsrp_val"],
                test_features=blocked["X_test"],
                test_labels=blocked["y_test"],
                test_rsrp=blocked["rsrp_test"],
                training=config.training,
                epochs=int(config.training.aware_epochs),
                kpi=config.kpi,
                device=device,
                seed=int(config.training.seed),
                epoch_callback=epoch_callback,
                job_name=f"BL-AW blockage {int(blockage_pct)}",
            )
            atomic_torch_save(aware_dir / "weights_best.pt", result.best_state_dict)
            save_epoch_history(
                aware_dir / "history.csv",
                result.epoch_records,
                primary_key=primary_key,
                metric_keys=metric_keys,
            )
            save_train_result(aware_dir / "train_result.json", result, primary_key=primary_key)

            row = make_results_row(
                seed=int(config.training.seed),
                method="BL-AW",
                freeze="none",
                blockage_pct=int(blockage_pct),
                train_frac=1.0,
                train_samples=int(len(prepared.train_idx)),
                ft_epochs=0,
                selected_epoch=int(result.best_epoch),
                val_metrics=result.best_val_metrics,
                test_metrics=result.best_test_metrics,
                primary_key=primary_key,
                metric_keys=metric_keys,
                total_params=int(result.total_params),
                trainable_params=int(result.trainable_params),
                train_time_s=float(result.train_time_s),
                eval_time_s=0.0,
            )
            write_csv(aware_dir / "rows.csv", [row], fieldnames)

    export_baseline_checkpoints(run_dir, config, job, overwrite=overwrite)


def run_finetune_stage(
    run_dir: Path,
    config: ExperimentConfig,
    job: BatchJob,
    prepared: PreparedData,
    *,
    device: torch.device,
    overwrite: bool,
    epoch_callback: Optional[Callable[[Dict[str, object]], None]] = None,
) -> None:
    """Run the FT sweep over train fractions, blockage levels, and freeze modes."""
    if "FT" not in config.experiments:
        return

    jobs_dir = run_dir / "jobs" / "finetune"
    ensure_dir(jobs_dir)

    arrays = split_prepared_arrays(prepared)
    metric_keys = build_metric_keys(config.kpi)
    primary_key = config.kpi.primary_key()
    fieldnames = results_row_fieldnames(metric_keys)
    max_ft_epochs = int(max(config.training.ft_epoch_sweep))
    bl_ag_weights = find_bl_ag_checkpoint(run_dir)
    if bl_ag_weights is None:
        candidates = ", ".join(str(path) for path in bl_ag_checkpoint_candidates(run_dir))
        raise FileNotFoundError(f"FT requires BL-AG weights at one of these paths: {candidates}")
    initial_state = load_torch_state_dict(bl_ag_weights)

    bl_ag_rows: Dict[int, Dict[str, str]] = {}
    bl_ag_eval_path = run_dir / "jobs" / "baseline" / "bl_ag_eval" / "rows.csv"
    if bl_ag_eval_path.exists():
        for row in read_csv_rows(bl_ag_eval_path):
            bl_ag_rows[int(float(row["blockage_%"]))] = row

    for train_frac in config.training.train_fracs_sweep:
        # One deterministic subset is chosen per train fraction and then reused for
        # every blockage/freeze branch under that fraction.
        train_frac_tag = slugify(f"trainfrac_{float(train_frac):.6f}")
        train_frac_root = jobs_dir / train_frac_tag
        ensure_dir(train_frac_root)

        train_indices_path = train_frac_root / "train_indices.npy"
        if overwrite or not train_indices_path.exists():
            train_indices = get_ft_train_indices(
                prepared.train_idx,
                float(train_frac),
                seed=int(config.training.seed),
                sampling=str(config.training.ft_sampling),
            )
            atomic_save_npy(train_indices_path, np.asarray(train_indices, dtype=np.int64))
        train_indices = load_npy(train_indices_path)
        train_sample_count = int(len(train_indices))

        clean_train_features = prepared.features[train_indices]
        clean_train_labels = prepared.labels[train_indices]
        clean_train_rsrp = prepared.rsrp_tx[train_indices]

        for blockage_pct in config.blockage_levels:
            blocked_eval = blocked_named_splits(
                arrays,
                ("val", "test"),
                job=job,
                global_min_db=float(prepared.global_min_db),
                blockage_pct=int(blockage_pct),
            )

            for freeze_mode in config.training.ft_freeze_modes:
                freeze_root = train_frac_root / f"blockage_{int(blockage_pct):03d}" / f"freeze_{slugify(str(freeze_mode))}"
                ensure_dir(freeze_root)

                needs_training = (
                    float(train_frac) > 0.0
                    and int(blockage_pct) != 0
                    and str(freeze_mode) != "freeze_all"
                )
                required = ["rows.csv"] if not needs_training else ["rows.csv", "weights_best.pt", "history.csv", "train_result.json"]
                if not overwrite and files_exist(freeze_root, required):
                    continue

                if not needs_training:
                    # These branches are intentionally shortcut back to BL-AG-style
                    # evaluation because there is no meaningful update to fit:
                    #   - tf=0 has no training samples
                    #   - blockage=0 is still the clean world
                    #   - freeze_all has no trainable parameters
                    base_row_source = bl_ag_rows.get(int(blockage_pct))
                    total_params, trainable_params = parameter_counts_for_freeze(
                        setb_size=int(arrays["X_train"].shape[1]),
                        freeze_mode=str(freeze_mode),
                    )
                    if base_row_source is None:
                        model = CNNBeamPredictor(setb_size=int(arrays["X_train"].shape[1]), num_classes=32)
                        model.load_state_dict(initial_state, strict=True)
                        model.to(device)
                        eval_start = time.perf_counter()
                        val_metrics, test_metrics = evaluate_model_on_val_test(
                            model,
                            val_features=blocked_eval["X_val"],
                            val_labels=blocked_eval["y_val"],
                            val_rsrp=blocked_eval["rsrp_val"],
                            test_features=blocked_eval["X_test"],
                            test_labels=blocked_eval["y_test"],
                            test_rsrp=blocked_eval["rsrp_test"],
                            kpi=config.kpi,
                            batch_size=int(config.training.batch_size),
                            device=device,
                        )
                        eval_time_s = time.perf_counter() - eval_start
                        base_row = make_results_row(
                            seed=int(config.training.seed),
                            method="FT",
                            freeze=str(freeze_mode),
                            blockage_pct=int(blockage_pct),
                            train_frac=float(train_frac),
                            train_samples=int(train_sample_count),
                            ft_epochs=0,
                            selected_epoch=0,
                            val_metrics=val_metrics,
                            test_metrics=test_metrics,
                            primary_key=primary_key,
                            metric_keys=metric_keys,
                            total_params=total_params,
                            trainable_params=trainable_params,
                            train_time_s=0.0,
                            eval_time_s=eval_time_s,
                        )
                    else:
                        base_row = {
                            "seed": int(config.training.seed),
                            "method": "FT",
                            "freeze": str(freeze_mode),
                            "blockage_%": int(blockage_pct),
                            "train_frac": f"{float(train_frac):.6f}",
                            "train_samples": int(train_sample_count),
                            "ft_epochs": 0,
                            "selected_epoch": 0,
                            "total_params": int(total_params),
                            "trainable_params": int(trainable_params),
                            "train_time_s": format_metric(0.0),
                            "eval_time_s": "",
                            "val_acc_%": base_row_source.get("val_acc_%", "nan"),
                            "test_acc_%": base_row_source.get("test_acc_%", "nan"),
                        }
                        for key in metric_keys:
                            base_row[f"val_{key}"] = base_row_source.get(f"val_{key}", "nan")
                            base_row[f"test_{key}"] = base_row_source.get(f"test_{key}", "nan")

                    rows = []
                    for ft_epochs in config.training.ft_epoch_sweep:
                        row = dict(base_row)
                        row["ft_epochs"] = int(ft_epochs)
                        rows.append(row)
                    write_csv(freeze_root / "rows.csv", rows, fieldnames)
                    continue

                train_features, train_labels, train_rsrp = apply_blockage_to_split(
                    clean_train_features,
                    clean_train_labels,
                    clean_train_rsrp,
                    job=job,
                    global_min_db=float(prepared.global_min_db),
                    blockage_pct=int(blockage_pct),
                )
                freeze_policy = {
                    "no_freeze": "none",
                    "freeze_cnn": "freeze_cnn",
                    "freeze_head": "freeze_head",
                }.get(str(freeze_mode))
                if freeze_policy is None:
                    raise ValueError(f"Unknown FT freeze mode: {freeze_mode}")

                result = train_model(
                    init_state_dict=initial_state,
                    freeze_policy=freeze_policy,
                    train_features=train_features,
                    train_labels=train_labels,
                    train_rsrp=train_rsrp,
                    val_features=blocked_eval["X_val"],
                    val_labels=blocked_eval["y_val"],
                    val_rsrp=blocked_eval["rsrp_val"],
                    test_features=blocked_eval["X_test"],
                    test_labels=blocked_eval["y_test"],
                    test_rsrp=blocked_eval["rsrp_test"],
                    training=config.training,
                    epochs=int(max_ft_epochs),
                    kpi=config.kpi,
                    device=device,
                    seed=int(config.training.seed),
                    epoch_callback=epoch_callback,
                    job_name=f"FT {freeze_mode} tf={float(train_frac):.3f} blockage={int(blockage_pct)}",
                )
                atomic_torch_save(freeze_root / "weights_best.pt", result.best_state_dict)
                save_epoch_history(
                    freeze_root / "history.csv",
                    result.epoch_records,
                    primary_key=primary_key,
                    metric_keys=metric_keys,
                )
                save_train_result(freeze_root / "train_result.json", result, primary_key=primary_key)

                rows = []
                for ft_epochs in config.training.ft_epoch_sweep:
                    # One full FT run is trained once, then the configured epoch
                    # budgets are read back as "best within budget" selections.
                    selected_epoch, selected_val_metrics, selected_test_metrics = compute_budget_metrics(
                        result.epoch_records,
                        int(ft_epochs),
                        primary_key=primary_key,
                    )
                    rows.append(
                        make_results_row(
                            seed=int(config.training.seed),
                            method="FT",
                            freeze=str(freeze_mode),
                            blockage_pct=int(blockage_pct),
                            train_frac=float(train_frac),
                            train_samples=int(train_sample_count),
                            ft_epochs=int(ft_epochs),
                            selected_epoch=int(selected_epoch),
                            val_metrics=selected_val_metrics,
                            test_metrics=selected_test_metrics,
                            primary_key=primary_key,
                            metric_keys=metric_keys,
                            total_params=int(result.total_params),
                            trainable_params=int(result.trainable_params),
                            train_time_s=float(result.train_time_s),
                            eval_time_s=0.0,
                        )
                    )
                write_csv(freeze_root / "rows.csv", rows, fieldnames)


def artifact_dir_for_result_row(run_dir: Path, row: Dict[str, str], relative_job_path: str) -> Optional[Path]:
    path = run_dir / relative_job_path
    if (path / "train_result.json").exists() or (path / "history.csv").exists():
        return path
    if row.get("method") == "BL-AG":
        return run_dir / "jobs" / "baseline" / "bl_ag_train"
    return None


def freeze_mode_for_result_row(row: Dict[str, str]) -> str:
    method = str(row.get("method", ""))
    freeze = str(row.get("freeze", "none") or "none")
    if method in {"BL-AG", "BL-AW"}:
        return "none"
    if method in NON_ML_BASELINE_METHODS:
        return "freeze_all"
    return freeze


def enrich_result_row_metadata(run_dir: Path, row: Dict[str, str], relative_job_path: str) -> None:
    """Fill parameter/timing metadata during aggregation when rows are older."""
    method = str(row.get("method", ""))
    if method in NON_ML_BASELINE_METHODS:
        if not str(row.get("total_params", "")).strip():
            row["total_params"] = "0"
        if not str(row.get("trainable_params", "")).strip():
            row["trainable_params"] = "0"
        if not str(row.get("train_time_s", "")).strip():
            row["train_time_s"] = format_metric(0.0)
        return

    artifact_dir = artifact_dir_for_result_row(run_dir, row, relative_job_path)
    metadata = read_train_metadata(artifact_dir) if artifact_dir is not None else {}
    if not str(row.get("total_params", "")).strip() or not str(row.get("trainable_params", "")).strip():
        total_params, trainable_params = parameter_counts_for_freeze(
            setb_size=8,
            freeze_mode=freeze_mode_for_result_row(row),
        )
        metadata.setdefault("total_params", int(total_params))
        metadata.setdefault("trainable_params", int(trainable_params))
    for key in ("total_params", "trainable_params"):
        if not str(row.get(key, "")).strip() and key in metadata:
            row[key] = str(int(metadata[key]))
    if not str(row.get("train_time_s", "")).strip() and "train_time_s" in metadata:
        row["train_time_s"] = format_metric(float(metadata["train_time_s"]))
    if not str(row.get("train_time_s", "")).strip() and method == "FT":
        train_frac = optional_float(row.get("train_frac"))
        blockage = optional_int(row.get("blockage_%"))
        freeze = str(row.get("freeze", ""))
        if (train_frac is not None and train_frac <= 0.0) or blockage == 0 or freeze == "freeze_all":
            row["train_time_s"] = format_metric(0.0)


def aggregate_stage(run_dir: Path, *, kpi: KpiConfig) -> None:
    """Merge all stage-level `rows.csv` files into the final run-level `results.csv`."""
    jobs_root = run_dir / "jobs"
    if not jobs_root.exists():
        raise FileNotFoundError(f"Missing jobs directory: {jobs_root}")

    metric_keys = build_metric_keys(kpi)
    fieldnames = results_row_fieldnames(metric_keys)
    fieldnames += ["results_schema_version", "job_path"]

    rows_by_key: Dict[Tuple[str, str, str, str, str], Dict[str, str]] = {}
    for rows_path in jobs_root.rglob("rows.csv"):
        relative_job_path = str(rows_path.parent.relative_to(run_dir))
        for row in read_csv_rows(rows_path):
            normalized = dict(row)
            normalized["job_path"] = relative_job_path
            normalized["results_schema_version"] = RESULTS_SCHEMA_VERSION
            enrich_result_row_metadata(run_dir, normalized, relative_job_path)
            # The aggregate table is keyed by the experiment identity, not by the
            # filesystem path. Conflicting rows here would mean two stages claimed
            # to produce the same experiment result with different contents.
            key = (
                normalized.get("method", ""),
                normalized.get("freeze", ""),
                normalized.get("blockage_%", ""),
                normalized.get("train_frac", ""),
                normalized.get("ft_epochs", ""),
            )
            previous = rows_by_key.get(key)
            if previous is not None and previous != normalized:
                raise RuntimeError(
                    f"Found conflicting aggregate rows for key={key}. "
                    f"Paths: {previous.get('job_path')} and {normalized.get('job_path')}"
                )
            rows_by_key[key] = normalized

    rows = [{key: row.get(key, "") for key in fieldnames} for row in rows_by_key.values()]

    def sort_key(row: Dict[str, str]) -> Tuple[object, ...]:
        return (
            row.get("method", ""),
            row.get("freeze", ""),
            float(row.get("train_frac", "0") or 0),
            int(float(row.get("blockage_%", "0") or 0)),
            int(float(row.get("ft_epochs", "0") or 0)),
        )

    rows.sort(key=sort_key)
    write_csv(run_dir / "results.csv", rows, fieldnames)
    atomic_write_json(
        run_dir / "aggregate_done.json",
        {
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pipeline_version": PIPELINE_VERSION,
            "results_schema_version": RESULTS_SCHEMA_VERSION,
            "n_rows": int(len(rows)),
        },
    )


# ---------------------------------------------------------------------------
# Run metadata and top-level orchestration
# ---------------------------------------------------------------------------
def write_setb_mapping(run_dir: Path, *, pattern: int) -> None:
    """Write the Set-B beam mapping CSV beside the run outputs."""
    write_csv(
        run_dir / "setb_mapping.csv",
        build_setb_mapping_rows(int(pattern)),
        [
            "setb_order",
            "beam_name",
            "az_deg",
            "el_deg",
            "az_idx",
            "el_idx",
            "tx_index",
            "setb_pattern",
        ],
    )


def write_run_config_snapshot(run_dir: Path, config: ExperimentConfig, job: BatchJob) -> None:
    """Save the exact config snapshot used for one run directory."""
    snapshot = {
        "pipeline_version": PIPELINE_VERSION,
        "results_schema_version": RESULTS_SCHEMA_VERSION,
        "values_cube_path": str(config.values_cube_path),
        "outdir": str(config.outdir),
        "run_name_prefix": config.run_name_prefix,
        "run_name": config.run_name(pattern=job.pattern, blocked_beam_index=job.blocked_beam_index),
        "setb_pattern": int(job.pattern),
        "blocked_setb_index": int(job.blocked_beam_index),
        "blocked_beam_name": f"B{int(job.blocked_beam_index)}",
        "blockage_model": config.blockage_model,
        "stages": list(config.stages),
        "experiments": list(config.experiments),
        "blockage_levels": list(config.blockage_levels),
        "split": asdict(config.split),
        "training": asdict(config.training),
        "kpi": asdict(config.kpi),
        "plotting": dict(config.plotting),
        "representative_beams": dict(config.representative_beams),
        "scenario_provenance": dict(config.scenario_provenance),
        "config_source": str(config.config_path),
        "system_settings": load_system_settings(config.config_path),
    }
    atomic_write_json(run_dir / "config.json", snapshot)


def report_stage(callback: Optional[StageCallback], stage: str, **payload: object) -> None:
    """Send stage updates to the outer runner when one is attached."""
    if callback is None:
        return
    callback(stage, payload)


def run_single_experiment(
    config: ExperimentConfig,
    job: BatchJob,
    *,
    overwrite: Optional[bool] = None,
    stage_callback: Optional[StageCallback] = None,
    epoch_callback: Optional[Callable[[Dict[str, object]], None]] = None,
) -> Path:
    """Run the requested stages for one pattern/blocked-beam combination."""
    overwrite_flag = bool(config.overwrite if overwrite is None else overwrite)
    run_dir = config.run_dir(job)
    validate_run_dependencies(config, job, overwrite=overwrite_flag)
    device = configure_torch_runtime(config.config_path)
    ensure_dir(run_dir)
    write_run_config_snapshot(run_dir, config, job)
    write_setb_mapping(run_dir, pattern=int(job.pattern))

    report_stage(stage_callback, "run_start", run_dir=str(run_dir), job=asdict(job))

    prepared: Optional[PreparedData] = None
    needs_prepared = bool({"prepare", "baseline", "finetune"} & set(config.stages))

    if needs_prepared:
        if "prepare" in config.stages:
            report_stage(stage_callback, "prepare", status="running")
            prepared = prepare_stage(run_dir, config, job, overwrite=overwrite_flag)
            report_stage(stage_callback, "prepare", status="completed")
        else:
            prepared = load_prepared_data(run_dir)

        if prepared is None:
            raise RuntimeError("Prepared data was not loaded.")

    if "baseline" in config.stages:
        report_stage(stage_callback, "baseline", status="running")
        run_baseline_stage(
            run_dir,
            config,
            job,
            prepared,
            device=device,
            overwrite=overwrite_flag,
            epoch_callback=epoch_callback,
        )
        report_stage(stage_callback, "baseline", status="completed")

    if "finetune" in config.stages:
        report_stage(stage_callback, "finetune", status="running")
        run_finetune_stage(
            run_dir,
            config,
            job,
            prepared,
            device=device,
            overwrite=overwrite_flag,
            epoch_callback=epoch_callback,
        )
        report_stage(stage_callback, "finetune", status="completed")

    if "aggregate" in config.stages:
        report_stage(stage_callback, "aggregate", status="running")
        aggregate_stage(run_dir, kpi=config.kpi)
        report_stage(stage_callback, "aggregate", status="completed")

    report_stage(stage_callback, "run_end", run_dir=str(run_dir), job=asdict(job))
    return run_dir
