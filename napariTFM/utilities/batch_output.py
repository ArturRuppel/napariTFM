"""Resolve where batch-derived output lands — the ``processed/`` bucket (ROADMAP §4).

All derived output goes into a bucket named ``processed/``, never mixed with raw
inputs. Two modes:

- **Processed root set** → the input-folder tree is *mirrored* under it. Each
  folder is reproduced relative to the *longest common parent* of all input
  folders. When the folders are genuinely disconnected (no shared parent — e.g.
  different Windows drives) or there is only a single folder, fall back to folder
  *basenames* flat under the processed root and emit a warning.
- **Processed root empty → in-place**: a ``processed/`` subfolder is created
  *inside each input folder*, so derived files still never mix with raw.

The resolver is pure path arithmetic so it is fully unit-testable without a
filesystem.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

#: Name of the in-place bucket created inside each input folder.
PROCESSED_BUCKET = "processed"


@dataclass
class OutputPlan:
    """Mapping from each input folder (original string) to its output directory."""

    output_dirs: Dict[str, Path]
    warnings: List[str] = field(default_factory=list)


def resolve_output_plan(
    input_folders: Sequence[str],
    processed_root: Optional[str] = None,
) -> OutputPlan:
    """Map every input folder to the directory that holds its derived output.

    Each output directory holds ``<experiment>.ntfm`` (the experiment name is the
    input folder's basename), ``figures/`` and ``batch.log``.

    Args:
        input_folders: the batch input folders, as originally supplied.
        processed_root: the configured processed root, or empty/``None`` for the
            in-place mode.

    Returns:
        An :class:`OutputPlan` keyed by the original folder strings, plus any
        warnings to surface to the user.
    """
    warnings: List[str] = []

    if not processed_root:
        # In-place: a processed/ subfolder inside each raw input folder.
        output_dirs = {
            original: Path(original) / PROCESSED_BUCKET for original in input_folders
        }
        return OutputPlan(output_dirs, warnings)

    root = Path(processed_root)
    base = _mirror_base(input_folders, warnings)

    output_dirs: Dict[str, Path] = {}
    for original in input_folders:
        folder = Path(original)
        if base is None:
            output_dirs[original] = root / folder.name
        else:
            output_dirs[original] = root / folder.relative_to(base)

    if base is None:
        _warn_basename_collisions(input_folders, warnings)

    return OutputPlan(output_dirs, warnings)


def experiment_output_dir(
    experiment_path: str, processed_root: Optional[str] = None
) -> Path:
    """The single directory that holds one experiment's derived output.

    The one resolver every consumer must share: the batch writer, the
    experiments-list status dots, and the interactive persist path all call this
    so they can never disagree about *where* an experiment's ``.ntfm`` lives.
    """
    plan = resolve_output_plan([str(experiment_path)], processed_root)
    return plan.output_dirs[str(experiment_path)]


def experiment_ntfm_path(
    experiment_path: str, processed_root: Optional[str] = None
) -> Path:
    """The canonical ``.ntfm`` path for one experiment (basename names the file)."""
    out_dir = experiment_output_dir(experiment_path, processed_root)
    return out_dir / f"{Path(experiment_path).name}.ntfm"


def _mirror_base(input_folders: Sequence[str], warnings: List[str]) -> Optional[Path]:
    """Longest common parent of all folders, or ``None`` to signal basename fallback."""
    if len(input_folders) < 2:
        # A single folder has no meaningful mirror tree — use its basename.
        return None
    try:
        return Path(os.path.commonpath([str(f) for f in input_folders]))
    except ValueError:
        warnings.append(
            "Input folders are on disconnected roots (no common parent); "
            "falling back to folder basenames flat under the processed root."
        )
        return None


def _warn_basename_collisions(
    input_folders: Sequence[str], warnings: List[str]
) -> None:
    """Surface basename clashes that would overwrite each other in fallback mode."""
    names = [Path(f).name for f in input_folders]
    duplicates = sorted({name for name, count in Counter(names).items() if count > 1})
    if duplicates:
        warnings.append(
            "Folder basenames collide under the processed root: "
            f"{', '.join(duplicates)}. Outputs would overwrite each other."
        )
