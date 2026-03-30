"""
Multi-Cell experiment support for OpenLift.

Enables simultaneous testing of multiple treatment groups (channels/campaigns)
against a shared holdout control group, with pairwise comparisons and
optional synergy detection.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, field_validator, model_validator
import pandas as pd
import numpy as np
import logging

from .experiment import Period
from .pipeline import _run_pipeline, ExperimentConfig
from .inference import compare_cells

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Config models
# ------------------------------------------------------------------

class CellConfig(BaseModel):
    """A single treatment cell in a multi-cell experiment."""
    name: str                    # e.g. "YouTube Only"
    test_geos: List[str]         # e.g. ["Lagos", "Ibadan"]
    label: str = ""              # e.g. "Cell A"

    @field_validator("test_geos")
    @classmethod
    def check_test_geos(cls, v):
        if len(v) < 1:
            raise ValueError("Each cell must have at least 1 test geo.")
        return v


class MultiCellExperimentConfig(BaseModel):
    """Configuration for a multi-cell (cross-channel) experiment."""
    name: str
    cells: List[CellConfig]
    control_geos: List[str]
    pre_period: Period
    post_period: Period

    @field_validator("cells")
    @classmethod
    def check_cells(cls, v):
        if len(v) < 2:
            raise ValueError("Multi-cell experiments require at least 2 cells.")
        if len(v) > 8:
            raise ValueError("Maximum 8 cells supported.")
        return v

    @field_validator("control_geos")
    @classmethod
    def check_controls(cls, v):
        if len(v) < 2:
            raise ValueError("Need at least 2 control geos.")
        return v

    @model_validator(mode="after")
    def check_no_overlap(self) -> "MultiCellExperimentConfig":
        """Ensure no geo appears in multiple cells or in both cells and controls."""
        all_test_geos = []
        for cell in self.cells:
            all_test_geos.extend(cell.test_geos)

        # Check for duplicates across cells
        if len(all_test_geos) != len(set(all_test_geos)):
            seen = set()
            dupes = set()
            for g in all_test_geos:
                if g in seen:
                    dupes.add(g)
                seen.add(g)
            raise ValueError(f"Geos appear in multiple cells: {dupes}")

        # Check for overlap with controls
        overlap = set(all_test_geos) & set(self.control_geos)
        if overlap:
            raise ValueError(f"Geos cannot be in both a cell and the control group: {overlap}")

        return self

    @model_validator(mode="after")
    def check_pre_post_continuity(self) -> "MultiCellExperimentConfig":
        pre_len = (self.pre_period.end_date - self.pre_period.start_date).days + 1
        if pre_len < 14:
            raise ValueError(f"Pre-period must be at least 14 days. Current: {pre_len} days.")
        if self.post_period.start_date <= self.pre_period.end_date:
            raise ValueError("Post-period start_date must be after Pre-period end_date.")
        return self


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

def run_multi_cell_experiment(
    df: pd.DataFrame,
    config: MultiCellExperimentConfig,
    date_col: str = "date",
    geo_col: str = "geo",
    outcome_col: str = "outcome",
) -> Dict[str, Any]:
    """
    Run a multi-cell geo-lift experiment.

    For each cell, runs the standard Bayesian pipeline against the
    shared control group, then computes pairwise comparisons.

    Parameters
    ----------
    df : pd.DataFrame — long-format data (date, geo, outcome)
    config : MultiCellExperimentConfig
    date_col, geo_col, outcome_col : column names

    Returns
    -------
    Dict with:
        - "cells": per-cell results (same structure as single-cell)
        - "comparisons": pairwise P(A > B) for every cell pair
        - "synergy": synergy metrics if 3+ cells exist
        - "config": the experiment configuration
    """
    df[date_col] = pd.to_datetime(df[date_col])
    df_wide = df.pivot(index=date_col, columns=geo_col, values=outcome_col).sort_index()

    cell_results = {}
    cell_lift_samples = {}

    for cell in config.cells:
        label = cell.label or cell.name
        logger.info(f"Running cell: {label} ({cell.name})")

        # For cells with multiple test geos, we average them into
        # a single synthetic "treatment" series.
        if len(cell.test_geos) == 1:
            test_geo = cell.test_geos[0]
        else:
            # Create a synthetic aggregated test geo
            synthetic_name = f"__cell_{label}__"
            df_wide[synthetic_name] = df_wide[cell.test_geos].mean(axis=1)
            test_geo = synthetic_name

        # Build ExperimentConfig for this cell
        exp_config = ExperimentConfig(
            name=f"{config.name}__{label}",
            test_geo=test_geo,
            control_geos=config.control_geos,
            pre_period=config.pre_period,
            post_period=config.post_period,
        )

        try:
            result = _run_pipeline(df_wide, exp_config)
            result["cell_name"] = cell.name
            result["cell_label"] = label
            result["cell_test_geos"] = cell.test_geos
            cell_results[label] = result
        except Exception as e:
            logger.error(f"Cell {label} failed: {e}")
            cell_results[label] = {
                "cell_name": cell.name,
                "cell_label": label,
                "cell_test_geos": cell.test_geos,
                "error": str(e),
            }

    # ------------------------------------------------------------------
    # Pairwise comparisons
    # ------------------------------------------------------------------
    comparisons = {}
    labels = [c.label or c.name for c in config.cells]

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a_label = labels[i]
            b_label = labels[j]
            a_result = cell_results.get(a_label, {})
            b_result = cell_results.get(b_label, {})

            if "error" in a_result or "error" in b_result:
                comparisons[f"{a_label}_vs_{b_label}"] = {
                    "error": "One or both cells failed.",
                }
                continue

            comp = compare_cells(a_result, b_result, a_label, b_label)
            comparisons[f"{a_label}_vs_{b_label}"] = comp

    # ------------------------------------------------------------------
    # Synergy detection (if 3+ cells — look for "both" type cell)
    # ------------------------------------------------------------------
    synergy = None
    if len(config.cells) >= 3:
        synergy = _detect_synergy(cell_results, labels)

    return {
        "experiment_name": config.name,
        "cells": cell_results,
        "comparisons": comparisons,
        "synergy": synergy,
        "config": {
            "name": config.name,
            "num_cells": len(config.cells),
            "control_geos": config.control_geos,
            "pre_period": {
                "start_date": str(config.pre_period.start_date),
                "end_date": str(config.pre_period.end_date),
            },
            "post_period": {
                "start_date": str(config.post_period.start_date),
                "end_date": str(config.post_period.end_date),
            },
        },
    }


def _detect_synergy(
    cell_results: Dict[str, Dict],
    labels: List[str],
) -> Optional[Dict[str, Any]]:
    """
    If there are 3+ cells, check whether the last cell's lift is greater
    than the sum of the other cells' lifts (super-additive synergy).

    Convention: the last cell is assumed to be the "combined" treatment.
    """
    valid = [l for l in labels if "error" not in cell_results.get(l, {})]
    if len(valid) < 3:
        return None

    try:
        combined_label = valid[-1]
        individual_labels = valid[:-1]

        combined_lift = cell_results[combined_label]["metrics"]["incremental_outcome_mean"]
        sum_individual = sum(
            cell_results[l]["metrics"]["incremental_outcome_mean"]
            for l in individual_labels
        )

        synergy_value = combined_lift - sum_individual
        synergy_pct = (synergy_value / sum_individual * 100) if sum_individual != 0 else 0

        return {
            "combined_cell": combined_label,
            "individual_cells": individual_labels,
            "combined_lift": combined_lift,
            "sum_individual_lifts": sum_individual,
            "synergy_delta": synergy_value,
            "synergy_pct": synergy_pct,
            "is_super_additive": synergy_value > 0,
        }
    except Exception as e:
        logger.error(f"Synergy detection failed: {e}")
        return None
