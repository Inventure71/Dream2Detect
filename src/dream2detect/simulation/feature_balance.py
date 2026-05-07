from __future__ import annotations

import csv
import html
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ..bands import BAND_DEFINITIONS
from ..features.catalog import ASSIGNMENT_FIELD_NAMES, FEATURE_AXES
from ..features.models import FeatureAssignment
from ..features.sampler import sample_feature_assignments


@dataclass(frozen=True)
class FeatureBalanceSimulationResult:
    band_plan: dict[str, int]
    sampled_assignments: list[tuple[str, FeatureAssignment]]

    @property
    def total_assignments(self) -> int:
        return len(self.sampled_assignments)


class _InMemoryAssignmentRepository:
    def __init__(self) -> None:
        self._assignments_by_band: dict[str, list[FeatureAssignment]] = {}

    def add_assignment(self, *, score_band: str, assignment: FeatureAssignment) -> None:
        self._assignments_by_band.setdefault(score_band, []).append(assignment)

    def list_feature_assignments(self, *, score_band: str) -> list[FeatureAssignment]:
        return list(self._assignments_by_band.get(score_band, []))

    def get_feature_value_counts(self, *, score_band: str, axis_name: str) -> Counter[str]:
        counter: Counter[str] = Counter()
        for assignment in self._assignments_by_band.get(score_band, []):
            counter[getattr(assignment, axis_name)] += 1
        return counter


def build_even_band_plan(total_prompts: int) -> dict[str, int]:
    if total_prompts <= 0:
        raise ValueError("total_prompts must be positive")
    band_names = list(BAND_DEFINITIONS)
    base = total_prompts // len(band_names)
    remainder = total_prompts % len(band_names)
    plan: dict[str, int] = {}
    for index, band_name in enumerate(band_names):
        plan[band_name] = base + (1 if index < remainder else 0)
    return plan


def run_feature_balance_simulation(
    *,
    total_prompts: int,
    seed: int | None = None,
    band_plan: dict[str, int] | None = None,
) -> FeatureBalanceSimulationResult:
    resolved_band_plan = band_plan or build_even_band_plan(total_prompts)
    if sum(resolved_band_plan.values()) != total_prompts:
        raise ValueError("band_plan counts must sum to total_prompts")

    repository = _InMemoryAssignmentRepository()
    sampled_assignments: list[tuple[str, FeatureAssignment]] = []

    for band_index, (score_band, count) in enumerate(resolved_band_plan.items()):
        if count <= 0:
            continue
        band_seed = None if seed is None else seed + band_index
        assignments = sample_feature_assignments(
            score_band=score_band,
            count=count,
            repository=repository,
            seed=band_seed,
        )
        for assignment in assignments:
            repository.add_assignment(score_band=score_band, assignment=assignment)
            sampled_assignments.append((score_band, assignment))

    return FeatureBalanceSimulationResult(
        band_plan=dict(resolved_band_plan),
        sampled_assignments=sampled_assignments,
    )


def _assignment_csv_rows(result: FeatureBalanceSimulationResult) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index, (score_band, assignment) in enumerate(result.sampled_assignments, start=1):
        row = {
            "simulated_prompt_id": f"sim_prompt_{index:04d}",
            "score_band": score_band,
        }
        row.update(assignment.as_dict())
        rows.append(row)
    return rows


def _axis_count_rows(result: FeatureBalanceSimulationResult) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    total = result.total_assignments
    for axis in FEATURE_AXES:
        counts: Counter[str] = Counter()
        for _, assignment in result.sampled_assignments:
            counts[getattr(assignment, axis.name)] += 1
        for option in axis.options:
            count = counts[option.value]
            target_count = 0.0
            for score_band, band_count in result.band_plan.items():
                allowed_values_for_band = [axis_option for axis_option in axis.allowed_options(score_band)]
                if not allowed_values_for_band:
                    continue
                if any(axis_option.value == option.value for axis_option in allowed_values_for_band):
                    target_count += band_count / len(allowed_values_for_band)
            rows.append(
                {
                    "axis_name": axis.name,
                    "axis_category": axis.category,
                    "value": option.value,
                    "count": count,
                    "share": round((count / total) if total else 0.0, 6),
                    "target_mean": round(target_count, 4),
                    "delta_from_mean": round(count - target_count, 4),
                }
            )
    return rows


def _band_count_rows(result: FeatureBalanceSimulationResult) -> list[dict[str, str | int]]:
    counter: Counter[str] = Counter(score_band for score_band, _ in result.sampled_assignments)
    return [
        {"score_band": score_band, "count": counter[score_band]}
        for score_band in BAND_DEFINITIONS
    ]


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _render_axis_section(axis_name: str, rows: list[dict[str, str | int | float]]) -> str:
    max_count = max(int(row["count"]) for row in rows) if rows else 1
    bars: list[str] = []
    for row in rows:
        count = int(row["count"])
        width_pct = 0 if max_count == 0 else (count / max_count) * 100
        bars.append(
            f"""
            <div class="bar-row">
              <div class="bar-label">{html.escape(str(row["value"]))}</div>
              <div class="bar-track">
                <div class="bar-fill" style="width: {width_pct:.2f}%"></div>
              </div>
              <div class="bar-meta">count={count} | mean={float(row["target_mean"]):.2f} | delta={float(row["delta_from_mean"]):+.2f}</div>
            </div>
            """
        )
    return f"""
    <section class="axis-card">
      <h2>{html.escape(axis_name)}</h2>
      {''.join(bars)}
    </section>
    """


def _render_html_report(result: FeatureBalanceSimulationResult, axis_rows: list[dict[str, str | int | float]]) -> str:
    grouped_rows: dict[str, list[dict[str, str | int | float]]] = {}
    for row in axis_rows:
        grouped_rows.setdefault(str(row["axis_name"]), []).append(row)

    axis_sections = "".join(
        _render_axis_section(axis_name, grouped_rows[axis_name])
        for axis_name in ASSIGNMENT_FIELD_NAMES
    )
    band_items = "".join(
        f"<li><strong>{html.escape(score_band)}</strong>: {count}</li>"
        for score_band, count in result.band_plan.items()
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Dream2Detect Offline Feature Balance Simulation</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      margin: 24px;
      color: #111827;
      background: #f8fafc;
    }}
    h1, h2 {{
      margin: 0 0 12px 0;
    }}
    .summary, .axis-card {{
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 16px;
    }}
    .bar-row {{
      display: grid;
      grid-template-columns: 220px 1fr 240px;
      gap: 12px;
      align-items: center;
      margin: 8px 0;
    }}
    .bar-track {{
      height: 18px;
      background: #e5e7eb;
      border-radius: 999px;
      overflow: hidden;
    }}
    .bar-fill {{
      height: 100%;
      background: #2563eb;
    }}
    .bar-label, .bar-meta {{
      font-size: 13px;
      line-height: 1.3;
    }}
    ul {{
      margin: 8px 0 0 20px;
      padding: 0;
    }}
    code {{
      background: #eef2ff;
      padding: 2px 6px;
      border-radius: 4px;
    }}
  </style>
</head>
<body>
  <section class="summary">
    <h1>Offline Feature Balance Simulation</h1>
    <p>Total simulated prompts: <strong>{result.total_assignments}</strong></p>
    <p>This report is offline only. It uses the feature sampler and never calls the OpenAI API.</p>
    <p>Band plan:</p>
    <ul>{band_items}</ul>
  </section>
  {axis_sections}
</body>
</html>
"""


def write_feature_balance_outputs(
    *,
    result: FeatureBalanceSimulationResult,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    assignment_rows = _assignment_csv_rows(result)
    axis_rows = _axis_count_rows(result)
    band_rows = _band_count_rows(result)

    assignments_csv = output_dir / "sampled_assignments.csv"
    axis_counts_csv = output_dir / "axis_counts.csv"
    band_counts_csv = output_dir / "band_counts.csv"
    report_html = output_dir / "report.html"

    _write_csv(
        assignments_csv,
        ["simulated_prompt_id", "score_band", *ASSIGNMENT_FIELD_NAMES],
        assignment_rows,
    )
    _write_csv(
        axis_counts_csv,
        ["axis_name", "axis_category", "value", "count", "share", "target_mean", "delta_from_mean"],
        axis_rows,
    )
    _write_csv(
        band_counts_csv,
        ["score_band", "count"],
        band_rows,
    )
    report_html.write_text(_render_html_report(result, axis_rows), encoding="utf-8")

    return {
        "assignments_csv": assignments_csv,
        "axis_counts_csv": axis_counts_csv,
        "band_counts_csv": band_counts_csv,
        "report_html": report_html,
    }
