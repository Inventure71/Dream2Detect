from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import load_settings


@dataclass(frozen=True)
class BandDefinition:
    score_band: str
    coarse_class: str
    representative_score: int
    spec_filename: str

    @property
    def spec_path(self) -> Path:
        settings = load_settings()
        return settings.band_specs_dir / self.spec_filename


BAND_DEFINITIONS: dict[str, BandDefinition] = {
    "0-10": BandDefinition("0-10", "intact", 5, "00-10-intact.md"),
    "11-20": BandDefinition("11-20", "minor", 15, "11-20-minor-low.md"),
    "21-30": BandDefinition("21-30", "minor", 25, "21-30-minor-mid.md"),
    "31-35": BandDefinition("31-35", "minor", 33, "31-35-minor-high.md"),
    "36-45": BandDefinition("36-45", "moderate", 40, "36-45-moderate-low.md"),
    "46-55": BandDefinition("46-55", "moderate", 50, "46-55-moderate-mid.md"),
    "56-65": BandDefinition("56-65", "moderate", 60, "56-65-moderate-high.md"),
    "66-75": BandDefinition("66-75", "severe", 70, "66-75-severe-low.md"),
    "76-85": BandDefinition("76-85", "severe", 80, "76-85-severe-mid.md"),
    "86-100": BandDefinition("86-100", "severe", 93, "86-100-severe-high.md"),
}


def get_band_definition(score_band: str) -> BandDefinition:
    try:
        return BAND_DEFINITIONS[score_band]
    except KeyError as exc:
        supported = ", ".join(BAND_DEFINITIONS)
        raise ValueError(f"Unknown score band '{score_band}'. Supported bands: {supported}") from exc


def load_band_spec_text(score_band: str) -> str:
    band = get_band_definition(score_band)
    return band.spec_path.read_text(encoding="utf-8")
