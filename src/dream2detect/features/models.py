from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FeatureOption:
    axis_name: str
    value: str
    prompt_hint: str
    allowed_bands: frozenset[str] | None = None
    compatible_damage_locations: frozenset[str] | None = None
    stress_weight: int = 0
    text_risk_weight: int = 0

    def is_allowed_for_band(self, score_band: str) -> bool:
        return self.allowed_bands is None or score_band in self.allowed_bands


@dataclass(frozen=True)
class FeatureAxis:
    name: str
    category: str
    description: str
    options: tuple[FeatureOption, ...]

    def allowed_options(self, score_band: str) -> tuple[FeatureOption, ...]:
        return tuple(option for option in self.options if option.is_allowed_for_band(score_band))


@dataclass(frozen=True)
class FeatureAssignment:
    damage_profile_primary: str
    damage_profile_secondary: str
    damage_location_primary: str
    box_form_factor: str
    box_pattern: str
    label_presence: str
    tape_profile: str
    background_context: str
    camera_angle: str
    lighting_style: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)

    def signature(self) -> tuple[str, ...]:
        values = self.as_dict()
        return tuple(values[key] for key in sorted(values))
