from __future__ import annotations

import random
from collections import Counter

from .catalog import ASSIGNMENT_FIELD_NAMES, FEATURE_AXES, get_feature_axis
from .models import FeatureAssignment, FeatureOption

MAX_STRESS_WEIGHT_PER_ASSIGNMENT = 1
MAX_TEXT_RISK_PER_ASSIGNMENT = 2


def _candidate_value_pool(counts: Counter[str], values: list[str]) -> list[str]:
    if not values:
        return []
    ranked = sorted(values, key=lambda value: (counts[value], value))
    min_count = counts[ranked[0]]
    return [value for value in ranked if counts[value] <= min_count + 1]


def _build_existing_combo_counts(
    score_band: str,
    existing_assignments: list[FeatureAssignment],
) -> Counter[tuple[str, ...]]:
    counter: Counter[tuple[str, ...]] = Counter()
    for assignment in existing_assignments:
        counter[assignment.signature()] += 1
    return counter


def _build_existing_concept_counts(existing_assignments: list[FeatureAssignment]) -> Counter[tuple[str, ...]]:
    counter: Counter[tuple[str, ...]] = Counter()
    for assignment in existing_assignments:
        counter[_concept_signature(assignment)] += 1
    return counter


def sample_feature_assignments(
    *,
    score_band: str,
    count: int,
    repository,
    seed: int | None = None,
) -> list[FeatureAssignment]:
    rng = random.Random(seed)
    axis_counts = {
        axis.name: repository.get_feature_value_counts(score_band=score_band, axis_name=axis.name)
        for axis in FEATURE_AXES
    }
    existing_assignments = repository.list_feature_assignments(score_band=score_band)
    combo_counts = _build_existing_combo_counts(score_band, existing_assignments)
    concept_counts = _build_existing_concept_counts(existing_assignments)
    sampled: list[FeatureAssignment] = []

    for _ in range(count):
        best_candidate: FeatureAssignment | None = None
        best_score: tuple[int, int, int] | None = None

        for _attempt in range(64):
            chosen_values: dict[str, str] = {}
            total_axis_cost = 0

            for axis in FEATURE_AXES:
                allowed = axis.allowed_options(score_band)
                if axis.name == "damage_location_primary":
                    primary_value = chosen_values.get("damage_profile_primary")
                    if primary_value is not None:
                        allowed = tuple(
                            option
                            for option in allowed
                            if _is_location_compatible_with_primary(
                                primary_value=primary_value,
                                location_value=option.value,
                            )
                        )
                if axis.name == "camera_angle":
                    location_value = chosen_values.get("damage_location_primary")
                    if location_value is not None:
                        allowed = tuple(
                            option
                            for option in allowed
                            if _is_camera_compatible_with_location(
                                camera_option=option,
                                location_value=location_value,
                            )
                        )
                allowed_values = [option.value for option in allowed]
                if not allowed_values:
                    raise RuntimeError(f"No allowed feature values for axis {axis.name!r} in band {score_band!r}")
                effective_counts = Counter(axis_counts[axis.name])
                for prior in sampled:
                    effective_counts[getattr(prior, axis.name)] += 1
                if axis.name != "damage_location_primary":
                    pool = _stress_filtered_pool(
                        axis_name=axis.name,
                        score_band=score_band,
                        chosen_values=chosen_values,
                        effective_counts=effective_counts,
                        allowed_values=allowed_values,
                    )
                else:
                    pool = _candidate_value_pool(effective_counts, allowed_values)
                value = rng.choice(pool)
                chosen_values[axis.name] = value
                total_axis_cost += effective_counts[value]

            candidate = FeatureAssignment(**chosen_values)
            signature = candidate.signature()
            duplication_penalty = combo_counts[signature]
            for prior in sampled:
                if prior.signature() == signature:
                    duplication_penalty += 1
            concept_penalty = concept_counts[_concept_signature(candidate)]
            for prior in sampled:
                if _concept_signature(prior) == _concept_signature(candidate):
                    concept_penalty += 1
            score = (duplication_penalty, concept_penalty, total_axis_cost)

            if best_score is None or score < best_score:
                best_candidate = candidate
                best_score = score
                if score[0] == 0:
                    break

        if best_candidate is None:
            raise RuntimeError(f"Could not sample a feature assignment for band {score_band!r}")

        sampled.append(best_candidate)

    return sampled


def _is_location_compatible_with_primary(*, primary_value: str, location_value: str) -> bool:
    primary_axis = get_feature_axis("damage_profile_primary")
    primary_option = next(option for option in primary_axis.options if option.value == primary_value)
    compatible_locations = primary_option.compatible_damage_locations
    return compatible_locations is None or location_value in compatible_locations


def _is_camera_compatible_with_location(*, camera_option: FeatureOption, location_value: str) -> bool:
    compatible_locations = camera_option.compatible_damage_locations
    return compatible_locations is None or location_value in compatible_locations


def _stress_filtered_pool(
    *,
    axis_name: str,
    score_band: str,
    chosen_values: dict[str, str],
    effective_counts: Counter[str],
    allowed_values: list[str],
) -> list[str]:
    axis = get_feature_axis(axis_name)
    current_stress = _current_assignment_stress(chosen_values)
    current_text_risk = _current_assignment_text_risk(chosen_values)
    stress_safe_values = [
        option.value
        for option in axis.allowed_options(score_band)
        if option.value in allowed_values
        and current_stress + option.stress_weight <= MAX_STRESS_WEIGHT_PER_ASSIGNMENT
        and current_text_risk + option.text_risk_weight <= MAX_TEXT_RISK_PER_ASSIGNMENT
    ]
    candidate_values = stress_safe_values or allowed_values
    return _candidate_value_pool(effective_counts, candidate_values)


def _current_assignment_stress(chosen_values: dict[str, str]) -> int:
    total = 0
    for axis_name, value in chosen_values.items():
        axis = get_feature_axis(axis_name)
        option = next(option for option in axis.options if option.value == value)
        total += option.stress_weight
    return total


def _current_assignment_text_risk(chosen_values: dict[str, str]) -> int:
    total = 0
    for axis_name, value in chosen_values.items():
        axis = get_feature_axis(axis_name)
        option = next(option for option in axis.options if option.value == value)
        total += option.text_risk_weight
    return total


def _concept_signature(assignment: FeatureAssignment) -> tuple[str, ...]:
    return (
        assignment.damage_profile_primary,
        assignment.background_context,
    )
