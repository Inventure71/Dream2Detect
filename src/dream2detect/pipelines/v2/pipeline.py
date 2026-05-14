from __future__ import annotations

import csv
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dream2detect.bands import BAND_DEFINITIONS, get_band_definition
from dream2detect.features.catalog import ASSIGNMENT_FIELD_NAMES, FEATURE_AXES, get_feature_axis
from dream2detect.features.models import FeatureAssignment, FeatureOption
from dream2detect.services.batch_image_generation import BATCH_ENDPOINT


V2_PIPELINE_VERSION = "v2_real_failure_targeted"


@dataclass(frozen=True)
class ImageGenerationSpec:
    model: str = "gpt-image-2"
    quality: str = "medium"
    size: str = "1024x1024"


@dataclass(frozen=True)
class V2PipelineConfig:
    run_name: str
    count_per_band: int = 12
    band_counts: dict[str, int] | None = None
    seed: int = 20260509
    output_dir: Path = Path("data/pipelines/v2")
    image_spec: ImageGenerationSpec = ImageGenerationSpec()


@dataclass(frozen=True)
class RealDomainScenario:
    scenario_id: str
    family: str
    eligible_coarse_classes: tuple[str, ...]
    real_gap_priority: str
    target_failure_mode: str
    preferred_values: dict[str, tuple[str, ...]]
    scene_directive: str
    image_quality_directive: str
    label_safety_directive: str
    hard_negative: bool = False
    boundary_focus: str = "none"
    eligible_score_bands: tuple[str, ...] = ()


@dataclass(frozen=True)
class V2PromptRow:
    prompt_uid: str
    custom_id: str
    pipeline_version: str
    score_band: str
    coarse_class: str
    representative_score: int
    band_spec_file: str
    scenario_id: str
    scenario_family: str
    real_gap_priority: str
    target_failure_mode: str
    hard_negative: bool
    boundary_focus: str
    prompt_title: str
    prompt_text: str
    qc_required_checks: str
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


@dataclass(frozen=True)
class V2PipelineArtifacts:
    run_dir: Path
    prompt_manifest_path: Path
    input_jsonl_path: Path
    manifest_json_path: Path
    metadata_path: Path
    run_config_path: Path
    summary_path: Path
    qc_rubric_path: Path
    prompt_count: int


def default_real_domain_scenarios() -> tuple[RealDomainScenario, ...]:
    return (
        RealDomainScenario(
            scenario_id="hard_intact_tape_label_clutter",
            family="hard_negative",
            eligible_coarse_classes=("intact",),
            real_gap_priority="reduce false damage cues on intact packages",
            target_failure_mode="intact package that looks visually busy but is not structurally damaged",
            hard_negative=True,
            boundary_focus="intact_vs_minor",
            preferred_values={
                "damage_profile_primary": ("visually_intact",),
                "damage_profile_secondary": ("tiny_handling_scuffs", "none"),
                "box_form_factor": ("flat_mailer_box", "medium_standard", "long_rectangular"),
                "box_pattern": ("shipping_label_patch", "tape_heavy_plain", "subtle_printed_markings"),
                "label_presence": ("small_shipping_label", "barcode_label_only", "multiple_small_stickers"),
                "tape_profile": ("cross_tape", "reinforced_edge_tape", "old_peeling_tape"),
                "background_context": ("delivery_sorting_table", "packing_station", "warehouse_floor"),
                "camera_angle": ("slightly_top_down", "high_three_quarter_left", "high_three_quarter_right"),
                "lighting_style": ("mixed_indoor", "cool_fluorescent", "harsh_side_shadow"),
            },
            scene_directive="Make the scene look like a real phone photo taken during shipping intake, with normal clutter and tape but no actual structural defect.",
            image_quality_directive="Use realistic smartphone exposure, mild compression, imperfect framing, and non-studio lighting.",
            label_safety_directive="Any label or barcode must be small, generic, and unreadable; no logos or personal data.",
        ),
        RealDomainScenario(
            scenario_id="hard_intact_near_damage_context",
            family="hard_negative",
            eligible_coarse_classes=("intact",),
            real_gap_priority="teach intact despite clutter, tape wrinkles, shadows, and label noise",
            target_failure_mode="intact package surrounded by visual distractions that should not count as damage",
            hard_negative=True,
            boundary_focus="intact_vs_minor",
            preferred_values={
                "damage_profile_primary": ("visually_intact",),
                "damage_profile_secondary": ("tiny_handling_scuffs", "none"),
                "box_form_factor": ("flat_mailer_box", "medium_standard", "long_rectangular"),
                "box_pattern": ("shipping_label_patch", "tape_heavy_plain", "recycled_patchwork_cardboard"),
                "label_presence": ("small_shipping_label", "barcode_label_only", "multiple_small_stickers"),
                "tape_profile": ("old_peeling_tape", "single_center_tape", "reinforced_edge_tape"),
                "background_context": ("outdoor_doorstep", "van_cargo_area", "warehouse_floor"),
                "camera_angle": ("slightly_top_down", "top_down", "high_three_quarter_left", "high_three_quarter_right"),
                "lighting_style": ("harsh_side_shadow", "dim_ambient", "mixed_indoor"),
            },
            scene_directive="Include wrinkles in tape, shadows, scuffs, or background clutter, but keep the cardboard faces and edges structurally intact.",
            image_quality_directive="Use a real handheld phone-photo look with imperfect alignment and mild compression.",
            label_safety_directive="Labels and barcodes must be generic, small, and unreadable.",
        ),
        RealDomainScenario(
            scenario_id="flat_taped_mailer_subtle_damage",
            family="real_failure_repair",
            eligible_coarse_classes=("minor", "moderate"),
            real_gap_priority="repair synthetic model tendency to call subtle flat-mailer damage intact",
            target_failure_mode="flat or shallow package with subtle real damage previously confused with intact",
            boundary_focus="minor_vs_intact",
            preferred_values={
                "damage_profile_primary": (
                    "light_corner_softening",
                    "small_face_dent",
                    "light_edge_crease",
                    "clear_corner_crease",
                    "face_dent_with_crease",
                ),
                "damage_profile_secondary": ("tiny_handling_scuffs", "light_crease_cluster", "minor_tape_peel"),
                "damage_location_primary": ("front_face_center", "top_flap_edge", "front_top_left_corner", "front_top_right_corner"),
                "box_form_factor": ("flat_mailer_box", "long_rectangular"),
                "box_pattern": ("tape_heavy_plain", "shipping_label_patch", "recycled_patchwork_cardboard"),
                "label_presence": ("small_shipping_label", "barcode_label_only"),
                "tape_profile": ("old_peeling_tape", "single_center_tape", "reinforced_edge_tape"),
                "background_context": ("delivery_sorting_table", "warehouse_floor", "outdoor_doorstep"),
                "camera_angle": ("slightly_top_down", "top_down", "high_three_quarter_left", "high_three_quarter_right"),
                "lighting_style": ("mixed_indoor", "soft_even", "dim_ambient"),
            },
            scene_directive="Show a flat mailer-style cardboard package where the damage is visible but easy to miss at a glance.",
            image_quality_directive="Keep the damage readable at 384px training resolution; avoid perfect studio sharpness.",
            label_safety_directive="Keep shipping labels present but visually secondary and unreadable.",
        ),
        RealDomainScenario(
            scenario_id="minor_low_contrast_real_recall",
            family="minor_recall_repair",
            eligible_coarse_classes=("minor",),
            real_gap_priority="increase minor recall without turning minor into moderate",
            target_failure_mode="low-contrast minor damage that must be visible but not exaggerated",
            boundary_focus="minor_core",
            preferred_values={
                "damage_profile_primary": (
                    "light_corner_softening",
                    "small_face_dent",
                    "light_edge_crease",
                    "clear_corner_crease",
                    "face_dent_with_crease",
                ),
                "damage_profile_secondary": ("tiny_handling_scuffs", "light_crease_cluster", "minor_tape_peel"),
                "damage_location_primary": ("front_face_center", "top_flap_edge", "front_top_left_corner", "front_top_right_corner"),
                "box_form_factor": ("flat_mailer_box", "medium_standard", "long_rectangular"),
                "box_pattern": ("plain_brown", "shipping_label_patch", "tape_heavy_plain"),
                "label_presence": ("small_shipping_label", "barcode_label_only", "no_visible_label"),
                "tape_profile": ("single_center_tape", "old_peeling_tape", "reinforced_edge_tape"),
                "background_context": ("delivery_sorting_table", "warehouse_floor", "outdoor_doorstep"),
                "camera_angle": ("slightly_top_down", "diagonal_corner_view", "high_three_quarter_left", "high_three_quarter_right"),
                "lighting_style": ("soft_even", "mixed_indoor", "dim_ambient"),
            },
            scene_directive="Make the minor defect legible on close inspection while preserving the sense that the package is still usable and mostly intact.",
            image_quality_directive="Avoid cartoon creases; use real cardboard texture and modest phone-camera softness.",
            label_safety_directive="No readable addresses, logos, or large label text.",
        ),
        RealDomainScenario(
            scenario_id="minor_moderate_boundary_edge_split",
            family="boundary_case",
            eligible_coarse_classes=("minor", "moderate"),
            real_gap_priority="sharpen minor/moderate boundary decisions around edge damage",
            target_failure_mode="minor-to-moderate boundary edge and seam damage",
            boundary_focus="minor_vs_moderate",
            preferred_values={
                "damage_profile_primary": (
                    "corner_crumple_without_opening",
                    "short_partial_tear",
                    "bent_flap_or_edge",
                    "edge_split_without_cavity",
                    "frayed_corner_plus_face_dent",
                    "single_small_puncture_with_denting",
                ),
                "damage_profile_secondary": ("light_crease_cluster", "secondary_corner_softening", "minor_tape_peel"),
                "damage_location_primary": ("top_flap_edge", "front_bottom_left_corner", "front_bottom_right_corner", "left_side_panel", "right_side_panel"),
                "box_form_factor": ("medium_standard", "long_rectangular", "flat_mailer_box"),
                "box_pattern": ("plain_brown", "tape_heavy_plain", "recycled_patchwork_cardboard"),
                "label_presence": ("no_visible_label", "small_shipping_label"),
                "tape_profile": ("single_center_tape", "reinforced_edge_tape", "old_peeling_tape"),
                "background_context": ("warehouse_floor", "delivery_sorting_table", "van_cargo_area"),
                "camera_angle": ("bottom_edge_low", "diagonal_corner_view", "side_profile_left", "side_profile_right"),
                "lighting_style": ("soft_even", "cool_fluorescent", "harsh_side_shadow"),
            },
            scene_directive="Frame the affected edge clearly so a reviewer can decide whether it is still minor or truly moderate.",
            image_quality_directive="Use realistic perspective and mild shadows, but do not hide the seam or edge condition.",
            label_safety_directive="Avoid large readable markings; small generic handling marks are acceptable.",
        ),
        RealDomainScenario(
            scenario_id="moderate_not_intact_real_geometry",
            family="moderate_recall_repair",
            eligible_coarse_classes=("moderate",),
            real_gap_priority="reduce high-confidence real moderate predicted as intact",
            target_failure_mode="moderate real cardboard deformation that should never look intact",
            boundary_focus="moderate_core",
            preferred_values={
                "damage_profile_primary": (
                    "partial_side_compression",
                    "clear_corner_split",
                    "face_dent_with_visible_tear",
                    "multiple_damaged_edges",
                    "narrow_opening_from_edge_split",
                    "controlled_side_crush",
                ),
                "damage_profile_secondary": ("secondary_face_dent", "light_crease_cluster", "minor_tape_peel", "small_stain_patch"),
                "damage_location_primary": ("left_side_panel", "right_side_panel", "front_face_center", "top_flap_edge"),
                "box_form_factor": ("flat_mailer_box", "medium_standard", "long_rectangular"),
                "box_pattern": ("shipping_label_patch", "tape_heavy_plain", "recycled_patchwork_cardboard"),
                "label_presence": ("small_shipping_label", "barcode_label_only", "no_visible_label"),
                "tape_profile": ("reinforced_edge_tape", "cross_tape", "old_peeling_tape"),
                "background_context": ("warehouse_floor", "delivery_sorting_table", "outdoor_doorstep", "van_cargo_area"),
                "camera_angle": ("three_quarter_left", "three_quarter_right", "diagonal_corner_view", "slightly_top_down"),
                "lighting_style": ("mixed_indoor", "harsh_side_shadow", "soft_even"),
            },
            scene_directive="Show clear loss of cardboard geometry while keeping the package recognizable and not fully destroyed.",
            image_quality_directive="Use realistic clutter, shadows, tape reflections, and non-square-phone-photo framing cues.",
            label_safety_directive="Any shipping information must be unreadable and generic.",
        ),
        RealDomainScenario(
            scenario_id="moderate_real_crush_with_tape",
            family="real_failure_repair",
            eligible_coarse_classes=("moderate", "severe"),
            real_gap_priority="make moderate and low-severe damage look like real received packages",
            target_failure_mode="moderate crushed or split package with tape and real-world clutter",
            boundary_focus="moderate_core",
            preferred_values={
                "damage_profile_primary": (
                    "partial_side_compression",
                    "clear_corner_split",
                    "face_dent_with_visible_tear",
                    "multiple_damaged_edges",
                    "narrow_opening_from_edge_split",
                    "controlled_side_crush",
                ),
                "damage_profile_secondary": ("secondary_face_dent", "minor_tape_peel", "small_stain_patch"),
                "damage_location_primary": ("left_side_panel", "right_side_panel", "front_face_center", "top_flap_edge"),
                "box_form_factor": ("medium_standard", "long_rectangular", "tall_rectangular"),
                "box_pattern": ("tape_heavy_plain", "shipping_label_patch", "recycled_patchwork_cardboard"),
                "label_presence": ("small_shipping_label", "barcode_label_only", "no_visible_label"),
                "tape_profile": ("reinforced_edge_tape", "cross_tape", "old_peeling_tape"),
                "background_context": ("warehouse_floor", "delivery_sorting_table", "van_cargo_area"),
                "camera_angle": ("three_quarter_left", "three_quarter_right", "high_three_quarter_left", "high_three_quarter_right"),
                "lighting_style": ("mixed_indoor", "cool_fluorescent", "harsh_side_shadow"),
            },
            scene_directive="Make the package look like a real received shipment, not a staged product shot.",
            image_quality_directive="Preserve clear geometry loss while adding realistic shadows, tape reflections, and normal camera noise.",
            label_safety_directive="Labels must not be readable or brand-like.",
        ),
        RealDomainScenario(
            scenario_id="severe_realistic_not_cartoon",
            family="severe_reality_anchor",
            eligible_coarse_classes=("severe",),
            real_gap_priority="increase severe recall without render-like or exaggerated damage",
            target_failure_mode="severe damage that remains photographically realistic rather than exaggerated",
            boundary_focus="moderate_vs_severe",
            preferred_values={
                "damage_profile_primary": (
                    "large_corner_opening",
                    "strong_side_crushing",
                    "broad_edge_tear",
                    "major_side_collapse",
                    "large_torn_opening",
                    "heavy_crushing_with_split_seams",
                    "partially_flattened_box",
                ),
                "damage_profile_secondary": ("additional_crushed_edge", "broad_surface_grime", "extra_small_puncture"),
                "damage_location_primary": ("front_face_center", "left_side_panel", "right_side_panel", "front_top_left_corner", "front_top_right_corner"),
                "box_form_factor": ("medium_standard", "long_rectangular", "tall_rectangular"),
                "box_pattern": ("plain_brown", "tape_heavy_plain", "recycled_patchwork_cardboard"),
                "label_presence": ("no_visible_label", "small_shipping_label"),
                "tape_profile": ("reinforced_edge_tape", "cross_tape", "old_peeling_tape"),
                "background_context": ("warehouse_floor", "van_cargo_area", "outdoor_doorstep"),
                "camera_angle": ("three_quarter_left", "three_quarter_right", "slightly_top_down", "diagonal_corner_view"),
                "lighting_style": ("soft_even", "mixed_indoor", "dim_ambient"),
            },
            scene_directive="Show severe structural failure with plausible cardboard material behavior and no surreal tearing.",
            image_quality_directive="Use documentary realism; avoid dramatic cinematic lighting or artificially clean edges.",
            label_safety_directive="No readable logos, addresses, brand marks, or large text.",
        ),
        RealDomainScenario(
            scenario_id="severe_not_intact_opening_visible",
            family="severe_recall_repair",
            eligible_coarse_classes=("severe",),
            real_gap_priority="repair severe real examples predicted as intact",
            target_failure_mode="severe structural opening or collapse that remains obvious under real capture artifacts",
            boundary_focus="severe_core",
            preferred_values={
                "damage_profile_primary": (
                    "large_corner_opening",
                    "strong_side_crushing",
                    "broad_edge_tear",
                    "major_side_collapse",
                    "large_torn_opening",
                    "heavy_crushing_with_split_seams",
                    "partially_flattened_box",
                ),
                "damage_profile_secondary": ("additional_crushed_edge", "broad_surface_grime", "extra_small_puncture"),
                "damage_location_primary": ("left_side_panel", "right_side_panel", "front_face_center", "front_top_left_corner", "front_top_right_corner"),
                "box_form_factor": ("medium_standard", "long_rectangular", "tall_rectangular", "flat_mailer_box"),
                "box_pattern": ("plain_brown", "tape_heavy_plain", "shipping_label_patch", "recycled_patchwork_cardboard"),
                "label_presence": ("small_shipping_label", "barcode_label_only", "no_visible_label"),
                "tape_profile": ("old_peeling_tape", "cross_tape", "reinforced_edge_tape"),
                "background_context": ("warehouse_floor", "van_cargo_area", "outdoor_doorstep", "delivery_sorting_table"),
                "camera_angle": ("three_quarter_left", "three_quarter_right", "diagonal_corner_view", "slightly_top_down"),
                "lighting_style": ("mixed_indoor", "harsh_side_shadow", "dim_ambient"),
            },
            scene_directive="Keep the severe opening, collapse, or split seam visible even with clutter or imperfect framing.",
            image_quality_directive="Use documentary realism and phone-camera noise; avoid staged product photography and unrealistic torn edges.",
            label_safety_directive="No readable personal information, real logos, or dominant text.",
        ),
        RealDomainScenario(
            scenario_id="severe_high_extreme_collapse_anchor",
            family="severe_high_repair",
            eligible_coarse_classes=("severe",),
            eligible_score_bands=("86-100",),
            real_gap_priority="generate truly top-band severe-high examples instead of mid-severe damage",
            target_failure_mode="extreme but still recognizable package collapse with dominant multi-panel structural failure",
            boundary_focus="severe_high_reserved",
            preferred_values={
                "damage_profile_primary": (
                    "extreme_collapse",
                    "partially_flattened_box",
                    "very_large_open_tear",
                    "severe_multi_side_failure",
                ),
                "damage_profile_secondary": ("broad_surface_grime",),
                "damage_location_primary": (
                    "front_face_center",
                    "left_side_panel",
                    "right_side_panel",
                    "front_top_left_corner",
                    "front_top_right_corner",
                    "top_flap_edge",
                ),
                "box_form_factor": ("medium_standard", "long_rectangular", "tall_rectangular"),
                "box_pattern": ("plain_brown", "tape_heavy_plain", "recycled_patchwork_cardboard"),
                "label_presence": ("small_shipping_label", "barcode_label_only", "no_visible_label"),
                "tape_profile": ("old_peeling_tape", "cross_tape", "reinforced_edge_tape"),
                "background_context": ("warehouse_floor", "van_cargo_area", "outdoor_doorstep"),
                "camera_angle": ("three_quarter_left", "three_quarter_right", "slightly_top_down", "diagonal_corner_view"),
                "lighting_style": ("mixed_indoor", "harsh_side_shadow", "dim_ambient"),
            },
            scene_directive=(
                "Make the package look among the worst examples in the dataset: "
                "dominant collapse, large open tearing, or multi-side failure should be obvious immediately, "
                "while the object remains recognizable as a cardboard package."
            ),
            image_quality_directive=(
                "Use documentary phone-photo realism, but do not soften the damage; "
                "the severe-high failure must dominate the image at 384px."
            ),
            label_safety_directive="No readable labels, addresses, logos, or brand marks.",
        ),
        RealDomainScenario(
            scenario_id="severe_high_open_cavity_anchor",
            family="severe_high_repair",
            eligible_coarse_classes=("severe",),
            eligible_score_bands=("86-100",),
            real_gap_priority="force severe-high prompts to show a dominant large opening or major material failure",
            target_failure_mode="very large torn opening or exposed cavity with heavy surrounding collapse",
            boundary_focus="severe_high_reserved",
            preferred_values={
                "damage_profile_primary": (
                    "very_large_open_tear",
                    "extreme_collapse",
                    "severe_multi_side_failure",
                    "partially_flattened_box",
                ),
                "damage_profile_secondary": ("broad_surface_grime",),
                "damage_location_primary": (
                    "front_face_center",
                    "front_top_left_corner",
                    "front_top_right_corner",
                    "left_side_panel",
                    "right_side_panel",
                ),
                "box_form_factor": ("medium_standard", "tall_rectangular", "long_rectangular"),
                "box_pattern": ("tape_heavy_plain", "plain_brown", "recycled_patchwork_cardboard"),
                "label_presence": ("no_visible_label", "small_shipping_label", "barcode_label_only"),
                "tape_profile": ("old_peeling_tape", "cross_tape", "reinforced_edge_tape"),
                "background_context": ("warehouse_floor", "van_cargo_area", "outdoor_doorstep"),
                "camera_angle": ("three_quarter_left", "three_quarter_right", "diagonal_corner_view", "slightly_top_down"),
                "lighting_style": ("harsh_side_shadow", "mixed_indoor", "dim_ambient"),
            },
            scene_directive=(
                "Show a dominant large opening or exposed cavity with surrounding crushed cardboard, "
                "split seams, and severe material failure. This should not look like ordinary severe-low damage."
            ),
            image_quality_directive=(
                "Keep the photo realistic and not cartoonish, but make the top-band severity unmistakable."
            ),
            label_safety_directive="Keep all labels generic or unreadable; no personal information or logos.",
        ),
        RealDomainScenario(
            scenario_id="compression_shadow_occlusion",
            family="capture_robustness",
            eligible_coarse_classes=("minor", "moderate", "severe"),
            real_gap_priority="train robustness to shadows, compression, blur, and imperfect framing",
            target_failure_mode="real capture artifacts that can make damaged packages look intact",
            boundary_focus="visibility_stress",
            preferred_values={
                "damage_profile_primary": (
                    "light_edge_crease",
                    "face_dent_with_crease",
                    "edge_split_without_cavity",
                    "partial_side_compression",
                    "controlled_side_crush",
                    "strong_side_crushing",
                ),
                "damage_profile_secondary": ("tiny_handling_scuffs", "light_crease_cluster", "small_stain_patch", "minor_tape_peel"),
                "box_form_factor": ("flat_mailer_box", "medium_standard", "long_rectangular"),
                "box_pattern": ("shipping_label_patch", "tape_heavy_plain", "plain_brown"),
                "label_presence": ("small_shipping_label", "barcode_label_only", "no_visible_label"),
                "tape_profile": ("single_center_tape", "cross_tape", "reinforced_edge_tape"),
                "background_context": ("delivery_sorting_table", "warehouse_floor", "packing_station"),
                "camera_angle": ("high_three_quarter_left", "high_three_quarter_right", "slightly_top_down", "low_angle_front"),
                "lighting_style": ("harsh_side_shadow", "dim_ambient", "backlit_but_readable"),
            },
            scene_directive="Include a mild real-world visibility challenge, such as partial shadow, close framing, or background clutter, while keeping the label-defining damage visible.",
            image_quality_directive="Use phone-camera noise, compression, and slight blur; do not obscure the defect beyond recognition.",
            label_safety_directive="Keep any labels tiny, generic, and unreadable.",
        ),
    )


def generate_v2_prompt_rows(config: V2PipelineConfig) -> list[V2PromptRow]:
    if config.count_per_band <= 0:
        raise ValueError("count_per_band must be positive.")

    band_counts = _resolve_band_counts(config)
    rng = random.Random(config.seed)
    scenarios = default_real_domain_scenarios()
    rows: list[V2PromptRow] = []
    row_index = 1
    for score_band in BAND_DEFINITIONS:
        eligible = _eligible_scenarios_for_band(score_band, scenarios)
        for band_index in range(band_counts[score_band]):
            scenario = eligible[band_index % len(eligible)]
            assignment = _build_assignment(score_band, scenario, rng, band_index)
            rows.append(_build_prompt_row(row_index, score_band, scenario, assignment))
            row_index += 1
    return rows


def _resolve_band_counts(config: V2PipelineConfig) -> dict[str, int]:
    counts = {score_band: config.count_per_band for score_band in BAND_DEFINITIONS}
    if not config.band_counts:
        return counts

    unknown_bands = sorted(set(config.band_counts) - set(BAND_DEFINITIONS))
    if unknown_bands:
        raise ValueError(f"Unknown score bands in band_counts: {', '.join(unknown_bands)}")

    for score_band, count in config.band_counts.items():
        if count < 0:
            raise ValueError(f"band_counts for {score_band} must be non-negative.")
        counts[score_band] = count
    return counts


def build_v2_pipeline_artifacts(config: V2PipelineConfig) -> V2PipelineArtifacts:
    rows = generate_v2_prompt_rows(config)
    run_dir = config.output_dir / config.run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    prompt_manifest_path = run_dir / "prompt_manifest.csv"
    input_jsonl_path = run_dir / "input.jsonl"
    manifest_json_path = run_dir / "manifest.json"
    metadata_path = run_dir / "metadata.json"
    run_config_path = run_dir / "run_config.json"
    summary_path = run_dir / "summary.json"
    qc_rubric_path = run_dir / "qc_rubric.md"

    _write_prompt_manifest(prompt_manifest_path, rows)
    _write_batch_input(input_jsonl_path, rows, config.image_spec)
    _write_json(manifest_json_path, _build_manifest(rows, config))
    _write_json(metadata_path, _build_batch_metadata(rows, config, run_dir))
    _write_json(run_config_path, _build_run_config(config))
    _write_json(summary_path, _build_summary(rows))
    qc_rubric_path.write_text(build_qc_rubric(), encoding="utf-8")

    return V2PipelineArtifacts(
        run_dir=run_dir,
        prompt_manifest_path=prompt_manifest_path,
        input_jsonl_path=input_jsonl_path,
        manifest_json_path=manifest_json_path,
        metadata_path=metadata_path,
        run_config_path=run_config_path,
        summary_path=summary_path,
        qc_rubric_path=qc_rubric_path,
        prompt_count=len(rows),
    )


def _eligible_scenarios_for_band(
    score_band: str,
    scenarios: tuple[RealDomainScenario, ...],
) -> tuple[RealDomainScenario, ...]:
    coarse_class = get_band_definition(score_band).coarse_class
    exact_band_scenarios = tuple(
        scenario
        for scenario in scenarios
        if score_band in scenario.eligible_score_bands
    )
    if exact_band_scenarios:
        return exact_band_scenarios

    eligible = tuple(
        scenario
        for scenario in scenarios
        if coarse_class in scenario.eligible_coarse_classes
        and not scenario.eligible_score_bands
    )
    if not eligible:
        raise RuntimeError(f"No v2 scenarios are eligible for score_band={score_band!r}")
    return eligible


def _build_assignment(
    score_band: str,
    scenario: RealDomainScenario,
    rng: random.Random,
    variant_index: int,
) -> FeatureAssignment:
    chosen: dict[str, str] = {}
    for axis in FEATURE_AXES:
        allowed = list(axis.allowed_options(score_band))
        if axis.name == "damage_location_primary" and "damage_profile_primary" in chosen:
            allowed = _filter_location_options(chosen["damage_profile_primary"], allowed)
        if axis.name == "camera_angle" and "damage_location_primary" in chosen:
            allowed = _filter_camera_options(chosen["damage_location_primary"], allowed)
        if not allowed:
            raise RuntimeError(f"No allowed options for axis={axis.name!r}, band={score_band!r}")

        preferred = scenario.preferred_values.get(axis.name, ())
        preferred_allowed = [option for option in allowed if option.value in preferred]
        pool = preferred_allowed or allowed
        rng.shuffle(pool)
        chosen[axis.name] = pool[variant_index % len(pool)].value
    return FeatureAssignment(**chosen)


def _filter_location_options(primary_value: str, allowed: list[FeatureOption]) -> list[FeatureOption]:
    primary_axis = get_feature_axis("damage_profile_primary")
    primary_option = next(option for option in primary_axis.options if option.value == primary_value)
    compatible = primary_option.compatible_damage_locations
    if compatible is None:
        return allowed
    return [option for option in allowed if option.value in compatible]


def _filter_camera_options(location_value: str, allowed: list[FeatureOption]) -> list[FeatureOption]:
    return [
        option
        for option in allowed
        if option.compatible_damage_locations is None
        or location_value in option.compatible_damage_locations
    ]


def _build_prompt_row(
    row_index: int,
    score_band: str,
    scenario: RealDomainScenario,
    assignment: FeatureAssignment,
) -> V2PromptRow:
    band = get_band_definition(score_band)
    prompt_uid = f"v2_prompt_{row_index:06d}"
    prompt_title = f"{band.score_band} {scenario.scenario_id.replace('_', ' ')}"
    feature_block = _feature_prompt_block(assignment, coarse_class=band.coarse_class)
    visibility_sentence = _visibility_sentence(band.coarse_class)
    material_sentence = _material_sentence(band.coarse_class)
    prompt_text = (
        f"Create a realistic documentary smartphone photo for package-defect ML training. "
        f"Target severity band: {band.score_band} ({band.coarse_class}, representative score {band.representative_score}). "
        f"Scenario: {scenario.target_failure_mode}. "
        f"Real-domain gap target: {scenario.real_gap_priority}. "
        f"{scenario.scene_directive} "
        f"{scenario.image_quality_directive} "
        f"Feature contract: {feature_block}. "
        f"{visibility_sentence} "
        f"{material_sentence} "
        f"Do not create an illustration, render, collage, text overlay, dramatic product ad, or impossible geometry. "
        f"All labels, barcodes, addresses, logos, and large text must be absent, generic, or unreadable. "
        f"{scenario.label_safety_directive}"
    )
    qc_checks = (
        "realistic_photo; assigned_band_visible; no_artistic_render; no_readable_personal_text; "
        "package_central; defect_matches_assignment; damage_not_overstated; damage_not_hidden"
    )
    return V2PromptRow(
        prompt_uid=prompt_uid,
        custom_id=prompt_uid,
        pipeline_version=V2_PIPELINE_VERSION,
        score_band=band.score_band,
        coarse_class=band.coarse_class,
        representative_score=band.representative_score,
        band_spec_file=band.spec_filename,
        scenario_id=scenario.scenario_id,
        scenario_family=scenario.family,
        real_gap_priority=scenario.real_gap_priority,
        target_failure_mode=scenario.target_failure_mode,
        hard_negative=scenario.hard_negative,
        boundary_focus=scenario.boundary_focus,
        prompt_title=prompt_title,
        prompt_text=prompt_text,
        qc_required_checks=qc_checks,
        **assignment.as_dict(),
    )


def _feature_prompt_block(assignment: FeatureAssignment, *, coarse_class: str) -> str:
    parts: list[str] = []
    for axis_name, value in assignment.as_dict().items():
        axis = get_feature_axis(axis_name)
        option = next(option for option in axis.options if option.value == value)
        prompt_hint = option.prompt_hint
        if coarse_class == "intact":
            prompt_hint = _intact_safe_prompt_hint(axis_name, prompt_hint)
        parts.append(f"{axis_name}={value} ({prompt_hint})")
    return "; ".join(parts)


def _visibility_sentence(coarse_class: str) -> str:
    if coarse_class == "intact":
        return (
            "The package must be the central subject, structurally intact, and "
            "recognizably undamaged at 384px training resolution."
        )
    return (
        "The package must be the central subject and the assigned defect must "
        "be visible at 384px training resolution."
    )


def _material_sentence(coarse_class: str) -> str:
    if coarse_class == "intact":
        return (
            "Use ordinary cardboard material, plausible tape, labels, and "
            "background context, but do not add dents, tears, crushed corners, "
            "holes, split seams, or structural deformation."
        )
    return (
        "Use ordinary cardboard material, plausible folds, dents, tape, labels, "
        "and background context."
    )


def _intact_safe_prompt_hint(axis_name: str, prompt_hint: str) -> str:
    if axis_name == "damage_location_primary":
        return prompt_hint.replace("primary damage", "main visible package area")
    return prompt_hint


def _write_prompt_manifest(path: Path, rows: list[V2PromptRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _write_batch_input(path: Path, rows: list[V2PromptRow], image_spec: ImageGenerationSpec) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_build_batch_request(row, image_spec)) + "\n")


def _build_batch_request(row: V2PromptRow, image_spec: ImageGenerationSpec) -> dict[str, Any]:
    return {
        "custom_id": row.custom_id,
        "method": "POST",
        "url": BATCH_ENDPOINT,
        "body": {
            "model": image_spec.model,
            "prompt": row.prompt_text,
            "quality": image_spec.quality,
            "size": image_spec.size,
        },
    }


def _build_manifest(rows: list[V2PromptRow], config: V2PipelineConfig) -> dict[str, Any]:
    return {
        "pipeline_version": V2_PIPELINE_VERSION,
        "run_name": config.run_name,
        "endpoint": BATCH_ENDPOINT,
        "completion_window": "24h",
        "image_spec": asdict(config.image_spec),
        "request_count": len(rows),
        "requests": [
            {
                "custom_id": row.custom_id,
                "prompt_uid": row.prompt_uid,
                "score_band": row.score_band,
                "coarse_class": row.coarse_class,
                "scenario_id": row.scenario_id,
                "hard_negative": row.hard_negative,
            }
            for row in rows
        ],
    }


def _build_batch_metadata(
    rows: list[V2PromptRow],
    config: V2PipelineConfig,
    run_dir: Path,
) -> dict[str, Any]:
    return {
        "batch_name": config.run_name,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "prepared",
        "endpoint": BATCH_ENDPOINT,
        "completion_window": "24h",
        "request_count": len(rows),
        "input_jsonl_path": str(run_dir / "input.jsonl"),
        "manifest_json_path": str(run_dir / "manifest.json"),
        "output_jsonl_path": str(run_dir / "output.jsonl"),
        "error_jsonl_path": str(run_dir / "errors.jsonl"),
        "generated_output_subdir": config.run_name,
        "model": config.image_spec.model,
        "quality": config.image_spec.quality,
        "size": config.image_spec.size,
        "pipeline_version": V2_PIPELINE_VERSION,
        "prompt_manifest_path": str(run_dir / "prompt_manifest.csv"),
        "input_file_id": None,
        "batch_id": None,
        "output_file_id": None,
        "error_file_id": None,
    }


def _build_run_config(config: V2PipelineConfig) -> dict[str, Any]:
    return {
        "pipeline_version": V2_PIPELINE_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "run_name": config.run_name,
        "count_per_band": config.count_per_band,
        "band_counts": _resolve_band_counts(config),
        "seed": config.seed,
        "output_dir": str(config.output_dir),
        "image_spec": asdict(config.image_spec),
        "submission_policy": "prepare_only_no_api_submission",
        "design_goal": "repair synthetic-to-real domain gap using real-failure-targeted prompts",
    }


def _build_summary(rows: list[V2PromptRow]) -> dict[str, Any]:
    return {
        "pipeline_version": V2_PIPELINE_VERSION,
        "prompt_count": len(rows),
        "score_band_counts": dict(Counter(row.score_band for row in rows)),
        "coarse_class_counts": dict(Counter(row.coarse_class for row in rows)),
        "scenario_counts": dict(Counter(row.scenario_id for row in rows)),
        "scenario_family_counts": dict(Counter(row.scenario_family for row in rows)),
        "real_gap_priority_counts": dict(Counter(row.real_gap_priority for row in rows)),
        "hard_negative_count": sum(1 for row in rows if row.hard_negative),
        "boundary_focus_counts": dict(Counter(row.boundary_focus for row in rows)),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_qc_rubric() -> str:
    return """# V2 Synthetic Image QC Rubric

Reject the image if any of these are true:

- It looks like an illustration, 3D render, product ad, collage, or stylized scene.
- The package is not the central readable subject.
- The assigned damage is missing, hidden, or materially different from the manifest.
- The severity is obviously outside the score band.
- A supposed intact hard negative has structural damage.
- A damaged scenario could plausibly be mistaken for intact because the defect is too small,
  hidden, or visually ambiguous.
- A minor/moderate boundary case is too obvious or too ambiguous to label.
- Labels, barcodes, addresses, brands, or large text dominate the image.
- Damage geometry is physically impossible or cartoonishly exaggerated.

Prefer images that look like ordinary real package photos: imperfect framing, realistic tape, labels,
lighting, clutter, and cardboard material, while keeping the defect readable at training resolution.
"""
