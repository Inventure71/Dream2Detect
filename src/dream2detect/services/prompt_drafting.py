from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent
from uuid import uuid4

from pydantic import BaseModel, Field

from ..bands import get_band_definition, load_band_spec_text
from ..config import load_settings
from ..features.catalog import get_feature_axis
from ..features.models import FeatureAssignment
from ..features.sampler import sample_feature_assignments
from ..openai_connector import build_openai_client
from ..storage.repository import Dream2DetectRepository


@dataclass(frozen=True)
class DraftedPrompt:
    assignment_key: str
    assignment: FeatureAssignment
    prompt_uid: str
    prompt_title: str
    prompt_text: str
    prompt_rationale: str
    uniqueness_notes: str


class PromptCandidate(BaseModel):
    assignment_key: str = Field(min_length=1)
    prompt_title: str = Field(min_length=1)
    prompt_text: str = Field(min_length=1)
    prompt_rationale: str = Field(min_length=1)
    uniqueness_notes: str = Field(min_length=1)


class PromptDraftResponse(BaseModel):
    prompts: list[PromptCandidate]


def _load_prompt_template() -> str:
    settings = load_settings()
    template_path = settings.generated_images_dir.parent / "data" / "templates" / "prompt_writing_template.md"
    return template_path.read_text(encoding="utf-8")


def _format_assignment_block(index: int, assignment: FeatureAssignment) -> str:
    values = assignment.as_dict()
    lines = [f"- assignment_key: slot_{index}"]
    for key, value in values.items():
        axis = get_feature_axis(key)
        option = next(option for option in axis.options if option.value == value)
        lines.append(f"- {key}: {value}")
        lines.append(f"  meaning: {option.prompt_hint}")
    lines.append("- hard_rule: keep the primary defect and location semantically exact")
    lines.append("- variation_rule: vary only non-label details such as cardboard shade, tiny wear, camera distance, or shadow shape")
    return "\n".join(lines)


def _build_prompt_request(score_band: str, assignments: list[FeatureAssignment]) -> str:
    band = get_band_definition(score_band)
    band_spec = load_band_spec_text(score_band)
    template = _load_prompt_template()
    assignment_blocks = "\n\n".join(
        _format_assignment_block(index=index, assignment=assignment)
        for index, assignment in enumerate(assignments, start=1)
    )
    return dedent(
        f"""
        {template}

        Produce exactly {len(assignments)} prompt candidates for the requested band.

        Return data that matches the required structured schema exactly.

        Band metadata:
        - score_band: {band.score_band}
        - coarse_class: {band.coarse_class}
        - representative_score: {band.representative_score}
        - band_spec_file: {band.spec_filename}

        Band specification:
        {band_spec}

        Structured feature assignments:
        {assignment_blocks}

        Fidelity rules:
        - the final prompt must preserve each assignment semantically, not approximately
        - do not move the main defect from the assigned location to a different region
        - do not replace the assigned primary defect with a nearby but different defect type
        - keep incidental creativity in non-label details only
        - if you need variety, vary texture, framing distance, secondary scene detail, or wording while keeping the assignment intact

        Uniqueness requirements:
        - prompts must remain inside the requested band
        - prompts must faithfully use the provided structured feature assignments
        - prompts must vary defect layout, scene composition, or context in useful ways
        - prompts must not collapse into near-duplicates
        - prompts must keep the package clearly visible and central
        - prompts must avoid text overlays, watermarks, logos dominating the image, or irrelevant clutter
        - any allowed label, barcode, sticker, or printed marking must remain physically small and visually secondary
        - packing-station or work-area scenes must not introduce large readable signs or text-heavy props
        - prompts must be suitable for package-defect ML training, not for artistic showcase output
        """
    ).strip()


def draft_prompts_for_band(
    score_band: str,
    count: int,
    repository: Dream2DetectRepository,
    *,
    seed: int | None = None,
) -> list[DraftedPrompt]:
    settings = load_settings()
    client = build_openai_client()
    assignments = sample_feature_assignments(
        score_band=score_band,
        count=count,
        repository=repository,
        seed=seed,
    )
    request_text = _build_prompt_request(score_band, assignments)
    response = client.responses.parse(
        model=settings.openai.prompt_model,
        input=request_text,
        text_format=PromptDraftResponse,
    )
    parsed = response.output_parsed
    if parsed is None:
        raise RuntimeError("Prompt drafting returned no structured output.")
    prompts = parsed.prompts
    if len(prompts) != count:
        raise RuntimeError(f"Expected {count} prompts, got {len(prompts)}")
    assignment_by_key = {
        f"slot_{index}": assignment for index, assignment in enumerate(assignments, start=1)
    }
    drafted_prompts: list[DraftedPrompt] = []
    seen_assignment_keys: set[str] = set()
    for item in prompts:
        assignment_key = item.assignment_key.strip()
        assignment = assignment_by_key.get(assignment_key)
        if assignment is None:
            supported = ", ".join(sorted(assignment_by_key))
            raise RuntimeError(
                f"Prompt drafting returned unknown assignment_key={assignment_key!r}. Supported keys: {supported}"
            )
        if assignment_key in seen_assignment_keys:
            raise RuntimeError(f"Prompt drafting returned duplicate assignment_key={assignment_key!r}")
        seen_assignment_keys.add(assignment_key)
        drafted_prompts.append(
            DraftedPrompt(
                assignment_key=assignment_key,
                assignment=assignment,
                prompt_uid=f"prompt_{uuid4().hex[:16]}",
                prompt_title=item.prompt_title.strip(),
                prompt_text=item.prompt_text.strip(),
                prompt_rationale=item.prompt_rationale.strip(),
                uniqueness_notes=item.uniqueness_notes.strip(),
            )
        )
    if len(seen_assignment_keys) != count:
        missing = sorted(set(assignment_by_key) - seen_assignment_keys)
        raise RuntimeError(f"Prompt drafting missed assignment keys: {', '.join(missing)}")
    return drafted_prompts


def save_drafted_prompts(score_band: str, drafted_prompts: list[DraftedPrompt], repository: Dream2DetectRepository) -> None:
    settings = load_settings()
    band = get_band_definition(score_band)
    for prompt in drafted_prompts:
        repository.insert_prompt(
            prompt_uid=prompt.prompt_uid,
            score_band=band.score_band,
            coarse_class=band.coarse_class,
            representative_score=band.representative_score,
            band_spec_file=band.spec_filename,
            damage_profile_primary=prompt.assignment.damage_profile_primary,
            damage_profile_secondary=prompt.assignment.damage_profile_secondary,
            damage_location_primary=prompt.assignment.damage_location_primary,
            box_form_factor=prompt.assignment.box_form_factor,
            box_pattern=prompt.assignment.box_pattern,
            label_presence=prompt.assignment.label_presence,
            tape_profile=prompt.assignment.tape_profile,
            background_context=prompt.assignment.background_context,
            camera_angle=prompt.assignment.camera_angle,
            lighting_style=prompt.assignment.lighting_style,
            prompt_model=settings.openai.prompt_model,
            prompt_title=prompt.prompt_title,
            prompt_text=prompt.prompt_text,
            prompt_rationale=prompt.prompt_rationale,
            uniqueness_notes=prompt.uniqueness_notes,
            prompt_status="drafted",
        )
