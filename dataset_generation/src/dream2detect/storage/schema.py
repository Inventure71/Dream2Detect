from __future__ import annotations

PROMPTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS prompts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_uid TEXT NOT NULL UNIQUE,
    score_band TEXT NOT NULL,
    coarse_class TEXT NOT NULL,
    representative_score INTEGER NOT NULL,
    band_spec_file TEXT NOT NULL,
    damage_profile_primary TEXT,
    damage_profile_secondary TEXT,
    damage_location_primary TEXT,
    box_form_factor TEXT,
    box_pattern TEXT,
    label_presence TEXT,
    tape_profile TEXT,
    background_context TEXT,
    camera_angle TEXT,
    lighting_style TEXT,
    prompt_model TEXT NOT NULL,
    prompt_title TEXT NOT NULL,
    prompt_text TEXT NOT NULL,
    prompt_rationale TEXT,
    uniqueness_notes TEXT,
    prompt_status TEXT NOT NULL DEFAULT 'drafted',
    review_notes TEXT,
    image_status TEXT NOT NULL DEFAULT 'pending',
    image_ref TEXT,
    image_error TEXT,
    qc_status TEXT NOT NULL DEFAULT 'unreviewed',
    reviewed_score_band TEXT,
    reviewed_coarse_class TEXT,
    final_score_band TEXT,
    final_coarse_class TEXT,
    qc_notes TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

GENERATED_IMAGES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS generated_images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_uid TEXT NOT NULL UNIQUE,
    prompt_id INTEGER NOT NULL,
    image_model TEXT NOT NULL,
    image_quality TEXT NOT NULL,
    image_size TEXT NOT NULL,
    image_ref TEXT NOT NULL UNIQUE,
    revised_prompt TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(prompt_id) REFERENCES prompts(id)
);
"""


def schema_statements() -> list[str]:
    return [PROMPTS_TABLE_SQL, GENERATED_IMAGES_TABLE_SQL]
