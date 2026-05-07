from __future__ import annotations

from .models import FeatureAxis, FeatureOption


BACKGROUND_CONTEXT_AXIS = FeatureAxis(
    name="background_context",
    category="scene_context",
    description="Background setting around the package.",
    options=(
        FeatureOption("background_context", "warehouse_floor", "a clean warehouse floor background"),
        FeatureOption("background_context", "delivery_sorting_table", "a delivery sorting or packing table background"),
        FeatureOption("background_context", "plain_studio_floor", "a plain studio-like floor background"),
        FeatureOption("background_context", "outdoor_doorstep", "an outdoor doorstep or porch delivery background"),
        FeatureOption("background_context", "van_cargo_area", "a delivery van cargo-area background"),
        FeatureOption(
            "background_context",
            "packing_station",
            "a packing-station background with work surfaces but no dominant readable signage",
            text_risk_weight=1,
        ),
    ),
)


CAMERA_ANGLE_AXIS = FeatureAxis(
    name="camera_angle",
    category="image_capture",
    description="Primary camera viewpoint used for the package photo.",
    options=(
        FeatureOption("camera_angle", "front_eye_level", "front eye-level view"),
        FeatureOption("camera_angle", "three_quarter_left", "slight three-quarter view from the left"),
        FeatureOption("camera_angle", "three_quarter_right", "slight three-quarter view from the right"),
        FeatureOption("camera_angle", "slightly_top_down", "slightly top-down angle"),
        FeatureOption("camera_angle", "low_angle_front", "low frontal angle that still keeps defects readable"),
    ),
)


LIGHTING_STYLE_AXIS = FeatureAxis(
    name="lighting_style",
    category="image_capture",
    description="Lighting conditions affecting the package photo.",
    options=(
        FeatureOption("lighting_style", "soft_even", "soft even lighting"),
        FeatureOption("lighting_style", "cool_fluorescent", "cool fluorescent indoor lighting"),
        FeatureOption("lighting_style", "mixed_indoor", "mixed indoor lighting with moderate variation"),
        FeatureOption(
            "lighting_style",
            "dim_ambient",
            "dimmer ambient lighting while defects remain readable",
            stress_weight=1,
        ),
        FeatureOption(
            "lighting_style",
            "harsh_side_shadow",
            "one harsher side-light that creates visible shadows but keeps the package legible",
            stress_weight=1,
        ),
        FeatureOption(
            "lighting_style",
            "backlit_but_readable",
            "mild backlighting while the package defects remain clearly readable",
            stress_weight=1,
        ),
    ),
)
