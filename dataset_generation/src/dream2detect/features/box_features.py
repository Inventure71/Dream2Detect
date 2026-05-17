from __future__ import annotations

from .models import FeatureAxis, FeatureOption


BOX_FORM_FACTOR_AXIS = FeatureAxis(
    name="box_form_factor",
    category="package_appearance",
    description="Overall cardboard box shape or format.",
    options=(
        FeatureOption("box_form_factor", "small_cube", "a small cube-shaped shipping box"),
        FeatureOption("box_form_factor", "medium_standard", "a medium standard shipping box"),
        FeatureOption("box_form_factor", "long_rectangular", "a long rectangular shipping box"),
        FeatureOption("box_form_factor", "tall_rectangular", "a tall rectangular cardboard box"),
        FeatureOption("box_form_factor", "flat_mailer_box", "a flatter mailer-style cardboard box"),
    ),
)


BOX_PATTERN_AXIS = FeatureAxis(
    name="box_pattern",
    category="package_appearance",
    description="Visible surface patterning or styling on the cardboard box.",
    options=(
        FeatureOption("box_pattern", "plain_brown", "plain brown cardboard with minimal printed markings"),
        FeatureOption(
            "box_pattern",
            "subtle_printed_markings",
            "subtle printed handling symbols or very small markings",
        ),
        FeatureOption(
            "box_pattern",
            "shipping_label_patch",
            "a visible but limited shipping-label area printed or adhered to one face",
            text_risk_weight=1,
        ),
        FeatureOption("box_pattern", "tape_heavy_plain", "mostly plain cardboard with more visible tape lines"),
        FeatureOption("box_pattern", "recycled_patchwork_cardboard", "recycled cardboard panels with slightly varied cardboard tones"),
    ),
)


LABEL_PRESENCE_AXIS = FeatureAxis(
    name="label_presence",
    category="package_appearance",
    description="How much visible shipping-label or sticker material is present.",
    options=(
        FeatureOption("label_presence", "no_visible_label", "no visible shipping label"),
        FeatureOption(
            "label_presence",
            "small_shipping_label",
            "one small shipping label that does not dominate the image",
            text_risk_weight=1,
        ),
        FeatureOption(
            "label_presence",
            "barcode_label_only",
            "one simple barcode-style label that stays small and secondary",
            text_risk_weight=1,
        ),
        FeatureOption(
            "label_presence",
            "multiple_small_stickers",
            "a few small stickers or routing labels that remain clearly secondary",
            text_risk_weight=2,
        ),
    ),
)


TAPE_PROFILE_AXIS = FeatureAxis(
    name="tape_profile",
    category="package_appearance",
    description="How tape appears on the package surface.",
    options=(
        FeatureOption("tape_profile", "minimal_tape", "minimal visible tape"),
        FeatureOption("tape_profile", "single_center_tape", "one central tape seam"),
        FeatureOption("tape_profile", "cross_tape", "crossed tape strips across the closing seam"),
        FeatureOption("tape_profile", "reinforced_edge_tape", "extra tape reinforcement along an edge or corner"),
        FeatureOption("tape_profile", "old_peeling_tape", "older tape with slight peeling or wear"),
    ),
)
