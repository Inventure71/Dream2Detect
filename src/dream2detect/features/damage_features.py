from __future__ import annotations

from .models import FeatureAxis, FeatureOption


LOW_MINOR_BANDS = frozenset({"11-20"})
MID_MINOR_BANDS = frozenset({"21-30"})
HIGH_MINOR_BANDS = frozenset({"31-35"})
LOW_MODERATE_BANDS = frozenset({"36-45"})
MID_MODERATE_BANDS = frozenset({"46-55"})
HIGH_MODERATE_BANDS = frozenset({"56-65"})
LOW_SEVERE_BANDS = frozenset({"66-75"})
MID_SEVERE_BANDS = frozenset({"76-85"})
HIGH_SEVERE_BANDS = frozenset({"86-100"})
ALL_DAMAGE_BANDS = frozenset(
    {
        "0-10",
        "11-20",
        "21-30",
        "31-35",
        "36-45",
        "46-55",
        "56-65",
        "66-75",
        "76-85",
        "86-100",
    }
)

CORNER_LOCATIONS = frozenset(
    {
        "front_top_left_corner",
        "front_top_right_corner",
        "front_bottom_left_corner",
        "front_bottom_right_corner",
    }
)
FACE_PANEL_LOCATIONS = frozenset({"front_face_center", "left_side_panel", "right_side_panel"})
EDGE_OR_FLAP_LOCATIONS = frozenset({"top_flap_edge", "left_side_panel", "right_side_panel"})
CORNER_OR_EDGE_LOCATIONS = CORNER_LOCATIONS | EDGE_OR_FLAP_LOCATIONS
ANY_DAMAGE_LOCATION = CORNER_LOCATIONS | FACE_PANEL_LOCATIONS | frozenset({"top_flap_edge"})


PRIMARY_DAMAGE_AXIS = FeatureAxis(
    name="damage_profile_primary",
    category="defect_content",
    description="Main visible defect pattern that defines the package condition.",
    options=(
        FeatureOption(
            "damage_profile_primary",
            "visually_intact",
            "no meaningful defect, only negligible handling traces",
            frozenset({"0-10"}),
            ANY_DAMAGE_LOCATION,
        ),
        FeatureOption(
            "damage_profile_primary",
            "light_corner_softening",
            "one lightly softened corner with no opening",
            LOW_MINOR_BANDS,
            CORNER_LOCATIONS,
        ),
        FeatureOption(
            "damage_profile_primary",
            "small_face_dent",
            "one small dent on a single face with normal overall geometry",
            LOW_MINOR_BANDS,
            FACE_PANEL_LOCATIONS,
        ),
        FeatureOption(
            "damage_profile_primary",
            "light_edge_crease",
            "one visible but limited edge crease without tearing",
            LOW_MINOR_BANDS,
            CORNER_OR_EDGE_LOCATIONS,
        ),
        FeatureOption(
            "damage_profile_primary",
            "light_surface_scuffing",
            "noticeable scuffs and handling wear without structural compromise",
            LOW_MINOR_BANDS,
            FACE_PANEL_LOCATIONS,
        ),
        FeatureOption(
            "damage_profile_primary",
            "clear_corner_crease",
            "one clearly creased corner with more obvious handling damage",
            MID_MINOR_BANDS,
            CORNER_LOCATIONS,
        ),
        FeatureOption(
            "damage_profile_primary",
            "small_corner_fray",
            "one frayed corner with limited cardboard fiber wear but no cavity",
            MID_MINOR_BANDS,
            CORNER_LOCATIONS,
        ),
        FeatureOption(
            "damage_profile_primary",
            "face_dent_with_crease",
            "one face dent combined with a short crease, still clearly minor",
            MID_MINOR_BANDS,
            FACE_PANEL_LOCATIONS,
        ),
        FeatureOption(
            "damage_profile_primary",
            "edge_wear_cluster",
            "several small edge wear marks concentrated on one side",
            MID_MINOR_BANDS,
            CORNER_OR_EDGE_LOCATIONS,
        ),
        FeatureOption(
            "damage_profile_primary",
            "corner_crumple_without_opening",
            "one visibly crumpled corner but no opening into the box interior",
            HIGH_MINOR_BANDS,
            CORNER_LOCATIONS,
        ),
        FeatureOption(
            "damage_profile_primary",
            "short_partial_tear",
            "one short partial tear that does not create a visible cavity",
            HIGH_MINOR_BANDS,
            CORNER_LOCATIONS | FACE_PANEL_LOCATIONS | frozenset({"top_flap_edge"}),
        ),
        FeatureOption(
            "damage_profile_primary",
            "pronounced_face_dent",
            "one pronounced dent that clearly deforms a face but stops short of collapse",
            HIGH_MINOR_BANDS,
            FACE_PANEL_LOCATIONS,
        ),
        FeatureOption(
            "damage_profile_primary",
            "bent_flap_or_edge",
            "one bent flap or edge with visible deformation but no broad failure",
            HIGH_MINOR_BANDS,
            CORNER_OR_EDGE_LOCATIONS,
        ),
        FeatureOption(
            "damage_profile_primary",
            "frayed_corner_plus_face_dent",
            "one damaged corner with frayed cardboard plus one noticeable face dent, still no large opening",
            LOW_MODERATE_BANDS,
            CORNER_LOCATIONS,
        ),
        FeatureOption(
            "damage_profile_primary",
            "edge_split_without_cavity",
            "one edge split or seam failure without a wide opening or exposed interior cavity",
            LOW_MODERATE_BANDS,
            CORNER_OR_EDGE_LOCATIONS,
        ),
        FeatureOption(
            "damage_profile_primary",
            "single_small_puncture_with_denting",
            "one small puncture accompanied by local denting, moderate but not severe",
            LOW_MODERATE_BANDS,
            FACE_PANEL_LOCATIONS | CORNER_LOCATIONS,
        ),
        FeatureOption(
            "damage_profile_primary",
            "partial_side_compression",
            "one side panel compressed enough to look clearly damaged but not collapsed",
            MID_MODERATE_BANDS,
            frozenset({"left_side_panel", "right_side_panel"}),
        ),
        FeatureOption(
            "damage_profile_primary",
            "clear_corner_split",
            "one clearly split corner with limited opening and surrounding deformation",
            MID_MODERATE_BANDS,
            CORNER_LOCATIONS,
        ),
        FeatureOption(
            "damage_profile_primary",
            "face_dent_with_visible_tear",
            "one strong face dent plus one visible tear, still below severe failure",
            MID_MODERATE_BANDS,
            FACE_PANEL_LOCATIONS,
        ),
        FeatureOption(
            "damage_profile_primary",
            "multiple_damaged_edges",
            "multiple damaged edges concentrated on one package side",
            MID_MODERATE_BANDS,
            CORNER_OR_EDGE_LOCATIONS,
        ),
        FeatureOption(
            "damage_profile_primary",
            "narrow_opening_from_edge_split",
            "one narrow opening created by an edge split, with a small visible opening that does not dominate the package condition",
            HIGH_MODERATE_BANDS,
            CORNER_OR_EDGE_LOCATIONS,
        ),
        FeatureOption(
            "damage_profile_primary",
            "controlled_side_crush",
            "one strongly compressed side panel with localized geometry loss, but without broad breakage or obvious severe collapse",
            HIGH_MODERATE_BANDS,
            frozenset({"left_side_panel", "right_side_panel"}),
        ),
        FeatureOption(
            "damage_profile_primary",
            "multiple_damaged_corners",
            "multiple damaged corners without full collapse",
            HIGH_MODERATE_BANDS,
            CORNER_LOCATIONS,
        ),
        FeatureOption(
            "damage_profile_primary",
            "small_cavity_with_deformation",
            "one small visible cavity with surrounding deformation and material failure",
            HIGH_MODERATE_BANDS,
            CORNER_LOCATIONS | frozenset({"top_flap_edge"}),
        ),
        FeatureOption(
            "damage_profile_primary",
            "large_corner_opening",
            "one large torn-open corner with clearly exposed interior",
            LOW_SEVERE_BANDS,
            CORNER_LOCATIONS,
        ),
        FeatureOption(
            "damage_profile_primary",
            "strong_side_crushing",
            "one side heavily crushed with major geometric failure",
            LOW_SEVERE_BANDS,
            frozenset({"left_side_panel", "right_side_panel", "front_face_center"}),
        ),
        FeatureOption(
            "damage_profile_primary",
            "collapsed_corner_cluster",
            "several corners badly compromised with collapse concentrated in one area",
            LOW_SEVERE_BANDS,
            CORNER_LOCATIONS,
        ),
        FeatureOption(
            "damage_profile_primary",
            "broad_edge_tear",
            "one broad tear running along an edge or seam with strong structural failure",
            LOW_SEVERE_BANDS,
            CORNER_OR_EDGE_LOCATIONS,
        ),
        FeatureOption(
            "damage_profile_primary",
            "major_side_collapse",
            "major collapse affecting one large side or edge region",
            MID_SEVERE_BANDS,
            frozenset({"left_side_panel", "right_side_panel", "front_face_center"}),
        ),
        FeatureOption(
            "damage_profile_primary",
            "large_torn_opening",
            "one large torn opening exposing a substantial interior cavity",
            MID_SEVERE_BANDS,
            CORNER_LOCATIONS | FACE_PANEL_LOCATIONS,
        ),
        FeatureOption(
            "damage_profile_primary",
            "collapsed_corner_cluster_with_open_seam",
            "multiple collapsed corners with one visible split seam or opening, clearly beyond severe-low",
            MID_SEVERE_BANDS,
            CORNER_LOCATIONS,
        ),
        FeatureOption(
            "damage_profile_primary",
            "heavy_crushing_with_split_seams",
            "heavy crushing with split seams and broad structural failure",
            MID_SEVERE_BANDS,
            FACE_PANEL_LOCATIONS | EDGE_OR_FLAP_LOCATIONS,
        ),
        FeatureOption(
            "damage_profile_primary",
            "extreme_collapse",
            "extreme collapse with major shape destruction",
            HIGH_SEVERE_BANDS,
            ANY_DAMAGE_LOCATION,
        ),
        FeatureOption(
            "damage_profile_primary",
            "partially_flattened_box",
            "box partially flattened with severe multi-panel deformation",
            HIGH_SEVERE_BANDS,
            FACE_PANEL_LOCATIONS,
        ),
        FeatureOption(
            "damage_profile_primary",
            "very_large_open_tear",
            "very large open tear with dominant exposed interior and severe material failure",
            HIGH_SEVERE_BANDS,
            CORNER_LOCATIONS | FACE_PANEL_LOCATIONS,
        ),
        FeatureOption(
            "damage_profile_primary",
            "severe_multi_side_failure",
            "severe failure across multiple sides, edges, or corners",
            HIGH_SEVERE_BANDS,
            ANY_DAMAGE_LOCATION,
        ),
    ),
)


SECONDARY_DAMAGE_AXIS = FeatureAxis(
    name="damage_profile_secondary",
    category="defect_content",
    description="Secondary visible defect or supporting defect cue that should not override the band.",
    options=(
        FeatureOption("damage_profile_secondary", "none", "no secondary defect; keep the package condition simple", ALL_DAMAGE_BANDS),
        FeatureOption(
            "damage_profile_secondary",
            "tiny_handling_scuffs",
            "tiny handling scuffs or light cosmetic wear only",
            frozenset({"0-10", "11-20", "21-30"}),
        ),
        FeatureOption(
            "damage_profile_secondary",
            "light_crease_cluster",
            "a few limited creases supporting the main defect",
            frozenset({"11-20", "21-30", "31-35", "36-45"}),
        ),
        FeatureOption(
            "damage_profile_secondary",
            "small_stain_patch",
            "one small stain or dirt patch that does not dominate the condition",
            frozenset({"11-20", "21-30", "31-35", "36-45", "46-55"}),
        ),
        FeatureOption(
            "damage_profile_secondary",
            "secondary_corner_softening",
            "a second corner with mild softening or wear",
            frozenset({"21-30", "31-35", "36-45", "46-55"}),
        ),
        FeatureOption(
            "damage_profile_secondary",
            "secondary_face_dent",
            "one secondary smaller face dent supporting the main defect",
            frozenset({"31-35", "36-45", "46-55", "56-65"}),
        ),
        FeatureOption(
            "damage_profile_secondary",
            "minor_tape_peel",
            "minor peeling or lifted tape near the main damage area",
            frozenset({"21-30", "31-35", "36-45", "46-55", "56-65"}),
        ),
        FeatureOption(
            "damage_profile_secondary",
            "extra_small_puncture",
            "one additional small puncture or nick that stays below the main severity signal",
            frozenset({"46-55", "56-65", "66-75"}),
        ),
        FeatureOption(
            "damage_profile_secondary",
            "additional_crushed_edge",
            "an additional visibly crushed edge supporting a stronger damaged impression",
            frozenset({"56-65", "66-75", "76-85"}),
        ),
        FeatureOption(
            "damage_profile_secondary",
            "broad_surface_grime",
            "broad surface dirt or grime that contributes to condition but does not replace structural damage",
            frozenset({"56-65", "66-75", "76-85", "86-100"}),
        ),
    ),
)


DAMAGE_LOCATION_AXIS = FeatureAxis(
    name="damage_location_primary",
    category="defect_layout",
    description="Main visible location of the primary defect.",
    options=(
        FeatureOption("damage_location_primary", "front_top_left_corner", "primary damage on the front top-left corner"),
        FeatureOption("damage_location_primary", "front_top_right_corner", "primary damage on the front top-right corner"),
        FeatureOption("damage_location_primary", "front_bottom_left_corner", "primary damage on the front bottom-left corner"),
        FeatureOption("damage_location_primary", "front_bottom_right_corner", "primary damage on the front bottom-right corner"),
        FeatureOption("damage_location_primary", "front_face_center", "primary damage centered on the main visible front face"),
        FeatureOption("damage_location_primary", "left_side_panel", "primary damage on the left side panel"),
        FeatureOption("damage_location_primary", "right_side_panel", "primary damage on the right side panel"),
        FeatureOption("damage_location_primary", "top_flap_edge", "primary damage concentrated along the top flap edge"),
    ),
)
