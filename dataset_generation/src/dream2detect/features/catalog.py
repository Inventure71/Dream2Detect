from __future__ import annotations

from .box_features import BOX_FORM_FACTOR_AXIS, BOX_PATTERN_AXIS, LABEL_PRESENCE_AXIS, TAPE_PROFILE_AXIS
from .damage_features import DAMAGE_LOCATION_AXIS, PRIMARY_DAMAGE_AXIS, SECONDARY_DAMAGE_AXIS
from .models import FeatureAxis
from .scene_features import BACKGROUND_CONTEXT_AXIS, CAMERA_ANGLE_AXIS, LIGHTING_STYLE_AXIS


FEATURE_AXES: tuple[FeatureAxis, ...] = (
    PRIMARY_DAMAGE_AXIS,
    SECONDARY_DAMAGE_AXIS,
    DAMAGE_LOCATION_AXIS,
    BOX_FORM_FACTOR_AXIS,
    BOX_PATTERN_AXIS,
    LABEL_PRESENCE_AXIS,
    TAPE_PROFILE_AXIS,
    BACKGROUND_CONTEXT_AXIS,
    CAMERA_ANGLE_AXIS,
    LIGHTING_STYLE_AXIS,
)

ASSIGNMENT_FIELD_NAMES: tuple[str, ...] = tuple(axis.name for axis in FEATURE_AXES)

_AXIS_BY_NAME = {axis.name: axis for axis in FEATURE_AXES}


def get_feature_axis(axis_name: str) -> FeatureAxis:
    try:
        return _AXIS_BY_NAME[axis_name]
    except KeyError as exc:
        supported = ", ".join(sorted(_AXIS_BY_NAME))
        raise ValueError(f"Unknown feature axis '{axis_name}'. Supported axes: {supported}") from exc
