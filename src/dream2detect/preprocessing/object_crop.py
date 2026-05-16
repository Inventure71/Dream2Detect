from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


DEFAULT_SAM3_PROMPTS = (
    "cardboard box",
    "shipping box",
    "damaged cardboard package",
)


@dataclass(frozen=True)
class BoundingBox:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return max(0, self.x1 - self.x0)

    @property
    def height(self) -> int:
        return max(0, self.y1 - self.y0)

    @property
    def area(self) -> int:
        return self.width * self.height

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass(frozen=True)
class CropResult:
    raw_bbox: BoundingBox
    bbox: BoundingBox
    status: str
    method: str
    prompt: str
    confidence: float
    mask_area_ratio: float | None = None
    notes: str = ""


class ObjectDetector(Protocol):
    def detect(self, image: Image.Image) -> CropResult | None:
        ...


def clamp_box(box: BoundingBox, *, image_width: int, image_height: int) -> BoundingBox:
    x0 = max(0, min(image_width - 1, box.x0))
    y0 = max(0, min(image_height - 1, box.y0))
    x1 = max(x0 + 1, min(image_width, box.x1))
    y1 = max(y0 + 1, min(image_height, box.y1))
    return BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)


def expand_box(
    box: BoundingBox,
    *,
    image_width: int,
    image_height: int,
    padding_fraction: float,
) -> BoundingBox:
    if padding_fraction < 0:
        raise ValueError(f"padding_fraction must be non-negative, got {padding_fraction}")
    pad_x = int(round(box.width * padding_fraction))
    pad_y = int(round(box.height * padding_fraction))
    return clamp_box(
        BoundingBox(
            x0=box.x0 - pad_x,
            y0=box.y0 - pad_y,
            x1=box.x1 + pad_x,
            y1=box.y1 + pad_y,
        ),
        image_width=image_width,
        image_height=image_height,
    )


def build_square_box(
    box: BoundingBox,
    *,
    image_width: int,
    image_height: int,
    padding_fraction: float,
) -> BoundingBox:
    if padding_fraction < 0:
        raise ValueError(f"padding_fraction must be non-negative, got {padding_fraction}")
    side = int(round(max(box.width, box.height) * (1.0 + 2.0 * padding_fraction)))
    side = max(1, side)

    center_x = (box.x0 + box.x1) / 2.0
    center_y = (box.y0 + box.y1) / 2.0
    x0 = int(round(center_x - side / 2.0))
    y0 = int(round(center_y - side / 2.0))
    if side <= image_width:
        x0 = max(0, min(image_width - side, x0))
    if side <= image_height:
        y0 = max(0, min(image_height - side, y0))
    return BoundingBox(x0=x0, y0=y0, x1=x0 + side, y1=y0 + side)


def build_center_prior_box(
    *,
    image_width: int,
    image_height: int,
    crop_fraction: float,
) -> BoundingBox:
    if not 0.0 < crop_fraction <= 1.0:
        raise ValueError(f"crop_fraction must be in (0.0, 1.0], got {crop_fraction}")
    crop_width = max(1, int(round(image_width * crop_fraction)))
    crop_height = max(1, int(round(image_height * crop_fraction)))
    x0 = (image_width - crop_width) // 2
    y0 = (image_height - crop_height) // 2
    return BoundingBox(x0=x0, y0=y0, x1=x0 + crop_width, y1=y0 + crop_height)


def crop_pad_resize(
    image: Image.Image,
    box: BoundingBox,
    *,
    image_size: int,
) -> Image.Image:
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")
    cropped = image.crop(box.as_tuple())
    cropped.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
    output = Image.new("RGB", (image_size, image_size), (245, 245, 245))
    offset = ((image_size - cropped.width) // 2, (image_size - cropped.height) // 2)
    output.paste(cropped, offset)
    return output


def crop_stretch_resize(
    image: Image.Image,
    box: BoundingBox,
    *,
    image_size: int,
) -> Image.Image:
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")
    cropped = image.crop(box.as_tuple())
    return cropped.resize((image_size, image_size), Image.Resampling.LANCZOS)


def crop_square_with_white_padding_resize(
    image: Image.Image,
    box: BoundingBox,
    *,
    image_size: int,
) -> Image.Image:
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")
    if box.width != box.height:
        raise ValueError(f"square crop requires a square box, got {box}")

    output = Image.new("RGB", (box.width, box.height), (255, 255, 255))
    image_box = BoundingBox(
        x0=max(0, box.x0),
        y0=max(0, box.y0),
        x1=min(image.width, box.x1),
        y1=min(image.height, box.y1),
    )
    if image_box.x1 > image_box.x0 and image_box.y1 > image_box.y0:
        region = image.crop(image_box.as_tuple())
        output.paste(region, (image_box.x0 - box.x0, image_box.y0 - box.y0))
    return output.resize((image_size, image_size), Image.Resampling.LANCZOS)


def crop_resize(
    image: Image.Image,
    box: BoundingBox,
    *,
    image_size: int,
    resize_mode: str,
) -> Image.Image:
    if resize_mode == "pad":
        return crop_pad_resize(image, box, image_size=image_size)
    if resize_mode == "stretch":
        return crop_stretch_resize(image, box, image_size=image_size)
    if resize_mode == "square_crop":
        return crop_square_with_white_padding_resize(image, box, image_size=image_size)
    raise ValueError(f"Unsupported resize_mode: {resize_mode}")


def _tensor_to_list(value: object) -> list[object]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)  # type: ignore[arg-type]


def _box_from_sam_value(value: object) -> BoundingBox:
    values = _tensor_to_list(value)
    if len(values) != 4:
        raise ValueError(f"Expected SAM box with four values, got {values!r}")
    x0, y0, x1, y1 = [int(round(float(item))) for item in values]
    return BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)


def _mask_area_ratio(mask: object, *, image_width: int, image_height: int) -> float | None:
    if mask is None:
        return None
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu()
    if hasattr(mask, "numpy"):
        array = mask.numpy()
        return float(array.astype(bool).sum()) / float(image_width * image_height)
    return None


def score_candidate(
    box: BoundingBox,
    *,
    image_width: int,
    image_height: int,
    model_score: float,
    min_area_ratio: float,
    max_area_ratio: float,
) -> float | None:
    area_ratio = float(box.area) / float(image_width * image_height)
    if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
        return None

    image_cx = image_width / 2.0
    image_cy = image_height / 2.0
    box_cx = (box.x0 + box.x1) / 2.0
    box_cy = (box.y0 + box.y1) / 2.0
    center_distance = (
        ((box_cx - image_cx) / image_width) ** 2
        + ((box_cy - image_cy) / image_height) ** 2
    ) ** 0.5
    center_score = max(0.0, 1.0 - 2.0 * center_distance)
    target_area = 0.55
    area_score = max(0.0, 1.0 - abs(area_ratio - target_area) / target_area)
    return 0.60 * model_score + 0.25 * center_score + 0.15 * area_score


class CenterPriorDetector:
    def __init__(self, *, crop_fraction: float = 0.9) -> None:
        self.crop_fraction = crop_fraction

    def detect(self, image: Image.Image) -> CropResult:
        box = build_center_prior_box(
            image_width=image.width,
            image_height=image.height,
            crop_fraction=self.crop_fraction,
        )
        return CropResult(
            raw_bbox=box,
            bbox=box,
            status="fallback_center",
            method="center_prior",
            prompt="",
            confidence=0.0,
            mask_area_ratio=None,
            notes="SAM crop unavailable or disabled; used deterministic center crop.",
        )


class Sam3TextDetector:
    def __init__(
        self,
        *,
        prompts: tuple[str, ...] = DEFAULT_SAM3_PROMPTS,
        device: str = "auto",
        min_area_ratio: float = 0.05,
        max_area_ratio: float = 0.98,
    ) -> None:
        if not prompts:
            raise ValueError("At least one SAM 3 prompt is required.")
        self.prompts = prompts
        self.device = device
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self._processor = None

    def _load_processor(self) -> object:
        if self._processor is not None:
            return self._processor
        try:
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "SAM 3 is not installed. Install facebookresearch/sam3 in a compatible "
                "environment and authenticate to Hugging Face for the SAM 3 checkpoints."
            ) from exc
        model_kwargs = {}
        if self.device != "auto" and "device" in inspect.signature(build_sam3_image_model).parameters:
            model_kwargs["device"] = self.device
        model = build_sam3_image_model(**model_kwargs)

        processor_kwargs = {}
        if self.device != "auto" and "device" in inspect.signature(Sam3Processor).parameters:
            processor_kwargs["device"] = self.device
        self._processor = Sam3Processor(model, **processor_kwargs)
        return self._processor

    def detect(self, image: Image.Image) -> CropResult | None:
        processor = self._load_processor()
        state = processor.set_image(image)  # type: ignore[attr-defined]
        best: tuple[float, CropResult] | None = None

        for prompt in self.prompts:
            output = processor.set_text_prompt(state=state, prompt=prompt)  # type: ignore[attr-defined]
            boxes = output.get("boxes", [])
            scores = output.get("scores", [])
            masks = output.get("masks", [])

            for index, raw_box in enumerate(boxes):
                box = clamp_box(
                    _box_from_sam_value(raw_box),
                    image_width=image.width,
                    image_height=image.height,
                )
                raw_score = scores[index] if index < len(scores) else 0.0
                model_score = float(raw_score.item() if hasattr(raw_score, "item") else raw_score)
                candidate_score = score_candidate(
                    box,
                    image_width=image.width,
                    image_height=image.height,
                    model_score=model_score,
                    min_area_ratio=self.min_area_ratio,
                    max_area_ratio=self.max_area_ratio,
                )
                if candidate_score is None:
                    continue
                mask = masks[index] if index < len(masks) else None
                result = CropResult(
                    raw_bbox=box,
                    bbox=box,
                    status="detected",
                    method="sam3_text",
                    prompt=prompt,
                    confidence=model_score,
                    mask_area_ratio=_mask_area_ratio(
                        mask,
                        image_width=image.width,
                        image_height=image.height,
                    ),
                    notes="",
                )
                if best is None or candidate_score > best[0]:
                    best = (candidate_score, result)

        return best[1] if best is not None else None


def resolve_detector(
    *,
    crop_backend: str,
    sam3_prompts: tuple[str, ...],
    sam3_device: str,
    center_crop_fraction: float,
    sam3_min_area_ratio: float,
    sam3_max_area_ratio: float,
) -> tuple[ObjectDetector | None, CenterPriorDetector]:
    center = CenterPriorDetector(crop_fraction=center_crop_fraction)
    if crop_backend == "center":
        return None, center
    if crop_backend in {"sam3", "hybrid"}:
        return (
            Sam3TextDetector(
                prompts=sam3_prompts,
                device=sam3_device,
                min_area_ratio=sam3_min_area_ratio,
                max_area_ratio=sam3_max_area_ratio,
            ),
            center,
        )
    raise ValueError(f"Unsupported crop_backend: {crop_backend}")


def choose_crop(
    image: Image.Image,
    *,
    crop_backend: str,
    sam_detector: ObjectDetector | None,
    center_detector: CenterPriorDetector,
    crop_padding_fraction: float,
) -> CropResult:
    if crop_backend == "center":
        result = center_detector.detect(image)
    else:
        try:
            result = sam_detector.detect(image) if sam_detector is not None else None
        except RuntimeError:
            if crop_backend == "sam3":
                raise
            result = None
        if result is None and crop_backend == "sam3":
            raise RuntimeError(
                "SAM 3 did not return a valid package/box crop for this image. "
                "Use --crop-backend hybrid only when center-prior fallback is acceptable."
            )
        if result is None:
            result = center_detector.detect(image)

    padded_box = expand_box(
        result.bbox,
        image_width=image.width,
        image_height=image.height,
        padding_fraction=crop_padding_fraction if result.method != "center_prior" else 0.0,
    )
    return CropResult(
        raw_bbox=result.raw_bbox,
        bbox=padded_box,
        status=result.status,
        method=result.method,
        prompt=result.prompt,
        confidence=result.confidence,
        mask_area_ratio=result.mask_area_ratio,
        notes=result.notes,
    )


def build_object_focused_manifest(
    source_manifest_path: Path,
    output_dir: Path,
    output_manifest_path: Path,
    *,
    image_size: int,
    source_image_column: str,
    crop_backend: str,
    sam3_prompts: tuple[str, ...],
    sam3_device: str = "auto",
    resize_mode: str = "square_crop",
    crop_padding_fraction: float = 0.01,
    center_crop_fraction: float = 0.9,
    sam3_min_area_ratio: float = 0.05,
    sam3_max_area_ratio: float = 0.98,
    limit: int | None = None,
    progress_every: int = 0,
) -> pd.DataFrame:
    frame = pd.read_csv(source_manifest_path)
    if len(frame) == 0:
        raise ValueError("Source manifest is empty.")
    if source_image_column not in frame.columns:
        raise ValueError(f"Source manifest must contain {source_image_column!r}.")
    if limit is not None:
        if limit <= 0:
            raise ValueError(f"limit must be positive when provided, got {limit}")
        frame = frame.head(limit).copy()

    sam_detector, center_detector = resolve_detector(
        crop_backend=crop_backend,
        sam3_prompts=sam3_prompts,
        sam3_device=sam3_device,
        center_crop_fraction=center_crop_fraction,
        sam3_min_area_ratio=sam3_min_area_ratio,
        sam3_max_area_ratio=sam3_max_area_ratio,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    output_paths: list[str] = []
    metadata_rows: list[dict[str, object]] = []

    total_rows = len(frame)
    for row_number, (_, row) in enumerate(frame.iterrows()):
        if progress_every > 0 and (row_number == 0 or row_number % progress_every == 0):
            print(f"Processing row {row_number + 1}/{total_rows}", flush=True)
        source_path = Path(str(row[source_image_column])).resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Row {row_number} missing source image: {source_path}")

        with Image.open(source_path) as raw_image:
            image = ImageOps.exif_transpose(raw_image).convert("RGB")
            crop = choose_crop(
                image,
                crop_backend=crop_backend,
                sam_detector=sam_detector,
                center_detector=center_detector,
                crop_padding_fraction=crop_padding_fraction,
            )
            processing_box = crop.bbox
            if resize_mode == "square_crop":
                processing_box = build_square_box(
                    crop.raw_bbox,
                    image_width=image.width,
                    image_height=image.height,
                    padding_fraction=crop_padding_fraction if crop.method != "center_prior" else 0.0,
                )
            processed = crop_resize(
                image,
                processing_box,
                image_size=image_size,
                resize_mode=resize_mode,
            )

        destination_name = f"{row_number:05d}_{source_path.stem}.png"
        destination_path = output_dir / destination_name
        processed.save(destination_path, format="PNG")
        output_paths.append(str(destination_path))
        metadata_rows.append(
            {
                "object_crop_status": crop.status,
                "object_crop_method": crop.method,
                "object_crop_prompt": crop.prompt,
                "object_crop_confidence": crop.confidence,
                "object_crop_mask_area_ratio": crop.mask_area_ratio,
                "object_crop_raw_bbox_x0": crop.raw_bbox.x0,
                "object_crop_raw_bbox_y0": crop.raw_bbox.y0,
                "object_crop_raw_bbox_x1": crop.raw_bbox.x1,
                "object_crop_raw_bbox_y1": crop.raw_bbox.y1,
                "object_crop_bbox_x0": processing_box.x0,
                "object_crop_bbox_y0": processing_box.y0,
                "object_crop_bbox_x1": processing_box.x1,
                "object_crop_bbox_y1": processing_box.y1,
                "object_crop_notes": crop.notes,
            }
        )
    if progress_every > 0:
        print(f"Processing row {total_rows}/{total_rows} complete", flush=True)

    derived = frame.copy()
    derived["source_image_path"] = [
        str(Path(value).resolve()) for value in frame[source_image_column].tolist()
    ]
    derived["image_path"] = output_paths
    derived["preprocessing_profile"] = f"rgb_object_focus_{resize_mode}_resize_{image_size}"
    derived["preprocessing_source_manifest"] = str(source_manifest_path.resolve())
    derived["object_crop_backend"] = crop_backend
    derived["object_crop_resize_mode"] = resize_mode
    for key in metadata_rows[0]:
        derived[key] = [metadata[key] for metadata in metadata_rows]

    derived.to_csv(output_manifest_path, index=False)
    return derived


def load_font(size: int) -> ImageFont.ImageFont:
    for font_path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        path = Path(font_path)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _row_box(row: pd.Series, *, prefix: str) -> BoundingBox | None:
    keys = [f"{prefix}_x0", f"{prefix}_y0", f"{prefix}_x1", f"{prefix}_y1"]
    if any(key not in row or pd.isna(row[key]) for key in keys):
        return None
    return BoundingBox(
        x0=int(row[keys[0]]),
        y0=int(row[keys[1]]),
        x1=int(row[keys[2]]),
        y1=int(row[keys[3]]),
    )


def _scale_box_to_tile(
    box: BoundingBox,
    *,
    image_width: int,
    image_height: int,
    tile_width: int,
    tile_height: int,
    offset_x: int,
    offset_y: int,
) -> tuple[int, int, int, int]:
    scale = min(tile_width / image_width, tile_height / image_height)
    rendered_width = int(round(image_width * scale))
    rendered_height = int(round(image_height * scale))
    inner_x = offset_x + (tile_width - rendered_width) // 2
    inner_y = offset_y + (tile_height - rendered_height) // 2
    scaled = (
        inner_x + int(round(box.x0 * scale)),
        inner_y + int(round(box.y0 * scale)),
        inner_x + int(round(box.x1 * scale)),
        inner_y + int(round(box.y1 * scale)),
    )
    return (
        max(offset_x, min(offset_x + tile_width, scaled[0])),
        max(offset_y, min(offset_y + tile_height, scaled[1])),
        max(offset_x, min(offset_x + tile_width, scaled[2])),
        max(offset_y, min(offset_y + tile_height, scaled[3])),
    )


def _paste_image_tile(
    *,
    sheet: Image.Image,
    draw: ImageDraw.ImageDraw,
    image: Image.Image,
    x: int,
    y: int,
    tile_size: int,
) -> None:
    image.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (tile_size, tile_size), (245, 245, 245))
    offset = ((tile_size - image.width) // 2, (tile_size - image.height) // 2)
    tile.paste(image, offset)
    sheet.paste(tile, (x, y))
    draw.rectangle((x, y, x + tile_size, y + tile_size), outline=(35, 35, 35), width=1)


def build_crop_contact_sheet(
    manifest_path: Path,
    output_path: Path,
    *,
    max_rows: int = 40,
    tile_size: int = 192,
) -> None:
    frame = pd.read_csv(manifest_path).head(max_rows)
    if len(frame) == 0:
        raise ValueError("Cannot build a contact sheet from an empty manifest.")
    font = load_font(12)
    columns = 2
    label_height = 60
    padding = 10
    rows_count = len(frame)
    width = columns * tile_size + (columns + 1) * padding
    height = rows_count * (tile_size + label_height) + (rows_count + 1) * padding
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)

    for row_position, (_, row) in enumerate(frame.iterrows()):
        y = padding + row_position * (tile_size + label_height + padding)
        for col, image_column in enumerate(("source_image_path", "image_path")):
            x = padding + col * (tile_size + padding)
            with Image.open(Path(str(row[image_column]))) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)
                tile = Image.new("RGB", (tile_size, tile_size), (245, 245, 245))
                offset = ((tile_size - image.width) // 2, (tile_size - image.height) // 2)
                tile.paste(image, offset)
            sheet.paste(tile, (x, y))
            draw.rectangle((x, y, x + tile_size, y + tile_size), outline=(35, 35, 35), width=1)

        label = (
            f"row={row_position} {row.get('training_score_band', '')} "
            f"{row.get('object_crop_method', '')} {row.get('object_crop_status', '')}"
        )
        draw.text((padding, y + tile_size + 5), label, font=font, fill=(0, 0, 0))
        bbox = (
            f"bbox=({row.get('object_crop_bbox_x0', '')},"
            f"{row.get('object_crop_bbox_y0', '')},"
            f"{row.get('object_crop_bbox_x1', '')},"
            f"{row.get('object_crop_bbox_y1', '')})"
        )
        draw.text((padding, y + tile_size + 25), bbox, font=font, fill=(50, 50, 50))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def build_crop_geometry_contact_sheet(
    manifest_path: Path,
    output_path: Path,
    *,
    max_rows: int = 24,
    tile_size: int = 192,
) -> None:
    frame = pd.read_csv(manifest_path).head(max_rows)
    if len(frame) == 0:
        raise ValueError("Cannot build a contact sheet from an empty manifest.")
    font = load_font(12)
    columns = 4
    label_height = 76
    padding = 10
    rows_count = len(frame)
    width = columns * tile_size + (columns + 1) * padding
    height = rows_count * (tile_size + label_height) + (rows_count + 1) * padding
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)

    column_labels = ("original", "SAM bbox", "square bbox", "result")
    for col, label in enumerate(column_labels):
        x = padding + col * (tile_size + padding)
        draw.text((x, 2), label, font=font, fill=(0, 0, 0))

    for row_position, (_, row) in enumerate(frame.iterrows()):
        y = padding + row_position * (tile_size + label_height + padding)
        y += 12
        source_path = Path(str(row["source_image_path"]))
        result_path = Path(str(row["image_path"]))
        raw_box = _row_box(row, prefix="object_crop_raw_bbox")
        square_box = _row_box(row, prefix="object_crop_bbox")

        with Image.open(source_path) as raw_image:
            source_image = ImageOps.exif_transpose(raw_image).convert("RGB")
            for col in range(3):
                x = padding + col * (tile_size + padding)
                tile_image = source_image.copy()
                _paste_image_tile(
                    sheet=sheet,
                    draw=draw,
                    image=tile_image,
                    x=x,
                    y=y,
                    tile_size=tile_size,
                )
                if col == 1 and raw_box is not None:
                    draw.rectangle(
                        _scale_box_to_tile(
                            raw_box,
                            image_width=source_image.width,
                            image_height=source_image.height,
                            tile_width=tile_size,
                            tile_height=tile_size,
                            offset_x=x,
                            offset_y=y,
                        ),
                        outline=(220, 20, 20),
                        width=3,
                    )
                if col == 2:
                    if raw_box is not None:
                        draw.rectangle(
                            _scale_box_to_tile(
                                raw_box,
                                image_width=source_image.width,
                                image_height=source_image.height,
                                tile_width=tile_size,
                                tile_height=tile_size,
                                offset_x=x,
                                offset_y=y,
                            ),
                            outline=(220, 20, 20),
                            width=2,
                        )
                    if square_box is not None:
                        draw.rectangle(
                            _scale_box_to_tile(
                                square_box,
                                image_width=source_image.width,
                                image_height=source_image.height,
                                tile_width=tile_size,
                                tile_height=tile_size,
                                offset_x=x,
                                offset_y=y,
                            ),
                            outline=(20, 120, 235),
                            width=3,
                        )

        x = padding + 3 * (tile_size + padding)
        with Image.open(result_path) as result_image:
            result_image = ImageOps.exif_transpose(result_image).convert("RGB")
            _paste_image_tile(
                sheet=sheet,
                draw=draw,
                image=result_image,
                x=x,
                y=y,
                tile_size=tile_size,
            )

        label = (
            f"row={row_position} {row.get('training_score_band', '')} "
            f"{row.get('object_crop_method', '')} {row.get('object_crop_resize_mode', '')}"
        )
        draw.text((padding, y + tile_size + 5), label, font=font, fill=(0, 0, 0))
        score = row.get("object_crop_confidence", "")
        prompt = row.get("object_crop_prompt", "")
        draw.text(
            (padding, y + tile_size + 25),
            f"score={score:.3f}" if isinstance(score, float) else f"score={score}",
            font=font,
            fill=(50, 50, 50),
        )
        draw.text(
            (padding + 150, y + tile_size + 25),
            f"prompt={prompt}",
            font=font,
            fill=(50, 50, 50),
        )
        draw.text(
            (padding, y + tile_size + 45),
            "red=SAM bbox blue=square crop",
            font=font,
            fill=(50, 50, 50),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)
