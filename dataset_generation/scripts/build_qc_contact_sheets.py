from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[1]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build contact sheets for manual image-label QC."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--queue-name", default="qc_queue.csv")
    parser.add_argument("--sheet-prefix", default="qc_sheet")
    parser.add_argument("--filter-qc-status", default="unreviewed")
    parser.add_argument("--images-per-sheet", type=int, default=20)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--tile-size", type=int, default=224)
    return parser


def load_font(size: int) -> ImageFont.ImageFont:
    for font_path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        path = Path(font_path)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    *,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    max_width: int,
    line_height: int,
) -> None:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)

    x, y = xy
    for line in lines[:4]:
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height


def make_sheet(
    rows: pd.DataFrame,
    *,
    output_path: Path,
    columns: int,
    tile_size: int,
) -> None:
    font = load_font(15)
    small_font = load_font(12)
    padding = 12
    label_height = 108
    rows_count = math.ceil(len(rows) / columns)
    sheet_width = columns * tile_size + (columns + 1) * padding
    sheet_height = rows_count * (tile_size + label_height) + (rows_count + 1) * padding
    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    draw = ImageDraw.Draw(sheet)

    for index, (_, row) in enumerate(rows.iterrows()):
        col = index % columns
        sheet_row = index // columns
        x = padding + col * (tile_size + padding)
        y = padding + sheet_row * (tile_size + label_height + padding)

        image_path = Path(str(row["image_path"]))
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (tile_size, tile_size), (245, 245, 245))
            ox = (tile_size - image.width) // 2
            oy = (tile_size - image.height) // 2
            tile.paste(image, (ox, oy))
        sheet.paste(tile, (x, y))
        draw.rectangle((x, y, x + tile_size, y + tile_size), outline=(35, 35, 35), width=1)

        label_y = y + tile_size + 5
        header = (
            f"row={row['row_index']} id={row['prompt_id']} "
            f"{row['training_score_band']} {row['training_coarse_class']}"
        )
        draw.text((x, label_y), header, font=font, fill=(0, 0, 0))
        draw_wrapped_text(
            draw,
            str(row.get("prompt_title", "")),
            (x, label_y + 21),
            font=small_font,
            fill=(30, 30, 30),
            max_width=tile_size,
            line_height=14,
        )
        detail = f"{row.get('damage_profile_primary', '')} | {row.get('camera_angle', '')}"
        draw_wrapped_text(
            draw,
            detail,
            (x, label_y + 73),
            font=small_font,
            fill=(80, 80, 80),
            max_width=tile_size,
            line_height=14,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def main() -> None:
    args = build_argument_parser().parse_args()
    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(manifest_path)
    if args.filter_qc_status:
        frame = frame[frame["qc_status"].astype(str) == args.filter_qc_status].copy()
    frame.insert(0, "row_index", frame.index)
    frame = frame.reset_index(drop=True)

    queue_path = output_dir / args.queue_name
    frame.to_csv(queue_path, index=False)

    for start in range(0, len(frame), args.images_per_sheet):
        chunk = frame.iloc[start : start + args.images_per_sheet]
        sheet_number = start // args.images_per_sheet + 1
        sheet_path = output_dir / f"{args.sheet_prefix}_{sheet_number:02d}.jpg"
        make_sheet(
            chunk,
            output_path=sheet_path,
            columns=args.columns,
            tile_size=args.tile_size,
        )

    print(f"Queued {len(frame)} rows for QC: {queue_path}")
    print(
        "Wrote "
        f"{math.ceil(len(frame) / args.images_per_sheet)} contact sheets to {output_dir}"
    )


if __name__ == "__main__":
    main()
