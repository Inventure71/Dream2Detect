from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH = 1920
HEIGHT = 1080
BG = "#F7F9FC"
TITLE = "#18243D"
SUBTITLE = "#6A7384"
CYAN = "#00B7E5"
PURPLE = "#8E61F8"
PANEL_BORDER = "#DCE6F4"
PANEL_BG = "#FFFFFF"
INSIGHT_BG = "#EAF8FD"
BODY = "#3A4458"
MUTED = "#6A7384"


def wrap_to_pixels(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    width: int,
    font: ImageFont.FreeTypeFont,
) -> list[str]:
    words = text.split()
    if not words:
        return [""]

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if draw.textlength(candidate, font=font) <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def load_font(size: int, *, bold: bool = False, italic: bool = False) -> ImageFont.FreeTypeFont:
    candidates: list[str] = []
    if bold and italic:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial Bold Italic.ttf",
                "/System/Library/Fonts/Supplemental/Helvetica.ttc",
            ]
        )
    elif bold:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "/System/Library/Fonts/Supplemental/Helvetica.ttc",
            ]
        )
    elif italic:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial Italic.ttf",
                "/System/Library/Fonts/Supplemental/Helvetica.ttc",
            ]
        )
    else:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial.ttf",
                "/System/Library/Fonts/Supplemental/Helvetica.ttc",
            ]
        )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    x: int,
    y: int,
    width: int,
    font: ImageFont.FreeTypeFont,
    fill: str,
    line_spacing: int = 8,
    bullet_indent: int = 26,
) -> int:
    current_y = y
    for raw_line in text.splitlines():
        if not raw_line:
            current_y += font.size + line_spacing
            continue
        bullet = raw_line.startswith("- ")
        content = raw_line[2:] if bullet else raw_line
        lines = wrap_to_pixels(
            draw,
            content,
            width=width - (bullet_indent if bullet else 0),
            font=font,
        )
        if not lines:
            lines = [""]
        for index, line in enumerate(lines):
            prefix = "- " if bullet and index == 0 else ""
            indent = bullet_indent if bullet and index > 0 else 0
            draw.text((x + indent, current_y), prefix + line, font=font, fill=fill)
            current_y += font.size + line_spacing
    return current_y


def main() -> None:
    out_dir = Path("artifacts/presentation_assets")
    out_dir.mkdir(parents=True, exist_ok=True)

    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)

    title_font = load_font(40, bold=True)
    subtitle_font = load_font(20, italic=True)
    header_font = load_font(24, bold=True)
    body_font = load_font(15)
    pill_font = load_font(21, bold=True)
    small_font = load_font(16)
    label_font = load_font(13, bold=True)
    takeaway_font = load_font(18)

    draw.text((96, 58), "Main Experimental Ladder: V1 to V5-B", font=title_font, fill=TITLE)
    draw.text(
        (96, 120),
        "The no-pretrain path was an experiment chain: each version changed one thing, measured it, then exposed the next bottleneck.",
        font=subtitle_font,
        fill=SUBTITLE,
    )

    main_box = (76, 188, 1844, 858)
    takeaway_box = (76, 884, 1844, 1004)
    draw.rectangle(main_box, fill=PANEL_BG, outline=PANEL_BORDER, width=2)
    draw.rectangle(takeaway_box, fill=INSIGHT_BG, outline="#A8E4F5", width=2)

    draw.text((main_box[0] + 28, main_box[1] + 18), "Step-by-Step Story", font=header_font, fill=CYAN)
    draw.text(
        (main_box[0] + 28, main_box[1] + 54),
        "Each version was a controlled answer to the previous failure, not a random architecture swap.",
        font=small_font,
        fill=MUTED,
    )

    versions = [
        (
            "V1",
            "First prove the task was learnable from synthetic labels.",
            "Scratch CNN at 224 px with basic diagnostics.",
            "There was learnable signal, but the model was fragile.",
        ),
        (
            "V2",
            "If V1 was capacity/data-limited, improve both before blaming transfer.",
            "Full QC labels, 384 px input, residual CNN, targeted synthetic scale-up.",
            "Synthetic results improved; the real-domain gap stayed visible.",
        ),
        (
            "V3",
            "Test whether the gap was caused by training setup rather than data.",
            "Searched augmentation, normalization, samplers, and ordinal/coarse variants.",
            "Low augmentation helped; GroupNorm/ordinal-coarse did not solve transfer.",
        ),
        (
            "V3.1",
            "Make the synthetic validation split less contaminated by prompt similarity.",
            "Moved to metadata_family_holdout by prompt/image family.",
            "Evaluation became harsher and more honest.",
        ),
        (
            "V4",
            "Stop simplifying the target: train the real 10 severity bands directly.",
            "Used score_band as the target with soft ordinal labels.",
            "Near-band synthetic behavior improved; real generalization was still weak.",
        ),
        (
            "V4.5",
            "Check whether the loss should punish far-away severity mistakes more.",
            "Added distance-aware EMD-style loss on the 10-band distribution.",
            "It gave small real gains, but not a decisive fix.",
        ),
        (
            "V5-A",
            "Ask whether severity is better represented as one ordered scalar.",
            "Regressed normalized representative score, then mapped back to bands.",
            "Real mean-bias improved, but synthetic exact-band accuracy fell.",
        ),
        (
            "V5-B",
            "Combine the useful signals instead of choosing one target form.",
            "Multitask scalar + 10-band + coarse supervision.",
            "Tolerance on real images widened; longer training showed scratch had plateaued.",
        ),
    ]

    card_w = 820
    card_h = 132
    gap_x = 62
    gap_y = 10
    start_x = main_box[0] + 32
    start_y = main_box[1] + 92

    def draw_story_row(prefix: str, text: str, *, x: int, y: int, width: int) -> int:
        draw.text((x, y), prefix, font=label_font, fill=MUTED)
        label_w = int(draw.textlength(prefix, font=label_font)) + 12
        return draw_wrapped_text(
            draw,
            text,
            x=x + label_w,
            y=y - 1,
            width=width - label_w,
            font=body_font,
            fill=BODY,
            line_spacing=4,
            bullet_indent=0,
        )

    for index, (version, why, changed, learned) in enumerate(versions):
        col = index % 2
        row = index // 2
        x1 = start_x + col * (card_w + gap_x)
        y1 = start_y + row * (card_h + gap_y)
        x2 = x1 + card_w
        y2 = y1 + card_h
        accent = CYAN if index < 4 else PURPLE
        draw.rectangle((x1, y1, x2, y2), fill="#FFFFFF", outline=PANEL_BORDER, width=2)
        draw.rectangle((x1, y1, x1 + 112, y2), fill="#F2FBFE" if index < 4 else "#F5F1FF", outline=PANEL_BORDER, width=1)
        draw.text((x1 + 28, y1 + 39), version, font=pill_font, fill=accent)
        text_x = x1 + 136
        text_w = card_w - 166
        next_y = draw_story_row("Why:", why, x=text_x, y=y1 + 13, width=text_w)
        next_y = draw_story_row("Changed:", changed, x=text_x, y=max(next_y + 3, y1 + 48), width=text_w)
        draw_story_row("Learned:", learned, x=text_x, y=max(next_y + 3, y1 + 84), width=text_w)

    draw.text((takeaway_box[0] + 28, takeaway_box[1] + 20), "Takeaway", font=header_font, fill=CYAN)
    takeaway = (
        "The ladder shows why pretrained is a comparison point, not the whole project: V1-V5-B proved the synthetic task was "
        "learnable, then narrowed the unresolved problem to real-domain transfer under a scratch CNN."
    )
    draw_wrapped_text(
        draw,
        takeaway,
        x=takeaway_box[0] + 28,
        y=takeaway_box[1] + 72,
        width=takeaway_box[2] - takeaway_box[0] - 56,
        font=takeaway_font,
        fill=BODY,
        line_spacing=8,
        bullet_indent=0,
    )

    png_path = out_dir / "dream2detect_v1_to_v5b_ladder_slide.png"
    pdf_path = out_dir / "dream2detect_v1_to_v5b_ladder_slide.pdf"
    image.save(png_path)
    image.save(pdf_path, "PDF", resolution=144.0)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
