from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from PIL import Image

from dream2detect.preprocessing.object_crop import (
    BoundingBox,
    CenterPriorDetector,
    build_center_prior_box,
    build_object_focused_manifest,
    build_square_box,
    choose_crop,
    crop_pad_resize,
    crop_stretch_resize,
    crop_square_with_white_padding_resize,
    expand_box,
)


class EmptyDetector:
    def detect(self, image: Image.Image):  # type: ignore[no-untyped-def]
        return None


class ObjectFocusedCacheTests(unittest.TestCase):
    def test_center_prior_box_uses_centered_fraction_of_image(self) -> None:
        box = build_center_prior_box(
            image_width=100,
            image_height=80,
            crop_fraction=0.8,
        )

        self.assertEqual(box, BoundingBox(x0=10, y0=8, x1=90, y1=72))

    def test_expand_box_clamps_to_image_bounds(self) -> None:
        box = expand_box(
            BoundingBox(x0=10, y0=20, x1=50, y1=60),
            image_width=70,
            image_height=80,
            padding_fraction=0.5,
        )

        self.assertEqual(box, BoundingBox(x0=0, y0=0, x1=70, y1=80))

    def test_build_square_box_contains_sam_box_without_distortion(self) -> None:
        box = build_square_box(
            BoundingBox(x0=10, y0=20, x1=90, y1=60),
            image_width=120,
            image_height=100,
            padding_fraction=0.0,
        )

        self.assertEqual(box.width, box.height)
        self.assertLessEqual(box.x0, 10)
        self.assertLessEqual(box.y0, 20)
        self.assertGreaterEqual(box.x1, 90)
        self.assertGreaterEqual(box.y1, 60)

    def test_build_square_box_keeps_long_side_even_when_image_is_shorter(self) -> None:
        box = build_square_box(
            BoundingBox(x0=0, y0=0, x1=95, y1=40),
            image_width=100,
            image_height=80,
            padding_fraction=0.0,
        )

        self.assertEqual(box.width, box.height)
        self.assertEqual(box.width, 95)
        self.assertGreaterEqual(box.x0, 0)
        self.assertLessEqual(box.x1, 100)
        self.assertTrue(box.y0 < 0 or box.y1 > 80)

    def test_crop_pad_resize_outputs_square_rgb_image(self) -> None:
        image = Image.new("RGB", (100, 60), color=(255, 255, 255))

        output = crop_pad_resize(
            image,
            BoundingBox(x0=10, y0=5, x1=90, y1=55),
            image_size=64,
        )

        self.assertEqual(output.mode, "RGB")
        self.assertEqual(output.size, (64, 64))

    def test_crop_stretch_resize_fills_square_without_letterbox(self) -> None:
        image = Image.new("RGB", (100, 60), color=(255, 255, 255))

        output = crop_stretch_resize(
            image,
            BoundingBox(x0=10, y0=5, x1=90, y1=55),
            image_size=64,
        )

        self.assertEqual(output.mode, "RGB")
        self.assertEqual(output.size, (64, 64))

    def test_square_crop_pads_out_of_bounds_with_white(self) -> None:
        image = Image.new("RGB", (80, 40), color=(10, 20, 30))

        output = crop_square_with_white_padding_resize(
            image,
            BoundingBox(x0=0, y0=-20, x1=80, y1=60),
            image_size=80,
        )

        self.assertEqual(output.mode, "RGB")
        self.assertEqual(output.size, (80, 80))
        self.assertEqual(output.getpixel((40, 0)), (255, 255, 255))
        self.assertEqual(output.getpixel((40, 40)), (10, 20, 30))
        self.assertEqual(output.getpixel((40, 79)), (255, 255, 255))

    def test_center_backend_writes_manifest_with_crop_metadata(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "image.jpg"
            Image.new("RGB", (100, 80), color=(255, 255, 255)).save(image_path)
            manifest_path = temp_path / "manifest.csv"
            pd.DataFrame(
                [
                    {
                        "prompt_id": 1,
                        "image_path": str(image_path),
                        "training_score_band": "36-45",
                        "training_coarse_class": "moderate",
                    }
                ]
            ).to_csv(manifest_path, index=False)

            output_manifest = temp_path / "object_manifest.csv"
            output_dir = temp_path / "cache"
            frame = build_object_focused_manifest(
                manifest_path,
                output_dir,
                output_manifest,
                image_size=64,
                source_image_column="image_path",
                crop_backend="center",
                sam3_prompts=("cardboard box",),
                center_crop_fraction=0.8,
            )

            self.assertTrue(output_manifest.exists())
            self.assertEqual(len(frame), 1)
            self.assertTrue(Path(frame.iloc[0]["image_path"]).exists())
            self.assertEqual(frame.iloc[0]["object_crop_method"], "center_prior")
            self.assertEqual(frame.iloc[0]["object_crop_status"], "fallback_center")
            self.assertEqual(frame.iloc[0]["preprocessing_profile"], "rgb_object_focus_square_crop_resize_64")
            self.assertEqual(frame.iloc[0]["object_crop_resize_mode"], "square_crop")
            self.assertEqual(int(frame.iloc[0]["object_crop_bbox_x0"]), 10)
            self.assertEqual(int(frame.iloc[0]["object_crop_raw_bbox_x0"]), 10)

    def test_sam3_backend_fails_when_sam_returns_no_crop(self) -> None:
        image = Image.new("RGB", (100, 80), color=(255, 255, 255))

        with self.assertRaisesRegex(RuntimeError, "SAM 3 did not return"):
            choose_crop(
                image,
                crop_backend="sam3",
                sam_detector=EmptyDetector(),
                center_detector=CenterPriorDetector(crop_fraction=0.8),
                crop_padding_fraction=0.1,
            )

    def test_hybrid_backend_falls_back_when_sam_returns_no_crop(self) -> None:
        image = Image.new("RGB", (100, 80), color=(255, 255, 255))

        result = choose_crop(
            image,
            crop_backend="hybrid",
            sam_detector=EmptyDetector(),
            center_detector=CenterPriorDetector(crop_fraction=0.8),
            crop_padding_fraction=0.1,
        )

        self.assertEqual(result.method, "center_prior")
        self.assertEqual(result.status, "fallback_center")


if __name__ == "__main__":
    unittest.main()
