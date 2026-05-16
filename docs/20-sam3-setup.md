# SAM3 Setup for Object-Focused Cropping

This project can use SAM3 as an optional preprocessing backend to crop each
image around the cardboard box/package before classifier training.

## Current Local Status

The current Mac environment is not a valid official SAM3 runtime:

- macOS ARM / Apple Silicon
- MPS available
- CUDA unavailable
- local PyTorch is below the official SAM3 requirement
- `sam3` is not importable

The official `facebookresearch/sam3` README currently lists:

- Python `3.12+`
- PyTorch `2.7+`
- CUDA-compatible GPU with CUDA `12.6+`
- Hugging Face access to the gated SAM3 checkpoints

There is also an open community PR for Apple Silicon / MPS support. Treat that
as an experimental runtime path until it is merged upstream. Local Mac strict
SAM3 cropping is acceptable only when the preflight script passes with
`--device mps` and a one-image smoke returns real boxes.

## Preflight Check

Run this before attempting strict SAM3 cropping:

```bash
python3 scripts/check_sam3_setup.py --device mps
```

The script checks:

- Python version
- PyTorch version
- CUDA availability and CUDA version
- whether `sam3` is importable
- whether a Hugging Face token is configured

If the MPS or CUDA SAM3 environment is ready, run a one-image smoke:

```bash
python3 scripts/check_sam3_setup.py \
  --device mps \
  --smoke-image data/real/source/redf0xwin_recognizing_defects_in_boxes_and_cardboard/images/0001.jpg \
  --prompt "cardboard box"
```

## Experimental MPS Environment Setup

Use a dedicated environment rather than installing experimental SAM3 into the
classifier training environment:

```bash
conda create -n sam3-mps python=3.12
conda activate sam3-mps

pip install --upgrade pip
pip install torch torchvision torchaudio
pip install git+https://github.com/provos/sam3.git@apple-silicon-support-v2
pip install pandas Pillow einops decord2 pycocotools opencv-python matplotlib psutil
```

The branch's wheel packaging currently omits required subpackages. Keep a local
source checkout and run through the project wrapper so Python imports the full
source tree:

```bash
mkdir -p ~/.cache/dream2detect
git clone --branch apple-silicon-support-v2 \
  https://github.com/provos/sam3.git \
  ~/.cache/dream2detect/sam3-apple-silicon-support-v2
```

Then authenticate to Hugging Face after requesting and receiving access to the
SAM3 checkpoint repository:

```bash
hf auth login
```

or configure `HF_TOKEN` in the environment.

Validate:

```bash
scripts/run_sam3_mps_python.sh scripts/check_sam3_setup.py --device mps
```

If this passes, run the one-image smoke above before generating a full cache.

## CUDA Environment Setup

On a CUDA machine, use a dedicated environment rather than installing SAM3 into
the classifier training environment:

```bash
conda create -n sam3 python=3.12
conda activate sam3

pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install git+https://github.com/facebookresearch/sam3.git
pip install pandas Pillow
```

Then authenticate to Hugging Face after requesting and receiving access to the
SAM3 checkpoint repository:

```bash
hf auth login
```

or configure `HF_TOKEN` in the environment.

## Strict SAM3 Crop Commands

Real images:

```bash
scripts/run_sam3_mps_python.sh scripts/build_object_focused_cache.py \
  --source-manifest data/datasets/real_labeled_dataset_current.csv \
  --output-dir data/cache/real_labeled_dataset_current_sam3_boxfocus_384 \
  --output-manifest data/datasets/real_labeled_dataset_current_sam3_boxfocus_384.csv \
  --image-size 384 \
  --source-image-column image_path \
  --crop-backend sam3 \
  --sam3-device mps \
  --contact-sheet data/qc/object_focus/real_sam3_boxfocus_sheet.jpg
```

Synthetic images:

```bash
scripts/run_sam3_mps_python.sh scripts/build_object_focused_cache.py \
  --source-manifest data/datasets/synthetic_full_qc_plus_v2_scale_processed_384.csv \
  --output-dir data/cache/synthetic_full_qc_plus_v2_scale_sam3_boxfocus_384 \
  --output-manifest data/datasets/synthetic_full_qc_plus_v2_scale_sam3_boxfocus_384.csv \
  --image-size 384 \
  --source-image-column source_image_path \
  --crop-backend sam3 \
  --sam3-device mps \
  --contact-sheet data/qc/object_focus/synthetic_sam3_boxfocus_sheet.jpg
```

Strict `sam3` mode must produce `object_crop_method=sam3_text`. If it fails, do
not continue to training until the failure is understood.
