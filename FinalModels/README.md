# Dream2Detect — Final Models

All models were trained exclusively on synthetic AI-generated images and evaluated on 385 real-world photos never seen during training.

## Download Checkpoints

Download each model folder from Google Drive and drag it into your `outputs/` folder.

| Model | Drive Link | ±1 Accuracy | MBE |
|---|---|---|---|
| EMD | https://drive.google.com/drive/folders/1Kdx9TlJyI42ftQ4njiAD7ltlPL2G41eD?usp=sharing | 58.44% | 1.4234 |
| SAM | https://drive.google.com/drive/folders/1yCo3wAUoKjcVRHv0nNk5Gc-3voKZDX1z?usp=sharing | 59.74% | 1.4182 |
| BaseModel | https://drive.google.com/drive/folders/1_SXjCcPRONZBoyzVr75bbbVQ1x9FXHhD?usp=sharing | 52.47% | 1.7039 |
| Dual | https://drive.google.com/drive/folders/1JjjkN6lKPVpifmlo6Qa3PtEnExiKUehR?usp=sharing | 55.06% | 1.6338 |

After downloading your `outputs/` folder should look like this:

```
outputs/
├── EMD/
│   └── best_model.pt
├── SAM/
│   └── best_model.pt
├── BaseModel/
│   └── best_model.pt
└── Dual/
    └── best_model.pt
```

---

## Requirements

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install timm albumentations pandas matplotlib scikit-learn
pip install segment-anything opencv-python
```

---

## Single Model Evaluation

> ⚠️ SAM model must use `--images_dir realWorld/sam_cropped/` — it was trained on SAM-cropped images and expects cropped input at evaluation time. All other models use `--images_dir realWorld/`.

**EMD:**
```bash
python evaluate3.py --checkpoint outputs/EMD/best_model.pt \
  --manifest realWorld/manifest.csv \
  --images_dir realWorld/ \
  --output_dir outputs/eval_emd \
  --model_name "EMD"
```

**SAM:**
```bash
python evaluate3.py --checkpoint outputs/SAM/best_model.pt \
  --manifest realWorld/manifest.csv \
  --images_dir realWorld/sam_cropped/ \
  --output_dir outputs/eval_sam \
  --model_name "SAM"
```

**Base Model:**
```bash
python evaluate3.py --checkpoint outputs/BaseModel/best_model.pt \
  --manifest realWorld/manifest.csv \
  --images_dir realWorld/ \
  --output_dir outputs/eval_base \
  --model_name "BaseModel"
```

**Dual (ConvNeXtDual):**
```bash
python evaluate3.py --checkpoint outputs/Dual/best_model.pt \
  --manifest realWorld/manifest.csv \
  --images_dir realWorld/ \
  --output_dir outputs/eval_dual \
  --model_name "Dual"
```

---

## Ensemble — Two Models

Use `--weight_a` to control how much weight model A gets (model B gets the remainder automatically).

**⭐ Best Model — EMD60+SAM40 (64.16% ±1, 1.3351 MBE):**
```bash
python evaluate_ensemble.py \
  --checkpoint_a outputs/EMD/best_model.pt \
  --checkpoint_b outputs/SAM/best_model.pt \
  --images_dir_b realWorld/sam_cropped/ \
  --weight_a 0.60 \
  --manifest realWorld/manifest.csv \
  --images_dir realWorld/ \
  --output_dir outputs/eval_emd60_sam40 \
  --model_name "EMD60+SAM40"
```

**EMD + Dual:**
```bash
python evaluate_ensemble.py \
  --checkpoint_a outputs/EMD/best_model.pt \
  --checkpoint_b outputs/Dual/best_model.pt \
  --weight_a 0.50 \
  --manifest realWorld/manifest.csv \
  --images_dir realWorld/ \
  --output_dir outputs/eval_emd_dual \
  --model_name "EMD+Dual"
```

**EMD + BaseModel:**
```bash
python evaluate_ensemble.py \
  --checkpoint_a outputs/EMD/best_model.pt \
  --checkpoint_b outputs/BaseModel/best_model.pt \
  --weight_a 0.50 \
  --manifest realWorld/manifest.csv \
  --images_dir realWorld/ \
  --output_dir outputs/eval_emd_base \
  --model_name "EMD+BaseModel"
```

---

## Triple Ensemble — Three Models

**EMD + SAM + Dual:**
```bash
python evaluate_triple.py \
  --checkpoint_a outputs/EMD/best_model.pt \
  --checkpoint_b outputs/SAM/best_model.pt \
  --checkpoint_c outputs/Dual/best_model.pt \
  --images_dir_b realWorld/sam_cropped/ \
  --manifest realWorld/manifest.csv \
  --images_dir realWorld/ \
  --output_dir outputs/eval_triple_emd_sam_dual \
  --model_name "EMD+SAM+Dual"
```

**EMD + SAM + BaseModel:**
```bash
python evaluate_triple.py \
  --checkpoint_a outputs/EMD/best_model.pt \
  --checkpoint_b outputs/SAM/best_model.pt \
  --checkpoint_c outputs/BaseModel/best_model.pt \
  --images_dir_b realWorld/sam_cropped/ \
  --manifest realWorld/manifest.csv \
  --images_dir realWorld/ \
  --output_dir outputs/eval_triple_emd_sam_base \
  --model_name "EMD+SAM+BaseModel"
```

Use `--weight_a`, `--weight_b`, `--weight_c` to adjust individual model weights (must sum to 1.0).

---

## Predict on a Single Image

```bash
python predict.py \
  --checkpoint outputs/EMD/best_model.pt \
  --image path/to/your/image.jpg
```

For SAM model, point at a SAM-cropped image:
```bash
python predict.py \
  --checkpoint outputs/SAM/best_model.pt \
  --image your_images_cropped/images/your_image.jpg
```

---

## Full Results

| Model | Exact | ±1 | MBE |
|---|---|---|---|
| ⭐ EMD60+SAM40 | 21.04% | 64.16% | 1.3351 |
| EMD65+SAM35 | 20.78% | 64.16% | 1.3299 |
| EMD+SAM 50/50 | 20.52% | 63.40% | 1.3636 |
| EMD+Dual 50/50 | 21.82% | 60.26% | 1.4338 |
| SAM alone | 21.04% | 59.74% | 1.4182 |
| EMD alone | 20.26% | 58.44% | 1.4234 |
| Dual alone | 18.18% | 55.06% | 1.6338 |
| BaseModel alone | 15.58% | 52.47% | 1.7039 |
