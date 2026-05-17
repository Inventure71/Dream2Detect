# V0 Final Model Checkpoints

The V0 pretrained pipeline used externally stored checkpoints. They are not
committed to this cleanup branch.

## Checkpoints

Download the model folder you want into `outputs/`:

| Model | Drive Link | Reported +-1 Accuracy | Reported MBE |
|---|---|---:|---:|
| EMD | https://drive.google.com/drive/folders/1Kdx9TlJyI42ftQ4njiAD7ltlPL2G41eD?usp=sharing | 58.44% | 1.4234 |
| SAM | https://drive.google.com/drive/folders/1yCo3wAUoKjcVRHv0nNk5Gc-3voKZDX1z?usp=sharing | 59.74% | 1.4182 |
| BaseModel | https://drive.google.com/drive/folders/1_SXjCcPRONZBoyzVr75bbbVQ1x9FXHhD?usp=sharing | 52.47% | 1.7039 |
| Dual | https://drive.google.com/drive/folders/1JjjkN6lKPVpifmlo6Qa3PtEnExiKUehR?usp=sharing | 55.06% | 1.6338 |

Expected layout:

```text
outputs/
  EMD/best_model.pt
  SAM/best_model.pt
  BaseModel/best_model.pt
  Dual/best_model.pt
```

## Evaluate One Model

```bash
python3 pipelines/v0-main-pretrained/EvaluartionFiles/evaluate3.py \
  --checkpoint outputs/EMD/best_model.pt \
  --manifest dataset/real/manifest.csv \
  --images_dir dataset/real \
  --output_dir outputs/eval_emd \
  --model_name "EMD"
```

## Evaluate An Ensemble

```bash
python3 pipelines/v0-main-pretrained/EvaluartionFiles/evaluate_ensemble.py \
  --checkpoint_a outputs/EMD/best_model.pt \
  --checkpoint_b outputs/Dual/best_model.pt \
  --weight_a 0.50 \
  --manifest dataset/real/manifest.csv \
  --images_dir dataset/real \
  --output_dir outputs/eval_emd_dual \
  --model_name "EMD+Dual"
```

SAM checkpoints expect SAM-cropped input. Use
`pipelines/v0-main-pretrained/TrainingFiles/preprocess_sam.py` to create a
cropped copy first.

