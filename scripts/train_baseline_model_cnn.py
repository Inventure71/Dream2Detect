from pathlib import Path
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    ROOT / "data"
    / "synthetic_full_qc_plus_v2_scale_384_not_cropped_dataset_20260516"
    / "synthetic_full_qc_plus_v2_scale_384_not_cropped_dataset_20260516"
)

MANIFEST = DATASET_DIR / "manifest.csv"

LABEL_COL = "final_score_band"

class_names = [
    "0-10",
    "11-20",
    "21-30",
    "31-35",
    "36-45",
    "46-55",
    "56-65",
    "66-75",
    "76-85",
    "86-100"
]

label_to_idx = {
    "0-10": 0,
    "11-20": 1,
    "21-30": 2,
    "31-35": 3,
    "36-45": 4,
    "46-55": 5,
    "56-65": 6,
    "66-75": 7,
    "76-85": 8,
    "86-100": 9
}

df = pd.read_csv(MANIFEST)
df = df.dropna(subset=["image_path", LABEL_COL])
df = df[df[LABEL_COL].isin(class_names)]

train_df, temp_df = train_test_split(
    df,
    test_size=0.3,
    random_state=42,
    stratify=df[LABEL_COL]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=temp_df[LABEL_COL]
)

class PackageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = DATASET_DIR / row["image_path"]
        image = Image.open(img_path).convert("RGB")
        label = label_to_idx[row[LABEL_COL]]

        if self.transform:
            image = self.transform(image)

        return image, label

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_loader = DataLoader(
    PackageDataset(train_df, train_transform),
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    PackageDataset(val_df, val_transform),
    batch_size=16,
    shuffle=False
)

test_loader = DataLoader(
    PackageDataset(test_df, val_transform),
    batch_size=16,
    shuffle=False
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = models.efficientnet_b0(
    weights=models.EfficientNet_B0_Weights.DEFAULT
)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    10
)
model = model.to(device)

checkpoint_path = ROOT / "best_model_acc_0.5185.pth"

if checkpoint_path.exists():
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print("Loaded checkpoint:", checkpoint_path)
else:
    print("Checkpoint not found")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

def within_k_accuracy(y_true, y_pred, k=1):
    correct = sum(abs(t - p) <= k for t, p in zip(y_true, y_pred))
    return correct / len(y_true)

best_acc = 0.5181

for epoch in range(8):
    model.train()
    train_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    acc_pm1 = within_k_accuracy(all_labels, all_preds, k=1)
    acc_pm2 = within_k_accuracy(all_labels, all_preds, k=2)

    print(f"Epoch {epoch+1}/8 | Loss: {train_loss:.4f} | Exact: {acc:.4f} | ±1: {acc_pm1:.4f} | ±2: {acc_pm2:.4f}")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), ROOT / f"best_model_acc_{best_acc:.4f}.pth")
        print("Saved new best model.")

print("\nBest validation accuracy:", best_acc)

print("\n--- TEST SET EVALUATION ---")

model.eval()

test_preds = []
test_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

        test_preds.extend(preds)
        test_labels.extend(labels.numpy())

test_exact = accuracy_score(test_labels, test_preds)
test_pm1 = within_k_accuracy(test_labels, test_preds, k=1)
test_pm2 = within_k_accuracy(test_labels, test_preds, k=2)

print(f"Test Exact Accuracy: {test_exact:.4f}")
print(f"Test ±1 Accuracy: {test_pm1:.4f}")
print(f"Test ±2 Accuracy: {test_pm2:.4f}")