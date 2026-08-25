import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix

# ─────────────────────────────────────────────
# 1. DATASET
# ─────────────────────────────────────────────
class FoodDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        self.classes = sorted([
            f for f in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, f))
        ])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        for cls in self.classes:
            cls_folder = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_folder):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    path = os.path.join(cls_folder, fname)
                    label = self.class_to_idx[cls]
                    self.samples.append((path, label))

        print(f"Found {len(self.samples)} images across {len(self.classes)} classes")
        print(f"Classes: {self.classes}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ─────────────────────────────────────────────
# 2. TRANSFORM
# ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ─────────────────────────────────────────────
# 3. LOAD DATASET
# ─────────────────────────────────────────────
dataset = FoodDataset(root_dir=r"C:\Users\harsh\Downloads\A,B,CNNS_with_Tim\veggie_heap_training", transform=transform)

img, label = dataset[0]
print(img.shape)
print(label)


# ─────────────────────────────────────────────
# 4. TRAIN / VAL SPLIT
# ─────────────────────────────────────────────
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)

print(f"Training images:   {train_size}")
print(f"Validation images: {val_size}")


# ─────────────────────────────────────────────
# 5. MODEL
# ─────────────────────────────────────────────
class FoodCNN(nn.Module):
    def __init__(self, num_classes=12):
        super(FoodCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # [3,64,64]  → [32,64,64]
            nn.ReLU(),
            nn.MaxPool2d(2),                               # [32,64,64] → [32,32,32]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # [32,32,32] → [64,32,32]
            nn.ReLU(),
            nn.MaxPool2d(2),                               # [64,32,32] → [64,16,16]

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # [64,16,16] → [128,16,16]
            nn.ReLU(),
            nn.MaxPool2d(2)                                # [128,16,16]→ [128,8,8]
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8192, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


model = FoodCNN(num_classes=len(dataset.classes))
print(model)

dummy  = torch.randn(4, 3, 64, 64)
output = model(dummy)
print(f"Output shape: {output.shape}")


# ─────────────────────────────────────────────
# 6. TRAINING LOOP
# ─────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_losses, val_losses         = [], []
train_accuracies, val_accuracies = [], []

EPOCHS = 15 

for epoch in range(EPOCHS):

    # ── TRAIN ──
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted   = torch.argmax(outputs, dim=1)
        correct    += (predicted == labels).sum().item()
        total      += labels.size(0)

    train_loss = total_loss / len(train_loader)
    train_acc  = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # ── VALIDATE ──
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total   = 0

    with torch.no_grad():
        for images, labels in val_loader:
            outputs   = model(images)
            loss      = criterion(outputs, labels)
            val_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            val_correct += (predicted == labels).sum().item()
            val_total   += labels.size(0)

    val_loss = val_loss / len(val_loader)
    val_acc  = val_correct / val_total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")


# ─────────────────────────────────────────────
# 7. HELPER: short class names
# ─────────────────────────────────────────────
def get_short_names(classes):
    short = []
    for c in classes:
        if "'s " in c:
            short.append(c.split("'s ")[-1])
        else:
            short.append(c)
    return short


# ─────────────────────────────────────────────
# 8. VISUALISATION 1 — Loss & Accuracy graphs
# ─────────────────────────────────────────────
def plot_training(train_losses, val_losses, train_accuracies, val_accuracies):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses,     label='Train Loss')
    ax1.plot(val_losses,       label='Val Loss')
    ax1.set_title('Loss over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(train_accuracies, label='Train Accuracy')
    ax2.plot(val_accuracies,   label='Val Accuracy')
    ax2.set_title('Accuracy over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('training_graphs.png')
    plt.show()

plot_training(train_losses, val_losses, train_accuracies, val_accuracies)


# ─────────────────────────────────────────────
# 9. VISUALISATION 2 — Confusion Matrix
# ─────────────────────────────────────────────
def plot_confusion_matrix(model, val_loader, classes):
    all_preds  = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            outputs   = model(images)
            predicted = torch.argmax(outputs, dim=1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())

    cm          = confusion_matrix(all_labels, all_preds)
    short_names = get_short_names(classes)

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        xticklabels=short_names,
        yticklabels=short_names,
        cmap='Blues'
    )
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

plot_confusion_matrix(model, val_loader, dataset.classes)


# ─────────────────────────────────────────────
# 10. VISUALISATION 3 — Sample Predictions
# ─────────────────────────────────────────────
def plot_predictions(model, test_dir, transform, classes, num_images=12):
    short_names = get_short_names(classes)

    test_images = []
    test_labels = []

    for cls in sorted(os.listdir(test_dir)):
        cls_folder = os.path.join(test_dir, cls)
        if not os.path.isdir(cls_folder):
            continue
        for fname in os.listdir(cls_folder):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                test_images.append(os.path.join(cls_folder, fname))
                test_labels.append(cls)
                break

    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    axes = axes.flatten()

    model.eval()
    with torch.no_grad():
        for i, (img_path, true_label) in enumerate(zip(test_images[:num_images], test_labels[:num_images])):
            img    = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0)

            output    = model(tensor)
            pred_idx  = torch.argmax(output, dim=1).item()
            pred_label = short_names[pred_idx]

            true_short = true_label.split("'s ")[-1] if "'s " in true_label else true_label

            axes[i].imshow(img)
            correct = pred_label == true_short
            color   = 'green' if correct else 'red'
            axes[i].set_title(f"True: {true_short}\nPred: {pred_label}", color=color)
            axes[i].axis('off')

        # hide empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

    plt.suptitle('Sample Predictions (Green=Correct, Red=Wrong)', fontsize=14)
    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.show()

plot_predictions(model, r"C:\Users\harsh\Downloads\A,B,CNNS_with_Tim\veggie_heap_testing", transform, dataset.classes)


# ─────────────────────────────────────────────
# 11. FINAL TEST ACCURACY
# ─────────────────────────────────────────────
def evaluate_test(model, test_dir, transform, classes):
    test_dataset = FoodDataset(root_dir=test_dir, transform=transform)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

    correct = 0
    total   = 0

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            outputs   = model(images)
            predicted = torch.argmax(outputs, dim=1)
            correct  += (predicted == labels).sum().item()
            total    += labels.size(0)

    accuracy = correct / total
    print("=" * 40)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print(f"Correct: {correct}/{total}")
    print("=" * 40)

evaluate_test(model, r"C:\Users\harsh\Downloads\A,B,CNNS_with_Tim\veggie_heap_testing", transform, dataset.classes)
