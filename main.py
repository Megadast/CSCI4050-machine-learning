"""
main.py
ML project about ASL image recognition
Dataset used: https://www.kaggle.com/datasets/prathumarikeri/american-sign-language-09az/data
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from utils import downloadAslDataset, getDataLoaders, getDevice


class AslClassifier(nn.Module):
    def __init__(self, numClasses: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, numClasses),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def computeAccuracy(logits, targets):
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean()


def trainOneEpoch(model, loader, optimizer, criterion, device):
    model.train()
    lossTotal = 0.0
    accTotal = 0.0

    print("[main] Starting training epoch...")

    for batchIndex, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        lossTotal += loss.item()
        accTotal += (torch.argmax(outputs, dim=1) == labels).float().mean().item()

        if batchIndex % 10 == 0:
            print(
                f"[main] Batch {batchIndex}/{len(loader)} | Loss {loss.item():.4f}",
                end="\r",
                flush=True
            )

    print()  # finish line cleanly
    return lossTotal / len(loader), accTotal / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    lossTotal = 0.0
    accTotal = 0.0
    yTrue = []
    yPred = []

    print("[main] Starting validation...", flush=True)

    with torch.no_grad():
        for batchIndex, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            lossTotal += loss.item()
            accTotal += (torch.argmax(outputs, dim=1) == labels).float().mean().item()

            if batchIndex % 10 == 0:
                print(
                    f"[main] Val batch {batchIndex}/{len(loader)} | Loss {loss.item():.4f}",
                    end="\r",
                    flush=True
                )

            preds = torch.argmax(outputs, dim=1)
            yTrue.extend(labels.cpu().tolist())
            yPred.extend(preds.cpu().tolist())

    print()
    return lossTotal / len(loader), accTotal / len(loader), yTrue, yPred


if __name__ == "__main__":
    device = getDevice()
    print(f"[main] Device: {device}")

    downloadAslDataset()

    trainLoader, valLoader, classNames = getDataLoaders()
    numClasses = len(classNames)

    model = AslClassifier(numClasses).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    maxEpochs = 10

    print(f"[main] Beginning training — total epochs: {maxEpochs}")

    bestValAcc = 0.0

    for epoch in range(maxEpochs):
        print(f"\n[main] === Epoch {epoch+1}/{maxEpochs} ===")

        trainLoss, trainAcc = trainOneEpoch(model, trainLoader, optimizer, criterion, device)
        valLoss, valAcc, yTrue, yPred = evaluate(model, valLoader, criterion, device)

        print(
            f"[main] Epoch {epoch + 1}/{maxEpochs} Complete | "
            f"Train Loss {trainLoss:.4f} Acc {trainAcc:.4f} | "
            f"Val Loss {valLoss:.4f} Acc {valAcc:.4f}"
        )

        if valAcc > bestValAcc:
            bestValAcc = valAcc
            os.makedirs("models", exist_ok=True)
            savePath = os.path.join("models", "asl_best.pth")
            torch.save(model.state_dict(), savePath)
            print(f"[main] Saved best model → {savePath}")

    print("\n[main] Training complete — running prediction tests...\n")
    os.system("python predict.py")