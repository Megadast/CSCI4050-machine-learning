"""
main.py
ML project about ASL image recognition
Dataset used: https://www.kaggle.com/datasets/prathumarikeri/american-sign-language-09az/data
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms

from utils import downloadAslDataset, getDataLoaders, getDevice


class AslResNet(nn.Module):
    def __init__(self, numClasses):
        super().__init__()
        self.model = models.resnet18(weights="IMAGENET1K_V1")

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.layer4.parameters():
            param.requires_grad = True

        inFeatures = self.model.fc.in_features
        self.model.fc = nn.Linear(inFeatures, numClasses)

    def forward(self, x):
        return self.model(x)


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
        accTotal += computeAccuracy(outputs, labels).item()

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
            accTotal += computeAccuracy(outputs, labels).item()

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

    #STEP 1 — Download if not present
    downloadAslDataset()

    #STEP 2 — check cropped otherwise run preprocessing
    if not os.path.isdir("data"):
        print("[main] Cropped dataset not found. Running preprocess_coco.py...")
        os.system("python preprocess_coco.py")
    else:
        print("[main] Cropped dataset found. Skipping preprocessing.")

    #STEP 3 — Load data
    trainLoader, valLoader, classNames = getDataLoaders()
    numClasses = len(classNames)

    model = AslResNet(numClasses).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )

    maxEpochs = 6
    print(f"[main] Beginning training — epochs: {maxEpochs}")

    bestValAcc = 0

    for epoch in range(maxEpochs):
        print(f"\n[main] === Epoch {epoch+1}/{maxEpochs} ===")

        trainLoss, trainAcc = trainOneEpoch(model, trainLoader, optimizer, criterion, device)
        valLoss, valAcc, yTrue, yPred = evaluate(model, valLoader, criterion, device)

        print(
            f"[main] Epoch {epoch+1} | "
            f"Train Loss {trainLoss:.4f} Acc {trainAcc:.4f} | "
            f"Val Loss {valLoss:.4f} Acc {valAcc:.4f}"
        )

        if valAcc > bestValAcc:
            bestValAcc = valAcc
            os.makedirs("models", exist_ok=True)
            savePath = "models/asl_best.pth"
            torch.save(model.state_dict(), savePath)
            print(f"[main] Saved best model → {savePath}")

    print("\n[main] Training complete — running prediction tests...\n")
    os.system("python predict.py")