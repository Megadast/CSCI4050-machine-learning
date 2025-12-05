#utils.py

import os
import subprocess
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from dotenv import load_dotenv

DATASET_SLUG = "prathumarikeri/american-sign-language-09az"
DEFAULT_DATA_DIR = "data/asl"


def ensureKaggleCredentials():
    load_dotenv()
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    apiToken = os.getenv("KAGGLE_API_TOKEN")

    if key is None and apiToken is not None:
        key = apiToken
        os.environ["KAGGLE_KEY"] = key

    if username is None:
        raise RuntimeError("KAGGLE_USERNAME missing in .env")
    if key is None:
        raise RuntimeError("KAGGLE_KEY or KAGGLE_API_TOKEN missing in .env")

    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key


def downloadAslDataset(dataDir: str = DEFAULT_DATA_DIR):
    ensureKaggleCredentials()
    os.makedirs(dataDir, exist_ok=True)

    if any(os.scandir(dataDir)):
        print(f"[utils] Dataset already exists in {dataDir}")
        return

    print(f"[utils] Downloading dataset into {dataDir}")

    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        DATASET_SLUG,
        "-p",
        dataDir,
        "--unzip",
    ]

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        raise RuntimeError("Kaggle CLI not installed")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Kaggle download failed: {e}")

    print("[utils] Download completed")

def getDataLoaders(
    dataDir: str = DEFAULT_DATA_DIR,
    batchSize: int = 32,
):
    print(f"[utils] Checking dataset folder: {dataDir}")

    rootCandidates = os.listdir(dataDir)
    if len(rootCandidates) == 1:
        potentialPath = os.path.join(dataDir, rootCandidates[0])
        if os.path.isdir(potentialPath):
            dataDir = potentialPath

    print(f"[utils] Dataset root: {dataDir}")
    print("[utils] Filtering to digits 0–9 only...")

    digitFolders = [str(i) for i in range(10)]

    filteredRoot = os.path.join(dataDir, "_digits_only")
    os.makedirs(filteredRoot, exist_ok=True)

    for d in digitFolders:
        src = os.path.join(dataDir, d)
        dst = os.path.join(filteredRoot, d)
        if os.path.isdir(src):
            if not os.path.isdir(dst):
                os.makedirs(dst)
            for fname in os.listdir(src):
                srcFile = os.path.join(src, fname)
                dstFile = os.path.join(dst, fname)
                if not os.path.exists(dstFile):
                    try:
                        import shutil
                        shutil.copy(srcFile, dstFile)
                    except:
                        pass

    print("[utils] Digits-only subset prepared.")
    
    print("[utils] Preparing transforms...")
    
    imagenetTransform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    print("[utils] Loading ImageFolder (digits only)...")
    fullDataset = datasets.ImageFolder(filteredRoot, transform=imagenetTransform)

    classNames = fullDataset.classes
    print(f"[utils] Classes included: {classNames}")

    totalSize = len(fullDataset)
    valSize = int(totalSize * 0.2)
    trainSize = totalSize - valSize

    print(f"[utils] Splitting dataset → Train: {trainSize}, Val: {valSize}")

    trainDataset, valDataset = random_split(
        fullDataset,
        [trainSize, valSize],
        generator=torch.Generator().manual_seed(42)
    )

    print("[utils] Split complete.")
    print(f"[utils] Train size: {len(trainDataset)}")
    print(f"[utils] Val size: {len(valDataset)}")

    print("[utils] Creating DataLoaders...")
    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, num_workers=0)
    valLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=False, num_workers=0)

    print("[utils] DataLoaders ready.")

    return trainLoader, valLoader, classNames


def getDevice() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")