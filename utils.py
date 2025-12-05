#utils.py
import os
import shutil
from dotenv import load_dotenv
from roboflow import Roboflow
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch

DEFAULT_DOWNLOADED = "data_downloaded"
DEFAULT_DATA_DIR = "data"


def downloadAslDataset():
    load_dotenv()
    apiKey = os.getenv("ROBOFLOW_API_KEY")
    workspace = os.getenv("ROBOFLOW_WORKSPACE")
    projectName = os.getenv("ROBOFLOW_PROJECT")
    versionNum = int(os.getenv("ROBOFLOW_VERSION", "1"))

    if apiKey is None:
        raise RuntimeError("ROBOFLOW_API_KEY missing in .env")

    if os.path.isdir(DEFAULT_DOWNLOADED):
        print("[utils] Dataset already downloaded")
        return

    print("[utils] Downloading dataset from Roboflow...")

    rf = Roboflow(api_key=apiKey)
    project = rf.workspace(workspace).project(projectName)
    version = project.version(versionNum)

    ds = version.download("coco")

    # Rename folder to data_downloaded/
    folderName = ds.location
    if os.path.isdir(folderName):
        os.rename(folderName, DEFAULT_DOWNLOADED)

    print("[utils] Download complete →", DEFAULT_DOWNLOADED)


def getDataLoaders(
    dataDir: str = DEFAULT_DATA_DIR,
    batchSize: int = 32,
):
    if not os.path.isdir(dataDir):
        raise RuntimeError(f"Dataset not found. Run preprocess_coco.py first.")

    print(f"[utils] Loading dataset from: {dataDir}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = datasets.ImageFolder(dataDir, transform=transform)
    classNames = dataset.classes

    print(f"[utils] Classes found: {classNames}")

    total = len(dataset)
    valSize = int(total * 0.2)
    trainSize = total - valSize

    print(f"[utils] Train: {trainSize}, Validation: {valSize}")

    trainSet, valSet = random_split(
        dataset,
        [trainSize, valSize],
        generator=torch.Generator().manual_seed(42),
    )

    trainLoader = DataLoader(trainSet, batch_size=batchSize, shuffle=True, num_workers=0)
    valLoader = DataLoader(valSet, batch_size=batchSize, shuffle=False, num_workers=0)

    return trainLoader, valLoader, classNames


def getDevice():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")