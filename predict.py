#predict.py

import os
import json
import shutil
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from utils import getDevice, getDataLoaders
from main import AslResNet
from hands import MediaPipeHandDetector


RAW_TEST_DIR = "data_downloaded/test"
ANNOT_FILE = os.path.join(RAW_TEST_DIR, "_annotations.coco.json")

COCO_OUTPUT_DIR = "test_output"
REAL_INPUT_DIR = "real"
REAL_OUTPUT_DIR = "real/output"

os.makedirs(COCO_OUTPUT_DIR, exist_ok=True)
os.makedirs(REAL_OUTPUT_DIR, exist_ok=True)

imagenetTransform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def getTrueLabelFromFilename(name):
    #Extracts first number or letter sequence
    base = os.path.splitext(name)[0]

    #If format is test_A or test_hello
    parts = base.split("_")
    if len(parts) > 1:
        return parts[1]

    #If filename is "hello.jpg"
    return parts[0]

def loadModel():
    _, _, classNames = getDataLoaders()
    device = getDevice()
    numClasses = len(classNames)

    model = AslResNet(numClasses)
    model.load_state_dict(torch.load("models/asl_best.pth", map_location=device))
    model.to(device)
    model.eval()
    return model, device, classNames

detector = MediaPipeHandDetector()

#Image prediction
def predictImage(model, device, img, classNames):
    x = imagenetTransform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        idx = torch.argmax(logits, dim=1).item()
    return classNames[idx]

#COCO dataset
def predictCOCO(model, device, classNames):
    print("\n[predict] Running COCO testset evaluation...")

    with open(ANNOT_FILE, "r") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

    correct = 0
    total = 0

    for ann in tqdm(coco["annotations"]):
        imgInfo = images[ann["image_id"]]
        fileName = imgInfo["file_name"]
        trueLabel = categories[ann["category_id"]]

        srcPath = os.path.join(RAW_TEST_DIR, fileName)
        if not os.path.isfile(srcPath):
            continue

        img = Image.open(srcPath).convert("RGB")

        x, y, w, h = ann["bbox"]
        x, y, w, h = int(x), int(y), int(w), int(h)
        crop = img.crop((x, y, x + w, y + h))

        predicted = predictImage(model, device, crop, classNames)
        total += 1
        if predicted == trueLabel:
            correct += 1

        #Move crop into folder named by predicted class
        classDir = os.path.join(COCO_OUTPUT_DIR, predicted)
        os.makedirs(classDir, exist_ok=True)

        crop.save(os.path.join(classDir, fileName))

    acc = (correct / total * 100) if total > 0 else 0
    print(f"[COCO] Accuracy: {correct}/{total} ({acc:.2f}%)")
    print(f"[COCO] Cropped predictions organized inside: {COCO_OUTPUT_DIR}/")
    
#REAL dataset
def predictReal(model, device, classNames):
    if not os.path.isdir(REAL_INPUT_DIR):
        print("[real] No real/ folder found. Skipping real-world testing.")
        return

    print("\n[predict] Running real-world evaluation")
    
    #TEMP
    print("[predict] TODO: FIX THIS, gives annotation, but prediction is wrong.\n")

    files = [f for f in os.listdir(REAL_INPUT_DIR)
             if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    for file in files:
        path = os.path.join(REAL_INPUT_DIR, file)
        img = Image.open(path).convert("RGB")

        # MediaPipe detection (for skeleton/box only)
        croppedHands, annotatedImg = detector.detectHands(img)

        # Always save annotated image (even if detection fails)
        annotatedSavePath = os.path.join(REAL_OUTPUT_DIR, f"annotated_{file}")
        annotatedImg.save(annotatedSavePath)

        if len(croppedHands) == 0:
            print(f"[real] {file} | No hand detected | Output saved: annotated_{file}")
            continue

        # Use first detected hand for prediction
        crop = croppedHands[0]
        predicted = predictImage(model, device, crop, classNames)
        trueLabel = getTrueLabelFromFilename(file)

    print(f"[real] Annotated images saved in: {REAL_OUTPUT_DIR}/")
    
if __name__ == "__main__":
    model, device, classNames = loadModel()

    predictCOCO(model, device, classNames)
    predictReal(model, device, classNames)

    print("\n[predict] All evaluations complete.")