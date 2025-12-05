#predict.py

import os
import re
import torch
from PIL import Image
from torchvision import transforms

from utils import getDevice
from main import AslResNet
from hands import MediaPipeHandDetector


MODEL_PATH = "models/asl_best.pth"
TEST_DIR = "test"
OUTPUT_DIR = "test/output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

imagenetTransform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def extractTrueLabel(filename):
    #Extract first number in any filename.
    match = re.search(r"\d+", filename)
    if match:
        return match.group(0)
    return None


def loadModel(classNames):
    device = getDevice()
    numClasses = len(classNames)
    
    model = AslResNet(numClasses)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model, device

detector = MediaPipeHandDetector()


def predictSingle(model, device, imgPath, classNames):
    
    img = Image.open(imgPath).convert("RGB")
    
    #Detect & crop
    croppedHands, annotatedImg = detector.detectHands(img)

    #Save annotated image
    outName = os.path.basename(imgPath)
    savePath = os.path.join(OUTPUT_DIR, f"annotated_{outName}")
    annotatedImg.save(savePath)

    if len(croppedHands) == 0:
        print(f"[predict] No hands found in {imgPath}.")
        return None

    #Classify first detected hand
    handImg = croppedHands[0]

    x = imagenetTransform(handImg).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        predIndex = torch.argmax(logits, dim=1).item()

    return classNames[predIndex]


if __name__ == "__main__":
    #Digits only (For now)
    classNames = [str(i) for i in range(10)]
    model, device = loadModel(classNames)

    if not os.path.isdir(TEST_DIR):
        print(f"[predict] Test folder '{TEST_DIR}' not found.")
        exit()

    files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    if not files:
        print("[predict] No images found in test folder.")
        exit()

    correct = 0
    total = 0

    print(f"[predict] Testing {len(files)} images...\n")

    for file in files:
        imgPath = os.path.join(TEST_DIR, file)

        #Ignore output folder
        if "annotated_" in file:
            continue

        trueLabel = extractTrueLabel(file)
        if trueLabel is None:
            print(f"[predict] WARNING: No numeric label in filename → {file}")
            continue

        pred = predictSingle(model, device, imgPath, classNames)

        print(f"Image: {file} | True: {trueLabel} | Predicted: {pred}")

        if pred == trueLabel:
            correct += 1
        total += 1

    if total > 0:
        print("\n[predict] Done.")
        acc = (correct / total) * 100
        print(f"[predict] Accuracy: {correct}/{total} ({acc:.2f}%)")
        print(f"[predict] Annotated images saved in: {OUTPUT_DIR}/")