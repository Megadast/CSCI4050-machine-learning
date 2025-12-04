#predict.py

import os
import torch
from PIL import Image
from torchvision import transforms

from utils import getDevice
from main import AslClassifier
from hands import simpleHandCrop


MODEL_PATH = "models/asl_best.pth"
TEST_DIR = "test"


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def loadModel(classNames):
    device = getDevice()
    numClasses = len(classNames)

    model = AslClassifier(numClasses)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model, device


def predictSingle(model, device, imgPath, classNames):
    img = Image.open(imgPath).convert("RGB")

    cropped = simpleHandCrop(img)
    
    x = transform(cropped).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        predIndex = torch.argmax(logits, dim=1).item()

    return classNames[predIndex]


if __name__ == "__main__":
    classNames = [str(i) for i in range(10)]  #only digits 0–9
    model, device = loadModel(classNames)

    if not os.path.isdir(TEST_DIR):
        print(f"[predict] Test folder '{TEST_DIR}' not found.")

    files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    if not files:
        print("[predict] No images found in test folder.")

    correct = 0
    total = 0

    print(f"[predict] Testing {len(files)} images...\n")

    for file in files:
        imgPath = os.path.join(TEST_DIR, file)

        #Extract ground truth label from file name: test_0.jpg → "0"
        trueLabel = file.split("_")[-1].split(".")[0]

        pred = predictSingle(model, device, imgPath, classNames)

        print(f"Image: {file} | True: {trueLabel} | Predicted: {pred}")

        if pred == trueLabel:
            correct += 1
        total += 1

    print("\n[predict] Done.")
    print(f"[predict] Accuracy: {correct}/{total} ({(correct/total)*100:.2f}%)")