import os
import json
from PIL import Image
from tqdm import tqdm

RAW_DIR = "data_downloaded"
OUT_DIR = "data"


def ensureDir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def processSplit(splitName):
    splitPath = os.path.join(RAW_DIR, splitName)
    annotPath = os.path.join(splitPath, "_annotations.coco.json")

    if not os.path.isfile(annotPath):
        print(f"[preprocess] No annotation file for {splitName}")
        return

    with open(annotPath, "r") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

    print(f"[preprocess] Processing {splitName}...")

    for ann in tqdm(coco["annotations"]):
        imgInfo = images[ann["image_id"]]
        fileName = imgInfo["file_name"]
        label = categories[ann["category_id"]]

        imgPath = os.path.join(splitPath, fileName)
        if not os.path.isfile(imgPath):
            continue

        img = Image.open(imgPath).convert("RGB")
        x, y, w, h = ann["bbox"]
        x, y, w, h = int(x), int(y), int(w), int(h)

        crop = img.crop((x, y, x + w, y + h))

        classDir = os.path.join(OUT_DIR, label)
        ensureDir(classDir)

        outPath = os.path.join(classDir, f"{splitName}_{ann['id']}.jpg")
        crop.save(outPath)


if __name__ == "__main__":
    ensureDir(OUT_DIR)

    for split in ["train", "valid", "test"]:
        processSplit(split)

    print("[preprocess] Done. Cropped data stored in 'data/'")