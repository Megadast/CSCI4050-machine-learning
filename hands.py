import cv2
import numpy as np
from PIL import Image


def simpleHandCrop(pilImage):

    img = np.array(pilImage)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    #Loose skin tone range (works for proof of concept)
    lower = np.array([0, 20, 40])
    upper = np.array([25, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    #Remove noise
    mask = cv2.medianBlur(mask, 7)

    #Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print("[crop] No hand detected — using original image.")
        return pilImage

    #Largest contour = hand
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)

    #Expand bounding box slightly
    pad = int(min(w, h) * 0.2)
    x = max(x - pad, 0)
    y = max(y - pad, 0)
    w = min(w + 2*pad, img.shape[1] - x)
    h = min(h + 2*pad, img.shape[0] - y)

    crop = img[y:y+h, x:x+w]
    return Image.fromarray(crop)