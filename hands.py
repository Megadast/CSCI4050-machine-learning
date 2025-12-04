#hands.py

import cv2
import mediapipe as mp
from PIL import Image
import numpy as np

mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils

'''
Utilizing: https://mediapipe.readthedocs.io/en/latest/solutions/hands.html
'''
class MediaPipeHandDetector:
    def __init__(self, maxHands=2, detectionConfidence=0.5, trackingConfidence=0.5):
        self.hands = mpHands.Hands(
            static_image_mode=True,
            max_num_hands=maxHands,
            min_detection_confidence=detectionConfidence,
            min_tracking_confidence=trackingConfidence
        )

    def detectHands(self, pilImage):
        """
        Detects hands and returns:
        - croppedHands: list of PIL images (cropped hand regions)
        - annotatedImg: annotated image with bounding boxes + skeleton
        """
        img = np.array(pilImage)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        results = self.hands.process(imgRGB)

        annotatedImg = img.copy()
        croppedHands = []

        if not results.multi_hand_landmarks:
            return [], Image.fromarray(annotatedImg)

        h, w, _ = img.shape

        for handLms in results.multi_hand_landmarks:
            
            #Draw landmarks on the annotated image
            mpDraw.draw_landmarks(annotatedImg, handLms, mpHands.HAND_CONNECTIONS)

            #Compute bounding box
            x_coords = [lm.x * w for lm in handLms.landmark]
            y_coords = [lm.y * h for lm in handLms.landmark]

            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))

            #Add padding
            pad = int(0.1 * (x_max - x_min))
            x_min = max(0, x_min - pad)
            x_max = min(w, x_max + pad)
            y_min = max(0, y_min - pad)
            y_max = min(h, y_max + pad)

            #Draw bounding box
            cv2.rectangle(
                annotatedImg,
                (x_min, y_min),
                (x_max, y_max),
                (0, 255, 0),
                2
            )

            #Crop the hand region
            crop = img[y_min:y_max, x_min:x_max]
            croppedHands.append(Image.fromarray(crop))

        return croppedHands, Image.fromarray(annotatedImg)