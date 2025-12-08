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
    def __init__(self, maxHands=2,
        detectionConfidence=0.5,
        trackingConfidence=0.5,
        static_image_mode=True):
        # Allow caller to choose static_image_mode (False is better for video)
        self.hands = mpHands.Hands(
            static_image_mode=static_image_mode,
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

            #Compute bounding box from landmarks
            xCoords = [lm.x * w for lm in handLms.landmark]
            yCoords = [lm.y * h for lm in handLms.landmark]

            xMin, xMax = int(min(xCoords)), int(max(xCoords))
            yMin, yMax = int(min(yCoords)), int(max(yCoords))

            # Improved padding: make crop square-ish and larger to capture full hand
            hand_width = xMax - xMin
            hand_height = yMax - yMin
            max_dim = max(hand_width, hand_height)
            
            # Add 30% padding around hand for context
            pad = int(0.3 * max_dim)
            
            # Center the hand in the padding
            center_x = (xMin + xMax) // 2
            center_y = (yMin + yMax) // 2
            half_size = (max_dim + 2 * pad) // 2
            
            xMin = max(0, center_x - half_size)
            xMax = min(w, center_x + half_size)
            yMin = max(0, center_y - half_size)
            yMax = min(h, center_y + half_size)
            
            # Ensure we have a valid crop
            if xMax <= xMin or yMax <= yMin:
                continue

            #Draw bounding box
            cv2.rectangle(
                annotatedImg,
                (xMin, yMin),
                (xMax, yMax),
                (0, 255, 0),
                2
            )

            #Crop the hand region
            crop = img[yMin:yMax, xMin:xMax]
            croppedHands.append(Image.fromarray(crop))

        return croppedHands, Image.fromarray(annotatedImg)