
# Run real-time ASL predictions from a webcam. Press 'q' to quit the window.
import argparse
import time
import json
import cv2
import torch
import numpy as np
from PIL import Image

import predict
from hands import MediaPipeHandDetector
from main import AslResNet


def load_class_names(path):
    if path is None:
        return None
    if path.lower().endswith('.json'):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            # support object with classes key
            return data.get('classes') or data.get('class_names') or list(data)
    else:
        # plain text, one class per line
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f.readlines() if line.strip()]


def build_model_from_names(model_path, class_names, device):
    model = AslResNet(len(class_names))
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def preprocess_hand_crop(crop_pil, transform):
    """
    Apply the standard ImageNet transform to a cropped hand PIL image.
    Ensures RGB conversion and proper tensor format.
    """
    if crop_pil.mode != 'RGB':
        crop_pil = crop_pil.convert('RGB')
    x = transform(crop_pil).unsqueeze(0)
    return x


def predict_hand(model, device, crop_pil, class_names, transform):
    """
    Predict ASL class from a PIL cropped hand image.
    Returns (class_name, confidence_score).
    """
    x = preprocess_hand_crop(crop_pil, transform).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        idx = torch.argmax(probs).item()
        conf = probs[idx].item()
        label = class_names[idx]
    return label, conf, probs


def get_top_predictions(probs, class_names, k=5):
    """
    Get top-k predictions with confidence scores.
    Returns list of (class_name, confidence) tuples.
    """
    top_probs, top_indices = torch.topk(probs, k=min(k, len(class_names)))
    return [(class_names[idx], prob.item()) for idx, prob in zip(top_indices, top_probs)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--model', type=str, default='models/asl_best.pth')
    parser.add_argument('--class-names', type=str, default=None,
                        help='Optional path to class names (json or txt)')
    parser.add_argument('--max-hands', type=int, default=1)
    parser.add_argument('--detection-confidence', type=float, default=0.5)
    parser.add_argument('--tracking-confidence', type=float, default=0.5)
    parser.add_argument('--debug', action='store_true',
                        help='Show cropped hand regions for debugging')
    parser.add_argument('--confidence-threshold', type=float, default=0.3,
                        help='Only show predictions above this confidence')
    parser.add_argument('--show-top-k', type=int, default=5,
                        help='Show top-k predictions in debug mode')
    args = parser.parse_args()

    # Attempt to reuse predict.loadModel() which provides the same transforms.
    try:
        model, device, classNames = predict.loadModel()
        transform = predict.imagenetTransform
    except Exception as e:
        # Fallback: allow user to provide class names file and load model manually
        print('[live] Warning: predict.loadModel() failed:', e)
        if args.class_names is None:
            print('[live] Provide --class-names or ensure dataset exists (data/). Exiting.')
            return
        classNames = load_class_names(args.class_names)
        if classNames is None:
            print('[live] Could not load class names. Exiting.')
            return
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = build_model_from_names(args.model, classNames, device)
        # Use the same transform as predict.py
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    print(f'[live] Device: {device}')
    print(f'[live] Classes: {classNames}')

    # Create detector optimized for video
    detector = MediaPipeHandDetector(maxHands=args.max_hands,
                                     detectionConfidence=args.detection_confidence,
                                     trackingConfidence=args.tracking_confidence,
                                     static_image_mode=False)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f'[live] Could not open camera index {args.camera}')
        return

    print('[live] Starting webcam. Press q to quit.')
    prev_time = time.time()
    frame_count = 0
    last_predictions = []  # Keep track of last N predictions for smoothing

    while True:
        ret, frame = cap.read()
        if not ret:
            print('[live] Failed to read frame. Exiting.')
            break

        frame_count += 1

        # Optionally resize for speed
        h, w = frame.shape[:2]
        if max(h, w) > 960:
            scale = 960.0 / max(h, w)
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        croppedHands, annotated = detector.detectHands(pil)

        annotated_np = np.array(annotated)
        annotated_bgr = cv2.cvtColor(annotated_np, cv2.COLOR_RGB2BGR)

        label_text = ''
        if len(croppedHands) > 0:
            crop = croppedHands[0]
            try:
                label, conf, probs = predict_hand(model, device, crop, classNames, transform)
                
                # Only display if confidence is above threshold
                if conf >= args.confidence_threshold:
                    label_text = f'{label} ({conf*100:.1f}%)'
                    last_predictions.append(label)
                    # Keep only last 10 predictions for smoothing
                    if len(last_predictions) > 10:
                        last_predictions.pop(0)
                    
                    # Show detailed prediction info in debug mode
                    if args.debug and frame_count % 10 == 0:
                        top_k = get_top_predictions(probs, classNames, k=args.show_top_k)
                        common = max(set(last_predictions), key=last_predictions.count) if last_predictions else label
                        print(f'\n[Frame {frame_count}] Top-{args.show_top_k} predictions:')
                        for i, (cls, prob) in enumerate(top_k, 1):
                            print(f'  {i}. {cls:15s} {prob*100:5.1f}%')
                        print(f'  Trend (last 10): {common}')
                else:
                    label_text = f'Low conf: {label} ({conf*100:.1f}%)'
                    if args.debug and frame_count % 10 == 0:
                        top_k = get_top_predictions(probs, classNames, k=3)
                        print(f'[Frame {frame_count}] Low confidence - Top-3: {", ".join([f"{cls} ({p*100:.0f}%)" for cls, p in top_k])}')
            except Exception as e:
                print(f'[live] Prediction error: {e}')
                label_text = 'ERROR'

        # Overlay label and FPS
        if label_text:
            cv2.putText(annotated_bgr, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 0), 2, cv2.LINE_AA)

        now = time.time()
        fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
        prev_time = now
        cv2.putText(annotated_bgr, f'FPS: {fps:.1f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('ASL Live', annotated_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f'[live] Processed {frame_count} frames.')


if __name__ == '__main__':
    main()
