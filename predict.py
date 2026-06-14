import sys
import joblib
import numpy as np
from features import extract_features

def predict(file_path):
    clf = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")

    feat = extract_features(file_path).reshape(1, -1)
    feat_scaled = scaler.transform(feat)

    pred = clf.predict(feat_scaled)[0]
    prob = clf.predict_proba(feat_scaled)[0]

    label = "Deepfake (AI-Generated)" if pred == 1 else "Genuine (Human)"
    confidence = prob[pred]

    print(f"Prediction: {label}")
    print(f"Confidence: {confidence*100:.2f}%")
    return label, confidence

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <audio_file_path>")
        sys.exit(1)
    predict(sys.argv[1])
