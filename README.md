# Deepfake Audio Detection

A machine learning system that classifies speech recordings as either **Genuine (Human)** or **Deepfake (AI-Generated)**, built using hand-engineered audio features and a Random Forest classifier trained from scratch — no pretrained models used.

🔗 **Live App:** https://deepfake-audio-detector-vedant.streamlit.app

---

## Project Overview

This project addresses the problem of detecting AI-generated (synthetic/TTS) speech versus genuine human speech. The system extracts a set of acoustic and spectral features from short audio clips and classifies them using a Random Forest model trained entirely from scratch on labeled data — no pretrained audio embeddings (e.g., wav2vec2) were used at any stage.

---

## Dataset

**Source:** [The Fake-or-Real (FoR) Dataset](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset)

- Used the `for-norm` subset's `training` split.
- Total available: 26,941 real / 26,927 fake clips.
- A balanced subset of **1,200 samples per class (2,400 total)** was used for training and evaluation, sufficient to meet all verification thresholds while keeping the pipeline fast and reproducible.
- 80/20 stratified train/test split.

---

## Preprocessing & Feature Extraction

Each audio clip is processed as follows:

1. Loaded at 16kHz sample rate using `librosa`.
2. Trimmed or zero-padded to a fixed duration of 2 seconds.
3. The following features are extracted and aggregated (mean and standard deviation) per clip:

| Feature | Description |
|---|---|
| MFCC (20 coefficients) | Captures timbral/spectral envelope characteristics |
| Delta-MFCC | First-order temporal derivative of MFCCs |
| Chroma STFT | Pitch class energy distribution |
| Spectral Centroid | "Brightness" of the sound |
| Spectral Bandwidth | Spread of the spectrum around the centroid |
| Spectral Rolloff | Frequency below which most spectral energy is contained |
| Zero-Crossing Rate | Rate of sign changes in the signal — useful for distinguishing noisiness/voicing |
| RMS Energy | Overall signal energy |

Each feature is aggregated using mean and standard deviation (and per-coefficient mean/std for multi-dimensional features like MFCC and chroma), resulting in a **120-dimensional feature vector** per clip.

This feature extraction logic is implemented in `features.py` and is shared across training, inference (`predict.py`), and the Streamlit app (`app.py`) to ensure consistency.

---

## Model Architecture

- **Classifier:** Random Forest (`scikit-learn`), 300 estimators, max depth 20.
- **Scaling:** Features are standardized using `StandardScaler` before being fed to the classifier.
- Trained entirely from scratch on the extracted feature vectors — no pretrained weights or embeddings used at any stage.

---

## Performance Metrics

Evaluated on a held-out 20% test split (480 samples, stratified):

| Metric | Result | Required Threshold |
|---|---|---|
| Overall Accuracy | **91.88%** | ≥ 80% |
| F1 Score | **0.918** | ≥ 80% |
| Genuine Class Accuracy | **92.9%** | ≥ 75% |
| Deepfake Class Accuracy | **90.8%** | ≥ 75% |
| Equal Error Rate (EER) | **8.13%** | ≤ 12% |

### Confusion Matrix

|  | Predicted Genuine | Predicted Deepfake |
|---|---|---|
| **Actual Genuine** | 223 | 17 |
| **Actual Deepfake** | 22 | 218 |

All verification thresholds are met with margin.

---

## Repository Structure
## Repository Structure

deepfake-audio-detection/
├── app.py
├── features.py
├── predict.py
├── model.pkl
├── scaler.pkl
├── requirements.txt
├── README.md
└── Untitled1.ipynb

---

## Usage

### Setup

```bash
pip install -r requirements.txt
```

### Run inference on a new audio file

```bash
python predict.py path/to/audio.wav
```

Output:Prediction: Genuine (Human)

Confidence: 92.10%
### Run the Streamlit app locally

```bash
streamlit run app.py
```

---

## Limitations

- The model is trained on synthetic speech samples from the Fake-or-Real dataset, which was generated using TTS systems available around the dataset's creation (~2019-2020). Generalization to **newer, higher-fidelity TTS and voice-cloning systems** not represented in this dataset may be limited, as the model has learned to detect artifacts characteristic of older synthesis methods.
- Audio that has been heavily compressed (e.g., voice messages sent through messaging apps using low-bitrate codecs like Opus) introduces its own spectral artifacts that were not present in the training data, which can affect prediction accuracy on such inputs.
- The model was trained on 2-second clips; very short or very long inputs are truncated/padded to this length, which may discard relevant information in longer recordings.

These limitations reflect the inherent challenge of generalizing to "previously unseen audio samples" outlined in the problem statement, and would be addressed in future work through training data covering a broader range of TTS systems and audio encoding conditions.

---

## Tech Stack

- Python, librosa (feature extraction)
- scikit-learn (Random Forest, StandardScaler)
- Streamlit (web app)
- Deployed on Streamlit Community Cloud

<img width="3196" height="1996" alt="image" src="https://github.com/user-attachments/assets/79b50753-f2d1-43a1-8d83-f7e3bbaa4c40" />
