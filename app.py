import streamlit as st
import joblib
import numpy as np
import tempfile
from features import extract_features

st.set_page_config(page_title="Deepfake Audio Detector", page_icon="🎙️")

@st.cache_resource
def load_model():
    clf = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return clf, scaler

clf, scaler = load_model()

st.title("🎙️ Deepfake Audio Detector")
st.write("Upload an audio clip to check whether it's genuine human speech or AI-generated (deepfake).")

uploaded = st.file_uploader(
    "Upload audio file",
    type=["wav", "mp3", "m4a", "mp4", "ogg"]
)

if uploaded is not None:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    st.audio(uploaded)

    with st.spinner("Analyzing audio..."):
        feat = extract_features(tmp_path).reshape(1, -1)
        feat_scaled = scaler.transform(feat)
        pred = clf.predict(feat_scaled)[0]
        prob = clf.predict_proba(feat_scaled)[0]

    label = "🟢 Genuine (Human)" if pred == 0 else "🔴 Deepfake (AI-Generated)"
    confidence = prob[pred] * 100

    st.subheader(f"Result: {label}")
    st.write(f"**Confidence:** {confidence:.2f}%")
    st.progress(int(confidence))
