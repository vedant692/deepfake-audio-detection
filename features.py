import librosa
import numpy as np

def extract_features(file_path, sr=16000, duration=2.0):
    y, sr = librosa.load(file_path, sr=sr, duration=duration)
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    delta_mfcc = librosa.feature.delta(mfcc)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    feats = []
    for f in [mfcc, delta_mfcc, chroma, spec_cent, spec_bw, rolloff, zcr, rms]:
        feats.append(f.mean())
        feats.append(f.std())
        if f.shape[0] > 1:
            feats.extend(f.mean(axis=1))
            feats.extend(f.std(axis=1))
    return np.array(feats, dtype=np.float32)
