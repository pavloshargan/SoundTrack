# model.py
# Shared model and feature pipeline
# All constants and functions mirror the training setup

import numpy as np
import librosa
import torch
import torch.nn as nn

# Audio and feature settings
SR = 16000
WIN_SEC = 3.0
HOP_SEC = 1.0

N_MFCC = 40
N_FFT = 1024
HOP_LENGTH = 160
T_TARGET = 300
FEATURE_SHAPE = (N_MFCC + 1, T_TARGET)

def ensure_finite(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

def mvn(x: np.ndarray) -> np.ndarray:
    mu = float(x.mean())
    std = float(x.std()) + 1e-8
    return (x - mu) / std

def make_features(wav_seg: np.ndarray) -> np.ndarray:
    from scipy.signal import wiener as _wiener
    x = mvn(ensure_finite(wav_seg))
    try:
        x_w = _wiener(x)
    except Exception:
        x_w = librosa.effects.preemphasis(x, coef=0.97)
    x_w = ensure_finite(x_w)
    if not np.all(np.isfinite(x_w)) or np.max(np.abs(x_w)) == 0.0:
        x_w = librosa.effects.preemphasis(x, coef=0.97)
    if x_w.size:
        x_w = x_w + 1e-6 * np.random.randn(*x_w.shape).astype(np.float32)
    x_w = ensure_finite(x_w)

    mfcc = librosa.feature.mfcc(
        y=x_w, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH,
        htk=True, center=True
    )
    rms = librosa.feature.rms(
        y=x_w, frame_length=N_FFT, hop_length=HOP_LENGTH, center=True
    )
    feat = np.concatenate([mfcc, rms], axis=0)
    feat = librosa.util.fix_length(feat, size=T_TARGET, axis=1)
    assert feat.shape == FEATURE_SHAPE, f"Feature shape mismatch. Expected {FEATURE_SHAPE}, got {feat.shape}"
    return ensure_finite(feat)

class MTL_MLP(nn.Module):
    def __init__(self, f: int = N_MFCC + 1, t: int = T_TARGET):
        super().__init__()
        in_dim = f * t
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU()
        )
        self.head_cad = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1)
        )
        self.head_gct = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor):
        z = x.view(x.shape[0], -1)
        h = self.trunk(z)
        return self.head_cad(h).squeeze(1), self.head_gct(h).squeeze(1)
