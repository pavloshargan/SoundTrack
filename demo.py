# demo.py
# 1) extract WAV from the given video with ffmpeg via subprocess
# 2) run the model on sliding 3 s windows
# 3) save Plotly HTML with cadence and GCT predictions
# Notes
# - uses hardcoded label stats from training: cadence mean=172.11 std=13.47, GCT mean=273.86 std=47.47
# - applies simple 3-point smoothing to match validation
# - adds light audio pre-processing to reduce domain shift

import os
import sys
import subprocess
import numpy as np
import librosa
import torch
from plotly.offline import plot as plotly_plot
import plotly.graph_objects as go

import model as mdl

# ------------- label stats from training -------------
CAD_MEAN = 172.11
CAD_STD = 13.47
GCT_MEAN = 273.86
GCT_STD = 47.47
# -----------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")


def ffmpeg_extract_audio(video_path, wav_out):
    cmd = ["ffmpeg", "-i", video_path, "-vn", "-ac", "1", "-ar", str(mdl.SR), "-y", wav_out, "-hide_banner", "-loglevel", "error"]
    subprocess.run(cmd, check=True)


def frame_audio(y, win_sec, hop_sec, sr):
    win = int(win_sec * sr)
    hop = int(hop_sec * sr)
    if y.size < win:
        return np.empty((0, win), dtype=np.float32)
    frames = librosa.util.frame(y, frame_length=win, hop_length=hop, axis=0)
    return frames


def pre_emphasis(x, coeff=0.97):
    if len(x) == 0:
        return x
    y = np.empty_like(x, dtype=np.float32)
    y[0] = x[0]
    y[1:] = x[1:] - coeff * x[:-1]
    return y


def rms_normalize(x, target_rms=0.03, eps=1e-8):
    rms = np.sqrt(np.mean(x.astype(np.float64) ** 2) + eps)
    gain = target_rms / max(rms, eps)
    y = x * gain
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def highpass_biquad(x, sr, fc=30.0, q=0.707):
    import math
    if len(x) == 0:
        return x
    w0 = 2.0 * math.pi * fc / sr
    alpha = math.sin(w0) / (2.0 * q)
    cosw = math.cos(w0)
    b0 = (1 + cosw) / 2
    b1 = -(1 + cosw)
    b2 = (1 + cosw) / 2
    a0 = 1 + alpha
    a1 = -2 * cosw
    a2 = 1 - alpha
    b0 /= a0
    b1 /= a0
    b2 /= a0
    a1 /= a0
    a2 /= a0
    y = np.zeros_like(x, dtype=np.float32)
    x1 = 0.0
    x2 = 0.0
    y1 = 0.0
    y2 = 0.0
    for i in range(len(x)):
        xi = float(x[i])
        yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        y[i] = yi
        x2 = x1
        x1 = xi
        y2 = y1
        y1 = yi
    return y


def roll3(x):
    if len(x) < 2:
        return x
    y = x.copy()
    y[0] = (x[0] + x[1]) / 2.0
    for i in range(1, len(x) - 1):
        y[i] = (x[i - 1] + x[i] + x[i + 1]) / 3.0
    y[-1] = (x[-2] + x[-1]) / 2.0
    return y


def main():
    if len(sys.argv) < 2:
        print("Usage: python demo.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print(f"Error: video not found: {video_path}")
        sys.exit(1)

    extracted_wav = "demo_audio.wav"
    checkpoint_path = "best_2.172_25.6.pt"
    plot_html = "demo_results.html"

    print(f"Extracting audio from {video_path} ...")
    ffmpeg_extract_audio(video_path, extracted_wav)
    print(f"WAV saved to {extracted_wav}")

    print("Loading audio...")
    y, _ = librosa.load(extracted_wav, sr=mdl.SR, mono=True)
    y = mdl.ensure_finite(y)
    y = highpass_biquad(y, mdl.SR, fc=30.0)
    y = pre_emphasis(y, 0.97)
    y = rms_normalize(y, target_rms=0.03)

    windows = frame_audio(y, mdl.WIN_SEC, mdl.HOP_SEC, mdl.SR)
    if windows.shape[0] == 0:
        raise RuntimeError("Audio shorter than window size. Nothing to process.")
    print(f"Creating {windows.shape[0]} windows of {mdl.WIN_SEC} s with {mdl.HOP_SEC} s hop")

    print("Computing features...")
    feats = [mdl.make_features(seg) for seg in windows]
    X = np.stack(feats, axis=0)

    print("Loading model...")
    net = mdl.MTL_MLP().to(DEVICE)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    net.load_state_dict(state_dict)
    net.eval()

    print("Running inference...")
    with torch.no_grad():
        xb = torch.from_numpy(X).to(DEVICE)
        pc_norm, pg_norm = net(xb)
        pc_norm = pc_norm.cpu().numpy()
        pg_norm = pg_norm.cpu().numpy()

    pc_norm = roll3(pc_norm)
    pg_norm = roll3(pg_norm)
    pc = pc_norm * CAD_STD + CAD_MEAN
    pg = pg_norm * GCT_STD + GCT_MEAN

    hop = int(mdl.HOP_SEC * mdl.SR)
    win = int(mdl.WIN_SEC * mdl.SR)
    centers_sec = (np.arange(len(pc)) * hop + win // 2) / mdl.SR

    print("Building Plotly figure...")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=centers_sec, y=pc, mode="lines+markers", name="Model cadence spm"))
    fig.add_trace(go.Scatter(x=centers_sec, y=pg, mode="lines+markers", name="Model GCT ms", yaxis="y2"))

    fig.update_layout(
        title="Model predictions over time",
        xaxis=dict(title="Time s"),
        yaxis=dict(title="Cadence steps per min"),
        yaxis2=dict(title="GCT ms", overlaying="y", side="right"),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        height=560,
        margin=dict(l=60, r=60, t=60, b=50)
    )

    plotly_plot(fig, filename=plot_html, auto_open=False)
    print(f"Saved HTML plot to {plot_html}")


if __name__ == "__main__":
    main()
