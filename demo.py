import os
import sys
import subprocess
import numpy as np
import pandas as pd
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
    x1 = x2 = y1 = y2 = 0.0
    for i in range(len(x)):
        xi = float(x[i])
        yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        y[i] = yi
        x2, x1, y2, y1 = x1, xi, y1, yi
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


def smooth_series(x, method="gaussian", win=7, sigma=1.5):
    """
    Simple 1D smoothing for small numeric arrays.
    method: "mean" | "median" | "gaussian"
    win: odd window length
    sigma: std for gaussian in index units
    """
    import numpy as np

    if x is None or len(x) == 0:
        return x
    win = int(win)
    if win < 3 or win % 2 == 0:
        win = 3
    n = len(x)
    pad = win // 2

    if method == "median":
        xp = np.pad(x, (pad, pad), mode="reflect")
        out = np.empty_like(xp, dtype=float)
        for i in range(n):
            out[i + pad] = np.median(xp[i:i + win])
        return out[pad:pad + n].astype(x.dtype)

    if method == "mean":
        k = np.ones(win, dtype=float) / win
        return np.convolve(x, k, mode="same").astype(x.dtype)

    idx = np.arange(win) - pad
    w = np.exp(-0.5 * (idx / float(sigma))**2)
    w /= np.sum(w)
    return np.convolve(x, w, mode="same").astype(x.dtype)


def load_step_annotations(csv_path):
    """
    Returns sorted numpy array of step times in seconds if the file exists.
    """
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)

    time_col = "video_time" if "video_time" in df.columns else ("time" if "time" in df.columns else None)
    if time_col is None:
        print(f"Warning: {csv_path} missing 'time' or 'video_time'. Skipping ground truth.")
        return None
    ts = np.asarray(df[time_col].values, dtype=float)
    ts = ts[np.isfinite(ts)]
    ts.sort()
    return ts


def cadence_from_events(event_times):
    """
    Instantaneous cadence from neighbor step differences.
    For consecutive times t_i, t_{i+1}:
      dt = t_{i+1} - t_i
      cadence = round(60 / dt) steps per minute
    Returns midpoints of intervals and the integer cadence values.
    """
    if event_times is None or len(event_times) < 2:
        return None, None
    diffs = np.diff(event_times)
    mids = (event_times[:-1] + event_times[1:]) / 2.0
    mask = diffs > 0
    if not np.any(mask):
        return None, None
    diffs = diffs[mask]
    mids = mids[mask]
    cad = np.round(60.0 / diffs).astype(int)
    return mids, cad


def main():
    # args: input media optional, annotation csv optional
    default_media = "RunningExample.wav"
    default_anno = "StepTimesAnnotation.csv"

    input_path = sys.argv[1] if len(sys.argv) > 1 else default_media
    anno_path = sys.argv[2] if len(sys.argv) > 2 else default_anno

    if not os.path.exists(input_path):
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    extracted_wav = "demo_audio.wav"
    checkpoint_path = "best_2.172_25.6.pt"
    plot_html = "demo_results.html"

    # audio source
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".wav":
        print(f"Using provided WAV file: {input_path}")
        extracted_wav = input_path
    else:
        print(f"Extracting audio from {input_path} ...")
        ffmpeg_extract_audio(input_path, extracted_wav)
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

    # ground truth cadence from neighbor step differences
    gt_times = load_step_annotations(anno_path) if os.path.exists(anno_path) else None
    gt_centers, gt_cad = cadence_from_events(gt_times) if gt_times is not None else (None, None)
    if gt_cad is None:
        print(f"No usable ground truth found at {anno_path}. Plot will contain model predictions only.")
        gt_cad_sm = None
    else:
        print(f"Loaded {len(gt_times)} step times from {anno_path}. Computed instantaneous GT cadence.")
        gt_cad_sm = smooth_series(gt_cad, method="gaussian", win=7, sigma=1.5)

    print("Building Plotly figure...")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=centers_sec, y=pc, mode="lines+markers", name="Model cadence spm"))
    if gt_cad_sm is not None:
        fig.add_trace(go.Scatter(x=gt_centers, y=gt_cad_sm, mode="lines", name="GT cadence smoothed", line=dict(dash="dot")))
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
