#!/usr/bin/env python3
# SoundTrackDB training script
# Uses shared model and feature pipeline from model.py
# Notes:
# - Scheduler is ReduceLROnPlateau with factor 0.5 and patience 5
# - TensorBoard logging is enabled
# - Optional --load_weights to warm start from a checkpoint

import os
import glob
import hashlib
import warnings
import random
import subprocess
import shutil
import argparse
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

# Import shared constants, feature maker, and model
from model import (
    SR, WIN_SEC, HOP_SEC,
    FEATURE_SHAPE, ensure_finite, make_features, MTL_MLP
)

# Silence decode warnings
warnings.filterwarnings("ignore", message="PySoundFile failed")
warnings.filterwarnings("ignore", message="librosa.core.audio.__audioread_load")
warnings.filterwarnings("ignore", category=UserWarning, module="librosa.core.audio")

# Hardcoded config for this project
DATA_DIR = "data"
AUDIO_WAV_DIR = os.path.join(DATA_DIR, "audio_wav")
AUDIO_M4A_DIR = os.path.join(DATA_DIR, "audio_m4a")
GCT_DIR = os.path.join(DATA_DIR, "gct_csv")
OUT_DIR = "runs/exp1"
CACHE_DIR = os.path.join(OUT_DIR, "preprocessed_cache")

EPOCHS = 40
BATCH_SIZE = 128
LR = 1e-3
NUM_WORKERS = 4
SEED = 1337

# Automatic device selection
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# FFmpeg utilities
def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg not found. Please install it and add to your PATH.")
    print("FFmpeg found.")

def convert_m4a_to_wav():
    if not os.path.isdir(AUDIO_M4A_DIR):
        return
    os.makedirs(AUDIO_WAV_DIR, exist_ok=True)
    m4a_files = glob.glob(os.path.join(AUDIO_M4A_DIR, "*.m4a"))

    files_to_convert = []
    for m4a in m4a_files:
        base = os.path.splitext(os.path.basename(m4a))[0]
        wav_path = os.path.join(AUDIO_WAV_DIR, f"{base}.wav")
        if not os.path.exists(wav_path):
            files_to_convert.append((m4a, wav_path))

    if not files_to_convert:
        print("All M4A files already have corresponding WAV siblings.")
        return

    print(f"Found {len(files_to_convert)} M4A files to convert to WAV...")
    for m4a_path, wav_path in tqdm(files_to_convert, desc="Converting M4A to WAV"):
        command = ["ffmpeg", "-i", m4a_path, "-ac", "1", "-ar", str(SR), wav_path, "-hide_banner", "-loglevel", "error"]
        try:
            subprocess.run(command, check=True)
        except Exception as e:
            print(f"Error converting {m4a_path}: {e}")

# Seeding utilities
def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = (SEED + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# Helpers
def hash_split_key(s: str) -> float:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF

def load_audio(path: str, sr: int = SR) -> np.ndarray:
    try:
        y, _ = librosa.load(path, sr=sr, mono=True)
    except Exception as e:
        print(f"Warning: Could not load {path}. Error: {e}. Returning empty.")
        return np.zeros(0, dtype=np.float32)
    y = ensure_finite(y)
    if not np.any(np.isfinite(y)):
        return np.zeros(0, dtype=np.float32)
    return y

def to_num(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)

def compute_stepwise_cadence(times_sec: np.ndarray) -> np.ndarray:
    cad = np.full_like(times_sec, np.nan, dtype=np.float32)
    if len(times_sec) >= 2:
        d = np.diff(times_sec)
        d = np.clip(d, 1e-3, None)
        cad[1:] = 60.0 / d
        cad[0] = cad[1]
    elif len(times_sec) == 1:
        cad[:] = np.nan
    return cad

def aggregate_window_labels(step_times_sec, step_cadence_spm, step_gct_ms, w_start, w_end):
    m = (step_times_sec >= w_start) & (step_times_sec < w_end)
    if not np.any(m):
        return None, None
    cad_vals, gct_vals = step_cadence_spm[m], step_gct_ms[m]
    cad = float(np.nanmean(cad_vals)) if np.any(~np.isnan(cad_vals)) else None
    gct = float(np.nanmean(gct_vals)) if np.any(~np.isnan(gct_vals)) else None
    return cad, gct

# Pre-computation
def preprocess_data(audio_files: List[str]):
    print(f"\nChecking for pre-processed data in {CACHE_DIR}...")
    os.makedirs(CACHE_DIR, exist_ok=True)

    for a_path in tqdm(audio_files, desc="Preprocessing WAV files"):
        base = os.path.splitext(os.path.basename(a_path))[0]
        feat_path = os.path.join(CACHE_DIR, f"{base}_feats.npy")
        label_path = os.path.join(CACHE_DIR, f"{base}_labels.npy")
        if os.path.exists(feat_path) and os.path.exists(label_path):
            continue

        c_path = os.path.join(GCT_DIR, base + ".csv")
        if not os.path.exists(c_path):
            continue
        df = pd.read_csv(c_path)

        required_cols = ["YOLO_Start_Time", "YOLO_Contact_Time", "YOLO_Cadence"]
        if not all(col in df.columns for col in required_cols):
            print(f"Warning: Skipping {base}, missing one of {required_cols}")
            continue

        y = load_audio(a_path, sr=SR)
        if y.size < int(WIN_SEC * SR):
            print(f"Warning: Skipping {base}, audio is too short.")
            continue

        step_t = to_num(df["YOLO_Start_Time"])
        step_gct_ms = to_num(df["YOLO_Contact_Time"])
        step_cad = to_num(df["YOLO_Cadence"])

        if np.all(np.isnan(step_cad)):
            step_cad = compute_stepwise_cadence(step_t)

        win_samples, hop_samples = int(WIN_SEC * SR), int(HOP_SEC * SR)
        wav_windows = librosa.util.frame(y, frame_length=win_samples, hop_length=hop_samples, axis=0)

        assert wav_windows.ndim == 2 and wav_windows.shape[1] == win_samples, \
            f"Framing error for {base}. Expected windows of {win_samples}, got {wav_windows.shape}"

        all_feats, all_labels = [], []
        for i, seg in enumerate(wav_windows):
            try:
                features = make_features(seg)
                all_feats.append(features)
            except Exception as e:
                print(f"Error making features for window {i} in {a_path}: {e}")
                continue

            w_start = (i * hop_samples) / SR
            w_end = (i * hop_samples + win_samples) / SR
            cad, gct = aggregate_window_labels(step_t, step_cad, step_gct_ms, w_start, w_end)
            cad_val = cad if cad is not None and np.isfinite(cad) else np.nan
            gct_val = gct if gct is not None and np.isfinite(gct) else np.nan
            all_labels.append([cad_val, gct_val])

        if not all_feats:
            continue
        np.save(feat_path, np.array(all_feats, dtype=np.float32))
        np.save(label_path, np.array(all_labels, dtype=np.float32))

# Dataset
@dataclass
class WindowIndex:
    feat_path: str
    window_idx: int
    label_cad: float
    label_gct: float
    session_id: str
    seq: int

class SoundTrackWindows(Dataset):
    def __init__(self, file_list: List[str], name: str = "dataset"):
        self.index: List[WindowIndex] = []
        print(f"\nBuilding index for {len(file_list)} sessions ({name}) from cache...")

        num_filtered_nan = 0
        num_filtered_outlier = 0

        for a_path in tqdm(file_list, desc=f"Indexing {name}", leave=True):
            base = os.path.splitext(os.path.basename(a_path))[0]
            feat_path = os.path.join(CACHE_DIR, f"{base}_feats.npy")
            label_path = os.path.join(CACHE_DIR, f"{base}_labels.npy")
            if not os.path.exists(feat_path) or not os.path.exists(label_path):
                continue

            labels = np.load(label_path)
            for i in range(labels.shape[0]):
                cad, gct = labels[i]

                if not (np.isfinite(cad) and np.isfinite(gct)):
                    num_filtered_nan += 1
                    continue

                if not (50.0 < cad < 250.0 and 100.0 < gct < 700.0):
                    num_filtered_outlier += 1
                    continue

                self.index.append(WindowIndex(
                    feat_path=feat_path,
                    window_idx=i,
                    label_cad=float(cad),
                    label_gct=float(gct),
                    session_id=base,
                    seq=i
                ))

        print(f" -> Filtered {num_filtered_nan} NaN windows.")
        print(f" -> Filtered {num_filtered_outlier} outlier windows.")
        print(f" -> Index built with {len(self.index)} valid windows.")

        self.cad_mean, self.cad_std = 0.0, 1.0
        self.gct_mean, self.gct_std = 0.0, 1.0

    def set_label_stats(self, cad_mean, cad_std, gct_mean, gct_std):
        self.cad_mean = cad_mean
        self.cad_std = cad_std
        self.gct_mean = gct_mean
        self.gct_std = gct_std

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        item = self.index[idx]
        try:
            all_features = np.load(item.feat_path, mmap_mode="r")
            x = all_features[item.window_idx]
        except Exception as e:
            raise IOError(f"Error loading feature from {item.feat_path} at index {item.window_idx}: {e}")

        assert x.shape == FEATURE_SHAPE, f"Feature shape mismatch: {x.shape} from {item.feat_path}"
        assert np.max(np.abs(x)) > 1e-9, f"Feature data is all zeros: {item.feat_path}"

        y1_val = (item.label_cad - self.cad_mean) / self.cad_std
        y2_val = (item.label_gct - self.gct_mean) / self.gct_std

        return (
            torch.from_numpy(x.copy()),
            torch.tensor(y1_val, dtype=torch.float32),
            torch.tensor(y2_val, dtype=torch.float32),
            item.session_id,
            item.seq,
        )

def collate_batch(batch):
    xs, yc, yg, session_ids, seqs = zip(*batch)
    return (
        torch.stack(xs),
        torch.stack(yc),
        torch.stack(yg),
        session_ids,
        torch.tensor(seqs, dtype=torch.long),
    )

# Evaluation
@torch.no_grad()
def evaluate(model, loader, device, cad_mean, cad_std, gct_mean, gct_std):
    model.eval()
    all_pc_raw, all_pg_raw = [], []
    all_ycad_raw, all_ygct_raw = [], []
    all_sessions, all_seqs = [], []

    for xb, ycad_norm, ygct_norm, session_ids, seqs in tqdm(loader, desc="Validating", leave=False):
        xb = xb.to(device)
        pc_norm, pg_norm = model(xb)

        all_pc_raw.extend(pc_norm.cpu().numpy())
        all_pg_raw.extend(pg_norm.cpu().numpy())
        all_ycad_raw.extend(ycad_norm.numpy())
        all_ygct_raw.extend(ygct_norm.numpy())
        all_sessions.extend(session_ids)
        all_seqs.extend(seqs.numpy())

    df = pd.DataFrame({
        "session": all_sessions,
        "seq": all_seqs,
        "pc_norm": all_pc_raw,
        "pg_norm": all_pg_raw,
        "ycad_norm": all_ycad_raw,
        "ygct_norm": all_ygct_raw,
    })

    df.sort_values(["session", "seq"], inplace=True)

    df["pc_smooth_norm"] = df.groupby("session")["pc_norm"].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    df["pg_smooth_norm"] = df.groupby("session")["pg_norm"].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)

    pc_smooth = (df["pc_smooth_norm"].values * cad_std) + cad_mean
    pg_smooth = (df["pg_smooth_norm"].values * gct_std) + gct_mean
    ycad = (df["ycad_norm"].values * cad_std) + cad_mean
    ygct = (df["ygct_norm"].values * gct_std) + gct_mean

    mae_c = np.mean(np.abs(pc_smooth - ycad))
    mae_g = np.mean(np.abs(pg_smooth - ygct))

    return float(mae_c), float(mae_g)

# Workflow
def find_audio_files():
    if not os.path.isdir(AUDIO_WAV_DIR):
        return []
    return sorted(glob.glob(os.path.join(AUDIO_WAV_DIR, "*.wav")))

def build_session_splits(audio_files: List[str]):
    train_files, val_files, test_files = [], [], []
    for p in audio_files:
        k = hash_split_key(os.path.basename(p))
        if k < 0.8:
            train_files.append(p)
        elif k < 0.9:
            val_files.append(p)
        else:
            test_files.append(p)
    return train_files, val_files, test_files

def main():
    parser = argparse.ArgumentParser(description="SoundTrackDB Training Script")
    parser.add_argument(
        "--load_weights",
        type=str,
        default=None,
        help="Path to weights (.pt) to load before training."
    )
    args = parser.parse_args()

    set_seed(SEED)
    print(f"--- Using device: {DEVICE} | Seed: {SEED} ---")
    os.makedirs(OUT_DIR, exist_ok=True)

    writer = SummaryWriter(log_dir=OUT_DIR)

    check_ffmpeg()
    convert_m4a_to_wav()
    audio_files = find_audio_files()
    if not audio_files:
        raise RuntimeError(f"No WAV files found in {AUDIO_WAV_DIR}.")
    preprocess_data(audio_files)

    train_files, val_files, test_files = build_session_splits(audio_files)
    print(f"\nSessions: train {len(train_files)} val {len(val_files)} test {len(test_files)}")

    train_ds = SoundTrackWindows(train_files, name="train")
    val_ds = SoundTrackWindows(val_files, name="val")
    test_ds = SoundTrackWindows(test_files, name="test")

    print(f"Total Windows: train {len(train_ds)} val {len(val_ds)} test {len(test_ds)}")
    if len(train_ds) == 0:
        raise RuntimeError("No training windows found.")

    print("\nCalculating label statistics from training set...")
    all_cads = np.array([item.label_cad for item in train_ds.index])
    all_gcts = np.array([item.label_gct for item in train_ds.index])

    cad_mean, cad_std = all_cads.mean(), all_cads.std()
    gct_mean, gct_std = all_gcts.mean(), all_gcts.std()

    print(f" -> Cadence: mean={cad_mean:.2f}, std={cad_std:.2f}")
    print(f" -> GCT:     mean={gct_mean:.2f}, std={gct_std:.2f}")

    assert cad_std > 1e-6, "Cadence standard deviation is zero. All training labels are identical."
    assert gct_std > 1e-6, "GCT standard deviation is zero. All training labels are identical."

    train_ds.set_label_stats(cad_mean, cad_std, gct_mean, gct_std)
    val_ds.set_label_stats(cad_mean, cad_std, gct_mean, gct_std)
    test_ds.set_label_stats(cad_mean, cad_std, gct_mean, gct_std)

    pin_memory = DEVICE == "cuda"
    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_ds, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        collate_fn=collate_batch, pin_memory=pin_memory, worker_init_fn=seed_worker, generator=g
    )
    val_loader = DataLoader(
        val_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        collate_fn=collate_batch, pin_memory=pin_memory, worker_init_fn=seed_worker
    )
    test_loader = DataLoader(
        test_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        collate_fn=collate_batch, pin_memory=pin_memory, worker_init_fn=seed_worker
    )

    device = torch.device(DEVICE)
    model = MTL_MLP().to(device)

    if args.load_weights:
        if os.path.exists(args.load_weights):
            print(f"\n--- Loading pre-trained weights from: {args.load_weights} ---")
            try:
                checkpoint = torch.load(args.load_weights, map_location=device)
                if isinstance(checkpoint, dict) and "model" in checkpoint:
                    state_dict = checkpoint["model"]
                else:
                    state_dict = checkpoint
                model.load_state_dict(state_dict)
                print("--- Weights loaded successfully. Starting training. ---")
            except Exception as e:
                print(f"!!! Error loading weights: {e}. Starting from scratch. !!!")
        else:
            print(f"!!! Warning: --load_weights path not found: {args.load_weights}. Starting from scratch. !!!")
    else:
        print("\n--- No pre-trained weights specified, starting from scratch. ---")

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)

    history = {"train_loss": [], "val_mae_cad": [], "val_mae_gct": [], "lr": []}
    best_val_score = float("inf")
    ckpt_path = os.path.join(OUT_DIR, "best.pt")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{EPOCHS}", leave=False)
        for xb, ycad_norm, ygct_norm, _, _ in pbar:
            xb = xb.to(device)
            ycad_norm = ycad_norm.to(device)
            ygct_norm = ygct_norm.to(device)

            opt.zero_grad()
            pc_norm, pg_norm = model(xb)
            loss = F.mse_loss(pc_norm, ycad_norm) + F.mse_loss(pg_norm, ygct_norm)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        tr_loss = running_loss / len(train_loader)
        val_c, val_g = evaluate(model, val_loader, device, cad_mean, cad_std, gct_mean, gct_std)
        print(f"Epoch {epoch:03d} | Train Loss {tr_loss:.4f} | Val MAE Cad {val_c:.3f} | Val MAE GCT {val_g:.1f}")

        writer.add_scalar("Loss/train", tr_loss, epoch)
        writer.add_scalar("MAE/val_cadence", val_c, epoch)
        writer.add_scalar("MAE/val_gct", val_g, epoch)

        current_score = (val_c if np.isfinite(val_c) else 999) + ((val_g if np.isfinite(val_g) else 9999) / 100.0)
        scheduler.step(current_score)

        current_lr = opt.param_groups[0]["lr"]
        history["train_loss"].append(tr_loss)
        history["val_mae_cad"].append(val_c)
        history["val_mae_gct"].append(val_g)
        history["lr"].append(current_lr)

        writer.add_scalar("LearningRate", current_lr, epoch)

        if current_score < best_val_score:
            best_val_score = current_score
            torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_path)
            print(f"  -> New best model saved (score: {current_score:.3f})")

    print("\nTraining complete. Evaluating best model on test set...")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    test_c, test_g = evaluate(model, test_loader, device, cad_mean, cad_std, gct_mean, gct_std)
    print(f"\nFinal Test Metrics (smoothed):\n  - MAE Cadence: {test_c:.3f} spm\n  - MAE GCT: {test_g:.1f} ms")

    writer.add_hparams(
        {"lr": LR, "batch_size": BATCH_SIZE, "seed": SEED, "epochs": EPOCHS},
        {"hparam/test_mae_cad": test_c, "hparam/test_mae_gct": test_g}
    )
    writer.close()

    print(f"\nResults saved to {OUT_DIR}")
    print(f"To view TensorBoard, run: tensorboard --logdir={os.path.dirname(OUT_DIR)}")

if __name__ == "__main__":
    main()
