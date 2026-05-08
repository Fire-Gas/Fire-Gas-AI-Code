"""CSV → 윈도우 텐서 변환 + train/val 분할."""
from __future__ import annotations
import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    CSV_COLUMNS, CLASS_NAMES, WINDOW_SIZE, WINDOW_STRIDE, VAL_SPLIT, SEED,
)
from features import engineer, feature_columns, fit_normalizer, normalize
from labeler import auto_label, severity_from_features


def load_csv(path: str, auto: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in CSV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV에 빠진 컬럼: {missing}")
    if "label" not in df.columns:
        if not auto:
            raise ValueError("label 컬럼 없음. --auto-label 사용하거나 수동 라벨 추가.")
        df["label"] = auto_label(df)
    df["severity"] = severity_from_features(df)
    return df


def _windowize(arr: np.ndarray, w: int, stride: int) -> np.ndarray:
    n = (len(arr) - w) // stride + 1
    if n <= 0:
        return np.empty((0, w, arr.shape[1]), dtype=arr.dtype)
    out = np.stack([arr[i * stride : i * stride + w] for i in range(n)], axis=0)
    return out


def _last_value(arr: np.ndarray, w: int, stride: int) -> np.ndarray:
    n = (len(arr) - w) // stride + 1
    if n <= 0:
        return np.empty((0,), dtype=arr.dtype)
    return np.array([arr[i * stride + w - 1] for i in range(n)])


def build_windows(df: pd.DataFrame):
    df = engineer(df)
    feat_cols = [c for c in feature_columns(df) if c not in ("label", "severity")]
    X_raw = df[feat_cols].to_numpy(dtype=np.float32)
    y_cls_raw = df["label"].to_numpy()
    y_sev_raw = df["severity"].to_numpy(dtype=np.float32)

    cls_to_idx = {n: i for i, n in enumerate(CLASS_NAMES)}
    y_cls_idx = np.array([cls_to_idx[v] for v in y_cls_raw], dtype=np.int32)

    Xw = _windowize(X_raw, WINDOW_SIZE, WINDOW_STRIDE)
    yc = _last_value(y_cls_idx, WINDOW_SIZE, WINDOW_STRIDE)
    ys = _last_value(y_sev_raw, WINDOW_SIZE, WINDOW_STRIDE)
    return Xw, yc, ys, feat_cols


def split_and_normalize(Xw, yc, ys):
    flat = Xw.reshape(-1, Xw.shape[-1])
    mean, std = fit_normalizer(flat)
    Xn = normalize(Xw, mean, std).astype(np.float32)

    X_tr, X_va, yc_tr, yc_va, ys_tr, ys_va = train_test_split(
        Xn, yc, ys, test_size=VAL_SPLIT, random_state=SEED, stratify=yc,
    )
    return (X_tr, yc_tr, ys_tr), (X_va, yc_va, ys_va), mean.astype(np.float32), std.astype(np.float32)


def save_norm(mean: np.ndarray, std: np.ndarray, feat_cols: list[str], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "normalizer.npz"), mean=mean, std=std)
    with open(os.path.join(out_dir, "feature_columns.json"), "w", encoding="utf-8") as f:
        json.dump(feat_cols, f, ensure_ascii=False, indent=2)
