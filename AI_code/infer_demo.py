"""스트리밍 추론 데모 — CSV 라인을 한 줄씩 흘려보내며 LED 명령 출력.

ESP32 펌웨어에서는 동일 로직이 TFLite Micro로 동작.
이 스크립트는 PC에서 동등성 검증용.
"""
from __future__ import annotations
import argparse
import json
import os
from collections import deque
import numpy as np
import pandas as pd
import tensorflow as tf

from config import (
    ARTIFACT_DIR, DEFAULT_CSV, WINDOW_SIZE, SAMPLE_PERIOD_S,
)
from features import engineer
from dose import DoseTracker
from risk import decide, render_frame


def _load_artifacts(art_dir: str):
    model = tf.keras.models.load_model(os.path.join(art_dir, "model.keras"), compile=False)
    nz = np.load(os.path.join(art_dir, "normalizer.npz"))
    with open(os.path.join(art_dir, "feature_columns.json"), "r", encoding="utf-8") as f:
        feat_cols = json.load(f)
    return model, nz["mean"], nz["std"], feat_cols


def stream(csv_path: str, art_dir: str, limit: int | None = None):
    model, mean, std, feat_cols = _load_artifacts(art_dir)
    raw = pd.read_csv(csv_path)
    eng = engineer(raw)
    # 학습 시 컬럼 순서 보존
    for c in feat_cols:
        if c not in eng.columns:
            eng[c] = 0.0
    eng = eng[feat_cols].astype(np.float32)

    buf = deque(maxlen=WINDOW_SIZE)
    tracker = DoseTracker()

    n = len(eng) if limit is None else min(limit, len(eng))
    for i in range(n):
        buf.append(eng.iloc[i].to_numpy(dtype=np.float32))
        if len(buf) < WINDOW_SIZE:
            continue
        x = np.stack(buf, axis=0)
        x = (x - mean) / np.where(std < 1e-6, 1.0, std)
        x = x.astype(np.float32)[None]

        cls_probs, sev = model.predict(x, verbose=0)
        cls_probs = cls_probs[0].tolist()
        sev = float(sev[0, 0])

        tracker.update_from_severity(sev, dt=SAMPLE_PERIOD_S)
        chronic = tracker.chronic_risk()

        frame = decide(cls_probs, sev, chronic, battery_ok=True)
        if i % 25 == 0 or sev > 0.5:
            print(f"t={i:5d} {render_frame(frame)} acute_TWA={tracker.acute_twa_ratio():.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=DEFAULT_CSV)
    p.add_argument("--art", default=ARTIFACT_DIR)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    stream(args.csv, args.art, args.limit)
