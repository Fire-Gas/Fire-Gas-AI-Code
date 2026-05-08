"""학습 스크립트.

사용:
  python synth.py                          # (선택) 합성 데이터 생성
  python train.py --csv data/sensors.csv   # 라벨 있는 CSV로 학습
  python train.py --auto-label             # 라벨 없는 CSV → pseudo-label로 학습
"""
from __future__ import annotations
import argparse
import json
import os
import numpy as np
import tensorflow as tf

from config import (
    DEFAULT_CSV, ARTIFACT_DIR, BATCH_SIZE, EPOCHS, LEARNING_RATE, SEED,
)
from dataset import load_csv, build_windows, split_and_normalize, save_norm
from model import build_model, compile_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=DEFAULT_CSV)
    p.add_argument("--auto-label", action="store_true")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--out", default=ARTIFACT_DIR)
    args = p.parse_args()

    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    print(f"[load] {args.csv}")
    df = load_csv(args.csv, auto=args.auto_label)
    print(f"  rows={len(df)}  classes={df['label'].value_counts().to_dict()}")

    Xw, yc, ys, feat_cols = build_windows(df)
    print(f"[windows] X={Xw.shape}  features={len(feat_cols)}")
    (X_tr, yc_tr, ys_tr), (X_va, yc_va, ys_va), mean, std = split_and_normalize(Xw, yc, ys)
    print(f"[split] train={len(X_tr)} val={len(X_va)}")

    os.makedirs(args.out, exist_ok=True)
    save_norm(mean, std, feat_cols, args.out)

    model = build_model(num_features=len(feat_cols))
    compile_model(model, LEARNING_RATE)
    model.summary()

    cb = [
        tf.keras.callbacks.EarlyStopping("val_loss", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=4, min_lr=1e-5),
    ]
    model.fit(
        X_tr, {"cls": yc_tr, "sev": ys_tr},
        validation_data=(X_va, {"cls": yc_va, "sev": ys_va}),
        epochs=args.epochs, batch_size=BATCH_SIZE, callbacks=cb, verbose=2,
    )

    keras_path = os.path.join(args.out, "model.keras")
    model.save(keras_path)
    print(f"[save] {keras_path}")

    # 평가 지표 저장
    val_loss, *_ = model.evaluate(X_va, {"cls": yc_va, "sev": ys_va}, verbose=0)
    eval_path = os.path.join(args.out, "metrics.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump({"val_loss": float(val_loss),
                   "n_train": int(len(X_tr)), "n_val": int(len(X_va))}, f, indent=2)
    print(f"[metrics] {eval_path}")


if __name__ == "__main__":
    main()
