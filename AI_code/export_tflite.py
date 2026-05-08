"""Keras → TFLite int8 변환 + ESP32용 C 헤더 추출.

산출물:
  artifacts/firegas.tflite
  artifacts/firegas_model.h     # const unsigned char firegas_model_tflite[] = {...};
  artifacts/firegas_model.cc    # 동일 배열의 C 정의
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import tensorflow as tf

from config import ARTIFACT_DIR, WINDOW_SIZE


def _representative_dataset(npz_path: str, n: int = 200):
    """대표 데이터셋 — train.py 출력의 정규화된 X 일부 사용 가능. 없으면 random."""
    if os.path.exists(npz_path):
        data = np.load(npz_path)
        X = data["X"][:n]
        for x in X:
            yield [x[None].astype(np.float32)]
    else:
        rng = np.random.default_rng(0)
        # 정규화된 입력은 평균 0, 표준편차 1에 가까움
        for _ in range(n):
            yield [rng.standard_normal((1, WINDOW_SIZE, _representative_dataset.num_features)).astype(np.float32)]


def to_c_array(data: bytes, name: str) -> str:
    lines = [f"const unsigned int {name}_len = {len(data)};",
             f"alignas(16) const unsigned char {name}[] = {{"]
    for i in range(0, len(data), 16):
        chunk = ", ".join(f"0x{b:02x}" for b in data[i : i + 16])
        lines.append(f"  {chunk},")
    lines.append("};")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--keras", default=os.path.join(ARTIFACT_DIR, "model.keras"))
    p.add_argument("--out-dir", default=ARTIFACT_DIR)
    p.add_argument("--rep", default=os.path.join(ARTIFACT_DIR, "rep.npz"),
                   help="대표 데이터(npz, key=X)")
    p.add_argument("--no-quant", action="store_true", help="float32로 출력")
    args = p.parse_args()

    model = tf.keras.models.load_model(args.keras, compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if not args.no_quant:
        # 모델의 입력 차원으로 대표 dataset 샘플 차원 동기화
        _representative_dataset.num_features = model.inputs[0].shape[-1]
        converter.representative_dataset = lambda: _representative_dataset(args.rep)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    tflite_bytes = converter.convert()
    os.makedirs(args.out_dir, exist_ok=True)
    tflite_path = os.path.join(args.out_dir, "firegas.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_bytes)
    print(f"[tflite] {tflite_path}  size={len(tflite_bytes)} bytes")

    header_body = to_c_array(tflite_bytes, "firegas_model_tflite")
    h_path = os.path.join(args.out_dir, "firegas_model.h")
    with open(h_path, "w", encoding="utf-8") as f:
        f.write("#pragma once\n#include <cstddef>\nextern const unsigned int firegas_model_tflite_len;\nextern const unsigned char firegas_model_tflite[];\n")
    cc_path = os.path.join(args.out_dir, "firegas_model.cc")
    with open(cc_path, "w", encoding="utf-8") as f:
        f.write('#include "firegas_model.h"\n#include <cstddef>\n')
        f.write(header_body + "\n")
    print(f"[header] {h_path}\n[source] {cc_path}")


if __name__ == "__main__":
    main()
