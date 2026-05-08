"""1D-CNN + LSTM 멀티헤드 — TinyML(TFLite Micro) 친화 경량 구조.

분류 헤드: 5-class 화재 시그니처(A/B/C/K/normal)
회귀 헤드: 0~1 severity (LED warn 강도 결정)

총 파라미터 < ~6K 목표. int8 양자화 후 ~10-20KB로 ESP32 SRAM에 적재 가능.
"""
from __future__ import annotations
import tensorflow as tf
from tensorflow.keras import layers, Model

from config import (
    WINDOW_SIZE, NUM_CLASSES,
    MODEL_CONV_FILTERS, MODEL_LSTM_UNITS, MODEL_DENSE_UNITS, DROPOUT,
)


def build_model(num_features: int) -> Model:
    inp = layers.Input(shape=(WINDOW_SIZE, num_features), name="window")

    x = layers.Conv1D(MODEL_CONV_FILTERS, 3, padding="same", activation="relu")(inp)
    x = layers.Conv1D(MODEL_CONV_FILTERS, 3, padding="same", activation="relu")(x)
    x = layers.Dropout(DROPOUT)(x)

    # LSTM의 unroll=True → TFLite Micro는 동적 RNN을 지원하지 않으므로 정적으로 풀어야 함.
    x = layers.LSTM(MODEL_LSTM_UNITS, unroll=True)(x)
    x = layers.Dropout(DROPOUT)(x)

    h = layers.Dense(MODEL_DENSE_UNITS, activation="relu")(x)

    out_cls = layers.Dense(NUM_CLASSES, activation="softmax", name="cls")(h)
    out_sev = layers.Dense(1, activation="sigmoid", name="sev")(h)

    return Model(inp, [out_cls, out_sev], name="firegas_tinyml")


def compile_model(model: Model, lr: float):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss={"cls": "sparse_categorical_crossentropy", "sev": "mse"},
        loss_weights={"cls": 1.0, "sev": 0.5},
        metrics={"cls": "accuracy", "sev": "mae"},
    )
    return model
