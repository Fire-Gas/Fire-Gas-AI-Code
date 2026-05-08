"""합성 데이터 생성기 — 실측 데이터 부족 시 cold-start 학습용.

Why: 실제 화재 노출 데이터는 수집 자체가 위험하고 라벨이 희소.
     5개 시나리오(A/B/C/K/normal)를 물리적 직관에 맞춰 시뮬레이션해 pretraining에 사용.
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd

from config import CSV_COLUMNS, CLASS_NAMES, DEFAULT_CSV, SEED


RNG = np.random.default_rng(SEED)


def _bg_walk(n: int, base: float, drift: float, noise: float) -> np.ndarray:
    """배경 변동 — 작은 random walk."""
    w = np.cumsum(RNG.normal(0, drift, n)) + base
    return np.maximum(0.0, w + RNG.normal(0, noise, n))


def _ramp(n: int, peak: float, t_peak: float = 0.6, decay: float = 0.4) -> np.ndarray:
    """발화→peak→감쇠 곡선."""
    t = np.linspace(0, 1, n)
    rise = np.exp(-((t - t_peak) ** 2) / (2 * decay ** 2))
    return peak * rise / rise.max()


def _scenario(label: str, n: int = 600) -> pd.DataFrame:
    co = _bg_walk(n, 50, 5, 10)
    no = _bg_walk(n, 30, 3, 5)
    nh = _bg_walk(n, 20, 2, 3)
    bme_raw = _bg_walk(n, 200000, 5000, 8000)
    bme_real = _bg_walk(n, 250000, 5000, 8000)
    co2 = _bg_walk(n, 450, 10, 20)
    temp = _bg_walk(n, 25, 0.3, 0.5)
    hum = _bg_walk(n, 40, 0.5, 1.0)
    mq5 = _bg_walk(n, 800, 30, 60)
    detected = np.zeros(n, dtype=int)

    if label == "A_fire":
        co += _ramp(n, 4000)        # 셀룰로오스 → CO 급증
        co2 += _ramp(n, 8000)
        bme_real += _ramp(n, 100000)
        temp += _ramp(n, 60)
        detected[co > 1000] = 1
    elif label == "B_fire":
        bme_real += _ramp(n, 400000)  # 유류 → VOC, 가연성가스 폭발적 증가
        bme_raw += _ramp(n, 300000)
        mq5 += _ramp(n, 3000)
        co += _ramp(n, 2500)
        temp += _ramp(n, 90)
        detected[mq5 > 2000] = 1
    elif label == "C_fire":
        no += _ramp(n, 2500)          # PVC/절연체 → NOx, HCl
        nh += _ramp(n, 1500)          # MICS-NH 채널 교차 감응
        bme_real += _ramp(n, 80000)
        co += _ramp(n, 1500)
        temp += _ramp(n, 50)
        detected[no > 800] = 1
    elif label == "K_fire":
        bme_real += _ramp(n, 350000)  # acrolein 등 VOC
        mq5 += _ramp(n, 2000)
        hum += _ramp(n, 30)           # 수증기
        co += _ramp(n, 1000)
        temp += _ramp(n, 70)
        detected[mq5 > 1800] = 1
    # normal: 변동 없음

    df = pd.DataFrame({
        "MICS_CO": co.astype("uint32"),
        "MICS_NH": nh.astype("uint32"),
        "MICS_NO": no.astype("uint32"),
        "BME_RAW_ADC": bme_raw.astype("uint32"),
        "BME_REAL_ADC": bme_real.astype("uint32"),
        "SCD_CO2": co2.astype("uint16"),
        "SCD_TEMP": temp.astype("int32"),
        "SCD_HUM": hum.astype("int32"),
        "MQ5_VOLTAGE_MV": mq5.astype("uint32"),
        "MQ5_GAS_DETECTED": detected.astype("int32"),
    })
    df["label"] = label
    return df[CSV_COLUMNS + ["label"]]


def generate(out_path: str, n_per_class: int = 600) -> pd.DataFrame:
    parts = [_scenario(c, n_per_class) for c in CLASS_NAMES]
    df = pd.concat(parts, ignore_index=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[synth] {len(df)} rows → {out_path}")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=DEFAULT_CSV)
    p.add_argument("--per-class", type=int, default=600)
    args = p.parse_args()
    generate(args.out, args.per_class)
