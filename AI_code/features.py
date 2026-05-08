"""센서 융합 피처 — 절댓값 + ΔC/Δt + 센서 간 비율."""
from __future__ import annotations
import numpy as np
import pandas as pd

from config import SENSOR_COLUMNS, SAMPLE_PERIOD_S


def add_rate_of_change(df: pd.DataFrame, dt: float = SAMPLE_PERIOD_S) -> pd.DataFrame:
    """ΔC/Δt 피처 추가. NaN(첫 샘플)은 0으로."""
    out = df.copy()
    for col in SENSOR_COLUMNS:
        if col == "MQ5_GAS_DETECTED":
            continue  # 이산값은 미분 의미 없음
        roc = (df[col].astype("float64").diff() / dt).fillna(0.0)
        out[f"{col}_ROC"] = roc
    return out


def add_ratios(df: pd.DataFrame, eps: float = 1.0) -> pd.DataFrame:
    """센서 간 비율 — 화재 유형 시그니처(예: NO/CO 비율 → 전기화재 추정).

    Why: 단일 센서로는 발암 가스 식별이 불가하므로 산업위생의 indicator-gas 접근에 따라
         이종 센서 비율을 명시적 피처로 노출.
    """
    out = df.copy()
    f = df.astype("float64")
    out["R_NO_OVER_CO"] = f["MICS_NO"] / (f["MICS_CO"] + eps)
    out["R_NH_OVER_CO"] = f["MICS_NH"] / (f["MICS_CO"] + eps)
    out["R_VOC_OVER_CO"] = f["BME_REAL_ADC"] / (f["MICS_CO"] + eps)
    out["R_VOC_OVER_CO2"] = f["BME_REAL_ADC"] / (f["SCD_CO2"] + eps)
    out["R_MQ5_OVER_CO"] = f["MQ5_VOLTAGE_MV"] / (f["MICS_CO"] + eps)
    out["R_NO_OVER_CO2"] = f["MICS_NO"] / (f["SCD_CO2"] + eps)
    return out


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = add_rate_of_change(df)
    df = add_ratios(df)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """label/timestamp 등을 제외한 모델 입력 컬럼."""
    skip = {"label", "timestamp", "ts"}
    return [c for c in df.columns if c not in skip]


def normalize(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (arr - mean) / np.where(std < 1e-6, 1.0, std)


def fit_normalizer(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return arr.mean(axis=0), arr.std(axis=0)
