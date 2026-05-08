"""규칙 기반 자동 라벨러 — 라벨 없는 CSV 부트스트랩용.

Why: 실측 라벨링 비용이 높고 화재 현장에서 ground truth 수집이 위험·고비용이므로,
     산업위생의 indicator-gas heuristic으로 pseudo-label 생성 후 후속 수동 보정 가능.
How to apply: train.py에서 --auto-label 플래그가 있을 때만 호출.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from config import CLASS_NAMES


def _percentile_norm(s: pd.Series) -> pd.Series:
    """0~1로 매핑(분포 기반). 센서 보정값 부재로 절대 ppm 환산 불가하기 때문."""
    s = s.astype("float64")
    lo, hi = s.quantile(0.05), s.quantile(0.95)
    if hi - lo < 1e-9:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return ((s - lo) / (hi - lo)).clip(0.0, 1.0)


def auto_label(df: pd.DataFrame) -> pd.Series:
    co = _percentile_norm(df["MICS_CO"])
    no = _percentile_norm(df["MICS_NO"])
    nh = _percentile_norm(df["MICS_NH"])
    voc = _percentile_norm(df["BME_REAL_ADC"])
    co2 = _percentile_norm(df["SCD_CO2"])
    mq5 = _percentile_norm(df["MQ5_VOLTAGE_MV"])
    hum = _percentile_norm(df["SCD_HUM"])

    activity = (co + co2 + voc + mq5) / 4.0  # 무언가 타고 있는지

    # 클래스별 시그니처 점수
    s_A = 0.6 * co + 0.4 * co2 - 0.2 * no              # 셀룰로오스: CO/CO2 우세
    s_B = 0.5 * voc + 0.5 * mq5 - 0.2 * nh             # 유류: VOC/가연성 가스 우세
    s_C = 0.6 * no + 0.3 * nh + 0.1 * voc              # 전기/PVC: NOx, HCl(→NH3 채널 교차)
    s_K = 0.4 * voc + 0.3 * hum + 0.3 * mq5            # 식용유: VOC + 수증기 + 가연성

    scores = np.stack([s_A, s_B, s_C, s_K], axis=1)
    cls = scores.argmax(axis=1)

    labels = np.where(activity < 0.25, len(CLASS_NAMES) - 1, cls)  # 활동 낮으면 normal
    return pd.Series([CLASS_NAMES[i] for i in labels], index=df.index, name="label")


def severity_from_features(df: pd.DataFrame) -> pd.Series:
    """0~1 연속 위험 지수 — 회귀 타겟 보조."""
    co = _percentile_norm(df["MICS_CO"])
    no = _percentile_norm(df["MICS_NO"])
    voc = _percentile_norm(df["BME_REAL_ADC"])
    co2 = _percentile_norm(df["SCD_CO2"])
    mq5 = _percentile_norm(df["MQ5_VOLTAGE_MV"])
    sev = (0.30 * co + 0.20 * no + 0.20 * voc + 0.15 * co2 + 0.15 * mq5)
    return sev.clip(0.0, 1.0).rename("severity")
