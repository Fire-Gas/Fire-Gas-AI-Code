"""전역 상수 — 노출 한계, 윈도우 크기, 모델 하이퍼파라미터."""
from __future__ import annotations

CSV_COLUMNS = [
    "MICS_CO", "MICS_NH", "MICS_NO",
    "BME_RAW_ADC", "BME_REAL_ADC",
    "SCD_CO2", "SCD_TEMP", "SCD_HUM",
    "MQ5_VOLTAGE_MV", "MQ5_GAS_DETECTED",
]

SENSOR_COLUMNS = CSV_COLUMNS

# 샘플링 주기 (초). ESP32 펌웨어에서 1Hz 가정 — 실제 주기에 맞춰 조정.
SAMPLE_PERIOD_S = 1.0

# 슬라이딩 윈도우: 10초 컨텍스트 → CNN-LSTM 입력
WINDOW_SIZE = 10
WINDOW_STRIDE = 1

# final_value enum 순서 (common_struct.h와 동일 — 절대 변경 금지)
LED_NAMES = ["A", "B", "C", "K", "batt", "warn"]
LED_A, LED_B, LED_C, LED_K, LED_BATT, LED_WARN = range(6)

# 5-class: A화재(일반), B(유류), C(전기), K(주방), N(정상/배경)
CLASS_NAMES = ["A_fire", "B_fire", "C_fire", "K_fire", "normal"]
NUM_CLASSES = 5

# OSHA/NIOSH/KOSHA TWA 기준 (ppm). 화재 시 흔히 측정되는 발암·자극성 가스.
# Why: 본 시스템은 절대 농도 측정이 어려우므로 위험 등급 임계는 percentile + TLV 비율로 운용.
#      그래도 도즈 적분의 reference dose로 쓰기 위해 TLV·IDLH는 보존.
TLV_PPM = {
    "CO": 25.0,
    "NO2": 0.2,
    "NH3": 25.0,
    "CO2": 5000.0,
    "VOC": 10.0,
}
IDLH_PPM = {
    "CO": 1200.0,
    "NO2": 20.0,
    "NH3": 300.0,
    "CO2": 40000.0,
    "VOC": 500.0,
}

# Haber 법칙: K = C * t. 단순 근사로 만성 영향 정량화. (직업 보건 표준은 아님)
# Reference dose = TLV * 8h 환산 (단위: ppm·s)
def reference_dose_ppm_s(gas: str) -> float:
    return TLV_PPM[gas] * 8 * 3600.0

# 위험 등급 임계 (정규화된 0~1 severity 기준)
RISK_THRESHOLDS = {
    "safe":      0.20,
    "caution":   0.45,
    "warning":   0.70,
    "danger":    0.85,
    # > 0.85 → emergency
}

# 모델 하이퍼파라미터 (TinyML 친화 — ESP32 SRAM 520KB 제약)
MODEL_CONV_FILTERS = 8
MODEL_LSTM_UNITS = 16
MODEL_DENSE_UNITS = 16
DROPOUT = 0.1

# 학습
BATCH_SIZE = 64
EPOCHS = 40
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.15
SEED = 42

# 경로
import os
ROOT = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(ROOT, "artifacts")
DEFAULT_CSV = os.path.join(ROOT, "data", "sensors.csv")
