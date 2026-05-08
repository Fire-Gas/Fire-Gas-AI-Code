# Fire-Gas-AI-Code

소방관 개인 발암가스 노출 모니터링용 온디바이스 AI 학습/추론 파이프라인.

ESP32 펌웨어가 만들어 낸 `sensors.csv`(MICS-6814 / BME680 / SCD41 / MQ-5 융합)을 입력으로 받아
화재 시그니처 5-class 분류 + 누적 노출 도즈를 추정하고, `final_value` enum의 LED 6채널을 제어한다.

## 디렉터리

```
AI_code/
├── config.py          전역 상수 (TLV/IDLH, 윈도우, LED 매핑)
├── features.py        ΔC/Δt + 센서 비율 피처
├── labeler.py         Indicator-gas 규칙 기반 자동 라벨러
├── synth.py           합성 데이터 생성기
├── dataset.py         CSV → 윈도우 텐서
├── model.py           1D-CNN + LSTM 멀티헤드 (TinyML)
├── dose.py            TWA + Haber 누적 도즈
├── risk.py            severity + 클래스 → LED 명령
├── train.py           학습
├── export_tflite.py   TFLite int8 + ESP32용 C 헤더
├── infer_demo.py      스트리밍 추론 시뮬레이터
└── requirements.txt
```

## 빠른 시작

```bash
cd AI_code
pip install -r requirements.txt

# 1) 합성 데이터로 cold-start (실측 CSV가 아직 없을 때)
python synth.py --out data/sensors.csv --per-class 600

# 2) 학습 — 라벨이 있는 경우
python train.py --csv data/sensors.csv

# 2') 학습 — 라벨이 없는 실측 CSV의 경우 (자동 라벨링)
python train.py --csv data/real_log.csv --auto-label

# 3) TFLite int8 변환 + ESP32 C 헤더 추출
python export_tflite.py

# 4) PC에서 추론 시뮬레이션
python infer_demo.py --csv data/sensors.csv
```

## 펌웨어 통합

`artifacts/firegas_model.h`, `firegas_model.cc`를 ESP-IDF 프로젝트의 컴포넌트로 추가하고
TFLite Micro 인터프리터로 로드. 입력 텐서는 `[1, 10, F]` shape의 int8(zero-point/scale은 헤더 안에 포함).

LED 출력 매핑은 `risk.py`의 상태 코드(0~4)를 펌웨어 측 `set_led(led_idx, state)`에서 동일하게 해석해야 함.

## 모델 구조

```
Input (10, F)
 → Conv1D(8, k=3) → Conv1D(8, k=3)
 → LSTM(16, unroll=True)
 → Dense(16, relu)
 → [Dense(5, softmax) "cls",  Dense(1, sigmoid) "sev"]
```

- 분류 헤드: A_fire / B_fire / C_fire / K_fire / normal
- 회귀 헤드: 0~1 severity (LED warn 등급 결정에 사용)

## 누적 노출량 산식

- TWA: `(Σ Cᵢ·Δt) / T_ref`, T_ref = 8h
- Haber: `K = C · t` → reference dose = `TLV × 8h`
- 만성 위험 = `cum_dose / reference_dose` 의 가스별 최댓값

`dose.py`는 ppm 환산값과 severity proxy 둘 다 입력 가능. PoC에서는 후자(sev) 사용.
