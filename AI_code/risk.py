"""모델 출력 + 누적 도즈 → LED 명령(`final_value` enum 인덱스).

LED 채널: A, B, C, K, batt, warn (common_struct.h 정의 순서)
출력: 채널별 (state, period_ms) 튜플
  state ∈ {"off", "solid", "blink_slow", "blink_fast", "blink_emergency"}
"""
from __future__ import annotations
from dataclasses import dataclass

from config import (
    CLASS_NAMES, LED_NAMES, LED_A, LED_B, LED_C, LED_K, LED_BATT, LED_WARN,
    RISK_THRESHOLDS,
)

# 펌웨어 측에서 해석할 정수 상태 코드 — 변경 시 ESP32 코드와 동기화 필요
STATE_OFF = 0
STATE_SOLID = 1
STATE_BLINK_SLOW = 2     # 1Hz
STATE_BLINK_FAST = 3     # 4Hz
STATE_BLINK_EMERG = 4    # 8Hz + 적색 우선


@dataclass
class LedFrame:
    states: list  # length = 6, index per LED_NAMES
    severity: float
    chronic: float
    cls: str

    def to_bytes(self) -> bytes:
        return bytes(self.states)


def _severity_state(sev: float) -> int:
    t = RISK_THRESHOLDS
    if sev < t["safe"]:
        return STATE_OFF
    if sev < t["caution"]:
        return STATE_BLINK_SLOW
    if sev < t["warning"]:
        return STATE_BLINK_FAST
    if sev < t["danger"]:
        return STATE_SOLID
    return STATE_BLINK_EMERG


def decide(class_probs, severity: float, chronic_ratio: float, battery_ok: bool = True) -> LedFrame:
    """모델 출력 → LED 6채널 상태 결정."""
    states = [STATE_OFF] * len(LED_NAMES)

    # 1) 화재 클래스 LED
    cls_idx = int(max(range(len(class_probs)), key=lambda i: class_probs[i]))
    cls_name = CLASS_NAMES[cls_idx]
    cls_conf = float(class_probs[cls_idx])

    if cls_name != "normal" and cls_conf > 0.5:
        led_map = {"A_fire": LED_A, "B_fire": LED_B, "C_fire": LED_C, "K_fire": LED_K}
        states[led_map[cls_name]] = STATE_SOLID

    # 2) warn LED — 급성 + 만성 중 더 높은 위험 반영
    sev_state = _severity_state(severity)
    chronic_state = _severity_state(min(chronic_ratio, 1.0))
    states[LED_WARN] = max(sev_state, chronic_state)

    # 3) battery LED — 펌웨어에서 ADC로 측정 후 inject; 여기선 입력 그대로 반영
    states[LED_BATT] = STATE_OFF if battery_ok else STATE_BLINK_FAST

    return LedFrame(states=states, severity=severity, chronic=chronic_ratio, cls=cls_name)


def render_frame(frame: LedFrame) -> str:
    """디버그용 텍스트 렌더."""
    label = {STATE_OFF: "·", STATE_SOLID: "■",
             STATE_BLINK_SLOW: "▭", STATE_BLINK_FAST: "▰", STATE_BLINK_EMERG: "✸"}
    parts = [f"{n}={label[s]}" for n, s in zip(LED_NAMES, frame.states)]
    return (f"[{' '.join(parts)}] cls={frame.cls} "
            f"sev={frame.severity:.2f} chronic={frame.chronic:.2f}")
