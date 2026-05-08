"""TWA + Haber 법칙 누적 노출량 추적기.

TWA: (Σ C_i · Δt_i) / T_ref  — 8시간 시간가중평균
Haber: K = C · t              — 누적 도즈; reference dose 대비 비율로 만성 위험 산출
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from config import TLV_PPM, IDLH_PPM, reference_dose_ppm_s, SAMPLE_PERIOD_S


@dataclass
class GasState:
    name: str
    twa_window_s: float = 8 * 3600.0   # 직업적 8h 기준
    cum_dose: float = 0.0              # ppm·s
    twa_buffer: list = field(default_factory=list)  # (t, C) — 슬라이딩 8h
    elapsed: float = 0.0

    def update(self, conc_ppm: float, dt: float = SAMPLE_PERIOD_S):
        self.elapsed += dt
        self.cum_dose += conc_ppm * dt
        self.twa_buffer.append((self.elapsed, conc_ppm))
        # 8h 윈도우 밖은 제거
        cutoff = self.elapsed - self.twa_window_s
        while self.twa_buffer and self.twa_buffer[0][0] < cutoff:
            self.twa_buffer.pop(0)

    def twa(self) -> float:
        if not self.twa_buffer:
            return 0.0
        cs = np.array([c for _, c in self.twa_buffer])
        T = max(self.twa_buffer[-1][0] - self.twa_buffer[0][0], SAMPLE_PERIOD_S)
        return float(cs.mean() * (len(cs) * SAMPLE_PERIOD_S) / T) if T > 0 else float(cs.mean())

    def twa_ratio(self) -> float:
        return self.twa() / TLV_PPM[self.name]

    def idlh_ratio(self, conc_ppm: float) -> float:
        return conc_ppm / IDLH_PPM[self.name]

    def haber_ratio(self) -> float:
        return self.cum_dose / reference_dose_ppm_s(self.name)


@dataclass
class DoseTracker:
    """다중 가스 누적 추적. 모델이 직접 ppm을 못 주는 경우 정규화 sev 값을 proxy로 가능."""
    gases: dict = field(default_factory=lambda: {g: GasState(g) for g in TLV_PPM})

    def update_from_estimates(self, ppm_estimates: dict[str, float], dt: float = SAMPLE_PERIOD_S):
        for g, c in ppm_estimates.items():
            if g in self.gases:
                self.gases[g].update(float(c), dt)

    def update_from_severity(self, sev: float, dt: float = SAMPLE_PERIOD_S):
        """절대 ppm 추정이 어려울 때 fallback: severity를 모든 가스의 TLV 비율로 환산.

        Why: 본 PoC는 센서 보정(ppm) 미적용 — sev(0~1)를 'TLV 대비 노출 강도'로 재해석.
             상용화 시 NDIR 등 보정 센서로 교체되면 update_from_estimates 경로 사용.
        """
        for g, gs in self.gases.items():
            gs.update(sev * TLV_PPM[g], dt)

    def chronic_risk(self) -> float:
        """0~1+ 범위. 1.0 도달 시 8h TLV 노출 누적 한계 도달."""
        return max(g.haber_ratio() for g in self.gases.values())

    def acute_twa_ratio(self) -> float:
        return max(g.twa_ratio() for g in self.gases.values())
