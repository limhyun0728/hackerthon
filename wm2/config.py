"""wm2 설정. 모든 스위치는 여기 dataclass 필드다 — env 변수 금지 (설계 0절).

런 시작 시 `snapshot()`으로 config 전체 + git hash를 출력 폴더에 남긴다.
demo_cem/noblock 사고(어느 스위치로 돌았는지 프로세스 env를 뒤져야 알 수 있던 것)의
재발 방지 장치이므로, 학습·평가 진입점은 반드시 이걸 호출한다.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    embedding_dim: int = 64
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    history_frames: int = 3          # a + h1 + h2
    pred_frames: int = 6
    # C-JEPA 대응 스위치. 기본 0 — 상태가 ground truth라 latent 지도가 불필요 (설계 0절).
    latent_weight: float = 0.0
    # ablation 전용, 기본 off (설계 13절)
    routing_keys_idpos: bool = False


@dataclass(frozen=True)
class MaskConfig:
    """h1·h2 유닛 슬롯 마스킹 (설계 7절). 양 팀 모두 대상, anchor·액션 노드는 불가."""
    max_masked_units: int = 3        # window당 0~N개 균등 추출
    mask_probability: float = 0.5    # 마스킹이 하나라도 걸릴 window 비율


@dataclass(frozen=True)
class LossConfig:
    """설계 8절. 전부 frame a 기준 누적 잔차."""
    position: float = 3.0            # L1
    damage: float = 64.0             # MSE. 8→32→64 스윕 중 (근거는 losses.py 주석.
                                     # run11=32: close 1.1×로 복원, hold 2.1×·approach 재상승 잔존)
    ammo: float = 1.0                # MSE
    heading: float = 1.0             # MSE (절대 cos,sin)
    completion: float = 1.0          # BCE


# ── 보상 정의 상수 (2026-08-15 레버 2: 판정 기준과 목적함수 정합) ─────────────
# 보상 δ_t = Δprogress_t + SURVIVAL_BETA·Δ(아군 HP 비율)_t.
# β=1 논거: 분대 전멸(ΔH=−1)의 비용 = 임무 전체 가치(+1) — "다 죽고 완수"는 본전,
# "살아서 완수"는 엄밀 우위. 최종 판정(생존+완료)의 형상을 밀도 있는 보상으로 옮긴 것.
# 세 소비처(train_value_rtg 라벨, score.py gain, episode.py 온라인 라벨)가 공유한다.
VALUE_GAMMA = 0.97       # rtg 할인율/틱 (반감기 ~23틱). CEM의 V 가중은 γ^HORIZON≈0.83
SURVIVAL_BETA = 1.0


@dataclass(frozen=True)
class ScoreConfig:
    """설계 10절. score = Σ(보상) + lam·V(끝 프레임)."""
    w_damage: float = 1.0
    w_loss: float = 1.0
    w_objective: float = 1.0         # destroy_all 임무에서는 자동 0
    lam: float = 1.0


@dataclass(frozen=True)
class CEMConfig:
    """원본 C-JEPA Push-T 설정과 동일 (config/eval.yaml: num_samples 300, topk 30, n_steps 30).

    구 시스템의 탐색량이 원본의 1/100(32×3)이었다는 진단이 있었다. 재계획당
    300×30 = 9,000 상상이라 계획 시간이 램 — 스모크·게이트 측정은 CLI로 줄여 쓴다.
    """
    candidates: int = 300
    elites: int = 30
    iterations: int = 30
    horizon: int = 6
    goal_directed: bool = True
    goal_temperature_range: tuple[float, float] = (0.1, 0.5)
    # 상상 feasibility 마스크 (2026-08-14): 상상 궤적상 실행 불가능한 ENGAGE(사거리·LOS·
    # 표적 생존 전부 불통과)가 든 후보를 elite 선발에서 배제. 실행층 _feasible_target의
    # 계획측 미러 — 팬텀 원거리 사격 복권을 채점 전에 끊는다.
    imagined_feasibility_mask: bool = True


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    seed: int = 42
    device: str = "cuda:0"


@dataclass(frozen=True)
class RunConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    score: ScoreConfig = field(default_factory=ScoreConfig)
    cem: CEMConfig = field(default_factory=CEMConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output_root: str = "output/wm2"

    def snapshot(self) -> Path:
        """config 전체 + git hash를 출력 폴더에 JSON으로 남긴다."""
        root = Path(self.output_root)
        root.mkdir(parents=True, exist_ok=True)
        try:
            git_hash = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
            ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            git_hash = "unknown"
        payload = {"git": git_hash, **asdict(self)}
        path = root / "run_config.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        return path
