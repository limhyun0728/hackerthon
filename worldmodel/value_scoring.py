"""CEM 후보를 value head로 채점한다.

기존 `score_future_features_torch`는 손으로 짠 휴리스틱이라, 그것으로 뽑은 액션을
학습해도 "좋은 것"을 배우기 어려웠다. value head는 실제 에피소드 종료 결과(Monte
Carlo)로 학습되므로 그 자리를 대체할 수 있다.

**월드모델을 거치지 않는다.** DEVS로 정확히 굴린 뒤 그 끝 상태만 평가하므로 월드모델의
예측 오차가 액션 선택에 개입하지 않는다. 바뀌는 것은 "무엇을 좋다고 볼 것인가" 하나다.

라벨이 부트스트랩이 아니라는 점이 중요하다. V가 어떤 상태를 과대평가하면 CEM이 그리로
가고, DEVS가 끝까지 굴린 실제 결과가 라벨이 되어 다음 라운드에 정정된다. 오차가 있는
곳으로 CEM이 데려다주므로 정정이 필요한 지점에 표본이 생긴다.
"""

from __future__ import annotations

import torch

from hackerthon.worldmodel.slots import ObjectType, SlotBatch, TeamId
from hackerthon.worldmodel.value_head import ValueHead

# slot feature 안의 체력 index. alive 판정에 쓴다.
UNIT_HP_INDEX = 1
# 임무 종류 개수. value head의 mission one-hot 차원과 같아야 한다.
NUM_MISSION_TYPES = 4


def _slot_masks(
    current_batch: SlotBatch, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """유닛 slot의 팀별 mask를 만든다. 지형 slot은 전부 False."""
    type_ids = torch.as_tensor(current_batch.type_ids, device=device).long()
    team_ids = torch.as_tensor(current_batch.team_ids, device=device).long()
    unit = type_ids == int(ObjectType.UNIT)
    return unit, unit & (team_ids == int(TeamId.BLUE)), unit & (team_ids == int(TeamId.RED))


def make_value_score_fn(
    *,
    value_head: ValueHead,
    current_batch: SlotBatch,
    mission_type: int,
    device: torch.device,
    chunk_size: int = 32,
):
    """CEM `score_fn` 규약(future_features -> (C,) 점수)에 맞는 함수를 만든다.

    horizon 끝 프레임만 본다. V가 "이 상태에서 계속 가면 임무를 얼마나 달성하는가"를
    예측하므로 중간 프레임을 더 볼 이유가 없다.
    """
    unit_mask, blue_mask, red_mask = _slot_masks(current_batch, device)
    feature_mask = torch.as_tensor(current_batch.feature_mask, device=device)
    type_ids = torch.as_tensor(current_batch.type_ids, device=device).long()
    team_ids = torch.as_tensor(current_batch.team_ids, device=device).long()
    mission_onehot = torch.eye(NUM_MISSION_TYPES, device=device)[int(mission_type)]

    def score_fn(future_features: torch.Tensor) -> torch.Tensor:
        # (C, H, N, F) -> horizon 끝 프레임 (C, N, F)
        final = future_features[:, -1].to(device=device, dtype=torch.float32)
        candidates = int(final.shape[0])
        scores = []
        # 슬롯 수가 많은 실측맵에서 어텐션이 O(N^2)라 청크로 나눠 메모리를 묶는다.
        for start in range(0, candidates, chunk_size):
            block = final[start : start + chunk_size]
            size = int(block.shape[0])
            alive = (block[..., UNIT_HP_INDEX] > 0.0) & unit_mask.unsqueeze(0)
            with torch.no_grad():
                prediction = value_head(
                    features=block,
                    feature_mask=feature_mask.unsqueeze(0).expand(size, -1, -1),
                    type_ids=type_ids.unsqueeze(0).expand(size, -1),
                    team_ids=team_ids.unsqueeze(0).expand(size, -1),
                    alive_mask=alive,
                    blue_mask=blue_mask.unsqueeze(0).expand(size, -1),
                    red_mask=red_mask.unsqueeze(0).expand(size, -1),
                    mission_onehot=mission_onehot.unsqueeze(0).expand(size, -1),
                )
            scores.append(prediction["progress"])
        return torch.cat(scores, dim=0)

    return score_fn
