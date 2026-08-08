# baseline-unitloss (2026-08-08)

지형·임무를 attention query에서 빼기 **전**의 코드다. git 태그 `baseline-unitloss`
(커밋 fc0a86e)와 같은 내용이다.

## 이 코드로 돌린 것

```
output/unitloss_rule    규칙 정책 150 에피소드 파인튜닝 (GPU 1)
                        -> checkpoints/cjepa_unitloss.pt
                           checkpoints/cjepa_unitloss_value.pt
output/jepa300_loop     CEM jepa 300x30 (GPU 0)
```

## 포함된 변경 (측정 기록 문서 28.1~28.9)

- MOVE 목적지: 목적 지향(GOAL_WEIGHTS/온도) 제거, 엘리트 분포(move_mean/std) 반영
- ENGAGE 표적: 예측 상태 기준 사거리 안 균등 추출 + 실행 시 갈아타기
- 손실: loss_future / loss_future_state를 unit slot으로 제한
- 미래 디코딩: unit만, 지형·임무는 마지막 관측값 대입
- rollout 후보 청킹 (CEM_ROLLOUT_CHUNK_SIZE, 기본 16)

## 되돌리는 법

```bash
git checkout baseline-unitloss      # 또는 이 폴더 내용을 복사
```

파인튜닝 결과 checkpoint로 CEM 학습을 돌릴 때 이 코드를 써야 한다. 이후 구조
변경(지형·임무를 query에서 제외)은 가중치가 쓰이는 방식이 달라 호환되지 않는다.
