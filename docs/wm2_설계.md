# wm2 설계 (2026-08-12 확정)

기존 `worldmodel/`(16k줄, env 변수 5개 조합)을 대체하는 재작성. 이 문서가 단일 기준이며,
여기 없는 동작은 구현하지 않는다. 참조 구현: [galilai-group/cjepa@swm-renewal](https://github.com/galilai-group/cjepa/tree/swm-renewal).

## 0. 원칙

- **의존 경계**: `wm2/`는 구 코드를 import하지 않는다. 시뮬레이터는 `wm2/sim/adapter.py`
  한 파일 뒤로 격리(시뮬 자체는 환경이므로 재작성하지 않는다). 학습 데이터는 디스크의
  에피소드 디렉터리(`soldier_log.csv`, `commands_log.csv`, `config.json`)로만 읽는다.
- **스위치는 전부 config dataclass 필드.** env 변수 금지. 런 시작 시 config + git hash를
  출력 폴더에 JSON 스냅샷 (demo_cem env 사고 재발 방지).
- **이미 값 치른 교훈은 기본값**: 이동상한 1.0 단일 상수, 홀드아웃 4맵
  (euljiro/jamsil/assembly/yongsan) 코드에 하드코딩 배제, 종료시각 horizon+1.5 가드.
- latent/EMA 손실은 config 필드로만 존재, 기본 0. 이유: C-JEPA는 관측이 픽셀이라 latent
  예측이 강제였지만 우리는 좌표·HP가 이미 ground truth다.

## 1. Window

```
raw:  [pre | a | h1 h2 | f1 .. f6]
             ↑    ↑          ↑
             │    │          └ 미래 6프레임: 항상 query (양 팀 전부 예측 대상)
             │    └ history 2프레임: 유닛 슬롯 단위로 마스킹 가능
             └ anchor 프레임: 전 유닛 항상 완전 공개 (C-JEPA frame 0 규칙)
pre = a의 vx,vy 계산용 직전 raw 프레임 (모델 입력 아님)
```

- **anchor = 프레임 a.** 모든 잔차 레이블과 조립의 유일한 기준. 학습·추론 동일.
  레이블이 빼는 프레임 = 조립이 더하는 프레임. 이 규칙 위반 금지.
- 추론(CEM) 시: a = 현재 관측의 2틱 전, h2 = 현재. 마스킹 없음.

## 2. Slot / 특징

```
UNIT    : team, hp_ratio, ammo_ratio, x, y, heading_cos, heading_sin, alive, vx, vy
TERRAIN : type, x, y, w, h, traversability, movement_cost, cover_value, los_block
MISSION : type, objective_x, objective_y, time_remaining, completion
```

- vx,vy = 직전 프레임 대비 변위, 이동상한 1.0으로 정규화. **RED 위상 잠김(2틱 사각파,
  decision_delay=1.0 artifact)의 관측 가능성이 이 특징의 존재 이유.**

## 3. 블록 layout (embedding 64)

```
[ position 16 | velocity 8 | state 16 | heading 8 | identity 16 ]
```

| 블록 | UNIT | TERRAIN | MISSION |
|---|---|---|---|
| position | x,y | x,y,**w,h** | objective x,y |
| velocity | vx,vy | 0 | 0 |
| state | hp, ammo, alive | trav, cost, cover, los | time_remaining, completion |
| heading | cos,sin | 학습된 상수 | 학습된 상수 |
| identity | type+team+**팀내 index** emb | type(+terrain_type) emb | type emb |

- 블록별 타입별 소형 MLP 인코딩 + **블록별 LayerNorm** (전체 LN 금지 — 블록이 섞인다).
- 지형 w,h는 position 블록 (구 설계의 identity 잔류 문제 청산).
- identity에 **팀내 index emb**(101→0.., 201→0..)를 추가해 프레임 간 같은 유닛 binding이
  위치 연속성에 의존하지 않게 한다 (RED는 10m씩 점프하므로 특히 중요).

## 4. Predictor

- history 3 + 미래 6 query를 한 번의 forward로 예측. **비자기회귀** (알려진 절충: 오차
  비누적 vs 먼 프레임 정보 제한. 유지).
- **identity 층별 리셋** (OA-WAM address reset): 각 층 통과 후 identity 슬라이스 16차원을
  인코더 출력값으로 되쓴다. identity는 (type, team, index)만의 함수 = 시간 불변이므로 무손실.
- 미래 query = mask token + 시간 임베딩 + anchor(a) 투영 (C-JEPA `anchor_queries`)
  + 액션 노드는 별도 (아래 6).
- 지형: KV-only (query 없음, 미래 토큰 없음). 임무: history에선 KV, 미래에는 completion
  예측용 query 토큰 1개. 죽은 유닛 attention 차단.

## 5. Attention 가시성 (최종)

```
상태 토큰 (a, h1, h2, f1..f6 — 마스킹 쿼리·미래 쿼리 포함):
    서로 전방향. 제약 없음.
    근거: 미래 상태 토큰은 ground truth가 아니라 자기 예측이므로 참조는 누설이 아니라
    상호 정제다. 창 안에서 ground truth인 미래 입력은 액션뿐이다.

액션 노드 (tick τ 발행 명령):
    프레임 ≥ τ+1 인 상태 토큰에게만 가시. ("각 시점 예측은 그 이전 발행 액션까지만")
    - belief 정합: 가려진 h 복원이 미래 액션(추론 시점에 없거나 후보마다 다름)에
      의존하지 못하게 된다.
    - CEM 채점: f1이 step5 액션에 직접 의존하는 길 차단.
    - 남는 것: f5→f1 2-hop 간접 경로. 수용한다 (레퍼런스는 전면 비인과로도 동작).
```

## 6. 액션 노드

- 유닛별·스텝별 토큰, 슬롯 축의 별도 노드 (C-JEPA `NodeEmbedder`/`action_node` 방식).
- **토큰 내용 = 명령 인코딩 + 발행틱 시간 임베딩 + 발신자(사수) identity 임베딩.**
  마지막 항이 없으면 평탄화 후 attention이 같은 틱·같은 표적의 ENGAGE를 사수 불문
  동일 토큰으로 봐서 사수별 사거리 귀속이 불가능해진다 (실측: 전원 한계사거리
  유지사격에 상상 73HP vs DEVS 4.8HP — 2026-08-13 발견·수정). C-JEPA는 단일
  agent라 이 문제가 없었다. MOVE는 목적지 좌표로 자가결합돼 증상이 가려졌다.
- **절대 마스킹 불가, 손실 타깃 아님** (`num_unmaskable_slots` 대응).
- **내용 = 계획된 명령 (실행된 명령 아님).** "사거리 밖 ENGAGE → 아무 일 없음"이 학습
  분포 안에 들어와야 하기 때문. CEM이 던지는 토큰(계획)과 학습 분포가 일치한다.
  ENGAGE 게이트 같은 추론 규칙은 두지 않는다 — 실행 의미론은 모델이 데이터에서 배운다.
- RED는 액션 노드 없음 (외생 — 행동까지 통째로 예측).

## 7. 마스킹 (학습 augmentation)

- 대상: **양 팀 유닛 슬롯**, h1·h2에서 mask token으로 교체. window당 0~N개 (config).
  anchor·액션 노드·지형·임무는 불가.
- 역할 분담:
  - 가려진 RED (+액션 없음) → 순수 상호작용 추론 (BLUE 배치에 대한 반응 정책)
  - 가려진 BLUE (+계획 액션 공개) → 실행 의미론 (명령→실제 결과: 우회·클램프·강등·사망)
- 별도 손실 없음: 가려진 유닛의 h1,h2가 예측 타깃에 추가될 뿐, 같은 잔차 손실.
- 추론에서는 마스킹 없음. belief(미관측 RED)는 이 메커니즘의 추론 사용이며 코어 범위 밖
  (13절).

## 8. 출력 head / 손실 (전부 frame a 기준 누적 잔차)

```
L = 3·L1(Δ위치) + 8·MSE(Δ피해) + 1·MSE(Δ탄약) + 1·MSE(heading cos,sin) + 1·BCE(completion)
```

| 출력 | 방식 | 비고 |
|---|---|---|
| Δ위치 | 잔차, L1 | 앵커 시점 사망 유닛은 손실 마스크 (자명한 0으로 지표 착시 방지) |
| Δ피해 | 잔차 | hp̂ = hp(a) − Δ̂, clamp ≥0. 유일한 HP 지도 |
| Δ탄약 | 잔차 | 동일 논리 |
| heading | 절대 cos,sin | 각도 잔차는 wraparound 때문에 제외 |
| completion | BCE | 목표 상호작용 지도 신호. **소비는 유도값** (자기모순 방지) |
| alive | 없음 | hp̂>0 유도 |
| 지형 | 없음 | 복사 |
| 임무 | completion 외 없음 | time_remaining은 −k/60 산술, objective 복사 |

- 절대좌표 디코더 없음. 이중 경로 없음. 적용 범위: 전 유닛의 f1..f6 + 가려진 유닛의 h1,h2.

## 9. 조립 (상상)

```
x̂_k = x(a) + Δ̂_k    (k마다 독립, 합산 없음)
```

- 물리 클램프: 프레임 간 이동 ≤ 1.0, 사망 후 동결 (추론 전용, gradient 무관).
- ENGAGE 게이트 **없음** (6절의 계획 토큰이 대체).
- ŝ에 임무 슬롯 완성: objective 복사, time_remaining 산술, completion 유도.

## 10. 채점

```
score(후보) = Σ_{k=1..6} [ 1·RED피해_k − 1·BLUE피해_k + 1·목표거리감소_k ] + 1·V(ŝ₆)
```

- 전 항 모델 예측에서 계산. 가중·λ는 config (초기값 위). destroy_all이면 목표거리 항 0.
- V만 쓰지 않는 이유(실측): 후보 간 차이(≤6유닛)가 만드는 V 차이가 라벨 분산(남은 54초)에
  묻혀 rho +0.119 부호 역전까지 관측됨. horizon 안은 dense 보상, 너머만 V.
- V: 슬롯 인코더+pooling → progress **15-step 증가분** 라벨 (진행 중 합의로 MC 최종값에서
  변경 — 채점과 망원 합 정합, 이중계산 방지).
- **V의 학습 입력은 상상 상태다**: 실제 실행 명령을 월드모델에 넣어 그린 ŝ₆
  (학습 분포 = 추론 분포). 실측 상태로 학습한 V는 상상 위에서 순위가 뒤집혔다
  (실측 분해: V↔연속실현 −0.31, 실측 상태 위에선 +0.56). 라벨은 로그의 실제
  15초 증가분 그대로 — "모델이 이 그림을 그릴 때 현실은 이만큼 간다"를 배운다.
- **월드모델을 재학습하면 V 표본도 재생성한다** — V는 자기 월드모델 버전에 결합된다.

## 11. CEM

- 목적 지향 제안(온도 방식) 유지, step0 hard mask 유지(현재 상태 기준, 정확).
  이후 스텝 마스크 없음 — 실행 의미론은 모델이 처리.
- 6틱 plan commit + 실행 계층의 재표적 fallback 유지.

## 12. 판정 기준

1. (2단계, 오프라인) RED t+2 위치 오차 < 9.8m — 기존 판정 기준 재사용. 위치 L1이
   실제로 하강하는가 (구 시스템은 1,000 epoch 동안 0.04 정체).
2. (3단계) ENGAGE 계획 대비 실행률 22% → 70%+, 순위상관 부호 정상.
3. (4단계) rule 대비 승률 (rule: destroy 계열 3.6~9.6%).

## 13. 비범위 (지금 안 함)

- belief 추론 경로 (마스킹 메커니즘의 추론 사용 — 아키텍처 변경 없이 나중에 켠다)
- policy head (규약: policy 학습 안 함), devs warmup/refresh, 자기회귀 롤아웃,
  attention key 제한(idpos) — ablation 플래그로만
- RED decision_delay 수정 (게임 밸런스 전체가 바뀜 — 해커톤 내 금지)

## 14. 미결 (구현 중 처리)

- **계획 명령 로깅**: 구 CEM 에피소드는 강등된 ENGAGE의 계획 표적 id가 로그에 없다.
  rule 에피소드(statickv_rule, blockfix 계열)는 계획=실행이라 그대로 사용 가능 —
  1~2단계는 이것으로 충분. wm2 에피소드 기록기는 계획 명령을 별도 필드로 남긴다.
- 마스킹 유닛 수/비율, 손실 가중 미세값: config 기본값으로 두고 측정으로 조정.

### 14b. 카운터팩추얼 v3 — 통제 패턴 (2026-08-13 추가)

run4 실측: 사수·표적 identity를 토큰에 묶어도 8~9유닛 유지사격의 상상 피해가
실측의 10~20배(112 vs 4.8HP)로 남았다. 원인은 아키텍처가 아니라 데이터 —
무작위 계획(v2)은 표본마다 사수들이 제각각 거리에 흩어져 "engage 토큰 수 → 피해"
상관만 남고, 사거리·LOS의 인과가 분리되지 않는다. v3는 후보의 60%를 통제 패턴으로
생성한다 (`counterfactual.PATTERN_PROBS`): hold_fire(전원 제자리 최근접 사격) /
close_fire(3틱 접근 후 3틱 사격) / single_shooter(1명만 사격). 전대가 같은 조건이
되므로 그 거리의 실제 화력이 그대로 레이블이 된다. 소스는 rule 에피소드 + loop CEM
에피소드(교착 거리대 상태 포함). 검증 게이트: 상상에서 close_fire > hold_fire 순서
재현 + 과대평가 ≤2~3배.

run5b 3대역 프로브(2026-08-13): 표본이 밀집된 4~7u만 보정됐다(2.2×)— 7.5~10u 8.9×,
10~14u 84×로 사격확률 계단(4/7/10u)이 완만한 경사로 뭉개져 있었다. 후속 조치 둘 다
데이터 배분 도구다: ① base tick을 접촉 거리 대역별로 균등 표집(계층화, 기본 동작화 —
자연 표집은 에피소드가 머무는 거리로 쏠린다) ② train_wm `--cf-repeat`로 CF gradient
비중 확대 (CF는 rule 대비 1/10 수준이라 그대로는 밀린다).

## 15. 패키지 구조

```
wm2/
  config.py          # 전 스위치 dataclass + 스냅샷
  data/episodes.py   # 에피소드 디렉터리 → 프레임 (CSV 직접 파싱, 구코드 import 없음)
  data/windows.py    # 프레임 → window (pre|a|h1 h2|f1..6), vx,vy, 마스킹, 잔차 레이블
  data/scenarios.py  # 맵 풀, 홀드아웃 하드코딩 배제
  model/features.py  # 슬롯 스키마, 블록 layout, 규약 상수(이동상한 1.0)의 유일 정의처
  model/encoder.py   # 블록 인코더
  model/predictor.py # transformer: 가시성 행렬(5절), identity 리셋, anchor query
  model/heads.py     # Δ위치·Δ피해·Δ탄약·heading·completion
  model/losses.py    # 8절 손실 + 블록별 로깅
  model/rollout.py   # 조립 + 물리 클램프
  value/head.py      # V 재구현
  plan/cem.py        # 제안 + step0 mask
  plan/score.py      # 10절 채점
  train/train_wm.py  # 오프라인 학습
  train/loop.py      # 에피소딕 루프 (adapter 경유)
  sim/adapter.py     # 구 시뮬 격리 (유일한 구코드 접점)
  eval/position.py   # 판정 1  /  eval/planning.py  # 판정 2
```

진행: 1단계 data(골든 테스트: 구 window와 수치 대조) → 2단계 model+오프라인 학습(판정 1)
→ 3단계 rollout+CEM+채점(판정 2) → 4단계 adapter+루프(판정 3). 2단계까지 시뮬 불필요 —
기존 rule 에피소드(statickv_rule 400 + blockfix 계열 ~600×3)로 바로 학습.
