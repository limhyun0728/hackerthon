"""지휘결심 플랫폼 ↔ wm2 value head 다리 (2026-08-15).

플랫폼의 DEVS 실측 롤아웃 결과(unit rows)를 wm2 value head 입력으로 변환해
안전형(β=1)·득점형(β=0) 두 관점의 점수를 만든다. 채점식은 wm2 학습·평가와 동일한
벨만 정합 형태:

    score = [진행(끝) − 진행(지금)] + β·[아군HP비(끝) − 아군HP비(지금)] + γ^H·V(끝 상태)

DEVS 롤아웃 상태는 실측 분포이므로 **실측 입력판** V를 쓴다:
    안전형 --wm2-safe-value  checkpoints/wm2_value_rtgs_real.pt (β=1)
    득점형 --wm2-score-value checkpoints/wm2_value_rtg_real.pt  (β=0)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from types import SimpleNamespace

from wm2.config import ModelConfig, SURVIVAL_BETA, VALUE_GAMMA
from wm2.data.windows import EpisodeLayout, Window, _terrain_features as _wm2_terrain, _unit_vector
from wm2.model.features import (
    ACTION_DIM,
    MAX_AMMO,
    MAX_HP,
    TeamId,
    denorm_x,
    denorm_y,
    norm_height,
    norm_width,
    norm_x,
    norm_y,
)
from wm2.model.heads import WM2Heads
from wm2.model.predictor import WM2Predictor
from wm2.model.rollout import assemble_hp, assemble_positions, clamp_physics
from wm2.plan.cem import HORIZON, PlanCandidates, _action_tokens
from wm2.plan.score import progress_batch
from wm2.value.head import load_value_head

LENS_BETA = {"safe": float(SURVIVAL_BETA), "score": 0.0}
GAMMA_H = float(VALUE_GAMMA) ** HORIZON   # ≈ 0.83 — 6틱 뒤 가치의 벨만 가중


def load_lens_heads(paths: dict[str, Path], device: torch.device) -> dict[str, Any]:
    heads = {}
    for lens, path in paths.items():
        heads[lens] = load_value_head(Path(path), device)
        heads[lens].eval()
        print(f"wm2 value head 로드 [{lens}]: {Path(path).name} (β={LENS_BETA[lens]:g})")
    return heads


def _terrain_tensor(obstacles, device) -> torch.Tensor:
    """wm2 windows._terrain_features 인코딩의 미러 — (T, 9)."""
    rows = []
    for xmin, ymin, xmax, ymax in obstacles:
        w, h = float(xmax) - float(xmin), float(ymax) - float(ymin)
        rows.append([
            1.0, norm_x(float(xmin) + w / 2.0), norm_y(float(ymin) + h / 2.0),
            norm_width(w), norm_height(h), 0.0, 1.0, 1.0, 1.0,
        ])
    array = np.asarray(rows, dtype=np.float32).reshape(-1, 9)
    return torch.from_numpy(array).to(device)


class LensScorer:
    """한 결심 시점의 후보들을 두 관점으로 채점한다. build_archive마다 새로 만든다."""

    def __init__(
        self,
        heads: dict[str, Any],
        *,
        device: torch.device,
        obstacles,
        mission_type: int,
        objective: tuple[float, float],
        duration_sec: float,
        current_rows: list[dict[str, Any]],
    ):
        self.heads = heads
        self.device = device
        self.mission_type = int(mission_type)
        self.objective = (float(objective[0]), float(objective[1]))
        self.duration = float(duration_sec)
        self.terrain = _terrain_tensor(obstacles, device)
        self.unit_ids = sorted(int(r["id"]) for r in current_rows)
        self.team_ids = torch.tensor(
            [int(TeamId.BLUE) if uid < 200 else int(TeamId.RED) for uid in self.unit_ids],
            dtype=torch.long, device=device,
        )
        self.blue_ids = [u for u in self.unit_ids if u < 200]
        self._blue_denom = max(len(self.blue_ids), 1) * MAX_HP
        self.progress_now = self._progress(current_rows)
        self.survival_now = self._survival(current_rows)

    # ── 내부 변환 ────────────────────────────────────────────────────────
    def _pos_hp(self, rows) -> tuple[torch.Tensor, torch.Tensor]:
        by_id = {int(r["id"]): r for r in rows}
        pos = torch.tensor(
            [[float(by_id[u]["x"]), float(by_id[u]["y"])] for u in self.unit_ids],
            dtype=torch.float32, device=self.device,
        ).unsqueeze(0)
        hp = torch.tensor(
            [max(0.0, float(by_id[u]["hp"])) for u in self.unit_ids],
            dtype=torch.float32, device=self.device,
        ).unsqueeze(0)
        return pos, hp

    def _progress(self, rows) -> float:
        pos, hp = self._pos_hp(rows)
        return float(progress_batch(pos, hp, self.team_ids, self.mission_type, self.objective)[0])

    def _survival(self, rows) -> float:
        by_id = {int(r["id"]): r for r in rows}
        return sum(max(0.0, float(by_id[u]["hp"])) for u in self.blue_ids) / self._blue_denom

    def _unit_features(self, rows_end, rows_prev) -> torch.Tensor:
        end = {int(r["id"]): r for r in rows_end}
        prev = {int(r["id"]): r for r in rows_prev}
        feats = []
        for uid in self.unit_ids:
            r = end[uid]
            p = prev.get(uid, r)
            hp = max(0.0, float(r["hp"]))
            alive = 1.0 if hp > 0.0 else 0.0
            heading = math.radians(float(r.get("heading", 0.0)))
            feats.append([
                float(int(TeamId.BLUE) if uid < 200 else int(TeamId.RED)),
                hp / MAX_HP,
                max(0.0, float(r.get("ammo", 0))) / MAX_AMMO,
                norm_x(float(r["x"])), norm_y(float(r["y"])),
                math.cos(heading), math.sin(heading),
                alive,
                (float(r["x"]) - float(p["x"])) * alive,
                (float(r["y"]) - float(p["y"])) * alive,
            ])
        return torch.tensor([feats], dtype=torch.float32, device=self.device)

    # ── 공개 API ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def score(self, rows_end, rows_prev, t_end: float) -> dict[str, float]:
        progress_end = self._progress(rows_end)
        survival_end = self._survival(rows_end)
        unit = self._unit_features(rows_end, rows_prev)
        mission = torch.tensor([[
            float(self.mission_type),
            norm_x(self.objective[0]), norm_y(self.objective[1]),
            max(0.0, (self.duration - float(t_end)) / self.duration),
            1.0 if progress_end >= 1.0 - 1e-6 else 0.0,
        ]], dtype=torch.float32, device=self.device)
        terrain = self.terrain.unsqueeze(0)
        out = {}
        for lens, head in self.heads.items():
            value = float(head(
                unit_features=unit, terrain_features=terrain,
                mission_features=mission, team_ids=self.team_ids,
            )[0])
            out[lens] = (
                (progress_end - self.progress_now)
                + LENS_BETA[lens] * (survival_end - self.survival_now)
                + GAMMA_H * value
            )
        return out


class Wm2Rollout:
    """플랫폼 후보 계획을 wm2 월드모델(run13) 상상으로 굴린다 (--archive-backend wm2).

    구 계약(FutureActionPlanBatch)을 wm2 PlanCandidates로 되돌리고, 현재 belief rows에서
    계획용 Window를 만들어(관측 이력은 현재 상태 복제 — 구 model 백엔드와 같은 규약)
    predictor를 돌린 뒤, 결과를 구 슬롯 특징 배열 (C, H, N_slots, F)로 복원한다.
    다운스트림(_rows_from_features, _build_path)이 읽는 유닛 슬롯 인덱스 0~7만 채운다.
    """

    def __init__(self, checkpoint: Path, device: torch.device):
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        self.model = WM2Predictor(ModelConfig()).to(device)
        self.model.load_state_dict(payload["model"]); self.model.eval()
        self.heads = WM2Heads(ModelConfig()).to(device)
        self.heads.load_state_dict(payload["heads"]); self.heads.eval()
        self.device = device
        print(f"wm2 월드모델 로드: {Path(checkpoint).name} (아카이브 상상 백엔드)")

    def _window(self, rows, obstacles, mission_type, objective, duration_sec, time_sec,
                history_rows=None):
        """history_rows: 직전 belief 프레임들(과거→최근, 최대 2개). 없으면 현재 복제.

        RED 미래 예측은 관측 이력의 속도 신호에 크게 의존한다 — 정지 이력을 복제하면
        run13이 RED를 제자리로 상상해 셀 미리보기의 적 궤적이 죽는다.
        """
        unit_ids = tuple(sorted(int(r["id"]) for r in rows))
        by_id = {int(r["id"]): r for r in rows}
        team_ids = np.asarray(
            [int(TeamId.BLUE) if u < 200 else int(TeamId.RED) for u in unit_ids],
            dtype=np.int64,
        )
        num_blue = sum(1 for u in unit_ids if u < 200)

        class _E:
            pass
        _E.obstacles = tuple(tuple(float(v) for v in r) for r in obstacles)
        layout = EpisodeLayout(
            unit_ids=unit_ids, team_ids=team_ids, num_blue=num_blue,
            terrain_features=_wm2_terrain(_E), mission_type=int(mission_type),
            objective=(float(objective[0]), float(objective[1])),
            duration_sec=float(duration_sec),
        )
        def _states(frame_rows):
            frame_by_id = {int(r["id"]): r for r in frame_rows}
            return {
                u: SimpleNamespace(
                    x=float(frame_by_id.get(u, by_id[u])["x"]),
                    y=float(frame_by_id.get(u, by_id[u])["y"]),
                    heading_deg=float(frame_by_id.get(u, by_id[u]).get("heading", 0.0)),
                    hp=max(0.0, float(frame_by_id.get(u, by_id[u])["hp"])),
                    ammo=max(0.0, float(frame_by_id.get(u, by_id[u]).get("ammo", 0))),
                )
                for u in unit_ids
            }

        history = list(history_rows or [])[-2:]
        frames = [_states(f) for f in history] + [_states(rows)]
        while len(frames) < 3:                     # 이력 부족분은 가장 오래된 프레임 복제
            frames.insert(0, frames[0])
        unit_features = np.zeros((9, len(unit_ids), 10), dtype=np.float32)
        for fi in range(3):
            prev = frames[fi - 1] if fi > 0 else frames[0]
            for ui, uid in enumerate(unit_ids):
                unit_features[fi, ui] = _unit_vector(frames[fi][uid], prev[uid], int(team_ids[ui]))
        mission_features = np.zeros((9, 5), dtype=np.float32)
        tr = max(0.0, (float(duration_sec) - float(time_sec)) / float(duration_sec))
        for fi in range(3):
            mission_features[fi] = np.asarray(
                [float(mission_type), norm_x(float(objective[0])), norm_y(float(objective[1])), tr, 0.0],
                dtype=np.float32,
            )
        zeros8 = np.zeros((8, len(unit_ids)), dtype=np.float32)
        return Window(
            layout=layout, anchor_tick=max(0, int(time_sec) - 2),
            unit_features=unit_features, mission_features=mission_features,
            dpos=np.zeros((8, len(unit_ids), 2), dtype=np.float32),
            ddmg=zeros8, dammo=zeros8,
            heading=np.zeros((8, len(unit_ids), 2), dtype=np.float32),
            completion=np.zeros(6, dtype=np.float32),
            pos_loss_mask=np.asarray([frames[-1][u].hp > 0 for u in unit_ids], dtype=bool),
            actions=np.zeros((8, num_blue, ACTION_DIM), dtype=np.float32),
        )

    @staticmethod
    def _to_candidates(plans, red_ids: list[int]) -> PlanCandidates:
        types = plans.action_type_ids.detach().cpu().numpy().astype(np.int64)
        issued = plans.issued_mask.detach().cpu().numpy().astype(bool)
        theta = plans.theta_radians.detach().cpu().numpy().astype(np.float32)
        move_norm = plans.move_xy_norm.detach().cpu().numpy()
        move_xy = np.stack(
            [np.vectorize(denorm_x)(move_norm[..., 0]), np.vectorize(denorm_y)(move_norm[..., 1])],
            axis=-1,
        ).astype(np.float32)
        slot_by_red = {int(rid): i for i, rid in enumerate(sorted(red_ids))}
        targets = plans.target_entity_ids.detach().cpu().numpy()
        target_slots = np.full(types.shape, -1, dtype=np.int64)
        for rid, slot in slot_by_red.items():
            target_slots[targets == rid] = slot
        return PlanCandidates(types, move_xy, target_slots, theta, issued)

    def make_fn(self, *, current_rows, obstacles, mission_type, objective,
                duration_sec, time_sec, batch, history_rows=None):
        window = self._window(
            current_rows, obstacles, mission_type, objective, duration_sec, time_sec,
            history_rows=history_rows,
        )
        layout = window.layout
        red_ids = [u for u in layout.unit_ids if u >= 200]
        slot_index = {int(e): i for i, e in enumerate(batch.entity_ids)}
        unit_slots = np.asarray([slot_index[int(u)] for u in layout.unit_ids])
        num_slots = int(batch.features.shape[0])
        feat_dim = int(batch.features.shape[-1])
        device = self.device

        feats0 = window.unit_features[0]
        anchor_xy_t = torch.from_numpy(np.stack(
            [[denorm_x(v) for v in feats0[:, 3]], [denorm_y(v) for v in feats0[:, 4]]], axis=-1
        ).astype(np.float32)).to(device)
        anchor_hp_t = torch.from_numpy((feats0[:, 1] * MAX_HP).astype(np.float32)).to(device)
        anchor_ammo_t = torch.from_numpy((feats0[:, 2] * MAX_AMMO).astype(np.float32)).to(device)
        unit_t = torch.from_numpy(window.unit_features).to(device)
        terrain_t = torch.from_numpy(layout.terrain_features).to(device)
        mission_t = torch.from_numpy(window.mission_features).to(device)
        team_t = torch.from_numpy(np.asarray(layout.team_ids)).to(device)
        team_val = np.asarray(layout.team_ids, dtype=np.float32)

        @torch.no_grad()
        def rollout_fn(plans) -> torch.Tensor:
            candidates = self._to_candidates(plans, red_ids)
            tokens = torch.from_numpy(_action_tokens(window, candidates)).to(device)
            c = tokens.shape[0]
            out = np.zeros((c, HORIZON, num_slots, feat_dim), dtype=np.float32)
            for start in range(0, c, 128):
                end = min(start + 128, c)
                b = end - start
                pred_in = self.model(
                    unit_features=unit_t.unsqueeze(0).expand(b, -1, -1, -1),
                    terrain_features=terrain_t.unsqueeze(0).expand(b, -1, -1),
                    mission_features=mission_t.unsqueeze(0).expand(b, -1, -1),
                    actions=tokens[start:end],
                    team_ids=team_t,
                    masked_units=torch.zeros(b, layout.num_units, dtype=torch.bool, device=device),
                )
                pred = self.heads(pred_in["unit_tokens"], pred_in["mission_tokens"])
                raw = assemble_positions(anchor_xy_t.unsqueeze(0).expand(b, -1, -1), pred["dpos"][:, 2:])
                hp = assemble_hp(anchor_hp_t.unsqueeze(0).expand(b, -1), pred["ddmg"][:, 2:])
                clamped = clamp_physics(
                    raw, anchor_xy_t.unsqueeze(0).expand(b, -1, -1),
                    hp, (anchor_hp_t > 0).unsqueeze(0).expand(b, -1),
                ).cpu().numpy()
                hp_np = hp.cpu().numpy()
                ammo_np = (
                    anchor_ammo_t.reshape(1, 1, -1)
                    - pred["dammo"][:, 2:].clamp_min(0.0) * MAX_AMMO
                ).clamp_min(0.0).cpu().numpy()
                head_np = pred["heading"][:, 2:].cpu().numpy()
                blk = out[start:end]
                blk[:, :, unit_slots, 0] = team_val.reshape(1, 1, -1)
                blk[:, :, unit_slots, 1] = np.clip(hp_np / MAX_HP, 0.0, 1.0)
                blk[:, :, unit_slots, 2] = np.clip(ammo_np / MAX_AMMO, 0.0, 1.0)
                blk[:, :, unit_slots, 3] = np.vectorize(norm_x)(clamped[..., 0])
                blk[:, :, unit_slots, 4] = np.vectorize(norm_y)(clamped[..., 1])
                blk[:, :, unit_slots, 5] = head_np[..., 0]
                blk[:, :, unit_slots, 6] = head_np[..., 1]
                if feat_dim > 7:
                    blk[:, :, unit_slots, 7] = (hp_np > 0.0).astype(np.float32)
            return torch.from_numpy(out)

        return rollout_fn
