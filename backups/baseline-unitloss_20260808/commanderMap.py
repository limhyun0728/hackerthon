"""commanderMap.py

이미지 기반 지휘관용 belief map 컴포넌트.

역할:
- 매 tick contact_buffer(교신 중 유닛 관측)와 out_of_contact 목록을 받아 지휘관의 belief 상태를 갱신
- belief를 (1) PNG 지도로 렌더(VLM 입력용), (2) hp/ammo 텍스트 사이드카로 출력, (3) 롱포맷 belief 로그 행으로 출력
- 적 위치는 fog-of-war:
    · 관측 순간 World의 적 x/y에 매 tick 작은 측정 오차를 더함
    · 현재 tick에 관측된 살아있는 적만 지도에 표시
    · DESTROYED 보고가 오면 alive memory에서 제거하고 지도에서는 숨김
- hp/ammo 같은 이산 수치는 지도에 안 그리고 텍스트로 넘긴다.

가드/폴백 없음:
필요한 키가 비면 그대로 에러로 멈춘다.
"""

import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from hackerthon.commander_helpers import (
    GRID_COLUMNS,
    GRID_ROWS,
    GRID_X_MAX,
    GRID_X_MIN,
    GRID_Y_MAX,
    GRID_Y_MIN,
)


FRIENDLY_COLORS = ["#1d4ed8", "#2563eb", "#0284c7", "#0891b2", "#0ea5e9"]
ENEMY_COLORS = ["#dc2626", "#b91c1c", "#e11d48", "#f97316", "#991b1b"]

# 관측 순간 World의 정확한 적 x/y에 tick별 측정 오차를 더한다.
# 전투 판정용 전역 random과 분리된 seed를 사용해 지도 노이즈가 명중 난수에 영향을 주지 않는다.
ENEMY_POSITION_NOISE_MAX = 0.5
ENEMY_POSITION_NOISE_SEED = 42000


def _friendly_color(uid):
    # 수정: 아군은 모두 blue-family 안에서 유닛별로 구분한다.
    return FRIENDLY_COLORS[(int(uid) - 101) % len(FRIENDLY_COLORS)]


def _enemy_color(eid):
    # 수정: 적군은 모두 red/orange-family 안에서 적별로 구분한다.
    return ENEMY_COLORS[(int(eid) - 201) % len(ENEMY_COLORS)]


def _label_box(color, alpha=0.82):
    return dict(facecolor="white", edgecolor=color, alpha=alpha, pad=0.25)


def _believed_enemy_xy(belief, target_id):
    """
    지휘관이 믿는 표적 위치.
    ENGAGE 사격선 끝점 계산용.
    """
    info = belief["enemies_alive"].get(target_id)

    return info["pos"] if info is not None else None


class CommanderMap:
    def __init__(self):
        # tick을 가로질러 유지하는 belief 상태.
        self.last_known_pos = {}             # friendly uid -> (x, y)
        self.dead_enemies = {}               # enemy id -> (x, y)
        self._belief_tick = 0

    def _enemy_position_noise(self, enemy_id):
        """현재 belief tick과 적 ID에 따라 매 tick 달라지는 위치 오차를 반환한다."""
        rng = random.Random(
            ENEMY_POSITION_NOISE_SEED
            + self._belief_tick * 1000
            + int(enemy_id)
        )
        return (
            rng.uniform(-ENEMY_POSITION_NOISE_MAX, ENEMY_POSITION_NOISE_MAX),
            rng.uniform(-ENEMY_POSITION_NOISE_MAX, ENEMY_POSITION_NOISE_MAX),
        )

    def update_belief(self, contact_buffer, out_of_contact_ids):
        """
        contact_buffer:
            {uid: observation}
            이번 tick에 교신 중인 아군만 포함.

        out_of_contact_ids:
            [uid, ...]
            통제 편제 중 이번 tick 관측 끊긴 아군.

        enemy memory policy:
            - 현재 tick에서 관측된 alive enemy만 지도에 표시.
            - 현재 tick에서 관측되지 않은 enemy는 지도에서 즉시 제거.
            - DESTROYED 보고가 들어온 enemy는 dead_enemies tombstone에 기록.
        """
        self._belief_tick += 1

        friendlies = {}

        # 1) 교신 중 아군: 실위치 + hp/ammo, last_known 갱신
        for uid, obs in contact_buffer.items():
            s = obs["self"]
            pos = (s["x"], s["y"])

            self.last_known_pos[uid] = pos

            friendlies[uid] = {
                "pos": pos,
                "status": "in_contact",
                "hp": s["hp"],
                "ammo": s["ammo"],
                "mode": s.get("mode", "IDLE"),
                "target_id": s.get("target_id"),
            }

        # 2) 두절 아군: 마지막 교신 위치에 freeze
        for uid in out_of_contact_ids:
            if uid in self.last_known_pos:
                friendlies[uid] = {
                    "pos": self.last_known_pos[uid],
                    "status": "out_of_contact",
                    "hp": None,
                    "ammo": None,
                    "mode": None,
                    "target_id": None,
                }

        # 3) 죽은 적: 사망 보고를 누적하고 alive memory에서 제거
        for uid, obs in contact_buffer.items():
            for ent in obs["visible_entities"]:
                if (
                    ent["type"] == "enemy"
                    and (
                        ent["state"] == "DESTROYED"
                        or float(ent.get("hp", 1)) <= 0.0
                    )
                ):
                    self.dead_enemies[ent["id"]] = (
                        float(ent["x"]),
                        float(ent["y"]),
                    )

        # 4) 현재 tick에서 관측된 살아있는 적 sighting 수집
        # Each entry: (enemy_x, enemy_y, hp_or_None, heading_or_None)
        sightings = {}  # enemy_id -> [(...), ...]

        for uid, obs in contact_buffer.items():
            for ent in obs["visible_entities"]:
                if ent["type"] != "enemy":
                    continue

                if ent["state"] == "DESTROYED" or float(ent.get("hp", 1)) <= 0.0:
                    continue

                if ent["id"] in self.dead_enemies:
                    continue

                sightings.setdefault(ent["id"], []).append(
                    (
                        float(ent["x"]),
                        float(ent["y"]),
                        ent.get("hp"),
                        ent.get("heading"),
                    )
                )

        # 5) 현재 tick 관측만으로 적 지도를 새로 구성한다.
        enemies_alive = {}

        # 6) 현재 tick에서 관측된 적 위치 갱신
        for eid, obs_list in sightings.items():
            fused = self._fuse_enemy(eid, obs_list)
            fused["observed_this_tick"] = True
            fused["last_seen_tick"] = self._belief_tick

            enemies_alive[eid] = fused

        return {
            "friendlies": friendlies,
            "enemies_alive": enemies_alive,
            "enemies_dead": dict(self.dead_enemies),
        }

    def _fuse_enemy(self, enemy_id, obs_list):
        """
        관측 순간 World가 제공한 적 x/y를 관측자 사이에서 평균한 뒤,
        tick별 위치 노이즈를 더한 추정 좌표를 만든다.
        obs_list entries: (enemy_x, enemy_y, hp_or_None, heading_or_None)
        """
        # Extract HP from the closest observer (most reliable reading)
        # Use the minimum known HP (most recent/conservative estimate)
        known_hps = [o[2] for o in obs_list if o[2] is not None]
        hp = min(known_hps) if known_hps else None
        known_headings = [float(o[3]) for o in obs_list if o[3] is not None]
        heading = None
        if known_headings:
            sx = sum(math.cos(math.radians(h)) for h in known_headings)
            sy = sum(math.sin(math.radians(h)) for h in known_headings)
            heading = math.degrees(math.atan2(sy, sx))

        mean_x = sum(float(obs[0]) for obs in obs_list) / len(obs_list)
        mean_y = sum(float(obs[1]) for obs in obs_list) / len(obs_list)
        noise_x, noise_y = self._enemy_position_noise(enemy_id)

        return {
            "method": "estimated_position",
            "pos": (mean_x + noise_x, mean_y + noise_y),
            "hp": hp,
            "heading": heading,
        }

    # ---- 출력 1: VLM 입력용 PNG ----
    def render(
        self,
        belief,
        sim_time,
        path,
        show_action_grid: bool = False,
        selection_view: bool = False,
    ):
        """Stage 1 또는 Stage 4 용도의 지휘관 지도를 이미지로 저장한다."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(9, 8))

        ax.set_xlim(GRID_X_MIN, GRID_X_MAX)
        ax.set_ylim(GRID_Y_MIN, GRID_Y_MAX)
        ax.set_aspect("equal")
        if selection_view:
            # Stage 4는 mark 선택에 필요 없는 수치 거리축을 숨긴다.
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"Action Mark Selection  t={sim_time}")
        else:
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.set_xticks(range(-20, 21, 5))
            ax.set_yticks(range(-15, 11, 5))
            ax.set_xlabel("x (East +)")
            ax.set_ylabel("y (North +)")
            ax.set_title(f"Commander Belief Map  t={sim_time}")

        # 8방위 방향 표시
        _dir_style = dict(fontsize=8, color="dimgray", alpha=0.75, zorder=1,
                          bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.55, edgecolor="none"))
        ax.text(0,     9.6,  "N ▲",  ha="center", va="top",    weight="bold", **_dir_style)
        ax.text(0,    -14.6, "▼ S",  ha="center", va="bottom", weight="bold", **_dir_style)
        ax.text(19.6, -2.5,  "E ▶",  ha="right",  va="center", weight="bold", **_dir_style)
        ax.text(-19.6,-2.5,  "◀ W",  ha="left",   va="center", weight="bold", **_dir_style)
        ax.text(18.5,  9.2,  "NE",   ha="right",  va="top",    **_dir_style)
        ax.text(-18.5, 9.2,  "NW",   ha="left",   va="top",    **_dir_style)
        ax.text(18.5, -14.2, "SE",   ha="right",  va="bottom", **_dir_style)
        ax.text(-18.5,-14.2, "SW",   ha="left",   va="bottom", **_dir_style)

        if show_action_grid:
            # 이동 선택 단계에서만 A1~J10 mark를 현재 지도 위에 겹친다.
            # 열은 왼쪽에서 오른쪽 A~J, 행은 아래에서 위로 1~10이다.
            cell_width = (GRID_X_MAX - GRID_X_MIN) / len(GRID_COLUMNS)
            cell_height = (GRID_Y_MAX - GRID_Y_MIN) / GRID_ROWS

            for column_index in range(1, len(GRID_COLUMNS)):
                x = GRID_X_MIN + column_index * cell_width
                ax.axvline(x, color="deepskyblue", lw=1.2, ls=":", alpha=0.65, zorder=1)
            for row_index in range(1, GRID_ROWS):
                y = GRID_Y_MIN + row_index * cell_height
                ax.axhline(y, color="deepskyblue", lw=1.2, ls=":", alpha=0.65, zorder=1)

            for column_index, column in enumerate(GRID_COLUMNS):
                for row_index in range(GRID_ROWS):
                    x = GRID_X_MIN + (column_index + 0.5) * cell_width
                    y = GRID_Y_MIN + (row_index + 0.5) * cell_height
                    ax.text(
                        x,
                        y,
                        f"{column}{row_index + 1}",
                        color="deepskyblue",
                        fontsize=7,
                        ha="center",
                        va="center",
                        weight="bold",
                        alpha=0.82,
                        bbox=_label_box("deepskyblue", alpha=0.42),
                        zorder=2,
                    )
        alive_blue_ids = sorted(
            int(uid)
            for uid, info in belief["friendlies"].items()
            if info["status"] == "in_contact" and float(info.get("hp") or 0) > 0.0
        )
        dead_blue_ids = sorted(
            int(uid)
            for uid, info in belief["friendlies"].items()
            if info["status"] == "in_contact" and info.get("hp") is not None and float(info["hp"]) <= 0.0
        )
        alive_red_ids = sorted(int(eid) for eid in belief["enemies_alive"].keys())
        blue_label = f"BLUE ALIVE: {alive_blue_ids}"
        if dead_blue_ids:
            blue_label += f"  KIA: {dead_blue_ids}"
        red_label = f"RED OBSERVED: {alive_red_ids}"
        # Unit status panel (top-left): HP + ammo for all alive blue units
        status_lines = ["BLUE STATUS"]
        for uid in alive_blue_ids:
            info = belief["friendlies"].get(uid) or belief["friendlies"].get(str(uid)) or {}
            hp = int(float(info.get("hp") or 0))
            ammo = int(float(info.get("ammo") or 0))
            status_lines.append(f"B{uid}  HP:{hp}  AM:{ammo}")
        if dead_blue_ids:
            for uid in dead_blue_ids:
                status_lines.append(f"B{uid}  KIA")
        ax.text(
            19.5, 6.5,
            "\n".join(status_lines),
            color="#1d4ed8",
            fontsize=8,
            ha="right",
            va="top",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#1d4ed8", alpha=0.88),
            zorder=8,
            visible=not selection_view,
        )

        # 수정: VLM이 이미지 숫자를 오독하더라도 살아있는 ID 목록은 지도 내부에 명시한다.
        ax.text(
            19.5,
            9.2,
            blue_label,
            color="#1d4ed8",
            fontsize=8,
            ha="right",
            va="top",
            weight="bold",
            bbox=_label_box("#1d4ed8", alpha=0.88),
            zorder=8,
            visible=not selection_view,
        )
        ax.text(
            19.5,
            8.0,
            red_label,
            color="#dc2626",
            fontsize=8,
            ha="right",
            va="top",
            weight="bold",
            bbox=_label_box("#dc2626", alpha=0.88),
            zorder=8,
            visible=not selection_view,
        )

        # Friendlies
        for uid, info in belief["friendlies"].items():
            x, y = info["pos"]
            fcolor = _friendly_color(uid)
            is_dead = info["status"] == "in_contact" and info.get("hp") is not None and float(info["hp"]) <= 0

            if is_dead:
                ax.plot(x, y, "X", color="navy", markersize=13,
                        markeredgecolor="black", markeredgewidth=1.5, zorder=3)
                ax.text(x, y + 0.7, f"B{uid} KIA", color="navy", fontsize=8,
                        ha="center", weight="bold",
                        bbox=_label_box("navy", alpha=0.65), zorder=5)

            elif info["status"] == "in_contact":
                ax.plot(
                    x,
                    y,
                    "o",
                    color=fcolor,
                    markersize=13,
                    markeredgecolor="black",
                    zorder=3,
                )

                hp_val = info.get("hp")
                hp_str = f" HP:{int(float(hp_val))}" if hp_val is not None else ""
                ax.text(
                    x,
                    y + 0.7,
                    f"B{uid}" if selection_view else f"B{uid}{hp_str}",
                    color=fcolor,
                    fontsize=9,
                    ha="center",
                    weight="bold",
                    bbox=_label_box(fcolor),
                    zorder=5,
                )

                # ENGAGE 중이면 표적까지 사격선
                if (
                    not selection_view
                    and info.get("mode") == "ENGAGE"
                    and info.get("target_id") is not None
                ):
                    tgt = _believed_enemy_xy(belief, int(info["target_id"]))

                    if tgt is not None:
                        ax.plot(
                            [x, tgt[0]],
                            [y, tgt[1]],
                            color="orange",
                            ls="-",
                            lw=1.8,
                            zorder=2,
                        )
                        ax.plot(
                            x,
                            y,
                            "o",
                            color="none",
                            markersize=17,
                            markeredgecolor="orange",
                            markeredgewidth=2.0,
                            zorder=4,
                        )

            else:
                ax.plot(
                    x,
                    y,
                    "s",
                    color="gray",
                    markersize=13,
                    markeredgecolor="black",
                    zorder=3,
                )

                ax.text(
                    x,
                    y + 0.7,
                    f"B{uid}",
                    color=fcolor,
                    fontsize=9,
                    ha="center",
                    weight="bold",
                    bbox=_label_box("gray", alpha=0.75),
                    zorder=5,
                )

                ax.text(
                    x,
                    y - 1.1,
                    "STOP/OOC",
                    color="gray",
                    fontsize=7,
                    ha="center",
                )

        # 현재 tick에 관측된 살아있는 적만 표시한다.
        for eid, info in belief["enemies_alive"].items():
            ecolor = _enemy_color(eid)

            x, y = info["pos"]

            ax.plot(
                x,
                y,
                "s",
                color=ecolor,
                markersize=13,
                markeredgecolor="black",
                zorder=3,
            )

            # 적 HP는 표적 우선순위를 암묵적으로 만들기 때문에 지도에 표시하지 않는다.
            ax.text(
                x,
                y + 0.7,
                f"R{eid}",
                color=ecolor,
                fontsize=9,
                ha="center",
                weight="bold",
                bbox=_label_box(ecolor, alpha=0.75),
                zorder=5,
            )

        # Distance lines: each alive blue unit → nearest alive red unit
        alive_reds = {
            eid: info for eid, info in belief["enemies_alive"].items()
            if "pos" in info
        }
        if alive_reds and not selection_view:
            # 우선 표적을 미리 지정하지 않고 유닛별 거리 관계만 제공한다.
            for uid, info in belief["friendlies"].items():
                if info["status"] != "in_contact" or float(info.get("hp") or 0) <= 0:
                    continue
                bx, by = info["pos"]
                nearest_eid, nearest_xy, nearest_dist = None, None, float("inf")
                for eid, einfo in alive_reds.items():
                    ex, ey = einfo["pos"]
                    d = math.hypot(ex - bx, ey - by)
                    if d < nearest_dist:
                        nearest_dist, nearest_eid, nearest_xy = d, eid, (ex, ey)
                if nearest_xy is not None:
                    mx, my = (bx + nearest_xy[0]) / 2, (by + nearest_xy[1]) / 2
                    ax.plot(
                        [bx, nearest_xy[0]], [by, nearest_xy[1]],
                        color="gray", lw=0.8, ls=":", alpha=0.5, zorder=1,
                    )
                    ax.text(
                        mx, my, f"{nearest_dist:.1f}u",
                        color="gray", fontsize=7, ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7),
                        zorder=6,
                    )

        if not selection_view:
            # 거리 판단은 Stage 1 전용이므로 Stage 4에서는 축척도 숨긴다.
            sb_x0, sb_y0 = 8.0, -11.5
            ax.plot([sb_x0, sb_x0 + 5.0], [sb_y0, sb_y0], "k-", lw=2.0, zorder=7)
            ax.plot([sb_x0, sb_x0], [sb_y0 - 0.25, sb_y0 + 0.25], "k-", lw=1.5, zorder=7)
            ax.plot([sb_x0 + 5.0, sb_x0 + 5.0], [sb_y0 - 0.25, sb_y0 + 0.25], "k-", lw=1.5, zorder=7)
            ax.text(sb_x0 + 2.5, sb_y0 + 0.4, "5 units", fontsize=7.5, ha="center", va="bottom", color="black", zorder=7)
            for i, (speed, dist, lw) in enumerate([("fast", 1.5, 2.2), ("normal", 1.0, 1.8), ("slow", 0.5, 1.4)]):
                sy = sb_y0 - 1.1 - i * 0.85
                ax.plot([sb_x0, sb_x0 + dist], [sy, sy], color="purple", lw=lw, zorder=7)
                ax.plot([sb_x0, sb_x0], [sy - 0.18, sy + 0.18], color="purple", lw=1.2, zorder=7)
                ax.plot([sb_x0 + dist, sb_x0 + dist], [sy - 0.18, sy + 0.18], color="purple", lw=1.2, zorder=7)
                ax.text(sb_x0 + dist + 0.15, sy, f"{speed} ({dist}u)", fontsize=6.5, va="center", color="purple", zorder=7)

        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="blue",
                markersize=11,
                label="friendly (in contact)",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor="gray",
                markersize=11,
                label="friendly (out of contact / STOP)",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="X",
                color="w",
                markerfacecolor="navy",
                markeredgecolor="black",
                markersize=11,
                label="friendly (KIA)",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor="red",
                markersize=11,
                label="enemy (current estimated)",
            ),
            plt.Line2D(
                [0],
                [0],
                color="orange",
                lw=1.8,
                label="firing at target",
            ),
        ]

        if not selection_view:
            ax.legend(handles=handles, loc="upper left", fontsize=8)

        fig.tight_layout()
        fig.savefig(str(path), dpi=130)
        plt.close(fig)

        return str(path)

    # ---- 출력 2: hp/ammo 텍스트 사이드카 ----
    def text_sidecar(self, belief):
        alive_in_contact = sorted(
            uid
            for uid, info in belief["friendlies"].items()
            if info["status"] == "in_contact" and float(info.get("hp") or 0) > 0
        )

        dead_in_contact = sorted(
            uid
            for uid, info in belief["friendlies"].items()
            if info["status"] == "in_contact" and info.get("hp") is not None and float(info["hp"]) <= 0
        )

        ooc = sorted(
            uid
            for uid, info in belief["friendlies"].items()
            if info["status"] == "out_of_contact"
        )

        current_enemy_ids = sorted(belief["enemies_alive"])

        dead_enemy_ids = sorted(belief["enemies_dead"].keys())

        lines = []

        lines.append(f"[IN-CONTACT UNIT IDS - assign EACH of these integer ids exactly once]: {alive_in_contact}")
        lines.append("[UNIT HP/AMMO]")

        for uid in alive_in_contact:
            info = belief["friendlies"][uid]
            lines.append(f"- unit {uid}: hp={info['hp']}, ammo={info['ammo']}")

        if dead_in_contact:
            lines.append(f"[FRIENDLY KIA - do NOT assign commands to these]: {dead_in_contact}")

        lines.append(f"[OUT-OF-CONTACT UNIT IDS - do NOT assign these]: {ooc}")

        lines.append("[CURRENT ENEMY OBSERVATIONS]")
        lines.append(f"- current_observed_enemy_ids={current_enemy_ids}")
        lines.append(f"- dead_enemy_ids={dead_enemy_ids}")

        return "\n".join(lines)

    # ---- 출력 3: belief 로그 행 ----
    def belief_rows(self, belief, sim_time):
        rows = []

        for uid, info in belief["friendlies"].items():
            x, y = info["pos"]

            rows.append(
                {
                    "time": sim_time,
                    "entity_id": uid,
                    "kind": "friendly",
                    "status": info["status"],
                    "x": round(x, 3),
                    "y": round(y, 3),
                    "bearing_deg": "",
                    "bucket": "",
                    "observed_this_tick": info["status"] == "in_contact",
                    "last_seen_tick": "",
                    "mode": info.get("mode") or "",
                    "target_id": info.get("target_id") if info.get("target_id") is not None else "",
                }
            )

        for eid, info in belief["enemies_alive"].items():
            last_seen_tick = info.get("last_seen_tick", "")

            x, y = info["pos"]

            rows.append(
                {
                    "time": sim_time,
                    "entity_id": eid,
                    "kind": "enemy",
                    "status": "estimated_current",
                    "x": round(x, 3),
                    "y": round(y, 3),
                    "bearing_deg": "",
                    "bucket": "",
                    "observed_this_tick": True,
                    "last_seen_tick": last_seen_tick,
                }
            )

        for eid, (x, y) in belief["enemies_dead"].items():
            rows.append(
                {
                    "time": sim_time,
                    "entity_id": eid,
                    "kind": "enemy",
                    "status": "dead",
                    "x": round(x, 3),
                    "y": round(y, 3),
                    "bearing_deg": "",
                    "bucket": "",
                    "observed_this_tick": True,
                    "last_seen_tick": "",
                }
            )

        return rows
