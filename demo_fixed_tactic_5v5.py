"""선택한 전술과 최초 역할 배정을 끝까지 유지하는 5대5 비교 데모.

기존 ``simulation_direct_commander_5v5.py``와 전투/실행 코드는 그대로
재사용한다. 이 파일에서 달라지는 것은 다음 두 가지뿐이다.

1. CLI에서 지정한 전술만 Stage 2가 선택할 수 있다.
2. Stage 2는 최초 한 번만 실행되며 전술과 생존 유닛의 기존 역할을 재배정하지 않는다.
3. Blue는 첫 관측부터 모든 Red 위치를 알 수 있도록 전역 360도 관측을 사용한다.

Stage 1 상황평가, Stage 3 매 tick 행동 선택, Stage 4 mark 선택, Stage 5 명령
변환은 기존 ``CommanderPolicy`` 구현을 그대로 사용한다. 따라서 이 데모는 전술
재선택을 제거했을 때 현재 이동 선택기가 전술을 얼마나 일관되게 수행하는지 보는
격리 실험이다.
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pypdevs.DEVS import CoupledDEVS
from pypdevs.simulator import Simulator

from hackerthon.Soldier import SoldierAtomic
from hackerthon.commanderAdapter import MISSION, CommanderPolicy
from hackerthon.commander_helpers import validate_tactical_plan
from hackerthon.commander_models import TacticalPlan
from hackerthon.commander_tactics import TACTIC_LIBRARY, selected_tactic_prompt
from hackerthon.ruleAdapter import RuleSoldierPolicy
from hackerthon.simulation_direct_commander_5v5 import (
    CSVLoggerAtomic,
    CommanderAtomic,
    RulePolicyAtomic,
)
from hackerthon.worldModel import WorldAtomic


DEMO_BLUE_PERCEPTION_RANGE = 1000.0
DEMO_BLUE_FOV_DEG = 360.0


class FixedTacticCommanderPolicy(CommanderPolicy):
    """전술과 최초 역할을 재계획하지 않는 데모용 CommanderPolicy."""

    def __init__(self, fixed_tactic: str, **kwargs):
        tactic = str(fixed_tactic).strip().upper()
        if tactic not in TACTIC_LIBRARY:
            raise ValueError(
                f"Unknown tactic {fixed_tactic!r}; choose one of {sorted(TACTIC_LIBRARY)}"
            )
        super().__init__(**kwargs)
        self.fixed_tactic = tactic

    def _tactical_plan_trigger(
        self,
        current_tick: int,
        friendly_kia_ids: List[int],
        first_damage_ids: List[int],
    ) -> str:
        """Stage 2를 최초 한 번만 실행해 선택 전술과 최초 역할을 유지한다."""
        del current_tick, friendly_kia_ids, first_damage_ids
        if self._active_tactical_plan is None:
            return f"FIXED_TACTIC_INITIAL:{self.fixed_tactic}"
        return ""

    def select_tactic_and_roles(
        self,
        intel_buffer: Dict[int, Any],
        assessment: Dict[str, Any],
        map_context: Dict[str, Any],
        allowed_targets: Dict[int, List[str]],
    ) -> Dict[str, Any]:
        """고정 전술에 허용된 역할만 최초 한 번 배정한다."""

        def validate(result: Dict[str, Any]) -> None:
            validate_tactical_plan(intel_buffer, result)
            if result["tactic"] != self.fixed_tactic:
                raise ValueError(
                    f"tactic must remain {self.fixed_tactic}, got {result['tactic']}"
                )

        prompt = (
            f"Mission: {MISSION}\n"
            f"Fixed demo tactic: {self.fixed_tactic}\n"
            "This is a controlled experiment. Do not choose a different tactic.\n"
            "All living Red units are observed from the first decision.\n"
            "The directly engageable list still enforces the normal fire range and ammo rules.\n"
            "Assign exactly one allowed role from the fixed tactic to every living Blue unit.\n"
            "The tactic and these initial role assignments remain fixed for the entire run.\n"
            f"Living Blue ids: {map_context['alive_blue_ids']}\n"
            f"Situation assessment: {json.dumps(assessment, ensure_ascii=False)}\n"
            "Directly engageable targets:\n"
            f"{self._target_lines(allowed_targets)}\n"
            "Fixed tactic definition:\n"
            f"{selected_tactic_prompt(self.fixed_tactic)}\n"
            "Keep rationale and each reason short. Return Stage 2 JSON."
        )
        return self._llm_json_retry(
            title=f"DEMO STAGE 2 - FIXED {self.fixed_tactic} ROLES",
            prompt=prompt,
            image_paths=[],
            response_schema=TacticalPlan.model_json_schema(),
            model_cls=TacticalPlan,
            validator=validate,
        )


class FixedTacticBattleModel(CoupledDEVS):
    """기존 5대5 시나리오에 FixedTacticCommanderPolicy만 주입한 모델."""

    def __init__(
        self,
        fixed_tactic: str,
        model_name: str = "qwen2.5vl:72b",
        temperature: float = 0.2,
        seed: int = 42,
        output_root: str = "output",
        name: str = "FixedTacticBattleModel",
    ):
        super().__init__(name)
        self.fixed_tactic = str(fixed_tactic).strip().upper()
        self.seed = int(seed)
        random.seed(self.seed)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_name = f"fixed_{self.fixed_tactic.lower()}_{timestamp}"
        self.run_dir = os.path.join(output_root, run_name)
        os.makedirs(self.run_dir, exist_ok=False)
        print(f"[Fixed Tactic Demo] tactic={self.fixed_tactic}")
        print(f"[Run] output dir: {self.run_dir}")

        config = {
            "demo": "fixed_tactic_5v5",
            "fixed_tactic": self.fixed_tactic,
            "model": model_name,
            "temperature": float(temperature),
            "seed": self.seed,
            "stage2_policy": "initial_once",
            "stage1_stage3_stage4_stage5": "existing_implementation",
            "blue_perception": "global_from_first_observation",
            "blue_perception_range": DEMO_BLUE_PERCEPTION_RANGE,
            "blue_fov_deg": DEMO_BLUE_FOV_DEG,
            "fire_range": "existing_implementation",
        }
        Path(self.run_dir, "demo_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        blue_positions = self._make_blue_positions()
        red_positions = self._make_red_positions()
        initial_entities = [
            {
                "id": sid,
                "type": "soldier",
                "x": x,
                "y": y,
                "heading": heading,
                "state": "ALIVE",
            }
            for sid, x, y, heading in blue_positions
        ]
        initial_entities.extend(
            {
                "id": sid,
                "type": "enemy",
                "x": x,
                "y": y,
                "heading": heading,
                "state": "ALIVE",
            }
            for sid, x, y, heading in red_positions
        )

        self.world = self.addSubModel(WorldAtomic(initial_entities=initial_entities))
        self.logger = self.addSubModel(
            CSVLoggerAtomic(os.path.join(self.run_dir, "soldier_commander_log.csv"))
        )

        blue_ids = [sid for sid, _, _, _ in blue_positions]
        policy = FixedTacticCommanderPolicy(
            fixed_tactic=self.fixed_tactic,
            model=model_name,
            temperature=temperature,
            io_dir=os.path.join(self.run_dir, "llm_io"),
            map_dir=os.path.join(self.run_dir, "maps"),
            run_log_dir=self.run_dir,
        )
        blue_commander = self.addSubModel(
            CommanderAtomic(
                name="Blue_Fixed_Tactic_Commander",
                controlled_ids=blue_ids,
                policy=policy,
                dt=1.0,
                run_dir=self.run_dir,
            )
        )

        for blue_id, x, y, heading in blue_positions:
            soldier = self.addSubModel(
                SoldierAtomic(
                    name=f"Blue_Soldier_{blue_id}",
                    soldier_id=blue_id,
                    initial_x=x,
                    initial_y=y,
                    initial_heading=heading,
                    perception_range=DEMO_BLUE_PERCEPTION_RANGE,
                    max_move_per_step=policy.max_move_per_step,
                    fov_deg=DEMO_BLUE_FOV_DEG,
                )
            )
            self.connectPorts(self.world.world_out, soldier.world_in)
            self.connectPorts(self.world.damage_out, soldier.damage_in)
            self.connectPorts(soldier.status_out, self.world.status_in)
            self.connectPorts(soldier.status_out, self.logger.status_in)
            self.connectPorts(soldier.observation_out, blue_commander.intel_in)
            self.connectPorts(blue_commander.orders_out[blue_id], soldier.command_in)

        for red_id, x, y, heading in red_positions:
            self._add_red_soldier(red_id, x, y, heading)

    @staticmethod
    def _make_blue_positions():
        """기존 시나리오와 동일한 Blue 선형 초기 배치."""
        positions = []
        for index in range(5):
            ratio = index / 4
            positions.append(
                (
                    101 + index,
                    round(-5.0 + 2.0 * ratio, 2),
                    round(-5.0 + 10.0 * ratio, 2),
                    0.0,
                )
            )
        return positions

    @staticmethod
    def _make_red_positions():
        """기존 seed와 동일하게 생성되는 Red 방어선 초기 배치."""
        return [
            (
                201 + index,
                round(random.uniform(5.0, 8.0), 2),
                round(random.uniform(-5.0, 5.0), 2),
                180.0,
            )
            for index in range(5)
        ]

    def _add_red_soldier(self, red_id, x, y, heading):
        soldier = self.addSubModel(
            SoldierAtomic(
                name=f"Red_Soldier_{red_id}",
                soldier_id=red_id,
                initial_x=x,
                initial_y=y,
                initial_heading=heading,
                fov_deg=120.0,
                active=True,
            )
        )
        brain = self.addSubModel(
            RulePolicyAtomic(
                name=f"Red_Rule_{red_id}",
                policy=RuleSoldierPolicy(target_type="soldier"),
                decision_delay=1.0,
            )
        )
        self.connectPorts(soldier.observation_out, brain.observation_in)
        self.connectPorts(brain.command_out, soldier.command_in)
        self.connectPorts(self.world.world_out, soldier.world_in)
        self.connectPorts(self.world.damage_out, soldier.damage_in)
        self.connectPorts(self.world.spawn_out, soldier.spawn_in)
        self.connectPorts(soldier.status_out, self.world.status_in)
        self.connectPorts(soldier.status_out, self.logger.status_in)


def parse_args():
    parser = argparse.ArgumentParser(
        description="선택한 전술과 최초 역할을 끝까지 유지하는 5대5 데모"
    )
    parser.add_argument(
        "--tactic",
        required=True,
        choices=sorted(TACTIC_LIBRARY),
        help="전체 실행 동안 유지할 Blue 전술",
    )
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--model", default="qwen2.5vl:72b")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", default="output")
    return parser.parse_args()


def main():
    args = parse_args()
    battle = FixedTacticBattleModel(
        fixed_tactic=args.tactic,
        model_name=args.model,
        temperature=args.temperature,
        seed=args.seed,
        output_root=args.output_root,
    )
    simulator = Simulator(battle)
    simulator.setTerminationTime(args.duration)
    simulator.simulate()
    print(f"시뮬레이션 완료. 결과 디렉토리: {battle.run_dir}")
    print(f"고정 전술: {battle.fixed_tactic}")


if __name__ == "__main__":
    main()
