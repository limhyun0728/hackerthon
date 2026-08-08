import csv
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import ollama
from pydantic import ValidationError

from hackerthon.combat_config import MAX_FIRE_RANGE
from hackerthon.commander_helpers import (
    GRID_COLUMNS,
    GRID_ROWS,
    alive_ids,
    build_allowed_target_marks,
    build_command_sequence,
    build_current_grid_marks,
    validate_marked_action_plan,
    validate_situation_assessment,
    validate_tactical_plan,
    validate_tick_action_plan,
)
from hackerthon.commander_models import (
    CommandSequence,
    MarkedActionPlan,
    SituationAssessment,
    TacticalPlan,
    TickActionPlan,
)
from hackerthon.commander_tactics import selected_tactic_prompt, tactic_library_prompt
from hackerthon.commanderMap import CommanderMap


MISSION = (
    "Red forces are holding defensive positions in the east. "
    "Blue must destroy all Red forces, then reach the eastern objective "
    "at (10, 0), grid H7."
)


class CommanderPolicy:
    """상황평가·전술·행동·mark·실행을 분리한 5-stage 지휘관."""

    def __init__(
        self,
        model: str = "qwen2.5vl:72b",
        temperature: float = 0.2,
        io_dir: str = "llm_io",
        map_dir: str = "output/commander_maps_ollama",
        shooting_range: float = MAX_FIRE_RANGE,
        max_move_per_step: float = 1.5,
        tactical_plan_interval_ticks: int = 5,
        validation_retries: int = 10,
        run_log_dir: Optional[str] = None,
    ):
        """모델 설정과 5틱 전술 상태를 초기화한다."""
        self.model = model
        self.temperature = float(temperature)
        self.io_dir = Path(io_dir)
        self.map_dir = Path(map_dir)
        self.map_dir.mkdir(parents=True, exist_ok=True)
        self.map_latest_path = self.map_dir / "commander_map_latest.png"
        self.marked_map_latest_path = self.map_dir / "commander_marked_map_latest.png"
        self._map_tick_index = 0
        self._marked_map_tick_index = 0

        self.shooting_range = float(shooting_range)
        self.max_move_per_step = float(max_move_per_step)
        self.tactical_plan_interval_ticks = int(tactical_plan_interval_ticks)
        self.validation_retries = int(validation_retries)
        self.cmap = CommanderMap()

        self._active_tactical_plan: Dict[str, Any] | None = None
        self._active_plan_start_tick: int | None = None
        self._active_plan_trigger = ""
        self._previous_alive_ids: set[int] | None = None
        self._previous_hp_by_id: Dict[int, float] = {}
        self._first_damage_replan_used = False
        self._previous_action_plan: Dict[str, Any] | None = None
        self._move_goal_cache: Dict[int, Dict[str, Any]] = {}

        self._log_dir = Path(run_log_dir) if run_log_dir else Path("stage_logs")
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def decide(
        self,
        intel_buffer: Dict[int, Any],
        memory=None,
        sim_time: float = 0.0,
        controlled_ids: List[int] | None = None,
    ) -> Dict[str, Any]:
        """한 tick의 Stage 1~5를 순서대로 실행한다."""
        del memory
        if controlled_ids is None:
            raise ValueError("controlled_ids is required")

        current_tick = int(round(sim_time))
        current_alive_ids = set(alive_ids(intel_buffer))
        current_hp_by_id = {
            uid: float(intel_buffer[uid]["self"]["hp"])
            for uid in current_alive_ids
        }
        friendly_kia_ids = []
        if self._previous_alive_ids is not None:
            friendly_kia_ids = sorted(self._previous_alive_ids - current_alive_ids)
        first_damage_ids = []
        if not self._first_damage_replan_used:
            first_damage_ids = sorted(
                uid
                for uid, current_hp in current_hp_by_id.items()
                if uid in self._previous_hp_by_id
                and current_hp < self._previous_hp_by_id[uid]
            )

        map_context = self._build_map_context(
            intel_buffer,
            sim_time,
            controlled_ids,
        )
        allowed_targets = build_allowed_target_marks(
            intel_buffer,
            max_fire_range=self.shooting_range,
        )
        current_grid_marks = build_current_grid_marks(intel_buffer)

        # Stage 1은 매 tick 현재 지도만 평가하며 행동을 결정하지 않는다.
        assessment = self.assess_situation(
            intel_buffer,
            map_context,
            allowed_targets,
        )
        self._log_stage1(sim_time, assessment)

        plan_trigger = self._tactical_plan_trigger(
            current_tick,
            friendly_kia_ids,
            first_damage_ids,
        )
        tactical_plan_updated = bool(plan_trigger)
        if tactical_plan_updated:
            # Stage 2는 최초·5틱 주기·아군 KIA·최초 피해에만 전술과 역할을 갱신한다.
            self._active_tactical_plan = self.select_tactic_and_roles(
                intel_buffer,
                assessment,
                map_context,
                allowed_targets,
            )
            self._active_plan_start_tick = current_tick
            self._active_plan_trigger = plan_trigger
            self._previous_action_plan = None
            self._move_goal_cache.clear()
            self._log_stage2(
                sim_time,
                plan_trigger,
                self._active_tactical_plan,
            )
        if first_damage_ids:
            self._first_damage_replan_used = True

        tactical_plan = self._active_tactical_plan
        if tactical_plan is None or self._active_plan_start_tick is None:
            raise ValueError("Stage 2 tactical plan is not active")

        # Stage 3은 최신 상황평가와 유지 중인 전술·역할로 행동 종류를 고른다.
        action_plan = self.select_tick_actions(
            intel_buffer,
            assessment,
            tactical_plan,
            allowed_targets,
            current_grid_marks,
        )
        self._log_stage3(sim_time, action_plan)

        has_move = any(
            item["action"] == "MOVE"
            for item in action_plan["actions"]
        )
        marked_image_path = self._render_stage4_map(
            map_context,
            show_action_grid=has_move,
        )

        # Stage 4는 Stage 3 행동과 동일한 전술 문맥으로 target 또는 cell을 고른다.
        marked_plan = self.select_action_marks(
            action_plan,
            assessment,
            tactical_plan,
            map_context,
            marked_image_path,
            allowed_targets,
            current_grid_marks,
        )
        self._log_stage4(sim_time, marked_plan)

        # Stage 5는 LLM 판단을 추가하지 않고 mark를 실행 명령으로 변환한다.
        sequence = build_command_sequence(
            intel_buffer,
            action_plan,
            marked_plan,
            allowed_targets,
            current_grid_marks,
            self._move_goal_cache,
            max_move_per_step=self.max_move_per_step,
        )
        sequence = CommandSequence.model_validate(sequence).model_dump()
        self._log_stage5(sim_time, sequence)

        self._previous_alive_ids = current_alive_ids
        self._previous_hp_by_id = current_hp_by_id
        self._previous_action_plan = action_plan

        plan = {
            "intent": tactical_plan["tactic"],
            "tactic": tactical_plan["tactic"],
            "rationale": tactical_plan["rationale"],
            "assignments": tactical_plan["assignments"],
            "started_tick": self._active_plan_start_tick,
            "trigger": self._active_plan_trigger,
        }
        decision = {
            "assessment": assessment,
            "plan": plan,
            "tactical_plan_updated": tactical_plan_updated,
            "action_plan": action_plan,
            "marked_plan": marked_plan,
            "sequence": sequence,
            "map_image_path": map_context["image_path"],
            "marked_image_path": marked_image_path,
            "belief_rows": map_context["belief_rows"],
        }
        self._print_decision(decision)
        return decision

    def assess_situation(
        self,
        intel_buffer: Dict[int, Any],
        map_context: Dict[str, Any],
        allowed_targets: Dict[int, List[str]],
    ) -> Dict[str, Any]:
        """Stage 1 VLM 호출로 현재 상황만 평가한다."""
        def validate(result: Dict[str, Any]) -> None:
            """Stage 1 구조화 출력의 현재 아군 상태를 검사한다."""
            validate_situation_assessment(intel_buffer, result)

        return self._llm_json_retry(
            title="STAGE 1 - SITUATION ASSESSMENT",
            prompt=self._build_assessment_prompt(
                intel_buffer,
                map_context,
                allowed_targets,
            ),
            image_paths=[map_context["image_path"]],
            response_schema=SituationAssessment.model_json_schema(),
            model_cls=SituationAssessment,
            validator=validate,
        )

    def select_tactic_and_roles(
        self,
        intel_buffer: Dict[int, Any],
        assessment: Dict[str, Any],
        map_context: Dict[str, Any],
        allowed_targets: Dict[int, List[str]],
    ) -> Dict[str, Any]:
        """Stage 2 LLM 호출로 전술 하나와 유닛별 역할을 선택한다."""
        def validate(result: Dict[str, Any]) -> None:
            """Stage 2 역할의 유닛 범위와 전술 호환성을 검사한다."""
            validate_tactical_plan(intel_buffer, result)

        return self._llm_json_retry(
            title="STAGE 2 - TACTIC AND ROLES",
            prompt=self._build_tactical_plan_prompt(
                assessment,
                map_context,
                allowed_targets,
            ),
            image_paths=[],
            response_schema=TacticalPlan.model_json_schema(),
            model_cls=TacticalPlan,
            validator=validate,
        )

    def select_tick_actions(
        self,
        intel_buffer: Dict[int, Any],
        assessment: Dict[str, Any],
        tactical_plan: Dict[str, Any],
        allowed_targets: Dict[int, List[str]],
        current_grid_marks: Dict[int, str],
    ) -> Dict[str, Any]:
        """Stage 3 LLM 호출로 현재 tick의 ENGAGE·MOVE·HOLD를 선택한다."""
        def validate(result: Dict[str, Any]) -> None:
            """Stage 3 행동의 유닛 범위와 교전 가능 여부를 검사한다."""
            validate_tick_action_plan(intel_buffer, result, allowed_targets)

        return self._llm_json_retry(
            title="STAGE 3 - TICK ACTIONS",
            prompt=self._build_action_prompt(
                assessment,
                tactical_plan,
                allowed_targets,
                current_grid_marks,
            ),
            image_paths=[],
            response_schema=TickActionPlan.model_json_schema(),
            model_cls=TickActionPlan,
            validator=validate,
        )

    def select_action_marks(
        self,
        action_plan: Dict[str, Any],
        assessment: Dict[str, Any],
        tactical_plan: Dict[str, Any],
        map_context: Dict[str, Any],
        marked_image_path: str,
        allowed_targets: Dict[int, List[str]],
        current_grid_marks: Dict[int, str],
    ) -> Dict[str, Any]:
        """Stage 4 VLM 호출로 행동별 target 또는 grid mark를 선택한다."""
        def validate(result: Dict[str, Any]) -> None:
            """Stage 4 mark를 행동과 허용 목록에 맞춰 검사한다."""
            validate_marked_action_plan(
                action_plan,
                result,
                allowed_targets,
                current_grid_marks,
            )

        return self._llm_json_retry(
            title="STAGE 4 - ACTION MARKS",
            prompt=self._build_mark_prompt(
                action_plan,
                assessment,
                tactical_plan,
                map_context,
                allowed_targets,
                current_grid_marks,
            ),
            image_paths=[marked_image_path],
            response_schema=MarkedActionPlan.model_json_schema(),
            model_cls=MarkedActionPlan,
            validator=validate,
        )

    def _tactical_plan_trigger(
        self,
        current_tick: int,
        friendly_kia_ids: List[int],
        first_damage_ids: List[int],
    ) -> str:
        """최초 계획·5틱 주기·아군 KIA·최초 피해를 Stage 2 호출 조건으로 판정한다."""
        if self._active_tactical_plan is None:
            return "INITIAL_PLAN"
        if friendly_kia_ids:
            return "FRIENDLY_KIA:" + ",".join(str(uid) for uid in friendly_kia_ids)
        if first_damage_ids:
            return "FRIENDLY_FIRST_DAMAGE:" + ",".join(
                str(uid) for uid in first_damage_ids
            )
        if current_tick - int(self._active_plan_start_tick) >= self.tactical_plan_interval_ticks:
            return "PLAN_INTERVAL"
        return ""

    def _build_map_context(
        self,
        intel_buffer: Dict[int, Any],
        sim_time: float,
        controlled_ids: List[int],
    ) -> Dict[str, Any]:
        """현재 관측을 belief 지도와 텍스트용 상태로 변환한다."""
        in_contact = {int(uid) for uid in intel_buffer}
        out_of_contact = sorted(
            int(uid)
            for uid in controlled_ids
            if int(uid) not in in_contact
        )
        belief = self.cmap.update_belief(intel_buffer, out_of_contact)
        image_path = self.cmap.render(
            belief,
            sim_time,
            self._make_map_path(sim_time),
        )
        shutil.copyfile(image_path, self.map_latest_path)

        alive_blue_ids = sorted(
            int(uid)
            for uid, info in belief["friendlies"].items()
            if info["status"] == "in_contact" and float(info["hp"]) > 0.0
        )
        return {
            "belief": belief,
            "image_path": image_path,
            "belief_rows": self.cmap.belief_rows(belief, sim_time),
            "sim_time": sim_time,
            "alive_blue_ids": alive_blue_ids,
            "known_enemy_ids": sorted(int(eid) for eid in belief["enemies_alive"]),
        }

    def _render_stage4_map(
        self,
        map_context: Dict[str, Any],
        show_action_grid: bool,
    ) -> str:
        """Stage 4 선택용 현재 지도에 필요한 경우 grid를 표시한다."""
        image_path = self.cmap.render(
            map_context["belief"],
            map_context["sim_time"],
            self._make_marked_map_path(map_context["sim_time"]),
            show_action_grid=show_action_grid,
            selection_view=True,
        )
        shutil.copyfile(image_path, self.marked_map_latest_path)
        return image_path

    def _make_map_path(self, sim_time: float) -> str:
        """Stage 1 지도 파일 경로를 tick 순서대로 만든다."""
        safe_time = f"{float(sim_time):010.2f}".replace("-", "m").replace(".", "p")
        path = self.map_dir / f"commander_map_{self._map_tick_index:06d}_t{safe_time}.png"
        self._map_tick_index += 1
        return str(path)

    def _make_marked_map_path(self, sim_time: float) -> str:
        """Stage 4 지도 파일 경로를 tick 순서대로 만든다."""
        safe_time = f"{float(sim_time):010.2f}".replace("-", "m").replace(".", "p")
        path = self.map_dir / f"commander_marked_{self._marked_map_tick_index:06d}_t{safe_time}.png"
        self._marked_map_tick_index += 1
        return str(path)

    @staticmethod
    def _target_lines(allowed_targets: Dict[int, List[str]]) -> str:
        """유닛별 직접 교전 가능 표적을 짧은 행으로 만든다."""
        return "\n".join(
            f"- B{uid}: {marks}"
            for uid, marks in sorted(allowed_targets.items())
        )

    @staticmethod
    def _role_by_unit(tactical_plan: Dict[str, Any]) -> Dict[int, str]:
        """활성 전술의 역할 배정을 유닛 ID 기준 사전으로 만든다."""
        return {
            int(item["unit_id"]): item["role"]
            for item in tactical_plan["assignments"]
        }

    def _build_assessment_prompt(
        self,
        intel_buffer: Dict[int, Any],
        map_context: Dict[str, Any],
        allowed_targets: Dict[int, List[str]],
    ) -> str:
        """Stage 1의 순수 상황평가 프롬프트를 만든다."""
        wounded_ids = sorted(
            uid
            for uid in alive_ids(intel_buffer)
            if float(intel_buffer[uid]["self"]["hp"]) < 100.0
        )
        return (
            f"Mission: {MISSION}\n"
            "The attached image is the current shared commander map.\n"
            f"Living Blue ids: {map_context['alive_blue_ids']}\n"
            f"Wounded Blue ids: {wounded_ids}\n"
            "Directly engageable target marks by unit:\n"
            f"{self._target_lines(allowed_targets)}\n"
            "In situation_summary, briefly describe the force geometry and name relevant unit IDs.\n"
            "Assess the current situation only. Do not assign tactics, roles, or actions.\n"
            "Return Stage 1 JSON."
        )

    def _build_tactical_plan_prompt(
        self,
        assessment: Dict[str, Any],
        map_context: Dict[str, Any],
        allowed_targets: Dict[int, List[str]],
    ) -> str:
        """Stage 2의 전술 선택과 역할 배정 프롬프트를 만든다."""
        return (
            f"Mission: {MISSION}\n"
            "Choose one tactic from the library and assign one allowed role to every living Blue unit.\n"
            f"The plan remains active for {self.tactical_plan_interval_ticks} ticks.\n"
            f"Living Blue ids: {map_context['alive_blue_ids']}\n"
            f"Situation assessment: {json.dumps(assessment, ensure_ascii=False)}\n"
            "Directly engageable targets:\n"
            f"{self._target_lines(allowed_targets)}\n"
            "Tactical library:\n"
            f"{tactic_library_prompt()}\n"
            "Keep rationale and each reason short. Return Stage 2 JSON."
        )

    def _build_action_prompt(
        self,
        assessment: Dict[str, Any],
        tactical_plan: Dict[str, Any],
        allowed_targets: Dict[int, List[str]],
        current_grid_marks: Dict[int, str],
    ) -> str:
        """Stage 3의 유닛별 현재 행동 선택 프롬프트를 만든다."""
        previous_actions = (
            json.dumps(self._previous_action_plan, ensure_ascii=False)
            if self._previous_action_plan is not None
            else "NONE"
        )
        return (
            f"Mission: {MISSION}\n"
            "Choose exactly one current action for every living Blue unit: ENGAGE, MOVE, or HOLD.\n"
            "Follow the active tactic and assigned role. A visible target does not force ENGAGE.\n"
            "Use ENGAGE only when that unit has a listed target.\n"
            f"Latest situation assessment: {json.dumps(assessment, ensure_ascii=False)}\n"
            f"Active tactical plan: {json.dumps(tactical_plan, ensure_ascii=False)}\n"
            "Selected tactic definition:\n"
            f"{selected_tactic_prompt(tactical_plan['tactic'])}\n"
            "Directly engageable targets:\n"
            f"{self._target_lines(allowed_targets)}\n"
            f"Current grid cells: {json.dumps(current_grid_marks, ensure_ascii=False)}\n"
            f"Previous tick actions: {previous_actions}\n"
            "Keep rationale and each reason short. Return Stage 3 JSON."
        )

    def _build_mark_prompt(
        self,
        action_plan: Dict[str, Any],
        assessment: Dict[str, Any],
        tactical_plan: Dict[str, Any],
        map_context: Dict[str, Any],
        allowed_targets: Dict[int, List[str]],
        current_grid_marks: Dict[int, str],
    ) -> str:
        """Stage 4의 target·grid mark 선택 프롬프트를 만든다."""
        role_by_unit = self._role_by_unit(tactical_plan)
        action_lines = []
        for item in action_plan["actions"]:
            uid = int(item["unit_id"])
            action = item["action"]
            if action == "ENGAGE":
                choices: Any = allowed_targets[uid]
            elif action == "MOVE":
                choices = f"{GRID_COLUMNS[0]}1~{GRID_COLUMNS[-1]}{GRID_ROWS}"
            else:
                choices = "HOLD"

            line = (
                f"- B{uid} role={role_by_unit[uid]} action={action} "
                f"allowed_marks={choices}"
            )
            if action == "MOVE":
                current_cell = current_grid_marks[uid]
                line += (
                    f" current_cell={current_cell} "
                    f"forbidden_marks=['{current_cell}']"
                )
            action_lines.append(line)

        grid_note = ""
        if any(item["action"] == "MOVE" for item in action_plan["actions"]):
            grid_note = (
                "The map uses A1~J10. Columns A to J run west to east; "
                "rows 1 to 10 run south to north.\n"
            )

        return (
            f"Mission: {MISSION}\n"
            "Select exactly one allowed mark for every Blue unit.\n"
            "ENGAGE selects R###, MOVE selects A1~J10, and HOLD selects HOLD.\n"
            "MOVE must not select the unit's current_cell.\n"
            f"{grid_note}"
            f"Latest situation assessment: {json.dumps(assessment, ensure_ascii=False)}\n"
            f"Active tactical plan: {json.dumps(tactical_plan, ensure_ascii=False)}\n"
            f"Current tick actions: {json.dumps(action_plan, ensure_ascii=False)}\n"
            f"Known Red ids on shared map: {map_context['known_enemy_ids']}\n"
            + "\n".join(action_lines)
            + "\nKeep rationale and each reason short. Return Stage 4 JSON."
        )

    def _llm_json_call(
        self,
        title: str,
        prompt: str,
        image_paths: List[str],
        response_schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Ollama에 스키마 기반 JSON 응답을 한 번 요청한다."""
        print(f"\n[Ollama START] title={title} model={self.model} images={len(image_paths)}")
        user_message: Dict[str, Any] = {
            "role": "user",
            "content": prompt,
        }
        if image_paths:
            user_message["images"] = image_paths

        response = ollama.Client(timeout=600.0).chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Return one valid JSON object matching the schema.",
                },
                user_message,
            ],
            options={"temperature": self.temperature, "num_ctx": 8192},
            format=response_schema,
            keep_alive="30m",
        )
        raw = response["message"]["content"].strip()
        print(f"[Ollama DONE] title={title} chars={len(raw)}")
        self._save_llm_output(title, prompt, raw, image_paths)
        return self._parse_json_object(raw, title)

    def _llm_json_retry(
        self,
        title: str,
        prompt: str,
        image_paths: List[str],
        response_schema: Dict[str, Any],
        model_cls,
        validator,
    ) -> Dict[str, Any]:
        """JSON 구조·Stage 검증 실패 또는 일시적 네트워크 오류만 전체 출력을 재요청한다."""
        last_error = ""
        last_failure = ""
        for attempt in range(1, self.validation_retries + 2):
            retry_prompt = prompt
            if last_error:
                retry_prompt += (
                    "\nPrevious output validation failed: "
                    f"{self._summarize_retry_error(last_error)}\n"
                    "Return the corrected full JSON matching every listed constraint."
                )

            try:
                data = self._llm_json_call(
                    title=f"{title} attempt {attempt}",
                    prompt=retry_prompt,
                    image_paths=image_paths,
                    response_schema=response_schema,
                )
                result = model_cls.model_validate(data).model_dump()
                validator(result)
                return result
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                # 서버 응답 지연·일시 단절은 프롬프트를 바꾸지 않고 같은 요청을 다시 보낸다.
                last_failure = f"network error: {exc!r}"
                print(
                    f"[Blue Commander NETWORK RETRY] {title} "
                    f"{attempt}/{self.validation_retries + 1}: {last_failure}"
                )
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_error = str(exc)
                last_failure = last_error
                print(
                    f"[Blue Commander RETRY] {title} "
                    f"{attempt}/{self.validation_retries + 1}: {last_error}"
                )

        raise ValueError(f"{title} failed after retries: {last_failure}")

    @staticmethod
    def _summarize_retry_error(error_text: str) -> str:
        """검증 오류를 재요청 프롬프트에 들어갈 길이로 줄인다."""
        text = re.sub(r"\s+", " ", str(error_text)).strip()
        return text[:500]

    @staticmethod
    def _parse_json_object(raw: str, title: str) -> Dict[str, Any]:
        """응답 전체가 하나의 JSON 객체인지 검사한다."""
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError(f"LLM response for {title} is not a JSON object")
        return data

    def _print_decision(self, decision: Dict[str, Any]) -> None:
        """현재 5-stage 판단을 읽기 쉬운 한 묶음으로 출력한다."""
        assessment = decision["assessment"]
        plan = decision["plan"]
        print(
            "\n[Blue Commander]\n"
            f"axis={assessment['main_enemy_axis']} | "
            f"enemy_count={assessment['enemy_count_estimate']}\n"
            f"tactic={plan['tactic']} | roles={json.dumps(plan['assignments'], ensure_ascii=False)}\n"
            f"actions={json.dumps(decision['action_plan']['actions'], ensure_ascii=False)}\n"
            f"marks={json.dumps(decision['marked_plan']['selections'], ensure_ascii=False)}\n"
            f"current_map={decision['map_image_path']}\n"
            f"marked_map={decision['marked_image_path']}\n"
        )

    def _append_csv(self, filename: str, row: Dict[str, Any]) -> None:
        """Stage별 한 행을 해당 CSV에 추가한다."""
        path = self._log_dir / filename
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _log_stage1(self, sim_time: float, assessment: Dict[str, Any]) -> None:
        """Stage 1 상황평가를 기록한다."""
        self._append_csv("stage1_assessment.csv", {
            "time": round(sim_time, 1),
            "summary": assessment["situation_summary"],
            "enemy_count": assessment["enemy_count_estimate"],
            "superiority": assessment["local_superiority"],
            "opportunity": assessment["opportunity"],
            "risk": assessment["risk"],
        })

    def _log_stage2(
        self,
        sim_time: float,
        trigger: str,
        tactical_plan: Dict[str, Any],
    ) -> None:
        """Stage 2 전술과 역할이 갱신된 tick만 기록한다."""
        assignments = " | ".join(
            f"{item['unit_id']}:{item['role']}({item['reason']})"
            for item in tactical_plan["assignments"]
        )
        self._append_csv("stage2_tactical_plan.csv", {
            "time": round(sim_time, 1),
            "trigger": trigger,
            "tactic": tactical_plan["tactic"],
            "assignments": assignments,
            "rationale": tactical_plan["rationale"],
        })

    def _log_stage3(self, sim_time: float, action_plan: Dict[str, Any]) -> None:
        """Stage 3 유닛별 행동과 이유를 기록한다."""
        actions = " | ".join(
            f"{item['unit_id']}:{item['action']}({item['reason']})"
            for item in action_plan["actions"]
        )
        self._append_csv("stage3_actions.csv", {
            "time": round(sim_time, 1),
            "actions": actions,
            "rationale": action_plan["rationale"],
        })

    def _log_stage4(self, sim_time: float, marked_plan: Dict[str, Any]) -> None:
        """Stage 4 유닛별 mark와 이유를 기록한다."""
        selections = " | ".join(
            f"{item['unit_id']}:{item['mark']}({item['reason']})"
            for item in marked_plan["selections"]
        )
        self._append_csv("stage4_mark_selections.csv", {
            "time": round(sim_time, 1),
            "selections": selections,
            "rationale": marked_plan["rationale"],
        })

    def _log_stage5(self, sim_time: float, sequence: Dict[str, Any]) -> None:
        """Stage 5 최종 실행 명령을 기록한다."""
        commands = []
        for frame in sequence["frames"]:
            for command in frame["commands"]:
                suffix = command["target_id"]
                commands.append(
                    f"{command['unit_id']}:{command['action']}"
                    + (f"->{suffix}" if suffix is not None else "")
                )
        self._append_csv("stage5_commands.csv", {
            "time": round(sim_time, 1),
            "commands": " | ".join(commands),
        })

    def _save_llm_output(
        self,
        title: str,
        prompt: str,
        raw: str,
        image_paths: List[str],
    ) -> None:
        """각 Stage의 프롬프트와 원본 응답을 JSONL로 저장한다."""
        self.io_dir.mkdir(parents=True, exist_ok=True)
        path = self.io_dir / f"llm_outputs_{time.strftime('%Y-%m-%d')}.jsonl"
        record = {
            "time": time.time(),
            "title": title,
            "model": self.model,
            "image_paths": image_paths,
            "prompt": prompt,
            "raw_output": raw,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
