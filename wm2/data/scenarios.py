"""맵 풀과 홀드아웃 강제 (설계 0절).

구 시스템에서는 맵 분리가 명령줄에만 있어서 glob 하나로 홀드아웃이 조용히 섞일 수
있었다. 여기서는 에피소드의 장애물 집합을 홀드아웃 맵 config와 대조해 **코드로**
막는다. 에피소드 config.json에 맵 이름이 없으므로 장애물 서명으로 판별한다.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..model.features import HELDOUT_MAPS
from .episodes import Episode

# 실행 cwd와 무관하게 리포 안의 output/maps를 가리킨다. 상대경로면 cwd에 따라
# 서명이 조용히 비어 가드가 무력화된다.
_MAPS_ROOT = Path(__file__).resolve().parents[2] / "output" / "maps"


def _signature(obstacles) -> frozenset:
    return frozenset(tuple(round(float(v), 3) for v in rect) for rect in obstacles)


def heldout_signatures(maps_root: Path = _MAPS_ROOT) -> dict[frozenset, str]:
    """홀드아웃 맵 이름 → 장애물 서명. config가 없으면 그 맵은 건너뛴다."""
    result: dict[frozenset, str] = {}
    for name in HELDOUT_MAPS:
        config_path = maps_root / name / "config.json"
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text())
        result[_signature(config.get("obstacles", ()))] = name
    return result


def assert_not_heldout(episode: Episode, signatures: dict[frozenset, str]) -> None:
    name = signatures.get(_signature(episode.obstacles))
    if name is not None:
        raise ValueError(
            f"{episode.run_dir}: 홀드아웃 맵({name}) 에피소드다. 학습에 넣을 수 없다."
        )
