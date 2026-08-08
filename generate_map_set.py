"""VWorld 건물 데이터로 학습용 맵 세트를 한 번에 만든다.

월드모델이 특정 지형을 외우지 않게 하려면 여러 지역이 필요하다. 지역마다
vworld_building_obstacles로 건물 footprint를 받아 obstacles config를 쓰고,
실제로 걸어다닐 수 있는 공간이 충분한지 검사한 뒤 통과한 맵만 남긴다.

유닛 배치와 objective는 여기서 정하지 않는다. 학습 루프가 episode마다
자유 공간에서 새로 뽑기 때문에 config에는 지형만 있으면 된다.

사용법:
    python generate_map_set.py --out-root output/maps
    python generate_map_set.py --out-root output/maps --only gangnam gunja
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hackerthon.terrain import (
    largest_free_component,
    path_pad_for_unit_radius,
    set_path_pad,
)


@dataclass(frozen=True)
class MapSpec:
    """생성할 지역 한 곳."""

    name: str
    label: str
    lat: float
    lon: float


# 서울 시내에서 건물 밀도와 가로 구조가 서로 다른 지역들을 골랐다.
MAP_SPECS: tuple[MapSpec, ...] = (
    MapSpec("gangnam", "강남역", 37.497952, 127.027619),
    MapSpec("gunja", "군자역", 37.557192, 127.079344),
    MapSpec("seoultech", "서울과학기술대학교", 37.631940, 127.077090),
    MapSpec("yeouido", "여의도역", 37.521624, 126.924191),
    MapSpec("hongdae", "홍대입구역", 37.557192, 126.925381),
    # 아래는 홀드아웃 전용. 학습에 넣지 말 것 — 처음 보는 지형에서의 일반화를 잰다.
    # 학습 맵들과 최소 1.4km 떨어져 있어 400m 정사각 영역이 겹치지 않는다.
    MapSpec("assembly", "국회의사당", 37.531700, 126.913900),
    MapSpec("yongsan", "용산역", 37.529849, 126.964561),
    MapSpec("sinchon", "신촌역", 37.555134, 126.936893),
    MapSpec("euljiro", "을지로3가", 37.566100, 126.991700),
    MapSpec("itaewon", "이태원역", 37.534500, 126.994600),
    MapSpec("seongsu", "성수역", 37.544600, 127.055900),
    MapSpec("myeongdong", "명동역", 37.563600, 126.982700),
    MapSpec("jamsil", "잠실역", 37.513300, 127.100000),
    MapSpec("daerim", "대림역", 37.493000, 126.895500),
)

# 이 아래로 자유 공간이 좁으면 유닛이 서로 못 만나거나 배치가 실패한다.
MIN_FREE_CELLS = 600


def _generate_one(
    spec: MapSpec,
    *,
    out_root: Path,
    meters_per_unit: float,
    unit_radius_units: float,
    max_buildings: int,
    obstacle_cell_size: float,
    python_executable: str,
) -> Path:
    """vworld_building_obstacles를 호출해 지역 하나의 config를 만든다."""
    out_config = out_root / spec.name / "config.json"
    command = [
        python_executable,
        str(Path(__file__).resolve().parent / "vworld_building_obstacles.py"),
        "--origin-lat", str(spec.lat),
        "--origin-lon", str(spec.lon),
        "--meters-per-unit", str(meters_per_unit),
        "--unit-radius-units", str(unit_radius_units),
        "--max-buildings", str(max_buildings),
        "--obstacle-cell-size", str(obstacle_cell_size),
        "--out-config", str(out_config),
    ]
    subprocess.run(command, check=True)
    return out_config


def _validate(config_path: Path) -> tuple[bool, str]:
    """지형이 실제로 쓸 만한지 본다. 통과 여부와 사람이 읽을 요약을 낸다."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    obstacles = [tuple(float(value) for value in rect) for rect in config.get("obstacles", [])]
    if not obstacles:
        return False, "장애물이 없다 (건물을 못 받았거나 전부 걸러졌다)"

    real_map = config.get("real_map", {})
    radius = float(real_map.get("unit_radius_units", 0.035))
    # 검사도 학습 때와 같은 경로 여유폭을 써야 의미가 있다.
    set_path_pad(path_pad_for_unit_radius(radius))

    component = largest_free_component(obstacles)
    summary = (
        f"장애물 {len(obstacles)}개, 건물 {len(config.get('building_polygons', []))}동, "
        f"최대 이동 가능 영역 {len(component)}셀"
    )
    if len(component) < MIN_FREE_CELLS:
        return False, f"{summary} -> 자유 공간 부족 (최소 {MIN_FREE_CELLS})"
    return True, summary


def _parse_args(argv: Iterable[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="학습용 다중 맵 config 생성")
    parser.add_argument("--out-root", type=Path, default=Path("output/maps"))
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help=f"생성할 지역 이름. 생략하면 전부. 선택지: {', '.join(s.name for s in MAP_SPECS)}",
    )
    parser.add_argument("--meters-per-unit", type=float, default=10.0)
    parser.add_argument("--unit-radius-units", type=float, default=0.035)
    parser.add_argument("--max-buildings", type=int, default=120)
    parser.add_argument("--obstacle-cell-size", type=float, default=1.0)
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="vworld_building_obstacles를 실행할 인터프리터",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    specs = MAP_SPECS
    if args.only:
        known = {spec.name: spec for spec in MAP_SPECS}
        unknown = [name for name in args.only if name not in known]
        if unknown:
            raise ValueError(f"모르는 지역 이름: {unknown}. 선택지: {sorted(known)}")
        specs = tuple(known[name] for name in args.only)

    ok: list[Path] = []
    failed: list[tuple[str, str]] = []
    for spec in specs:
        print(f"\n=== {spec.label} ({spec.name}) lat={spec.lat} lon={spec.lon}")
        try:
            config_path = _generate_one(
                spec,
                out_root=args.out_root,
                meters_per_unit=args.meters_per_unit,
                unit_radius_units=args.unit_radius_units,
                max_buildings=args.max_buildings,
                obstacle_cell_size=args.obstacle_cell_size,
                python_executable=args.python,
            )
        except subprocess.CalledProcessError as error:
            failed.append((spec.name, f"VWorld 호출 실패 (exit={error.returncode})"))
            continue

        passed, summary = _validate(config_path)
        print(f"  {summary}")
        if passed:
            ok.append(config_path)
        else:
            failed.append((spec.name, summary))

    print("\n=== 결과")
    for path in ok:
        print(f"  사용 가능: {path}")
    for name, reason in failed:
        print(f"  제외: {name} - {reason}")
    if not ok:
        raise ValueError("쓸 수 있는 맵이 하나도 없다")

    print("\n학습 루프에 넘길 인자:")
    print("  --obstacle-configs " + " ".join(str(path) for path in ok))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
