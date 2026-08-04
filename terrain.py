"""장애물 지형의 단일 출처: 이동(A*)·시야·사격이 같은 기하를 공유한다.

장애물은 축정렬 사각형(AABB). 세 규칙이 하나의 LOS 함수를 쓰므로
"보이는데 못 쏘거나, 벽을 뚫고 걷는" 불일치가 생기지 않는다.
"""
import heapq
import math
from typing import Dict, List, Optional, Tuple

# (xmin, ymin, xmax, ymax)
Rect = Tuple[float, float, float, float]

# 월드 경계 (기존 commander_helpers 그리드 범위와 동일)
WORLD_X_MIN, WORLD_X_MAX = -20.0, 20.0
WORLD_Y_MIN, WORLD_Y_MAX = -15.0, 10.0

# 기본 시나리오 지형: 소규모 건물과 차량/차단물이 섞인 시가지.
# 거대한 정방형 블록 대신 폭 1~3u의 엄폐물을 떨어뜨려 배치해
# 넓은 도로, 좁은 골목, 모서리 peek 위치가 모두 생긴다.
URBAN_OBSTACLES: List[Rect] = [
    # 작은 건물/점포
    (-5.8, -6.8, -4.0, -4.2),
    (-5.7, -1.8, -3.9, 0.8),
    (-5.5, 3.7, -3.7, 6.3),
    (-1.2, -5.8, 0.8, -3.7),
    (-1.0, -0.7, 1.2, 1.2),
    (-1.1, 4.1, 0.9, 6.2),
    (3.0, -6.5, 5.0, -4.3),
    (3.2, -1.6, 5.3, 0.6),
    (3.0, 3.7, 5.0, 6.0),
    # 차량·낮은 차단물(기하 판정상 완전 엄폐물)
    (-7.0, 1.8, -5.8, 2.4),
    (-2.7, 2.3, -1.3, 2.9),
    (1.5, -3.0, 2.2, -1.8),
    (6.2, 1.8, 7.6, 2.4),
]

# 기존 import 계약을 유지한다.
DEFAULT_OBSTACLES = URBAN_OBSTACLES

# 경로 계산 시 유닛 반경만큼 장애물을 부풀린다(벽 스침 방지).
#
# 이 값은 통로 폭 하한을 정한다: 폭 2*PATH_PAD 미만 통로는 전부 통행 불가가 된다.
# 예전 0.45는 기본 URBAN_OBSTACLES(폭 1~3u 엄폐물) 기준으로 잡힌 값이라,
# 실측 건물맵처럼 격자 셀이 촘촘한 지형에서는 골목을 통째로 막아 유닛이
# 영구 정지한다. 실제 유닛 반경(예: 0.035u)보다는 넉넉하되 통로는 살리는
# 보수적인 값으로 낮췄다. set_path_pad()로 맵 config에서 덮어쓸 수 있다.
DEFAULT_PATH_PAD = 0.15
PATH_PAD = DEFAULT_PATH_PAD
GRID_RES = 0.5


def set_path_pad(pad: float) -> float:
    """경로 계산용 장애물 팽창 폭을 바꾼다. 맵 config에서 유도할 때 쓴다."""
    global PATH_PAD
    if pad <= 0.0:
        raise ValueError("PATH_PAD는 0보다 커야 한다")
    PATH_PAD = float(pad)
    return PATH_PAD


def path_pad_for_unit_radius(unit_radius_units: float, *, floor: float = 0.10) -> float:
    """유닛 반경에서 경로 여유폭을 유도한다. 너무 작아지지 않게 floor를 둔다."""
    return max(float(floor), 2.0 * float(unit_radius_units))


def _segment_intersects_rect(p: Tuple[float, float], q: Tuple[float, float], rect: Rect, pad: float = 0.0) -> bool:
    """선분 p-q가 (pad만큼 부풀린) 사각형과 교차하는지 slab 방식으로 판정한다."""
    xmin, ymin, xmax, ymax = rect
    xmin -= pad
    ymin -= pad
    xmax += pad
    ymax += pad

    (x0, y0), (x1, y1) = p, q
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for delta, lo, hi, origin in ((dx, xmin, xmax, x0), (dy, ymin, ymax, y0)):
        if abs(delta) < 1e-12:
            if origin < lo or origin > hi:
                return False
            continue
        ta = (lo - origin) / delta
        tb = (hi - origin) / delta
        if ta > tb:
            ta, tb = tb, ta
        t0 = max(t0, ta)
        t1 = min(t1, tb)
        if t0 > t1:
            return False
    return True


def has_los(p: Tuple[float, float], q: Tuple[float, float], obstacles: List[Rect], pad: float = 0.0) -> bool:
    """두 점 사이에 장애물이 없으면 True. 시야와 사격 판정이 함께 쓴다."""
    return not any(_segment_intersects_rect(p, q, rect, pad) for rect in obstacles)


def point_blocked(p: Tuple[float, float], obstacles: List[Rect], pad: float = 0.0) -> bool:
    x, y = p
    return any(
        xmin - pad <= x <= xmax + pad and ymin - pad <= y <= ymax + pad
        for xmin, ymin, xmax, ymax in obstacles
    )


def clamp_to_world(x: float, y: float, margin: float = 0.5) -> Tuple[float, float]:
    return (
        min(max(x, WORLD_X_MIN + margin), WORLD_X_MAX - margin),
        min(max(y, WORLD_Y_MIN + margin), WORLD_Y_MAX - margin),
    )


def _cell(p: Tuple[float, float]) -> Tuple[int, int]:
    return (round(p[0] / GRID_RES), round(p[1] / GRID_RES))


def _cell_center(c: Tuple[int, int]) -> Tuple[float, float]:
    return (c[0] * GRID_RES, c[1] * GRID_RES)


def _cell_free(c: Tuple[int, int], obstacles: List[Rect]) -> bool:
    x, y = _cell_center(c)
    if not (WORLD_X_MIN <= x <= WORLD_X_MAX and WORLD_Y_MIN <= y <= WORLD_Y_MAX):
        return False
    return not point_blocked((x, y), obstacles, pad=PATH_PAD)


def astar_path(
    start: Tuple[float, float],
    goal: Tuple[float, float],
    obstacles: List[Rect],
) -> Optional[List[Tuple[float, float]]]:
    """장애물을 피하는 점 목록을 반환한다. 직선이 뚫려 있으면 그대로 반환."""
    if has_los(start, goal, obstacles, pad=PATH_PAD):
        return [start, goal]

    start_c, goal_c = _cell(start), _cell(goal)
    if not _cell_free(goal_c, obstacles):
        return None

    open_heap: List[Tuple[float, Tuple[int, int]]] = [(0.0, start_c)]
    g_score: Dict[Tuple[int, int], float] = {start_c: 0.0}
    came: Dict[Tuple[int, int], Tuple[int, int]] = {}

    def h(c):
        return math.hypot(c[0] - goal_c[0], c[1] - goal_c[1])

    found = False
    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal_c:
            found = True
            break
        cx, cy = current
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nxt = (cx + dx, cy + dy)
                if not _cell_free(nxt, obstacles):
                    continue
                # 대각 이동은 양옆이 모두 비어 있어야 한다(모서리 끼임 방지).
                if dx and dy and not (
                    _cell_free((cx + dx, cy), obstacles) and _cell_free((cx, cy + dy), obstacles)
                ):
                    continue
                cost = g_score[current] + math.hypot(dx, dy)
                if cost < g_score.get(nxt, float("inf")):
                    g_score[nxt] = cost
                    came[nxt] = current
                    heapq.heappush(open_heap, (cost + h(nxt), nxt))

    if not found:
        return None

    cells = [goal_c]
    while cells[-1] != start_c:
        cells.append(came[cells[-1]])
    cells.reverse()
    points = [start] + [_cell_center(c) for c in cells[1:-1]] + [goal]

    # fat-LOS로 이어지는 노드를 건너뛰어 waypoint 수를 줄인다.
    smoothed = [points[0]]
    idx = 0
    while idx < len(points) - 1:
        far = idx + 1
        for j in range(len(points) - 1, idx, -1):
            if has_los(points[idx], points[j], obstacles, pad=PATH_PAD - 0.05):
                far = j
                break
        smoothed.append(points[far])
        idx = far
    return smoothed


def next_waypoint(
    start: Tuple[float, float],
    goal: Tuple[float, float],
    obstacles: List[Rect],
    max_step: float,
) -> Optional[Tuple[float, float]]:
    """이번 틱에 이동할 한 스텝 waypoint를 A* 경로 위에서 계산한다."""
    path = astar_path(start, goal, obstacles)
    if not path or len(path) < 2:
        return None
    target = path[1]
    dist = math.hypot(target[0] - start[0], target[1] - start[1])
    remaining = max_step
    # 경유점이 한 스텝보다 가까우면 다음 구간으로 이월한다.
    while dist < remaining - 1e-9 and len(path) > 2:
        remaining -= dist
        path = path[1:]
        start = path[0]
        target = path[1]
        dist = math.hypot(target[0] - start[0], target[1] - start[1])
    if dist <= remaining:
        return target
    scale = remaining / dist
    return (start[0] + (target[0] - start[0]) * scale, start[1] + (target[1] - start[1]) * scale)


def _neighbors(cell: Tuple[int, int], obstacles: List[Rect]):
    """astar_path와 같은 이동 규칙(8방향 + 모서리 컷 금지)으로 이웃 셀을 낸다."""
    cx, cy = cell
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nxt = (cx + dx, cy + dy)
            if not _cell_free(nxt, obstacles):
                continue
            if dx and dy and not (
                _cell_free((cx + dx, cy), obstacles) and _cell_free((cx, cy + dy), obstacles)
            ):
                continue
            yield nxt


def free_component(start: Tuple[float, float], obstacles: List[Rect]) -> set:
    """start 지점에서 실제로 걸어갈 수 있는 격자 셀 집합을 반환한다."""
    origin = _cell(start)
    if not _cell_free(origin, obstacles):
        return set()
    seen = {origin}
    stack = [origin]
    while stack:
        for nxt in _neighbors(stack.pop(), obstacles):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def largest_free_component(obstacles: List[Rect]) -> set:
    """월드 전체에서 가장 넓은 이동 가능 영역을 반환한다.

    시작 위치나 목표가 이 영역 밖이면 그 유닛은 어디로도 갈 수 없다.
    """
    x_cells = range(int(math.floor(WORLD_X_MIN / GRID_RES)), int(math.ceil(WORLD_X_MAX / GRID_RES)) + 1)
    y_cells = range(int(math.floor(WORLD_Y_MIN / GRID_RES)), int(math.ceil(WORLD_Y_MAX / GRID_RES)) + 1)
    unvisited = {(cx, cy) for cx in x_cells for cy in y_cells if _cell_free((cx, cy), obstacles)}
    best: set = set()
    while unvisited:
        seed = next(iter(unvisited))
        component = {seed}
        stack = [seed]
        while stack:
            for nxt in _neighbors(stack.pop(), obstacles):
                if nxt not in component:
                    component.add(nxt)
                    stack.append(nxt)
        unvisited -= component
        if len(component) > len(best):
            best = component
    return best


def cell_of(point: Tuple[float, float]) -> Tuple[int, int]:
    """좌표가 속한 격자 셀. 스폰 보정에서 중복 배치를 막을 때 쓴다."""
    return _cell(point)


def snap_to_component(
    point: Tuple[float, float],
    component: set,
    exclude: Optional[set] = None,
) -> Optional[Tuple[float, float]]:
    """point를 주어진 이동 가능 영역 안의 가장 가까운 셀 중심으로 옮긴다.

    exclude에 든 셀은 건너뛴다(유닛 스폰이 한 칸에 겹치지 않게).
    """
    if not component:
        return None
    blocked = exclude or set()
    origin = _cell(point)
    if origin in component and origin not in blocked:
        return point
    usable = component - blocked
    if not usable:
        return None
    best = min(usable, key=lambda c: math.dist(point, _cell_center(c)))
    return _cell_center(best)


def component_points(component: set) -> List[Tuple[float, float]]:
    """이동 가능 영역의 셀 중심 좌표 목록. 순회 순서는 결정적이다."""
    return [_cell_center(cell) for cell in sorted(component)]


def ring_positions(
    center: Tuple[float, float],
    radius: float,
    n: int = 24,
) -> List[Tuple[float, float]]:
    """표적 주위 반경 radius의 후보 사격 위치들."""
    out = []
    for k in range(n):
        angle = 2.0 * math.pi * k / n
        x, y = center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle)
        x, y = clamp_to_world(x, y)
        out.append((x, y))
    return out
