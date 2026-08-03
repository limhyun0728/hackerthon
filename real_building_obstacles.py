from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hackerthon.terrain import WORLD_X_MAX, WORLD_X_MIN, WORLD_Y_MAX, WORLD_Y_MIN


OVERPASS_ENDPOINT = "https://overpass-api.de/api/interpreter"
GANGNAM_STATION_LAT = 37.497952
GANGNAM_STATION_LON = 127.027619
DEFAULT_METERS_PER_UNIT = 10.0
DEFAULT_UNIT_RADIUS_UNITS = 0.035
NON_BLOCKING_NAME_KEYWORDS = (
    "지하",
    "underground",
    "subway",
    "metro",
)
NON_BLOCKING_BUILDING_VALUES = {
    "train_station",
    "transportation",
    "subway",
}


@dataclass(frozen=True)
class LocalBuilding:
    osm_id: int
    name: str
    points: tuple[tuple[float, float], ...]
    bounds: tuple[float, float, float, float]
    area: float


def lonlat_to_xy(
    *,
    lon: float,
    lat: float,
    origin_lon: float,
    origin_lat: float,
    meters_per_unit: float,
) -> tuple[float, float]:
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = meters_per_deg_lat * math.cos(math.radians(origin_lat))
    x_m = (float(lon) - float(origin_lon)) * meters_per_deg_lon
    y_m = (float(lat) - float(origin_lat)) * meters_per_deg_lat
    return x_m / meters_per_unit, y_m / meters_per_unit


def xy_to_lonlat(
    *,
    x: float,
    y: float,
    origin_lon: float,
    origin_lat: float,
    meters_per_unit: float,
) -> tuple[float, float]:
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = meters_per_deg_lat * math.cos(math.radians(origin_lat))
    lon = float(origin_lon) + float(x) * meters_per_unit / meters_per_deg_lon
    lat = float(origin_lat) + float(y) * meters_per_unit / meters_per_deg_lat
    return lon, lat


def world_bbox_lonlat(
    *,
    origin_lon: float,
    origin_lat: float,
    meters_per_unit: float,
    margin_m: float = 20.0,
) -> tuple[float, float, float, float]:
    west, south = xy_to_lonlat(
        x=WORLD_X_MIN - margin_m / meters_per_unit,
        y=WORLD_Y_MIN - margin_m / meters_per_unit,
        origin_lon=origin_lon,
        origin_lat=origin_lat,
        meters_per_unit=meters_per_unit,
    )
    east, north = xy_to_lonlat(
        x=WORLD_X_MAX + margin_m / meters_per_unit,
        y=WORLD_Y_MAX + margin_m / meters_per_unit,
        origin_lon=origin_lon,
        origin_lat=origin_lat,
        meters_per_unit=meters_per_unit,
    )
    return south, west, north, east


def _overpass_query(south: float, west: float, north: float, east: float) -> str:
    return f"""
[out:json][timeout:25];
(
  way["building"]({south:.8f},{west:.8f},{north:.8f},{east:.8f});
);
out body;
>;
out skel qt;
"""


def fetch_osm_buildings(
    *,
    origin_lon: float,
    origin_lat: float,
    meters_per_unit: float,
    endpoint: str = OVERPASS_ENDPOINT,
) -> dict[str, Any]:
    south, west, north, east = world_bbox_lonlat(
        origin_lon=origin_lon,
        origin_lat=origin_lat,
        meters_per_unit=meters_per_unit,
    )
    query = _overpass_query(south, west, north, east)
    payload = urllib.parse.urlencode({"data": query}).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={
            "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
            "User-Agent": "hackerthon-real-building-obstacles/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.loads(response.read().decode("utf-8"))


def _polygon_area(points: tuple[tuple[float, float], ...]) -> float:
    if len(points) < 3:
        return 0.0
    total = 0.0
    for idx, (x0, y0) in enumerate(points):
        x1, y1 = points[(idx + 1) % len(points)]
        total += x0 * y1 - x1 * y0
    return abs(total) / 2.0


def _is_non_blocking_footprint(tags: dict[str, Any]) -> bool:
    """Return True for mapped structures people can move through in this scenario."""
    name = " ".join(
        str(tags.get(key) or "")
        for key in ("name", "name:ko", "name:en")
    ).lower()
    building_value = str(tags.get("building") or "").lower()
    location = str(tags.get("location") or "").lower()
    layer = str(tags.get("layer") or "")

    if any(keyword in name for keyword in NON_BLOCKING_NAME_KEYWORDS):
        return True
    if building_value in NON_BLOCKING_BUILDING_VALUES:
        return True
    if location == "underground":
        return True
    try:
        if float(layer) < 0:
            return True
    except ValueError:
        pass
    return False


def _point_in_polygon(x: float, y: float, points: tuple[tuple[float, float], ...]) -> bool:
    inside = False
    count = len(points)
    if count < 3:
        return False
    prev_x, prev_y = points[-1]
    for curr_x, curr_y in points:
        crosses = (curr_y > y) != (prev_y > y)
        if crosses:
            at_x = (prev_x - curr_x) * (y - curr_y) / (prev_y - curr_y + 1e-12) + curr_x
            if x < at_x:
                inside = not inside
        prev_x, prev_y = curr_x, curr_y
    return inside


def _intersects_world(bounds: tuple[float, float, float, float]) -> bool:
    xmin, ymin, xmax, ymax = bounds
    return not (
        xmax < WORLD_X_MIN
        or xmin > WORLD_X_MAX
        or ymax < WORLD_Y_MIN
        or ymin > WORLD_Y_MAX
    )


def _clip_rect(bounds: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = bounds
    return (
        max(WORLD_X_MIN, xmin),
        max(WORLD_Y_MIN, ymin),
        min(WORLD_X_MAX, xmax),
        min(WORLD_Y_MAX, ymax),
    )


def _rects_from_polygon(
    building: LocalBuilding,
    *,
    cell_size: float,
) -> list[tuple[float, float, float, float]]:
    xmin, ymin, xmax, ymax = _clip_rect(building.bounds)
    if xmax - xmin <= 0.05 or ymax - ymin <= 0.05:
        return []

    x0 = math.floor(xmin / cell_size) * cell_size
    y0 = math.floor(ymin / cell_size) * cell_size
    rows: list[tuple[float, list[float]]] = []
    y = y0
    while y < ymax:
        cy = y + cell_size / 2.0
        filled: list[float] = []
        x = x0
        while x < xmax:
            cx = x + cell_size / 2.0
            if (
                WORLD_X_MIN <= cx <= WORLD_X_MAX
                and WORLD_Y_MIN <= cy <= WORLD_Y_MAX
                and _point_in_polygon(cx, cy, building.points)
            ):
                filled.append(x)
            x += cell_size
        if filled:
            rows.append((y, filled))
        y += cell_size

    rects: list[tuple[float, float, float, float]] = []
    for row_y, xs in rows:
        xs.sort()
        start = xs[0]
        prev = xs[0]
        for value in xs[1:]:
            if value > prev + cell_size * 1.01:
                rects.append(_clip_rect((start, row_y, prev + cell_size, row_y + cell_size)))
                start = value
            prev = value
        rects.append(_clip_rect((start, row_y, prev + cell_size, row_y + cell_size)))

    if not rects:
        rects.append(_clip_rect(building.bounds))
    return rects


def osm_payload_to_buildings(
    payload: dict[str, Any],
    *,
    origin_lon: float,
    origin_lat: float,
    meters_per_unit: float,
    min_area: float,
) -> list[LocalBuilding]:
    nodes: dict[int, tuple[float, float]] = {}
    ways: list[dict[str, Any]] = []
    for element in payload.get("elements", []):
        if element.get("type") == "node":
            nodes[int(element["id"])] = (
                float(element["lon"]),
                float(element["lat"]),
            )
        elif element.get("type") == "way":
            ways.append(element)

    buildings: list[LocalBuilding] = []
    for way in ways:
        local_points: list[tuple[float, float]] = []
        for node_id in way.get("nodes", []):
            lonlat = nodes.get(int(node_id))
            if lonlat is None:
                continue
            x, y = lonlat_to_xy(
                lon=lonlat[0],
                lat=lonlat[1],
                origin_lon=origin_lon,
                origin_lat=origin_lat,
                meters_per_unit=meters_per_unit,
            )
            local_points.append((x, y))
        if len(local_points) >= 2 and local_points[0] == local_points[-1]:
            local_points.pop()
        if len(local_points) < 3:
            continue

        xs = [point[0] for point in local_points]
        ys = [point[1] for point in local_points]
        bounds = (min(xs), min(ys), max(xs), max(ys))
        if not _intersects_world(bounds):
            continue
        area = _polygon_area(tuple(local_points))
        if area < min_area:
            continue
        tags = way.get("tags") or {}
        if _is_non_blocking_footprint(tags):
            continue
        buildings.append(
            LocalBuilding(
                osm_id=int(way["id"]),
                name=str(tags.get("name") or tags.get("name:ko") or ""),
                points=tuple(local_points),
                bounds=bounds,
                area=area,
            )
        )

    buildings.sort(key=lambda item: item.area, reverse=True)
    return buildings


def build_config_overlay(
    buildings: list[LocalBuilding],
    *,
    origin_lon: float,
    origin_lat: float,
    meters_per_unit: float,
    max_buildings: int,
    obstacle_cell_size: float,
    unit_radius_units: float,
) -> dict[str, Any]:
    selected = buildings[:max_buildings]
    obstacles = []
    building_polygons = []
    for building in selected:
        rects = _rects_from_polygon(building, cell_size=obstacle_cell_size)
        obstacles.extend(
            [round(xmin, 3), round(ymin, 3), round(xmax, 3), round(ymax, 3)]
            for xmin, ymin, xmax, ymax in rects
        )
        building_polygons.append(
            {
                "osm_id": building.osm_id,
                "name": building.name,
                "points": [
                    [round(x, 3), round(y, 3)]
                    for x, y in building.points
                ],
                "bounds": [round(value, 3) for value in building.bounds],
                "area": round(building.area, 3),
            }
        )

    return {
        "obstacles": obstacles,
        "building_polygons": building_polygons,
        "real_map": {
            "source": "OpenStreetMap Overpass building=* ways",
            "origin_name": "Gangnam Station",
            "origin_lat": origin_lat,
            "origin_lon": origin_lon,
            "meters_per_unit": meters_per_unit,
            "unit_radius_units": unit_radius_units,
            "unit_radius_meters": unit_radius_units * meters_per_unit,
            "world_bounds": [WORLD_X_MIN, WORLD_Y_MIN, WORLD_X_MAX, WORLD_Y_MAX],
            "building_count": len(building_polygons),
            "obstacle_count": len(obstacles),
            "obstacle_cell_size": obstacle_cell_size,
            "obstacle_model": "axis-aligned grid strips generated inside real building footprints",
        },
    }


def _read_base_config(path: Optional[Path]) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AABB terrain obstacles from real building footprints.")
    parser.add_argument("--preset", choices=("gangnam",), default="gangnam")
    parser.add_argument("--origin-lat", type=float, default=GANGNAM_STATION_LAT)
    parser.add_argument("--origin-lon", type=float, default=GANGNAM_STATION_LON)
    parser.add_argument("--meters-per-unit", type=float, default=DEFAULT_METERS_PER_UNIT)
    parser.add_argument("--unit-radius-units", type=float, default=DEFAULT_UNIT_RADIUS_UNITS)
    parser.add_argument("--max-buildings", type=int, default=60)
    parser.add_argument("--min-area", type=float, default=0.15)
    parser.add_argument("--obstacle-cell-size", type=float, default=1.0)
    parser.add_argument("--endpoint", default=OVERPASS_ENDPOINT)
    parser.add_argument("--base-config", type=Path, default=None)
    parser.add_argument("--out-config", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    payload = fetch_osm_buildings(
        origin_lon=args.origin_lon,
        origin_lat=args.origin_lat,
        meters_per_unit=args.meters_per_unit,
        endpoint=args.endpoint,
    )
    buildings = osm_payload_to_buildings(
        payload,
        origin_lon=args.origin_lon,
        origin_lat=args.origin_lat,
        meters_per_unit=args.meters_per_unit,
        min_area=args.min_area,
    )
    overlay = build_config_overlay(
        buildings,
        origin_lon=args.origin_lon,
        origin_lat=args.origin_lat,
        meters_per_unit=args.meters_per_unit,
        max_buildings=args.max_buildings,
        obstacle_cell_size=args.obstacle_cell_size,
        unit_radius_units=args.unit_radius_units,
    )
    config = _read_base_config(args.base_config)
    config.update(overlay)
    config.setdefault("objective", [11.0, 1.5])
    args.out_config.parent.mkdir(parents=True, exist_ok=True)
    args.out_config.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        "saved "
        f"{args.out_config} with {len(overlay['obstacles'])} obstacles "
        f"from {len(buildings)} fetched buildings"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
