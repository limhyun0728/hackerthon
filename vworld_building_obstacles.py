"""Generate terrain obstacles from VWorld 2D building footprint data."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from local_env import load_local_env
except ModuleNotFoundError:
    from hackerthon.local_env import load_local_env

from hackerthon.real_building_obstacles import (
    DEFAULT_METERS_PER_UNIT,
    DEFAULT_UNIT_RADIUS_UNITS,
    GANGNAM_STATION_LAT,
    GANGNAM_STATION_LON,
    LocalBuilding,
    _intersects_world,
    _is_non_blocking_footprint,
    _polygon_area,
    build_config_overlay,
    lonlat_to_xy,
    world_bbox_lonlat,
)


load_local_env()

VWORLD_DATA_ENDPOINT = "https://api.vworld.kr/req/data"
VWORLD_SPBD_LAYER = "LT_C_SPBD"
HARDCODED_VWORLD_API_KEY = ""
DEFAULT_VWORLD_DOMAIN = "http://100.89.147.58:8765"


def _feature_rings(geometry: dict[str, Any]) -> list[list[list[float]]]:
    geom_type = geometry.get("type")
    coords = geometry.get("coordinates") or []
    if geom_type == "Polygon":
        return [coords[0]] if coords else []
    if geom_type == "MultiPolygon":
        return [polygon[0] for polygon in coords if polygon]
    return []


def _stable_numeric_id(value: str) -> int:
    digits = "".join(ch for ch in value if ch.isdigit())
    if digits:
        return int(digits[-15:])
    total = 0
    for byte in value.encode("utf-8"):
        total = (total * 131 + byte) % 1_000_000_000_000_000
    return total


def _feature_name(properties: dict[str, Any]) -> str:
    return str(
        properties.get("buld_nm")
        or properties.get("buld_nm_dc")
        or " ".join(
            part
            for part in (str(properties.get("rd_nm") or ""), str(properties.get("buld_no") or ""))
            if part
        )
    )


def fetch_vworld_buildings(
    *,
    origin_lon: float,
    origin_lat: float,
    meters_per_unit: float,
    key: str,
    domain: str,
    endpoint: str = VWORLD_DATA_ENDPOINT,
    layer: str = VWORLD_SPBD_LAYER,
    size: int = 1000,
) -> dict[str, Any]:
    south, west, north, east = world_bbox_lonlat(
        origin_lon=origin_lon,
        origin_lat=origin_lat,
        meters_per_unit=meters_per_unit,
    )
    if not key:
        raise ValueError("VWorld API key가 필요하다. --vworld-key 또는 VWORLD_API_KEY를 설정하세요.")

    page = 1
    features: list[dict[str, Any]] = []
    bbox: list[float] | None = None
    while True:
        params = {
            "key": key,
            "domain": domain,
            "service": "data",
            "version": "2.0",
            "request": "getfeature",
            "format": "json",
            "size": str(size),
            "page": str(page),
            "geometry": "true",
            "attribute": "true",
            "crs": "EPSG:4326",
            "geomfilter": f"BOX({west},{south},{east},{north})",
            "data": layer,
        }
        url = endpoint + "?" + urllib.parse.urlencode(params)
        request = urllib.request.Request(url, headers={"User-Agent": "hackerthon-vworld-building-obstacles/1.0"})
        with urllib.request.urlopen(request, timeout=45) as response:
            payload = json.loads(response.read().decode("utf-8"))

        root = payload.get("response", {})
        status = root.get("status")
        if status != "OK":
            raise RuntimeError(f"VWorld API status={status}: {json.dumps(root, ensure_ascii=False)[:500]}")
        collection = root.get("result", {}).get("featureCollection", {})
        features.extend(collection.get("features", []) or [])
        if bbox is None and collection.get("bbox"):
            bbox = list(collection["bbox"])

        current_page = int(root.get("page", {}).get("current", page))
        total_page = int(root.get("page", {}).get("total", current_page))
        if current_page >= total_page:
            break
        page += 1

    return {
        "type": "FeatureCollection",
        "bbox": bbox,
        "features": features,
    }


def vworld_payload_to_buildings(
    payload: dict[str, Any],
    *,
    origin_lon: float,
    origin_lat: float,
    meters_per_unit: float,
    min_area: float,
) -> list[LocalBuilding]:
    buildings: list[LocalBuilding] = []
    for feature in payload.get("features", []) or []:
        geometry = feature.get("geometry") or {}
        properties = feature.get("properties") or {}
        tags = {
            "name": _feature_name(properties),
            "name:ko": _feature_name(properties),
            "building": "yes",
            "location": properties.get("pos_bul_nm") or "",
            "layer": properties.get("layer") or "",
        }
        if _is_non_blocking_footprint(tags):
            continue
        for index, ring in enumerate(_feature_rings(geometry)):
            local_points = [
                lonlat_to_xy(
                    lon=float(point[0]),
                    lat=float(point[1]),
                    origin_lon=origin_lon,
                    origin_lat=origin_lat,
                    meters_per_unit=meters_per_unit,
                )
                for point in ring
                if len(point) >= 2
            ]
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
            source_id = str(feature.get("id") or properties.get("bd_mgt_sn") or f"vworld_{len(buildings)}")
            buildings.append(
                LocalBuilding(
                    osm_id=_stable_numeric_id(f"{source_id}_{index}"),
                    name=_feature_name(properties),
                    points=tuple(local_points),
                    bounds=bounds,
                    area=area,
                )
            )
    buildings.sort(key=lambda item: item.area, reverse=True)
    return buildings


def _read_base_config(path: Optional[Path]) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AABB terrain obstacles from VWorld building footprints.")
    parser.add_argument("--preset", choices=("gangnam",), default="gangnam")
    parser.add_argument("--origin-lat", type=float, default=GANGNAM_STATION_LAT)
    parser.add_argument("--origin-lon", type=float, default=GANGNAM_STATION_LON)
    parser.add_argument("--meters-per-unit", type=float, default=DEFAULT_METERS_PER_UNIT)
    parser.add_argument("--unit-radius-units", type=float, default=DEFAULT_UNIT_RADIUS_UNITS)
    parser.add_argument("--max-buildings", type=int, default=120)
    parser.add_argument("--min-area", type=float, default=0.06)
    parser.add_argument("--obstacle-cell-size", type=float, default=1.0)
    parser.add_argument("--endpoint", default=VWORLD_DATA_ENDPOINT)
    parser.add_argument("--layer", default=VWORLD_SPBD_LAYER)
    parser.add_argument("--domain", default=os.getenv("VWORLD_DOMAIN", DEFAULT_VWORLD_DOMAIN))
    parser.add_argument("--vworld-key", default=os.getenv("VWORLD_API_KEY", "") or HARDCODED_VWORLD_API_KEY)
    parser.add_argument("--base-config", type=Path, default=None)
    parser.add_argument("--out-config", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    payload = fetch_vworld_buildings(
        origin_lon=args.origin_lon,
        origin_lat=args.origin_lat,
        meters_per_unit=args.meters_per_unit,
        key=args.vworld_key,
        domain=args.domain,
        endpoint=args.endpoint,
        layer=args.layer,
    )
    buildings = vworld_payload_to_buildings(
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
    overlay["real_map"].update(
        {
            "source": f"VWorld 2D Data API {args.layer}",
            "vworld_layer": args.layer,
            "vworld_domain": args.domain,
            "fetched_feature_count": len(payload.get("features", []) or []),
        }
    )
    config = _read_base_config(args.base_config)
    config.update(overlay)
    config.setdefault("objective", [11.0, 1.5])
    args.out_config.parent.mkdir(parents=True, exist_ok=True)
    args.out_config.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        "saved "
        f"{args.out_config} with {len(overlay['obstacles'])} obstacles "
        f"from {len(buildings)} VWorld buildings "
        f"({len(payload.get('features', []) or [])} fetched features)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
