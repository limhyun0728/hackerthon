"""Shared rendering helpers for terrain overlays and unit scale."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from matplotlib.axes import Axes
from matplotlib.patches import Polygon, Rectangle


DEFAULT_UNIT_RADIUS_UNITS = 0.035


def iter_building_polygons(config: Mapping[str, Any]) -> Iterator[list[tuple[float, float]]]:
    """Yield display polygons from config["building_polygons"]."""
    for building in config.get("building_polygons", []) or []:
        points = building.get("points", []) if isinstance(building, Mapping) else building
        polygon = []
        for point in points or []:
            if len(point) >= 2:
                polygon.append((float(point[0]), float(point[1])))
        if len(polygon) >= 3:
            yield polygon


def terrain_xy_values(config: Mapping[str, Any]) -> tuple[list[float], list[float]]:
    """Return x/y values that should be included in plot bounds."""
    xs: list[float] = []
    ys: list[float] = []
    for polygon in iter_building_polygons(config):
        for x, y in polygon:
            xs.append(x)
            ys.append(y)
    for rect in config.get("obstacles", []) or []:
        if len(rect) == 4:
            xmin, ymin, xmax, ymax = (float(value) for value in rect)
            xs.extend([xmin, xmax])
            ys.extend([ymin, ymax])
    return xs, ys


def draw_terrain(ax: Axes, config: Mapping[str, Any], *, zorder: int = 4) -> list[Any]:
    """Draw real building polygons when present, otherwise draw AABB obstacles."""
    artists: list[Any] = []
    polygons = list(iter_building_polygons(config))
    if polygons:
        for points in polygons:
            patch = Polygon(
                points,
                closed=True,
                fc="#536878",
                ec="#26323d",
                alpha=0.94,
                lw=0.7,
                zorder=zorder,
            )
            ax.add_patch(patch)
            artists.append(patch)
        return artists

    for xmin, ymin, xmax, ymax in config.get("obstacles", []) or []:
        xmin, ymin, xmax, ymax = float(xmin), float(ymin), float(xmax), float(ymax)
        width, height = xmax - xmin, ymax - ymin
        small_cover = width * height < 2.5
        patch = Rectangle(
            (xmin, ymin),
            width,
            height,
            fc="#687b8c" if small_cover else "0.55",
            ec="#43515d" if small_cover else "0.3",
            zorder=zorder,
        )
        ax.add_patch(patch)
        artists.append(patch)
    return artists


def unit_radius_units(config: Mapping[str, Any], default: float = DEFAULT_UNIT_RADIUS_UNITS) -> float:
    """Read unit body radius in sim units, falling back to meters/meters_per_unit."""
    real_map = config.get("real_map", {})
    if not isinstance(real_map, Mapping):
        real_map = {}

    raw_units = real_map.get("unit_radius_units", config.get("unit_radius_units"))
    if raw_units is not None:
        return max(0.01, float(raw_units))

    raw_meters = real_map.get("unit_radius_meters", config.get("unit_radius_meters"))
    raw_meters_per_unit = real_map.get("meters_per_unit", config.get("meters_per_unit"))
    if raw_meters is not None and raw_meters_per_unit:
        return max(0.01, float(raw_meters) / float(raw_meters_per_unit))

    return default
