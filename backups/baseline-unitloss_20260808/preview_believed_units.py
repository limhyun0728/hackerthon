"""미관측 RED 표시가 화면에서 어떻게 보이는지 이미지로 뽑는다.

플랫폼 UI의 <style>과 unitHtml을 platform_ui.PAGE_HTML에서 직접 뽑아 쓴다.
여기서 마크업을 다시 쓰면 미리보기가 실제 화면과 어긋나 거짓말이 되므로,
항상 원본에서 추출한다.

불확실 원의 크기 근거는 RED 최대 이동속도 x 미관측 경과 시간이다.
(로그 실측 p99/max 모두 1.0유닛/초 = 10m/s, 관측거리 100m에서 상한)

사용법:
    python preview_believed_units.py --out output/ui_previews
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hackerthon.platform_ui import PAGE_HTML

# 미리보기 기준 축척. zoom17 / 강남(위도 37.5) 에서의 m/px.
PREVIEW_METERS_PER_PIXEL = 0.9475
STAGE_PX = 250

# (설명, 유닛 dict). last_seen 은 S.view.time=60 기준 상대값이다.
SAMPLES: tuple[tuple[str, dict[str, object]], ...] = (
    ("BLUE 아군", {"id": 103, "hp": 100}),
    ("RED 관측중", {"id": 204, "hp": 72, "observed": True}),
    ("RED 미관측 1초", {"id": 205, "hp": 100, "observed": False, "last_seen": 59}),
    ("RED 미관측 3초", {"id": 206, "hp": 88, "observed": False, "last_seen": 57}),
    ("RED 미관측 6초", {"id": 207, "hp": 45, "observed": False, "last_seen": 54}),
    ("RED 미관측 12초", {"id": 208, "hp": 100, "observed": False, "last_seen": 48}),
)


def _extract(pattern: str) -> str:
    """PAGE_HTML에서 조각을 뽑는다. 못 찾으면 UI가 바뀐 것이므로 바로 실패시킨다."""
    found = re.search(pattern, PAGE_HTML, re.S)
    if found is None:
        raise ValueError(f"platform_ui.PAGE_HTML에서 조각을 못 찾았다: {pattern}")
    return found.group(1)


def build_html() -> str:
    """실제 UI 조각을 심은 단일 페이지를 만든다."""
    style = _extract(r"<style>(.*?)</style>")
    # 최상위 숫자 상수는 전부 가져온다. 하나만 빠져도 페이지가 ReferenceError로
    # 죽어 빈 이미지가 나오므로, 이름을 나열하지 않고 통째로 옮긴다.
    consts = "\n".join(re.findall(r"^const [A-Z_]+ = [\d.]+;", PAGE_HTML, re.M))
    if "MIN_BELIEVED_PX" not in consts or "RED_MAX_SPEED_MPS" not in consts:
        raise ValueError(f"상수 추출이 불완전하다: {consts!r}")
    unit_px = _extract(r"(function unitPx\(\) \{.*?\n\})")
    unit_html = _extract(r"(function unitHtml\(u, px\) \{.*?\n\})")
    if "ghost" not in unit_html:
        raise ValueError("unitHtml에 불확실 원(ghost)이 없다")

    cells = "".join(
        f'<div class="samp"><div class="cap">{name}</div>'
        f"<div class=\"stage\" data-u='{json.dumps(unit)}'></div></div>"
        for name, unit in SAMPLES
    )
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><style>{style}
 body {{ background:#0f1419; padding:24px; }}
 .row {{ display:flex; flex-wrap:wrap; }}
 .samp {{ width:{STAGE_PX}px; }}
 .cap {{ font:600 13px system-ui; color:#8b949e; margin-bottom:6px; text-align:center; }}
 /* 지도 배경 근사: drawTerrain의 건물 색/테두리와 같은 값 */
 .stage {{ position:relative; width:{STAGE_PX}px; height:{STAGE_PX}px;
   background:#1a1f26; border:1px solid #21262d; overflow:hidden; }}
 .stage::before {{ content:''; position:absolute; left:14px; top:14px; width:74px; height:96px;
   background:#2b3138; border:1.5px solid #f0f6fcE6; }}
 .stage::after {{ content:''; position:absolute; right:16px; bottom:18px; width:92px; height:70px;
   background:#2b3138; border:1.5px solid #f0f6fcE6; }}
 .anchor {{ position:absolute; left:{STAGE_PX // 2}px; top:{STAGE_PX // 2}px; }}
 .scalebar {{ position:absolute; left:10px; bottom:8px; font:11px system-ui; color:#8b949e; }}
</style></head><body>
<div class="row">{cells}</div>
<script>
{consts}
function metersPerPixel() {{ return {PREVIEW_METERS_PER_PIXEL}; }}
function rm() {{ return {{ unit_radius_meters:0.35, meters_per_unit:10.0, origin_lat:37.5 }}; }}
let S = {{ view:{{ time:60 }} }};
{unit_px}
{unit_html}
document.querySelectorAll('.stage').forEach(stage => {{
  const anchor = document.createElement('div');
  anchor.className = 'anchor';
  anchor.innerHTML = unitHtml(JSON.parse(stage.dataset.u), unitPx());
  stage.appendChild(anchor);
  const bar = document.createElement('div');
  bar.className = 'scalebar';
  bar.innerHTML = '&#8596; ' + Math.round(({STAGE_PX} - 20) * metersPerPixel()) + 'm';
  stage.appendChild(bar);
}});
// 캡처는 정지 화면이라 맥동을 최고점에서 고정한다. 실제 화면은 1.6초 주기로 뛴다.
document.querySelectorAll('.ghost').forEach(g => {{
  g.style.animation = 'none';
  g.style.opacity = '1';
  g.style.boxShadow = '0 0 14px #fb8500aa inset, 0 0 14px #fb8500aa';
}});
</script></body></html>"""


def _chrome_binary() -> str:
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        path = shutil.which(name)
        if path:
            return path
    raise ValueError("chrome 계열 브라우저를 못 찾았다")


def render(out_dir: Path) -> Path:
    """HTML을 만들고 headless chrome으로 PNG를 뽑는다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "believed_red.html"
    png_path = out_dir / "believed_red.png"
    html_path.write_text(build_html(), encoding="utf-8")

    with tempfile.TemporaryDirectory(prefix="chrome_ghost_preview_") as profile:
        subprocess.run(
            [
                _chrome_binary(),
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                "--hide-scrollbars",
                f"--user-data-dir={profile}",
                f"--window-size={STAGE_PX * len(SAMPLES) + 60},340",
                f"--screenshot={png_path}",
                "--virtual-time-budget=1500",
                html_path.resolve().as_uri(),
            ],
            check=True,
            capture_output=True,
        )
    if not png_path.exists():
        raise ValueError(f"캡처 실패: {png_path}")
    return png_path


def _parse_args(argv: Iterable[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="미관측 RED 표시 미리보기 생성")
    parser.add_argument("--out", type=Path, default=Path("output/ui_previews"))
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    png_path = render(args.out)
    print(f"png={png_path} ({png_path.stat().st_size} bytes)")
    print(f"html={png_path.with_suffix('.html')}")
    print("\n미관측 경과 시간별 불확실 반경 (RED 최대속도 10m/s, 관측거리 100m 상한)")
    for seconds in (0, 1, 3, 6, 10, 12):
        radius = min(100.0, seconds * 10.0)
        diameter = max(16.0, 2 * radius / PREVIEW_METERS_PER_PIXEL)
        note = "  (상한)" if radius >= 100.0 else ""
        print(f"  {seconds:>3}초  반경 {radius:>5.0f}m  화면 지름 {diameter:>5.0f}px{note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
