"""생성한 맵들을 NAVER Dynamic Map 위에 올려 PNG로 캡처한다.

맵마다 naver_dynamic_map_server를 띄우고, headless Chrome을 CDP로 붙잡아
한 프레임만 찍는다. render_naver_overlay_video와 같은 캡처 경로를 쓰되
soldier_log 없이 지형만 있는 config도 처리한다.

사용법:
    python capture_map_overlays.py --maps-root output/maps --out-dir output/map_previews
"""

from __future__ import annotations

import argparse
import base64
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

from hackerthon.render_naver_overlay_video import (
    CdpClient,
    _open_tab,
    _runtime_value,
    _wait_for_chrome,
    _with_query,
)


def _wait_for_server(url: str, timeout_sec: float = 20.0) -> dict:
    """서버가 state를 내려줄 때까지 기다린다."""
    deadline = time.time() + timeout_sec
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            response = requests.get(url.rstrip("/") + "/api/state", timeout=1.0)
            response.raise_for_status()
            return response.json()
        except Exception as error:  # 서버가 아직 안 떴을 뿐이다
            last_error = error
            time.sleep(0.2)
    raise TimeoutError(f"지도 서버가 준비되지 않았다: {last_error}")


def _wait_until_map_ready(client: CdpClient, timeout_sec: float) -> bool:
    """지도 타일과 오버레이가 그려질 때까지 기다린다.

    유닛 로그가 없는 지형 전용 페이지는 __naverTacticalReady를 세우지 않을 수
    있어서, 준비 플래그가 안 오면 실패로 보지 않고 고정 대기로 넘어간다.
    """
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            if _runtime_value(client, "window.__naverTacticalReady === true"):
                return True
        except Exception:
            pass
        time.sleep(0.2)
    return False


def _capture_one(
    *,
    map_dir: Path,
    out_path: Path,
    port: int,
    url_host: str,
    server_host: str,
    debug_port: int,
    width: int,
    height: int,
    zoom: int | None,
    chrome_bin: str,
    python_executable: str,
    ready_timeout: float,
    settle_sec: float,
) -> None:
    """맵 하나를 서버에 올리고 한 프레임 캡처한다."""
    server_command = [
        python_executable,
        str(Path(__file__).resolve().parent / "naver_dynamic_map_server.py"),
        "--run-dir", str(map_dir),
        "--port", str(port),
        "--host", server_host,
    ]
    if zoom is not None:
        server_command += ["--zoom", str(zoom)]

    server = subprocess.Popen(server_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    user_data_dir = tempfile.TemporaryDirectory(prefix="chrome_map_preview_")
    chrome = None
    client: CdpClient | None = None
    try:
        # 서버는 127.0.0.1에 바인딩하고, 브라우저 접속 URL만 등록한 호스트로 맞춘다.
        server_url = f"http://{url_host}:{port}/"
        state = _wait_for_server(f"http://127.0.0.1:{port}/")
        print(
            f"  건물 {len(state.get('buildingPolygons') or [])}동, "
            f"장애물 {len(state.get('obstacles') or [])}개"
        )

        chrome_cmd = [
            chrome_bin,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--remote-allow-origins=*",
            f"--remote-debugging-port={debug_port}",
            f"--user-data-dir={user_data_dir.name}",
            f"--window-size={width},{height}",
            "about:blank",
        ]
        chrome = subprocess.Popen(chrome_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        _wait_for_chrome(debug_port)

        capture_url = _with_query(server_url, {"capture": "1", "hud": "0", "t": "0"})
        client = CdpClient(_open_tab(debug_port, capture_url))
        client.call("Page.enable")
        client.call("Runtime.enable")
        client.call(
            "Emulation.setDeviceMetricsOverride",
            {"width": width, "height": height, "deviceScaleFactor": 2, "mobile": False},
        )
        ready = _wait_until_map_ready(client, ready_timeout)
        if not ready:
            print("  준비 플래그 없음 (유닛 로그 없는 지형 전용 페이지) - 고정 대기로 진행")
        # 네이버 타일은 비동기로 들어온다. 플래그와 무관하게 충분히 기다린다.
        time.sleep(settle_sec)
        try:
            _runtime_value(client, "window.__renderNaverTacticalAt && window.__renderNaverTacticalAt(0); true")
            time.sleep(0.5)
        except Exception:
            pass

        screenshot = client.call(
            "Page.captureScreenshot",
            {"format": "png", "fromSurface": True, "captureBeyondViewport": False},
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(base64.b64decode(screenshot["data"]))
        print(f"  저장: {out_path} ({out_path.stat().st_size // 1024}KB)")
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
        for process in (chrome, server):
            if process is not None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
        user_data_dir.cleanup()


def _parse_args(argv: Iterable[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="맵 config를 네이버 지도 위에 캡처")
    parser.add_argument("--maps-root", type=Path, default=Path("output/maps"))
    parser.add_argument("--out-dir", type=Path, default=Path("output/map_previews"))
    parser.add_argument("--only", nargs="+", default=None, help="캡처할 맵 이름")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--zoom", type=int, default=None, help="지도 확대 수준. 생략하면 서버 기본값")
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="지도 서버 포트. 네이버 콘솔에 등록한 Web 서비스 URL과 같은 포트여야 타일이 뜬다.",
    )
    parser.add_argument(
        "--url-host",
        default="localhost",
        help="브라우저가 접속할 호스트. 네이버는 localhost와 127.0.0.1을 다른 URL로 취급한다.",
    )
    parser.add_argument(
        "--server-host",
        default="0.0.0.0",
        help="지도 서버 바인딩 주소. Tailscale IP 같은 외부 주소로 접속하려면 0.0.0.0이어야 한다.",
    )
    parser.add_argument("--debug-port", type=int, default=9401)
    parser.add_argument("--ready-timeout", type=float, default=20.0)
    parser.add_argument("--settle-sec", type=float, default=6.0, help="타일 로딩 대기 시간")
    parser.add_argument("--chrome-bin", default="")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    chrome_bin = (
        args.chrome_bin
        or shutil.which("google-chrome")
        or shutil.which("google-chrome-stable")
        or shutil.which("chromium")
    )
    if not chrome_bin:
        raise FileNotFoundError("google-chrome/chromium 실행 파일을 찾을 수 없다")

    map_dirs = sorted(
        path for path in args.maps_root.iterdir() if (path / "config.json").exists()
    )
    if args.only:
        wanted = set(args.only)
        map_dirs = [path for path in map_dirs if path.name in wanted]
    if not map_dirs:
        raise ValueError(f"{args.maps_root} 아래에 config.json을 가진 맵이 없다")

    saved: list[Path] = []
    for index, map_dir in enumerate(map_dirs):
        print(f"\n=== {map_dir.name}")
        out_path = args.out_dir / f"{map_dir.name}.png"
        _capture_one(
            map_dir=map_dir,
            out_path=out_path,
            port=args.port,
            url_host=args.url_host,
            server_host=args.server_host,
            debug_port=args.debug_port + index,
            width=args.width,
            height=args.height,
            zoom=args.zoom,
            chrome_bin=chrome_bin,
            python_executable=args.python,
            ready_timeout=args.ready_timeout,
            settle_sec=args.settle_sec,
        )
        saved.append(out_path)

    print(f"\n총 {len(saved)}개 저장:")
    for path in saved:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
