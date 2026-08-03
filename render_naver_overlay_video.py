"""Render the NAVER Dynamic Map tactical overlay to an MP4 video.

The server page exposes window.__renderNaverTacticalAt(t). This script drives a
single headless Chrome tab through the Chrome DevTools Protocol, captures PNG
frames, then encodes them with ffmpeg.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
from pathlib import Path
from typing import Any

import requests
import websocket


DEFAULT_SERVER_URL = "http://100.89.147.58:8765/"


class CdpClient:
    """Tiny synchronous Chrome DevTools Protocol client."""

    def __init__(self, websocket_url: str) -> None:
        self.ws = websocket.create_connection(websocket_url, timeout=30)
        self.next_id = 1

    def close(self) -> None:
        self.ws.close()

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        message_id = self.next_id
        self.next_id += 1
        self.ws.send(json.dumps({"id": message_id, "method": method, "params": params or {}}))
        while True:
            response = json.loads(self.ws.recv())
            if response.get("id") == message_id:
                if "error" in response:
                    raise RuntimeError(f"CDP {method} failed: {response['error']}")
                return response.get("result", {})


def _with_query(url: str, params: dict[str, str]) -> str:
    parsed = urllib.parse.urlparse(url)
    query = dict(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
    query.update(params)
    return urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(query)))


def _api_url(server_url: str, path: str) -> str:
    parsed = urllib.parse.urlparse(server_url)
    return urllib.parse.urlunparse(parsed._replace(path=path, query="", fragment=""))


def _wait_for_chrome(port: int, timeout_sec: float = 15.0) -> None:
    deadline = time.time() + timeout_sec
    endpoint = f"http://127.0.0.1:{port}/json/version"
    while time.time() < deadline:
        try:
            requests.get(endpoint, timeout=0.5).raise_for_status()
            return
        except Exception:
            time.sleep(0.15)
    raise TimeoutError("Chrome remote debugging endpoint did not become ready")


def _open_tab(port: int, url: str) -> str:
    encoded = urllib.parse.quote(url, safe="")
    response = requests.put(f"http://127.0.0.1:{port}/json/new?{encoded}", timeout=10)
    if response.status_code >= 400:
        response = requests.get(f"http://127.0.0.1:{port}/json/new?{encoded}", timeout=10)
    response.raise_for_status()
    return response.json()["webSocketDebuggerUrl"]


def _runtime_value(client: CdpClient, expression: str) -> Any:
    result = client.call(
        "Runtime.evaluate",
        {
            "expression": expression,
            "returnByValue": True,
            "awaitPromise": True,
        },
    )
    remote = result.get("result", {})
    if "exceptionDetails" in result:
        raise RuntimeError(result["exceptionDetails"])
    return remote.get("value")


def _wait_until_ready(client: CdpClient, timeout_sec: float) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            if _runtime_value(client, "window.__naverTacticalReady === true"):
                return
        except Exception:
            pass
        time.sleep(0.2)
    raise TimeoutError("NAVER tactical map page did not become ready")


def _capture_frame(client: CdpClient, time_sec: float, out_path: Path, settle_ms: int) -> None:
    _runtime_value(client, f"window.__renderNaverTacticalAt({time_sec:.4f}); true")
    if settle_ms > 0:
        time.sleep(settle_ms / 1000.0)
    screenshot = client.call(
        "Page.captureScreenshot",
        {
            "format": "png",
            "fromSurface": True,
            "captureBeyondViewport": False,
        },
    )
    out_path.write_bytes(base64.b64decode(screenshot["data"]))


def _encode_video(frames_dir: Path, out_path: Path, fps: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "frame_%05d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def render_video(args: argparse.Namespace) -> None:
    server_url = args.server_url.rstrip("/") + "/"
    state = requests.get(_api_url(server_url, "/api/state"), timeout=10).json()
    time_min = float(args.start if args.start is not None else state.get("timeMin", 0.0))
    time_max = float(args.end if args.end is not None else state.get("timeMax", 0.0))
    if time_max < time_min:
        raise ValueError("--end must be greater than or equal to --start")
    frame_count = int(round((time_max - time_min) * args.fps)) + 1

    chrome_bin = args.chrome_bin or shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")
    if not chrome_bin:
        raise FileNotFoundError("google-chrome/chromium executable not found")

    frames_dir = args.frames_dir
    if frames_dir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="naver_overlay_frames_")
        frames_dir = Path(temp_dir.name)
    else:
        temp_dir = None
        frames_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in frames_dir.glob("frame_*.png"):
        old_frame.unlink()

    user_data_dir = tempfile.TemporaryDirectory(prefix="chrome_naver_overlay_")
    capture_url = _with_query(server_url, {"capture": "1", "hud": "0", "t": f"{time_min:.3f}"})
    chrome_cmd = [
        chrome_bin,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--disable-cache",
        "--disk-cache-size=0",
        "--disable-dev-shm-usage",
        "--remote-allow-origins=*",
        f"--remote-debugging-port={args.debug_port}",
        f"--user-data-dir={user_data_dir.name}",
        f"--window-size={args.width},{args.height}",
        "about:blank",
    ]
    chrome = subprocess.Popen(chrome_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    client: CdpClient | None = None
    try:
        _wait_for_chrome(args.debug_port)
        client = CdpClient(_open_tab(args.debug_port, capture_url))
        client.call("Page.enable")
        client.call("Runtime.enable")
        client.call(
            "Emulation.setDeviceMetricsOverride",
            {
                "width": args.width,
                "height": args.height,
                "deviceScaleFactor": 1,
                "mobile": False,
            },
        )
        _wait_until_ready(client, args.ready_timeout)
        time.sleep(args.initial_wait_ms / 1000.0)

        print(f"capturing {frame_count} frames from t={time_min:.2f} to t={time_max:.2f} at {args.fps} fps")
        for index in range(frame_count):
            time_sec = min(time_max, time_min + index / args.fps)
            _capture_frame(client, time_sec, frames_dir / f"frame_{index:05d}.png", args.settle_ms)
            if (index + 1) % max(1, args.fps * 5) == 0 or index + 1 == frame_count:
                print(f"captured {index + 1}/{frame_count}")

        _encode_video(frames_dir, args.out, args.fps)
        print(f"saved {args.out}")
    finally:
        if client is not None:
            client.close()
        chrome.terminate()
        try:
            chrome.wait(timeout=5)
        except subprocess.TimeoutExpired:
            chrome.kill()
        user_data_dir.cleanup()
        if temp_dir is not None and not args.keep_frames:
            temp_dir.cleanup()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render NAVER map overlay animation to MP4.")
    parser.add_argument("--server-url", default=os.getenv("NAVER_OVERLAY_SERVER_URL", DEFAULT_SERVER_URL))
    parser.add_argument("--out", type=Path, default=Path("hackerthon/output/naver_dynamic_demo_run/naver_dynamic_overlay.mp4"))
    parser.add_argument("--frames-dir", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--start", type=float, default=None)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--settle-ms", type=int, default=25)
    parser.add_argument("--initial-wait-ms", type=int, default=3500)
    parser.add_argument("--ready-timeout", type=float, default=30.0)
    parser.add_argument("--debug-port", type=int, default=9333)
    parser.add_argument("--chrome-bin", default="")
    parser.add_argument("--keep-frames", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    render_video(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
