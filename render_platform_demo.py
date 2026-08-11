"""지휘관 플랫폼을 실제로 조작하며 데모 영상을 만든다.

headless Chrome을 띄워 플랫폼 페이지를 열고, 사람이 하는 순서대로 클릭한다.
  배치 -> 작전 개시 -> 추천 시나리오 재생 -> 전개안 선택 -> 실행
각 단계에서 화면을 캡처해 mp4로 잇는다.

플랫폼이 이미 떠 있어야 한다:
    python commander_platform.py --port 8765 ...

사용법:
    python render_platform_demo.py --map gangnam --out output/demo/platform_demo.mp4
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import websocket

# NAVER 지도는 브라우저 Referer를 검사한다. 콘솔에 등록된 주소로 열어야
# 타일이 뜬다. 127.0.0.1로 열면 "인증 실패" 화면이 캡처된다.
DEFAULT_SERVER = "http://100.90.66.99:8765/"


class CdpClient:
    """Chrome DevTools Protocol 동기 클라이언트."""

    def __init__(self, websocket_url: str) -> None:
        self.ws = websocket.create_connection(websocket_url, timeout=60)
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
                    raise RuntimeError(f"CDP {method}: {response['error']}")
                return response.get("result", {})

    def evaluate(self, expression: str) -> Any:
        result = self.call(
            "Runtime.evaluate",
            {"expression": expression, "awaitPromise": True, "returnByValue": True},
        )
        if result.get("exceptionDetails"):
            raise RuntimeError(f"JS 오류: {result['exceptionDetails']}")
        return result.get("result", {}).get("value")

    def shot(self, path: Path) -> None:
        data = self.call("Page.captureScreenshot", {"format": "png", "fromSurface": True})
        path.write_bytes(base64.b64decode(data["data"]))


def _wait_for_chrome(port: int, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=1).read()
            return
        except OSError:
            time.sleep(0.3)
    raise RuntimeError("Chrome이 안 떴다")


def _open_tab(port: int, url: str) -> str:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/json/new?{urllib.parse.quote(url, safe='')}", method="PUT"
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())["webSocketDebuggerUrl"]


def _hold(client: CdpClient, frames: list[Path], directory: Path, seconds: float, fps: int) -> None:
    """현재 화면을 seconds만큼 붙잡아 둔다. 지휘관이 보는 시간을 준다."""
    for _ in range(max(1, int(seconds * fps))):
        path = directory / f"frame_{len(frames):05d}.png"
        client.shot(path)
        frames.append(path)
        time.sleep(1.0 / fps)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="플랫폼 데모 영상")
    parser.add_argument("--server-url", default=DEFAULT_SERVER)
    parser.add_argument("--map", default="gangnam")
    parser.add_argument("--mission", default="destroy_and_reach")
    parser.add_argument("--blue", type=int, default=5)
    parser.add_argument("--red", type=int, default=5)
    parser.add_argument("--decisions", type=int, default=3, help="시연할 결심 횟수")
    parser.add_argument("--out", type=Path, default=Path("output/demo/platform_demo.mp4"))
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--debug-port", type=int, default=9333)
    parser.add_argument("--chrome", default="/usr/bin/google-chrome")
    args = parser.parse_args(argv)

    user_data = tempfile.TemporaryDirectory(prefix="chrome_platform_demo_")
    frames_dir = Path(tempfile.mkdtemp(prefix="platform_demo_frames_"))
    chrome = subprocess.Popen(
        [
            args.chrome, "--headless=new", "--disable-gpu", "--no-sandbox",
            "--disable-dev-shm-usage", "--remote-allow-origins=*",
            f"--remote-debugging-port={args.debug_port}",
            f"--user-data-dir={user_data.name}",
            f"--window-size={args.width},{args.height}",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    frames: list[Path] = []
    client: CdpClient | None = None
    try:
        _wait_for_chrome(args.debug_port)
        client = CdpClient(_open_tab(args.debug_port, args.server_url))
        client.call("Page.enable")
        client.call("Runtime.enable")
        client.call("Emulation.setDeviceMetricsOverride", {
            "width": args.width, "height": args.height,
            "deviceScaleFactor": 1, "mobile": False,
        })
        time.sleep(4.0)

        # 1) 맵과 임무를 고르고 무작위 배치
        client.evaluate(f"""
            document.getElementById('map-select').value = {json.dumps(args.map)};
            document.getElementById('map-select').dispatchEvent(new Event('change'));
            document.getElementById('mission').value = {json.dumps(args.mission)};
            document.getElementById('rb').value = '{args.blue}';
            document.getElementById('rr').value = '{args.red}';
        """)
        time.sleep(2.5)
        _hold(client, frames, frames_dir, 1.5, args.fps)
        client.evaluate("document.getElementById('rand').click()")
        time.sleep(2.5)
        _hold(client, frames, frames_dir, 2.0, args.fps)

        # 2) 작전 개시. 추천 시나리오와 전개안 격자가 만들어질 때까지 기다린다.
        client.evaluate("document.getElementById('start').click()")
        for _ in range(240):
            time.sleep(1.0)
            ready = client.evaluate(
                "(function(){const g=document.getElementById('grid');"
                "return !!(g && g.querySelectorAll('.cell.on').length);})()"
            )
            if ready:
                break
        _hold(client, frames, frames_dir, 2.5, args.fps)

        # 3) 추천 시나리오 재생
        played = client.evaluate(
            "(function(){const b=document.getElementById('rec-play');"
            "if(b){b.click();return true;} return false;})()"
        )
        if played:
            _hold(client, frames, frames_dir, 6.0, args.fps)

        # 4) 결심 반복: 전개안을 고르고 실행한다
        for decision in range(args.decisions):
            # 격자 셀은 data-i로 후보 index를 들고 있다. 점수 최고 셀을 고른다.
            picked = client.evaluate(
                "(function(){const cells=[...document.querySelectorAll('.cell.on')];"
                "if(!cells.length) return null;"
                "let best=null, bestScore=-1e9;"
                "for(const c of cells){const i=Number(c.dataset.i);"
                "const s=(window.S&&S.cells&&S.cells[i])?S.cells[i].score:0;"
                "if(s>bestScore){bestScore=s;best=c;}}"
                "if(!best) return null; best.click();"
                "return best.textContent.trim().slice(0,40);})()"
            )
            if picked is None:
                break
            _hold(client, frames, frames_dir, 2.0, args.fps)
            # 선택한 안의 경로 재생
            client.evaluate(
                "(function(){const b=document.getElementById('rec-play');if(b) b.click();})()"
            )
            _hold(client, frames, frames_dir, 5.0, args.fps)
            client.evaluate("document.getElementById('commit').click()")
            for _ in range(240):
                time.sleep(1.0)
                ready = client.evaluate(
                    "(function(){const g=document.getElementById('grid');"
                    "return !!(g && g.querySelectorAll('.cell.on').length);})()"
                )
                if ready:
                    break
            _hold(client, frames, frames_dir, 2.0, args.fps)

        _hold(client, frames, frames_dir, 2.5, args.fps)
    finally:
        if client is not None:
            client.close()
        chrome.terminate()
        chrome.wait(timeout=10)

    if not frames:
        raise SystemExit("캡처된 프레임이 없다")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
         "-i", str(frames_dir / "frame_%05d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(args.out)],
        check=True,
    )
    print(f"{args.out}  ({len(frames)}프레임, {len(frames)/args.fps:.0f}초)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
