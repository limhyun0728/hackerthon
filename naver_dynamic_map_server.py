from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import urllib.parse
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

try:
    from local_env import load_local_env
except ModuleNotFoundError:
    from hackerthon.local_env import load_local_env


load_local_env()


DEFAULT_ORIGIN_LAT = 37.5665
DEFAULT_ORIGIN_LON = 126.9780
DEFAULT_METERS_PER_UNIT = 10.0
DEFAULT_ZOOM = 17
DEFAULT_UNIT_RADIUS_UNITS = 0.035

HARDCODED_NAVER_MAP_CLIENT_ID = "4mm5312jlo"


HTML = r"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Naver Dynamic Tactical Map</title>
  <style>
    :root {
      color-scheme: light;
      --panel: rgba(255, 255, 255, 0.94);
      --line: rgba(20, 26, 35, 0.14);
      --text: #111827;
      --muted: #4b5563;
      --blue: #1d4ed8;
      --red: #c62828;
      --green: #15803d;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    html, body {
      width: 100%;
      height: 100%;
      margin: 0;
      overflow: hidden;
      background: #f3f4f6;
      color: var(--text);
    }
    #map {
      position: fixed;
      inset: 0;
      width: 100vw;
      height: 100vh;
    }
    .hud {
      position: fixed;
      top: 12px;
      left: 12px;
      width: min(520px, calc(100vw - 24px));
      display: grid;
      gap: 8px;
      z-index: 10;
    }
    .bar, .status {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 12px 28px rgba(15, 23, 42, 0.12);
      backdrop-filter: blur(8px);
    }
    .bar {
      display: grid;
      grid-template-columns: 42px 1fr 84px 72px;
      gap: 8px;
      align-items: center;
      padding: 8px;
    }
    button, input {
      font: inherit;
    }
    button {
      height: 34px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #ffffff;
      color: var(--text);
      cursor: pointer;
      font-weight: 700;
    }
    button:hover {
      border-color: rgba(29, 78, 216, 0.4);
    }
    input[type="range"] {
      width: 100%;
      min-width: 0;
      accent-color: var(--blue);
    }
    .readout, .speed {
      height: 34px;
      display: flex;
      align-items: center;
      justify-content: center;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #ffffff;
      color: var(--muted);
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .speed {
      gap: 4px;
    }
    .speed input {
      width: 38px;
      border: 0;
      outline: none;
      text-align: right;
      color: var(--text);
      background: transparent;
    }
    .status {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      padding: 9px 10px;
      font-size: 12px;
      line-height: 1.35;
    }
    .status strong {
      display: block;
      font-size: 13px;
      margin-bottom: 2px;
    }
    .status span {
      color: var(--muted);
    }
    .pill {
      align-self: start;
      border-radius: 999px;
      padding: 4px 8px;
      background: #eef2ff;
      color: #3730a3;
      font-weight: 700;
      font-variant-numeric: tabular-nums;
    }
    .unit-marker {
      position: relative;
      width: 36px;
      height: 18px;
      border: 1px solid rgba(17, 24, 39, 0.32);
      border-radius: 4px;
      box-sizing: border-box;
      display: grid;
      place-items: center;
      background: rgba(255, 255, 255, 0.94);
      color: #111827;
      font-size: 10px;
      font-weight: 800;
      line-height: 1;
      text-shadow: none;
      box-shadow: 0 4px 10px rgba(15, 23, 42, 0.14);
    }
    .unit-marker.blue {
      color: var(--blue);
      border-color: rgba(29, 78, 216, 0.55);
    }
    .unit-marker.red {
      color: var(--red);
      border-color: rgba(198, 40, 40, 0.55);
    }
    .unit-marker.dead {
      background: #6b7280;
      color: #f9fafb;
    }
    .unit-marker .hp {
      position: absolute;
      left: 50%;
      top: 16px;
      transform: translateX(-50%);
      min-width: 26px;
      border-radius: 999px;
      padding: 1px 4px;
      background: rgba(255, 255, 255, 0.96);
      color: #111827;
      border: 1px solid rgba(17, 24, 39, 0.16);
      font-size: 9px;
      font-weight: 800;
      text-shadow: none;
      text-align: center;
    }
    .message {
      position: fixed;
      inset: auto 12px 12px 12px;
      z-index: 20;
      padding: 10px 12px;
      border-radius: 8px;
      background: #111827;
      color: #ffffff;
      font-size: 13px;
      box-shadow: 0 16px 34px rgba(15, 23, 42, 0.24);
      display: none;
    }
    .objective-marker {
      /* 별 하나로 둔다. 알약에 텍스트를 넣으면 시각적 무게가 커서 유닛을 가린다. */
      color: #a855f7;
      font-size: 30px;
      line-height: 30px;
      text-shadow: 0 0 4px #000, 0 0 8px #000;
      pointer-events: none;
    }
    @media (max-width: 560px) {
      .bar {
        grid-template-columns: 42px 1fr 70px;
      }
      .speed {
        display: none;
      }
      .status {
        grid-template-columns: 1fr;
      }
      .pill {
        justify-self: start;
      }
    }
  </style>
</head>
<body>
  <div id="map"></div>
  <div class="hud">
    <div class="bar">
      <button id="play" type="button">▶</button>
      <input id="time" type="range" min="0" max="0" step="0.1" value="0">
      <div id="readout" class="readout">t=0.0</div>
      <label class="speed"><input id="speed" type="number" min="0.1" max="8" step="0.1" value="1">x</label>
    </div>
    <div class="status">
      <div>
        <strong id="runName">loading</strong>
        <span id="planner">-</span>
      </div>
      <div id="counts" class="pill">B0 R0</div>
    </div>
  </div>
  <div id="message" class="message"></div>
  <script>
    const app = {
      config: null,
      data: null,
      map: null,
      playing: false,
      lastFrameMs: null,
      currentTime: 0,
      markers: new Map(),
      fovs: new Map(),
      showFov: false,
      showBelief: false,
      observedNow: new Set(),
      lastSeen: new Map(),
      rings: new Map(),
      unitBodies: new Map(),
      trails: new Map(),
      fireLines: new Map(),
      obstaclePolygons: [],
      objectiveMarker: null
    };

    const $ = (id) => document.getElementById(id);
    const urlParams = new URLSearchParams(window.location.search);

    function showMessage(text) {
      const node = $("message");
      node.textContent = text;
      node.style.display = text ? "block" : "none";
    }

    async function boot() {
      const config = await fetch("/api/config").then((res) => res.json());
      app.showFov = Boolean(config.showFov);
      app.showBelief = Boolean(config.showBelief);
      app.config = config;
      if (!config.clientId) {
        showMessage("NAVER_MAP_CLIENT_ID 또는 --client-id 값을 넣어야 네이버 Dynamic Map이 로드됩니다.");
        return;
      }
      window.__initNaverDynamicMap = initNaverDynamicMap;
      const script = document.createElement("script");
      script.src = "https://oapi.map.naver.com/openapi/v3/maps.js?ncpKeyId="
        + encodeURIComponent(config.clientId)
        + "&callback=__initNaverDynamicMap";
      script.onerror = () => showMessage("네이버 지도 SDK를 불러오지 못했습니다. 키와 Web 서비스 URL 등록을 확인하세요.");
      document.head.appendChild(script);
    }

    async function initNaverDynamicMap() {
      app.data = await fetch("/api/state").then((res) => res.json());
      setupMap();
      setupControls();
      applyPresentationParams();
      const requestedTime = Number(urlParams.get("t"));
      renderAt(Number.isFinite(requestedTime) ? requestedTime : (app.data.timeMin || 0));
      window.__renderNaverTacticalAt = renderAt;
      window.__naverTacticalReady = true;
      setTimeout(checkMapAuth, 1500);
      requestAnimationFrame(tick);
    }

    function setupMap() {
      const center = new naver.maps.LatLng(app.config.originLat, app.config.originLon);
      const cleanCapture = urlParams.get("capture") === "1" || urlParams.get("hud") === "0";
      app.map = new naver.maps.Map("map", {
        center,
        zoom: app.config.zoom,
        mapTypeControl: !cleanCapture,
        zoomControl: !cleanCapture,
        scaleControl: true
      });
      drawStaticLayers();
      if (app.data.bounds) {
        const sw = xyToLatLng(app.data.bounds.xMin, app.data.bounds.yMin);
        const ne = xyToLatLng(app.data.bounds.xMax, app.data.bounds.yMax);
        app.map.fitBounds(new naver.maps.LatLngBounds(sw, ne));
      }
      $("runName").textContent = app.data.runName || "run";
    }

    function applyPresentationParams() {
      if (urlParams.get("hud") === "0") {
        const hud = document.querySelector(".hud");
        if (hud) hud.style.display = "none";
      }
    }

    function checkMapAuth() {
      const bg = getComputedStyle($("map")).backgroundImage || "";
      if (bg.includes("auth_fail")) {
        showMessage("네이버 지도 인증이 실패했습니다. Maps Application의 Web 서비스 URL에 현재 접속 주소를 등록하세요.");
      }
    }

    function setupControls() {
      const slider = $("time");
      const minTime = app.data.timeMin || 0;
      const maxTime = app.data.timeMax || 0;
      slider.min = String(minTime);
      slider.max = String(maxTime);
      slider.value = String(minTime);
      slider.step = "0.1";
      slider.addEventListener("input", () => {
        app.playing = false;
        $("play").textContent = "▶";
        renderAt(Number(slider.value));
      });
      $("play").addEventListener("click", () => {
        app.playing = !app.playing;
        app.lastFrameMs = null;
        $("play").textContent = app.playing ? "Ⅱ" : "▶";
      });
    }

    function xyToLatLng(x, y) {
      const meters = app.config.metersPerUnit;
      const lat = app.config.originLat + (Number(y) * meters / 111320.0);
      const lng = app.config.originLon + (Number(x) * meters / (111320.0 * Math.cos(app.config.originLat * Math.PI / 180.0)));
      return new naver.maps.LatLng(lat, lng);
    }

    function drawStaticLayers() {
      const buildingPolygons = app.data.buildingPolygons || app.data.building_polygons || [];
      if (buildingPolygons.length > 0) {
        for (const building of buildingPolygons) drawBuildingPolygon(building);
      } else {
        for (const rect of app.data.obstacles || []) {
          drawObstacleRect(rect);
        }
      }
      if (app.data.objective) {
        app.objectiveMarker = new naver.maps.Marker({
          map: app.map,
          position: xyToLatLng(app.data.objective[0], app.data.objective[1]),
          icon: {
            content: '<div class="objective-marker">&#9733;</div>',
            anchor: new naver.maps.Point(17, 17)
          }
        });
      }
    }

    function drawObstacleRect(rect) {
        const [xMin, yMin, xMax, yMax] = rect.map(Number);
        const polygon = new naver.maps.Polygon({
          map: app.map,
          paths: [
            xyToLatLng(xMin, yMin),
            xyToLatLng(xMax, yMin),
            xyToLatLng(xMax, yMax),
            xyToLatLng(xMin, yMax)
          ],
          fillColor: "#475569",
          fillOpacity: 0.78,
          strokeColor: "#1f2937",
          strokeOpacity: 0.72,
          strokeWeight: 1,
          zIndex: 60
        });
        app.obstaclePolygons.push(polygon);
    }

    function drawBuildingPolygon(building) {
      const points = building.points || building;
      if (!points || points.length < 3) return;
      const polygon = new naver.maps.Polygon({
        map: app.map,
        paths: points.map((point) => xyToLatLng(point[0], point[1])),
        fillColor: "#334155",
        fillOpacity: 0.76,
        strokeColor: "#0f172a",
        strokeOpacity: 0.78,
        strokeWeight: 1,
        zIndex: 60
      });
      app.obstaclePolygons.push(polygon);
    }

    // 지휘관 플랫폼과 같은 표현을 쓴다. 점은 실축척(0.7m), 라벨과 HP바는 옆에
    // 붙여 화면 고정 크기로 읽는다. zoom19까지는 실축척이 하한보다 작아 하한이
    // 곧 표시 크기가 된다.
    const MIN_UNIT_PX = 7;
    const MIN_BELIEVED_PX = 11;
    const RED_MAX_SPEED_MPS = 10.0;
    const PERCEPTION_RANGE_M = 100.0;

    function metersPerPixel() {
      const lat = app.config.originLat || 37.5;
      return 156543.03392 * Math.cos(lat * Math.PI / 180) / Math.pow(2, app.map.getZoom());
    }

    function markerHtml(unitId, state, believed, staleSec) {
      const blue = Number(unitId) < 200;
      const alive = Number(state.hp) > 0;
      const bg = !alive ? "#484f58" : (blue ? "#1f6feb" : (believed ? "#fb8500" : "#da3633"));
      const ratio = Math.max(0, Math.min(1, Number(state.hp) / 100));
      const base = Math.max(MIN_UNIT_PX, (app.config.unitRadiusMeters * 2) / metersPerPixel());
      const dot = believed ? Math.max(base, MIN_BELIEVED_PX) : base;
      // 불확실 반경 = RED 최대속도 x 미관측 경과. 관측거리에서 자른다.
      const radiusM = Math.min(PERCEPTION_RANGE_M, (staleSec || 0) * RED_MAX_SPEED_MPS);
      const halo = believed ? Math.max(16, 2 * radiusM / metersPerPixel()) : 0;
      const haloHtml = halo
        ? `<div style="position:absolute;left:${-halo / 2}px;top:${-halo / 2}px;width:${halo}px;height:${halo}px;
             border-radius:50%;border:2px dashed #fb8500;background:radial-gradient(circle,#fb850040 0%,#fb850018 60%,#fb850000 100%)"></div>`
        : "";
      const shadow = believed
        ? "0 0 0 2px #ffe0b2,0 0 10px 3px #fb8500cc"
        : "0 0 0 1px #ffffffcc";
      const label = `${blue ? "B" : "R"}${Number(unitId) % 100}${believed ? ` ?${Math.round(staleSec)}s` : ""}`;
      const hpBar = alive
        ? `<div style="position:absolute;left:${dot / 2 + 5}px;top:8px;width:40px;height:6px;
             background:#000000cc;border-radius:3px;box-shadow:0 0 3px #000">
             <div style="width:${Math.round(ratio * 100)}%;height:100%;border-radius:3px;
               background:${ratio > 0.6 ? "#3fb950" : ratio > 0.3 ? "#d29922" : "#f85149"}"></div></div>`
        : "";
      return `<div style="position:relative;width:0;height:0">${haloHtml}
        <div style="position:absolute;left:${-dot / 2}px;top:${-dot / 2}px;width:${dot}px;height:${dot}px;
          border-radius:50%;background:${bg};box-shadow:${shadow}"></div>
        <div style="position:absolute;left:${dot / 2 + 5}px;top:-11px;white-space:nowrap;
          font:600 15px/17px system-ui;color:${believed ? "#fb8500" : "#fff"};
          text-shadow:0 0 4px #000,0 0 4px #000,0 0 4px #000">${label}</div>
        ${hpBar}</div>`;
    }

    // 사거리 안이고 시선이 트인 RED만 "관측"으로 본다. LosWorldAtomic의 피해
    // 판정과 같은 규칙이라, 쏠 수 있으면 빨강이고 아니면 주황이 된다.
    const MAX_FIRE_RANGE_UNITS = 10.0;

    function segmentBlocked(ax, ay, bx, by) {
      for (const rect of app.data.obstacles || []) {
        const [x0, y0, x1, y1] = rect;
        const dx = bx - ax;
        const dy = by - ay;
        let tMin = 0;
        let tMax = 1;
        let hit = true;
        for (const [pos, delta, low, high] of [[ax, dx, x0, x1], [ay, dy, y0, y1]]) {
          if (Math.abs(delta) < 1e-12) {
            if (pos < low || pos > high) { hit = false; break; }
            continue;
          }
          let t0 = (low - pos) / delta;
          let t1 = (high - pos) / delta;
          if (t0 > t1) { const tmp = t0; t0 = t1; t1 = tmp; }
          tMin = Math.max(tMin, t0);
          tMax = Math.min(tMax, t1);
          if (tMin > tMax) { hit = false; break; }
        }
        if (hit) return true;
      }
      return false;
    }

    function observedRedIds(states) {
      const blue = [];
      const seen = new Set();
      for (const [id, st] of states.entries()) {
        if (Number(id) < 200 && Number(st.hp) > 0) blue.push(st);
      }
      for (const [id, st] of states.entries()) {
        if (Number(id) < 200 || Number(st.hp) <= 0) continue;
        for (const b of blue) {
          const d = Math.hypot(st.x - b.x, st.y - b.y);
          if (d <= MAX_FIRE_RANGE_UNITS && !segmentBlocked(b.x, b.y, st.x, st.y)) {
            seen.add(String(id));
            break;
          }
        }
      }
      return seen;
    }

    function stateAt(series, timeSec) {
      if (!series || series.length === 0) return null;
      if (timeSec <= series[0].time) return series[0];
      const last = series[series.length - 1];
      if (timeSec >= last.time) return last;
      let lo = 0;
      let hi = series.length - 1;
      while (hi - lo > 1) {
        const mid = Math.floor((lo + hi) / 2);
        if (series[mid].time <= timeSec) lo = mid;
        else hi = mid;
      }
      const a = series[lo];
      const b = series[hi];
      const span = b.time - a.time || 1;
      const ratio = (timeSec - a.time) / span;
      const deltaHeading = ((b.heading - a.heading + 540) % 360) - 180;
      return {
        time: timeSec,
        x: a.x + ratio * (b.x - a.x),
        y: a.y + ratio * (b.y - a.y),
        heading: a.heading + ratio * deltaHeading,
        hp: a.hp,
        ammo: a.ammo,
        mode: a.mode,
        targetId: a.targetId
      };
    }

    function sectorPath(state, radiusUnit, halfAngleDeg) {
      const path = [xyToLatLng(state.x, state.y)];
      for (let i = 0; i <= 16; i += 1) {
        const deg = state.heading - halfAngleDeg + (2 * halfAngleDeg * i / 16);
        const rad = deg * Math.PI / 180;
        path.push(xyToLatLng(
          state.x + Math.cos(rad) * radiusUnit,
          state.y + Math.sin(rad) * radiusUnit
        ));
      }
      return path;
    }

    function trailPath(series, timeSec, current) {
      const points = [];
      for (const sample of series) {
        if (sample.time <= timeSec) points.push(xyToLatLng(sample.x, sample.y));
        else break;
      }
      if (current) points.push(xyToLatLng(current.x, current.y));
      return points;
    }

    function plannerAt(timeSec) {
      let current = "";
      for (const row of app.data.planner || []) {
        if (row.time <= timeSec) current = row.text;
        else break;
      }
      return current || "-";
    }

    function commandAt(unitId, timeSec) {
      const tick = Math.floor(timeSec);
      const rows = app.data.commandsByTick[String(tick)] || [];
      return rows.find((row) => Number(row.unitId) === Number(unitId)) || null;
    }

    function parseMoveTarget(detail) {
      if (!detail) return null;
      const match = String(detail).match(/\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)?/);
      return match ? { x: Number(match[1]), y: Number(match[2]) } : null;
    }

    function ensureUnitOverlays(unitId, state) {
      const isBlue = Number(unitId) < 200;
      const color = isBlue ? "#1d4ed8" : "#c62828";
      if (!app.markers.has(unitId)) {
        app.unitBodies.set(unitId, new naver.maps.Circle({
          map: app.map,
          center: xyToLatLng(state.x, state.y),
          radius: app.config.unitRadiusMeters,
          fillColor: color,
          fillOpacity: 0.92,
          strokeColor: "#111827",
          strokeOpacity: 0.95,
          strokeWeight: 1,
          zIndex: 90
        }));
        app.markers.set(unitId, new naver.maps.Marker({
          map: app.map,
          position: xyToLatLng(state.x, state.y),
          icon: { content: markerHtml(unitId, state), anchor: new naver.maps.Point(18, 30) },
          zIndex: 100
        }));
        app.fovs.set(unitId, new naver.maps.Polygon({
          map: app.map,
          paths: [],
          fillColor: color,
          fillOpacity: 0.08,
          strokeColor: color,
          strokeOpacity: 0.16,
          strokeWeight: 1,
          zIndex: 10
        }));
        app.trails.set(unitId, new naver.maps.Polyline({
          map: app.map,
          path: [],
          strokeColor: color,
          strokeOpacity: 0.56,
          strokeWeight: 3,
          zIndex: 25
        }));
        app.fireLines.set(unitId, new naver.maps.Polyline({
          map: app.map,
          path: [],
          strokeColor: "#f59e0b",
          strokeOpacity: 0.9,
          strokeWeight: 2,
          strokeStyle: "shortdash",
          zIndex: 30
        }));
        if (!isBlue) {
          app.rings.set(unitId, new naver.maps.Circle({
            map: app.map,
            center: xyToLatLng(state.x, state.y),
            radius: 7 * app.config.metersPerUnit,
            fillOpacity: 0,
            strokeColor: "#c62828",
            strokeOpacity: 0.24,
            strokeWeight: 1,
            zIndex: 15
          }));
        }
      }
    }

    function renderAt(timeSec) {
      if (!app.data || !app.map) return;
      app.currentTime = Math.min(Math.max(timeSec, app.data.timeMin || 0), app.data.timeMax || 0);
      $("time").value = String(app.currentTime);
      $("readout").textContent = `t=${app.currentTime.toFixed(1)}`;
      $("planner").textContent = plannerAt(app.currentTime);
      let aliveBlue = 0;
      let aliveRed = 0;
      const states = new Map();
      for (const [unitId, series] of Object.entries(app.data.units)) {
        const state = stateAt(series, app.currentTime);
        if (!state) continue;
        states.set(unitId, state);
      }
      app.observedNow = observedRedIds(states);
      for (const [unitId, state] of states.entries()) {
        ensureUnitOverlays(unitId, state);
        const dead = Number(state.hp) <= 0;
        if (!dead && Number(unitId) < 200) aliveBlue += 1;
        if (!dead && Number(unitId) >= 200) aliveRed += 1;
        const position = xyToLatLng(state.x, state.y);
        const body = app.unitBodies.get(unitId);
        body.setCenter(position);
        body.setRadius(app.config.unitRadiusMeters);
        body.setVisible(!dead);
        app.markers.get(unitId).setPosition(position);
        const isRed = Number(unitId) >= 200;
        const believed = app.showBelief && isRed && !dead && !app.observedNow.has(String(unitId));
        if (isRed && !believed) app.lastSeen.set(String(unitId), app.currentTime);
        const stale = believed
          ? Math.max(0, app.currentTime - (app.lastSeen.get(String(unitId)) ?? 0))
          : 0;
        app.markers.get(unitId).setIcon({
          content: markerHtml(unitId, state, believed, stale),
          anchor: new naver.maps.Point(0, 0)
        });
        // 시야각 부채꼴은 그리지 않는다. LosWorldAtomic이 피해 판정에 시야각을
        // 쓰지 않으므로 화면에만 있는 정보이고, 유닛이 겹치면 지형을 가린다.
        // NAVER_MAP_SHOW_FOV=1 로 다시 켤 수 있다.
        app.fovs.get(unitId).setPaths((dead || !app.showFov) ? [] : sectorPath(state, 10, 60));
        app.trails.get(unitId).setPath(trailPath(app.data.units[unitId], app.currentTime, state));
        const ring = app.rings.get(unitId);
        if (ring) {
          ring.setCenter(position);
          ring.setVisible(!dead);
        }
        const fire = app.fireLines.get(unitId);
        if (!dead && state.mode === "ENGAGE" && state.targetId != null && states.has(String(state.targetId))) {
          const target = states.get(String(state.targetId));
          fire.setPath([position, xyToLatLng(target.x, target.y)]);
        } else {
          fire.setPath([]);
        }
      }
      $("counts").textContent = `B${aliveBlue} R${aliveRed}`;
    }

    function tick(nowMs) {
      if (app.playing) {
        if (app.lastFrameMs == null) app.lastFrameMs = nowMs;
        const elapsed = (nowMs - app.lastFrameMs) / 1000;
        app.lastFrameMs = nowMs;
        const speed = Math.max(0.1, Number($("speed").value) || 1);
        let next = app.currentTime + elapsed * speed;
        if (next >= (app.data.timeMax || 0)) {
          next = app.data.timeMin || 0;
        }
        renderAt(next);
      } else {
        app.lastFrameMs = null;
      }
      requestAnimationFrame(tick);
    }

    boot().catch((error) => {
      console.error(error);
      showMessage(error.message || String(error));
    });
  </script>
</body>
</html>
"""


def _first_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return ""


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def _center_from_env() -> tuple[float, float]:
    center = os.getenv("NAVER_MAP_CENTER")
    if center:
        lon_text, lat_text = center.split(",", maxsplit=1)
        return float(lat_text.strip()), float(lon_text.strip())
    return (
        _env_float("NAVER_MAP_CENTER_LAT", DEFAULT_ORIGIN_LAT),
        _env_float("NAVER_MAP_CENTER_LON", DEFAULT_ORIGIN_LON),
    )


def _json_response(handler: BaseHTTPRequestHandler, payload: Any, status: int = 200) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(encoded)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(encoded)


def _text_response(handler: BaseHTTPRequestHandler, text: str, content_type: str = "text/html; charset=utf-8") -> None:
    encoded = text.encode("utf-8")
    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(encoded)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(encoded)


def _candidate_log_path(run_dir: Path, explicit: Optional[Path]) -> Path:
    if explicit is not None:
        return explicit
    candidates = (
        run_dir / "soldier_log.csv",
        run_dir / "soldier_commander_log.csv",
    )
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _parse_target_id(value: Any) -> Optional[int]:
    text = "" if value is None else str(value).strip()
    if text in ("", "None", "none", "null"):
        return None
    return int(float(text))


def _read_unit_rows(path: Path) -> dict[str, list[dict[str, Any]]]:
    units: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            unit_id = str(int(float(row["id"])))
            units[unit_id].append(
                {
                    "time": float(row["time"]),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "heading": float(row.get("heading") or 0.0),
                    "hp": float(row.get("hp") or 0.0),
                    "ammo": float(row.get("ammo") or 0.0),
                    "mode": str(row.get("mode") or ""),
                    "targetId": _parse_target_id(row.get("target_id")),
                }
            )
    return {
        unit_id: sorted(samples, key=lambda item: item["time"])
        for unit_id, samples in units.items()
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _planner_text(row: dict[str, str]) -> str:
    if "tactic" in row:
        return " ".join(part for part in (row.get("tactic"), row.get("decision")) if part)
    if "selector" in row:
        score = row.get("best_score")
        suffix = f" best={float(score):.1f}" if score not in (None, "") else ""
        return f"{row.get('selector', '')}{suffix}".strip()
    return " ".join(value for key, value in row.items() if key != "time" and value)


def _read_planner_rows(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "planner_log.csv"
    if not path.exists():
        return []
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append({"time": float(row["time"]), "text": _planner_text(row)})
    return sorted(rows, key=lambda item: item["time"])


def _read_command_rows(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    path = run_dir / "commands_log.csv"
    if not path.exists():
        return [], {}
    rows = []
    by_tick: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            item = {
                "time": float(row["time"]),
                "unitId": int(float(row["unit_id"])),
                "role": row.get("role", ""),
                "action": row.get("action", ""),
                "detail": row.get("detail", ""),
                "reason": row.get("reason", ""),
            }
            rows.append(item)
            by_tick[str(int(float(row["time"])))].append(item)
    return sorted(rows, key=lambda item: (item["time"], item["unitId"])), dict(by_tick)


def _bounds(units: dict[str, list[dict[str, Any]]], config: dict[str, Any]) -> Optional[dict[str, float]]:
    real_map = config.get("real_map", {}) if isinstance(config.get("real_map"), dict) else {}
    display_bounds = config.get("display_bounds") or real_map.get("display_bounds")
    if display_bounds and len(display_bounds) == 4:
        return {
            "xMin": float(display_bounds[0]),
            "yMin": float(display_bounds[1]),
            "xMax": float(display_bounds[2]),
            "yMax": float(display_bounds[3]),
        }

    xs = [sample["x"] for samples in units.values() for sample in samples]
    ys = [sample["y"] for samples in units.values() for sample in samples]
    for building in config.get("building_polygons", []) or []:
        points = building.get("points", []) if isinstance(building, dict) else building
        for point in points:
            if len(point) >= 2:
                xs.append(float(point[0]))
                ys.append(float(point[1]))
    for rect in config.get("obstacles", []) or []:
        if len(rect) == 4:
            xs.extend([float(rect[0]), float(rect[2])])
            ys.extend([float(rect[1]), float(rect[3])])
    objective = config.get("objective")
    if objective and len(objective) >= 2:
        xs.append(float(objective[0]))
        ys.append(float(objective[1]))
    if not xs or not ys:
        return None
    pad = 2.0
    return {
        "xMin": min(xs) - pad,
        "xMax": max(xs) + pad,
        "yMin": min(ys) - pad,
        "yMax": max(ys) + pad,
    }


def _state_payload(run_dir: Path, log_path: Path) -> dict[str, Any]:
    units = _read_unit_rows(log_path)
    config = _read_json(run_dir / "config.json")
    times = [sample["time"] for samples in units.values() for sample in samples]
    commands, commands_by_tick = _read_command_rows(run_dir)
    return {
        "runName": run_dir.name,
        "runDir": str(run_dir),
        "sourceLog": str(log_path),
        "units": units,
        "timeMin": min(times) if times else 0.0,
        "timeMax": max(times) if times else 0.0,
        "obstacles": config.get("obstacles", []),
        "buildingPolygons": config.get("building_polygons", []),
        "realMap": config.get("real_map", {}),
        "objective": config.get("objective"),
        "planner": _read_planner_rows(run_dir),
        "commands": commands,
        "commandsByTick": commands_by_tick,
        "bounds": _bounds(units, config),
    }


def _make_handler(args: argparse.Namespace):
    run_dir = Path(args.run_dir).resolve()
    log_path = _candidate_log_path(run_dir, Path(args.log_file).resolve() if args.log_file else None)
    run_config = _read_json(run_dir / "config.json")
    real_map = run_config.get("real_map", {}) if isinstance(run_config.get("real_map"), dict) else {}
    origin_lat = args.origin_lat
    origin_lon = args.origin_lon
    meters_per_unit = args.meters_per_unit
    if origin_lat == DEFAULT_ORIGIN_LAT and real_map.get("origin_lat") is not None:
        origin_lat = float(real_map["origin_lat"])
    if origin_lon == DEFAULT_ORIGIN_LON and real_map.get("origin_lon") is not None:
        origin_lon = float(real_map["origin_lon"])
    if meters_per_unit == DEFAULT_METERS_PER_UNIT and real_map.get("meters_per_unit") is not None:
        meters_per_unit = float(real_map["meters_per_unit"])
    unit_radius_units = args.unit_radius_units
    if unit_radius_units == DEFAULT_UNIT_RADIUS_UNITS:
        if real_map.get("unit_radius_units") is not None:
            unit_radius_units = float(real_map["unit_radius_units"])
        elif real_map.get("unit_radius_meters") is not None and meters_per_unit > 0:
            unit_radius_units = float(real_map["unit_radius_meters"]) / meters_per_unit
    unit_radius_meters = unit_radius_units * meters_per_unit
    client_id = args.client_id

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                _text_response(self, HTML)
                return
            if parsed.path == "/api/config":
                _json_response(
                    self,
                    {
                        "clientId": client_id,
                        "originLat": origin_lat,
                        "originLon": origin_lon,
                        "metersPerUnit": meters_per_unit,
                        "unitRadiusUnits": unit_radius_units,
                        "unitRadiusMeters": unit_radius_meters,
                        "zoom": args.zoom,
                        # 시야각 부채꼴 표시 여부. LosWorldAtomic이 피해 판정에
                        # 시야각을 쓰지 않으므로 기본은 끈다.
                        "showFov": os.getenv("NAVER_MAP_SHOW_FOV", "0") not in ("0", "", "false"),
                        # 미관측 RED를 주황 + 불확실 원으로 그릴지. 플랫폼은 안개
                        # 상황에서 지휘결심을 돕는 것이라 켜지만, 비교용 영상은
                        # 실제 DEVS 결과를 그대로 보여야 하므로 기본은 끈다.
                        "showBelief": os.getenv("NAVER_MAP_SHOW_BELIEF", "0") not in ("0", "", "false"),
                        "runDir": str(run_dir),
                    },
                )
                return
            if parsed.path == "/api/state":
                _json_response(self, _state_payload(run_dir, log_path))
                return
            _json_response(self, {"error": "not found"}, status=404)

        def log_message(self, fmt: str, *values: Any) -> None:
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % values))

    return Handler


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    origin_lat, origin_lon = _center_from_env()
    parser = argparse.ArgumentParser(description="Replay DEVS/world-model logs on NAVER Dynamic Map.")
    parser.add_argument("--run-dir", default=".", help="Directory containing soldier_log.csv and optional config/planner/commands logs.")
    parser.add_argument("--log-file", default="", help="Override soldier_log.csv path.")
    parser.add_argument(
        "--client-id",
        default=_first_env("NAVER_MAP_CLIENT_ID", "NAVER_MAP_KEY_ID", "NCP_MAP_CLIENT_ID")
        or HARDCODED_NAVER_MAP_CLIENT_ID,
    )
    parser.add_argument("--origin-lat", type=float, default=origin_lat)
    parser.add_argument("--origin-lon", type=float, default=origin_lon)
    parser.add_argument("--meters-per-unit", type=float, default=_env_float("NAVER_MAP_METERS_PER_UNIT", DEFAULT_METERS_PER_UNIT))
    parser.add_argument(
        "--unit-radius-units",
        type=float,
        default=_env_float("NAVER_MAP_UNIT_RADIUS_UNITS", DEFAULT_UNIT_RADIUS_UNITS),
        help="Displayed unit body radius in simulation units.",
    )
    parser.add_argument("--zoom", type=int, default=_env_int("NAVER_MAP_ZOOM", DEFAULT_ZOOM))
    parser.add_argument("--host", default=os.getenv("NAVER_MAP_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=_env_int("NAVER_MAP_PORT", 8765))
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    handler = _make_handler(args)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Naver Dynamic Tactical Map: {url}")
    print(f"run_dir={Path(args.run_dir).resolve()}")
    if args.log_file:
        print(f"log_file={Path(args.log_file).resolve()}")
    if not args.client_id:
        print("warning: NAVER_MAP_CLIENT_ID/--client-id is empty; the browser map will not load.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
