"""지휘관 플랫폼 UI 페이지.

commander_platform 서버가 그대로 서빙하는 단일 HTML이다. 배경은 NAVER Dynamic
Map을 쓰고 그 위에 건물 장애물, 부대, 목표를 오버레이한다. 지휘관이 지도를 눌러
아군/적군/목표를 직접 배치한 뒤 작전을 시작하면, 결심 시점마다 태세 축 아카이브를
보여주고 선택을 트리로 남긴다.
"""

PAGE_HTML = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>지휘관 시뮬레이션 플랫폼</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; background:#0f1419; color:#e6edf3; font:14px/1.5 system-ui, sans-serif; }
  header { padding:10px 16px; background:#161b22; border-bottom:1px solid #30363d; display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
  header h1 { font-size:15px; margin:0 10px 0 0; font-weight:600; }
  label { font-size:12px; color:#8b949e; }
  select, input, button { background:#0d1117; color:#e6edf3; border:1px solid #30363d; border-radius:6px; padding:4px 8px; font:inherit; }
  button { cursor:pointer; }
  button:hover { border-color:#58a6ff; }
  button.primary { background:#1f6feb; border-color:#1f6feb; }
  button.on { background:#1f6feb; border-color:#58a6ff; }
  main { display:grid; grid-template-columns: 1fr 400px; gap:12px; padding:12px; align-items:start; }
  #mapwrap { position:relative; height:74vh; border:1px solid #30363d; border-radius:8px; overflow:hidden; }
  #map { width:100%; height:100%; }
  .panel { background:#161b22; border:1px solid #30363d; border-radius:8px; padding:12px; }
  .panel h2 { font-size:13px; margin:0 0 8px; color:#8b949e; font-weight:600; }
  table.grid { border-collapse:collapse; width:100%; font-size:11px; }
  table.grid th { color:#6e7681; font-weight:500; padding:3px; font-size:10px; }
  table.grid td { border:1px solid #21262d; padding:0; height:42px; text-align:center; }
  .cell { width:100%; height:100%; display:flex; flex-direction:column; justify-content:center; cursor:pointer; }
  .cell.empty { background:#0d1117; cursor:default; color:#30363d; }
  .cell.on { background:#1f6feb33; }
  .cell.sel { outline:2px solid #58a6ff; outline-offset:-2px; }
  .cell b { font-size:12px; } .cell span { font-size:9px; color:#8b949e; }
  .tree { max-height:180px; overflow:auto; font-size:12px; }
  .tree div { padding:3px 6px; border-radius:4px; cursor:pointer; }
  .tree div:hover { background:#21262d; } .tree div.cur { background:#1f6feb44; }
  .status { font-size:12px; color:#8b949e; }
  .err { color:#f85149; font-size:12px; white-space:pre-wrap; }
  .toolbar { display:flex; gap:6px; align-items:center; flex-wrap:wrap; margin-bottom:8px; }
  .hint { position:absolute; left:10px; top:10px; z-index:5; background:#0d1117dd; border:1px solid #30363d;
          border-radius:6px; padding:6px 10px; font-size:12px; }
</style>
</head>
<body>
<header>
  <h1>지휘관 시뮬레이션 플랫폼</h1>
  <label>맵 <select id="map-select"></select></label>
  <label>임무 <select id="mission">
    <option value="destroy_all">적 격멸</option>
    <option value="destroy_and_reach">격멸 후 목표 확보</option>
    <option value="reach_objective">목표 침투</option>
    <option value="hold_objective">거점 방어</option>
  </select></label>
  <button class="primary" id="start">작전 개시</button>
  <button id="reset">배치 초기화</button>
  <span style="flex:1"></span>
  <span class="status" id="status">지도를 눌러 부대를 배치하세요</span>
</header>
<main>
  <div>
    <div class="toolbar">
      <span class="status">배치 모드:</span>
      <button id="m-blue" class="on">아군</button>
      <button id="m-red">적군</button>
      <button id="m-obj">목표</button>
      <span style="width:12px"></span>
      <label>아군 <input id="rb" type="number" value="5" min="1" max="10" style="width:46px"></label>
      <label>적 <input id="rr" type="number" value="7" min="1" max="10" style="width:46px"></label>
      <button id="rand">무작위 배치</button>
      <span class="status" id="counts"></span>
    </div>
    <div id="mapwrap"><div id="map"></div><div class="hint" id="hint">지도 클릭 = 배치 · 마커 클릭 = 삭제</div></div>
  </div>
  <div style="display:flex; flex-direction:column; gap:12px;">
    <div class="panel" id="rec-panel" style="display:none">
      <h2>추천 시나리오</h2>
      <div class="status" id="rec-info">계산 중…</div>
      <div style="display:flex; gap:6px; align-items:center; margin-top:8px">
        <button id="rec-play">▶ 재생</button>
        <input id="rec-slider" type="range" min="0" max="0" value="0" style="flex:1">
        <span class="status" id="rec-time">t=0s</span>
      </div>
      <div class="status" id="rec-picks" style="margin-top:6px; max-height:90px; overflow:auto"></div>
    </div>
    <div class="panel">
      <h2>전개안 — 교전태세 × 부대대형</h2>
      <div id="grid"><div class="status">작전 개시 후 표시됩니다</div></div>
      <div id="detail" class="status" style="margin-top:8px"></div>
      <button class="primary" id="commit" style="margin-top:8px; width:100%; display:none">선택한 안으로 6초 진행</button>
    </div>
    <div class="panel"><h2>결심 이력</h2><div class="tree" id="tree"></div></div>
    <div class="panel"><div class="err" id="err"></div></div>
  </div>
</main>
<script>
const $ = (id) => document.getElementById(id);
let S = { session:null, view:null, cells:[], sel:null, elabels:[], slabels:[],
          maps:{}, mapName:null, mode:'blue', place:{blue:[], red:[], obj:null},
          map:null, layers:[], unitLayers:[], pathLayers:[] };

const api = async (path, body) => {
  const opt = body ? {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)} : {};
  const r = await fetch(path, opt); const j = await r.json();
  if (j.error) { $('err').textContent = j.error; throw new Error(j.error); }
  $('err').textContent = ''; return j;
};

// 월드 좌표 <-> 위경도. naver_dynamic_map_server와 같은 변환식.
function rm() { return (S.maps[S.mapName]||{}).real_map || {}; }
function xyToLatLng(x, y) {
  const c = rm(), m = c.meters_per_unit || 10.0;
  return new naver.maps.LatLng(
    c.origin_lat + (y * m / 111320.0),
    c.origin_lon + (x * m / (111320.0 * Math.cos(c.origin_lat * Math.PI/180.0))));
}
function latLngToXY(ll) {
  const c = rm(), m = c.meters_per_unit || 10.0;
  return [ (ll.lng() - c.origin_lon) * (111320.0 * Math.cos(c.origin_lat*Math.PI/180.0)) / m,
           (ll.lat() - c.origin_lat) * 111320.0 / m ];
}

function clearLayers(key) { (S[key]||[]).forEach(o=>o.setMap(null)); S[key]=[]; }

function drawTerrain() {
  clearLayers('layers');
  const cfg = S.maps[S.mapName]; if (!cfg || !S.map) return;
  const polys = cfg.building_polygons||[];
  for (const b of polys) {
    const pts=(b.points||b); if(!pts||pts.length<3) continue;
    S.layers.push(new naver.maps.Polygon({ map:S.map, paths:[pts.map(p=>xyToLatLng(p[0],p[1]))],
      fillColor:'#2b3138', fillOpacity:0.88, strokeColor:'#f0f6fc', strokeOpacity:0.9, strokeWeight:1.5, clickable:false }));
  }
}

function marker(x, y, opts) {
  return new naver.maps.Marker({ map:S.map, position:xyToLatLng(x,y),
    icon:{ content:opts.html, anchor:new naver.maps.Point(9,9) }, zIndex:opts.z||100 });
}

function drawPlacement() {
  clearLayers('unitLayers');
  S.place.blue.forEach((p,i)=>{
    const m = marker(p[0],p[1],{html:`<div style="width:18px;height:18px;border-radius:50%;background:#1f6feb;border:2px solid #fff;color:#fff;font:10px/14px sans-serif;text-align:center">${i+1}</div>`});
    naver.maps.Event.addListener(m,'click',()=>{ S.place.blue.splice(i,1); drawPlacement(); });
    S.unitLayers.push(m);
  });
  S.place.red.forEach((p,i)=>{
    const m = marker(p[0],p[1],{html:`<div style="width:18px;height:18px;border-radius:50%;background:#da3633;border:2px solid #fff;color:#fff;font:10px/14px sans-serif;text-align:center">${i+1}</div>`});
    naver.maps.Event.addListener(m,'click',()=>{ S.place.red.splice(i,1); drawPlacement(); });
    S.unitLayers.push(m);
  });
  if (S.place.obj) S.unitLayers.push(marker(S.place.obj[0],S.place.obj[1],
    {html:'<div style="width:20px;height:20px;border-radius:50%;background:#a371f7;border:2px solid #fff;color:#fff;font:9px/16px sans-serif;text-align:center">OBJ</div>', z:200}));
  $('counts').textContent = `아군 ${S.place.blue.length} · 적군 ${S.place.red.length} · 목표 ${S.place.obj?'지정':'미지정'}`;
}

function drawUnits() {
  clearLayers('unitLayers'); clearLayers('pathLayers');
  const v = S.view; if (!v) return;
  const cell = S.sel!=null ? S.cells[S.sel] : null;
  if (cell) {
    const byId = {};
    for (const frame of cell.path) for (const u of frame) (byId[u.id] ||= []).push(u);
    for (const [uid, seq] of Object.entries(byId)) {
      const start = v.units.find(u=>u.id==+uid); if(!start) continue;
      S.pathLayers.push(new naver.maps.Polyline({ map:S.map,
        path:[xyToLatLng(start.x,start.y), ...seq.map(p=>xyToLatLng(p.x,p.y))],
        strokeColor:(+uid<200)?'#58a6ff':'#f85149', strokeOpacity:0.75, strokeWeight:2, strokeStyle:'shortdash' }));
    }
  }
  S.unitLayers.push(marker(v.objective[0], v.objective[1],
    {html:'<div style="width:20px;height:20px;border-radius:50%;background:#a371f7;border:2px solid #fff;color:#fff;font:9px/16px sans-serif;text-align:center">OBJ</div>', z:200}));
  for (const u of v.units) {
    const blue = u.id<200, alive = u.hp>0;
    const bg = !alive ? '#484f58' : (blue?'#1f6feb':'#da3633');
    S.unitLayers.push(marker(u.x,u.y,{ html:
      `<div style="width:18px;height:18px;border-radius:50%;background:${bg};border:2px solid #fff;
        color:#fff;font:9px/14px sans-serif;text-align:center" title="HP ${Math.round(u.hp)}">${u.id%100}</div>` }));
  }
}

function renderGrid() {
  const el = $('grid');
  if (!S.cells.length) { el.innerHTML = '<div class="status">후보 계산 중…</div>'; return; }
  const map = {};
  S.cells.forEach((c,i)=> map[`${c.engage_bin},${c.spread_bin}`] = i);
  let html = '<table class="grid"><tr><th></th>' + S.slabels.map(s=>`<th>${s}</th>`).join('') + '</tr>';
  for (let e=S.elabels.length-1; e>=0; e--) {
    html += `<tr><th>${S.elabels[e]}</th>`;
    for (let s=0; s<S.slabels.length; s++) {
      const i = map[`${e},${s}`];
      html += (i===undefined) ? '<td><div class="cell empty">·</div></td>'
        : `<td><div class="cell on ${S.sel===i?'sel':''}" data-i="${i}"><b>${S.cells[i].blue_alive}v${S.cells[i].red_alive}</b><span>적HP ${Math.round(S.cells[i].red_hp)}</span></div></td>`;
    }
    html += '</tr>';
  }
  el.innerHTML = html + '</table>';
  el.querySelectorAll('.cell.on').forEach(d => d.onclick = () => {
    stopPlay(); S.sel = +d.dataset.i; renderGrid(); drawUnits();
    const c = S.cells[S.sel];
    $('detail').textContent = `${c.label} — 아군 ${c.blue_alive}명 HP ${Math.round(c.blue_hp)} / 적 ${c.red_alive}명 HP ${Math.round(c.red_hp)} · 교전도 ${c.engage.toFixed(2)} 대형 ${c.spread.toFixed(1)}u`;
    $('commit').disabled = false;
  });
}

function renderTree() {
  const v = S.view; if(!v) return;
  $('tree').innerHTML = v.tree.sort((a,b)=>a.time-b.time)
    .map(n=>`<div class="${n.current?'cur':''}" data-n="${n.id}">t=${n.time.toFixed(0)}s  ${n.label||'작전 개시'}</div>`).join('');
  $('tree').querySelectorAll('div[data-n]').forEach(d => d.onclick = async () => {
    S.view = await api('/api/goto', {session:S.session, node:d.dataset.n});
    S.cells=[]; S.sel=null; renderTree(); drawUnits(); renderGrid(); loadCandidates();
  });
}

let REC = { frames:[], idx:0, timer:null };

function drawFrame(units) {
  clearLayers('unitLayers'); clearLayers('pathLayers');
  const v = S.view; if (!v) return;
  S.unitLayers.push(marker(v.objective[0], v.objective[1],
    {html:'<div style="width:20px;height:20px;border-radius:50%;background:#a371f7;border:2px solid #fff;color:#fff;font:9px/16px sans-serif;text-align:center">OBJ</div>', z:200}));
  for (const u of units) {
    const blue = u.id<200, alive = u.hp>0;
    const bg = !alive ? '#484f58' : (blue?'#1f6feb':'#da3633');
    S.unitLayers.push(marker(u.x,u.y,{ html:
      `<div style="width:18px;height:18px;border-radius:50%;background:${bg};border:2px solid #fff;
        color:#fff;font:9px/14px sans-serif;text-align:center" title="HP ${Math.round(u.hp)}">${u.id%100}</div>` }));
  }
}

function showFrame(i) {
  REC.idx = Math.max(0, Math.min(i, REC.frames.length-1));
  const f = REC.frames[REC.idx]; if (!f) return;
  $('rec-slider').value = REC.idx;
  $('rec-time').textContent = `t=${f.time.toFixed(0)}s`;
  drawFrame(f.units);
}

function stopPlay() { if (REC.timer) { clearInterval(REC.timer); REC.timer=null; } $('rec-play').textContent='▶ 재생'; }

$('rec-play').onclick = () => {
  if (REC.timer) { stopPlay(); return; }
  if (REC.idx >= REC.frames.length-1) REC.idx = 0;
  $('rec-play').textContent='❚❚ 일시정지';
  REC.timer = setInterval(() => {
    if (REC.idx >= REC.frames.length-1) { stopPlay(); return; }
    showFrame(REC.idx+1);
  }, 160);
};
$('rec-slider').oninput = (e) => { stopPlay(); showFrame(+e.target.value); };

async function loadRecommendation() {
  $('rec-panel').style.display='';
  $('rec-info').textContent = '추천 시나리오 계산 중… (결심마다 최고안을 이어붙입니다)';
  const j = await api('/api/recommend', {session:S.session});
  REC.frames = j.frames||[]; REC.idx = 0;
  $('rec-slider').max = Math.max(0, REC.frames.length-1);
  const last = REC.frames[REC.frames.length-1];
  const b = last ? last.units.filter(u=>u.id<200 && u.hp>0).length : 0;
  const r = last ? last.units.filter(u=>u.id>=200 && u.hp>0).length : 0;
  $('rec-info').textContent = `${REC.frames.length}프레임 · 결심 ${j.picks.length}회 · 종료 시 아군 ${b}명 / 적 ${r}명`;
  $('rec-picks').innerHTML = (j.picks||[]).map(p=>`t=${p.time.toFixed(0)}s ${p.label} → ${p.blue_alive}v${p.red_alive}`).join('<br>');
  showFrame(0);
}

async function loadCandidates() {
  $('status').textContent = '후보 계산 중…';
  const j = await api(`/api/candidates/${S.session}`);
  S.cells = j.cells||[]; S.sel=null; S.elabels=j.engage_labels||[]; S.slabels=j.spread_labels||[];
  $('status').textContent = j.finished ? '전투 종료' : `t=${S.view.time.toFixed(0)}s · 전개안 ${S.cells.length}개`;
  $('commit').style.display = S.cells.length ? '' : 'none';
  $('commit').disabled = true; $('detail').textContent='';
  renderGrid();
}

function initMap() {
  const cfg = S.maps[S.mapName]; if (!cfg) return;
  const c = cfg.real_map||{};
  if (!S.map) {
    S.map = new naver.maps.Map('map', { center:new naver.maps.LatLng(c.origin_lat, c.origin_lon), zoom:17 });
    naver.maps.Event.addListener(S.map, 'click', (e) => {
      if (S.session) return;                       // 작전 시작 후에는 배치 불가
      const xy = latLngToXY(e.coord);
      if (S.mode==='blue') {
        const cap = Math.max(1, +$('rb').value);
        if (S.place.blue.length >= cap) { $('err').textContent = `아군은 최대 ${cap}명입니다. 수를 늘리거나 마커를 눌러 지우세요`; return; }
        S.place.blue.push(xy);
      } else if (S.mode==='red') {
        const cap = Math.max(1, +$('rr').value);
        if (S.place.red.length >= cap) { $('err').textContent = `적군은 최대 ${cap}명입니다. 수를 늘리거나 마커를 눌러 지우세요`; return; }
        S.place.red.push(xy);
      } else S.place.obj = xy;
      $('err').textContent = '';
      drawPlacement();
    });
  } else {
    S.map.setCenter(new naver.maps.LatLng(c.origin_lat, c.origin_lon));
  }
  drawTerrain(); drawPlacement();
}

['blue','red','obj'].forEach(m => $(`m-${m}`).onclick = () => {
  S.mode = m; ['blue','red','obj'].forEach(k=>$(`m-${k}`).className = (k===m?'on':''));
});
$('rand').onclick = async () => {
  if (S.session) { $('err').textContent='작전 중에는 배치를 바꿀 수 없습니다. 배치 초기화를 먼저 누르세요'; return; }
  const j = await api('/api/random', { map:S.mapName, mission:$('mission').value,
    blue:+$('rb').value, red:+$('rr').value, seed:Math.floor(Math.random()*100000) });
  S.place = { blue:j.blue_positions, red:j.red_positions, obj:j.objective };
  drawPlacement();
  $('status').textContent = `무작위 배치 완료 (아군 ${S.place.blue.length} · 적 ${S.place.red.length}) — 작전 개시를 누르거나 마커를 눌러 수정하세요`;
};

function setPhase(running) {
  // 작전 중에는 배치 관련 조작을 잠근다. 개시 버튼과 실행 버튼의 역할이 섞이지 않게 한다.
  $('start').disabled = running;
  $('start').textContent = running ? '작전 진행 중' : '작전 개시';
  ['m-blue','m-red','m-obj','rand','rb','rr','map-select','mission'].forEach(id => $(id).disabled = running);
}

$('reset').onclick = () => { S.place={blue:[],red:[],obj:null}; S.session=null; S.view=null; S.cells=[]; setPhase(false);
  $('grid').innerHTML='<div class="status">작전 개시 후 표시됩니다</div>'; $('tree').innerHTML='';
  $('status').textContent='지도를 눌러 부대를 배치하세요'; clearLayers('pathLayers'); drawPlacement(); };

$('start').onclick = async () => {
  if (S.place.blue.length===0 || S.place.red.length===0) { $('err').textContent='아군과 적군을 최소 1명씩 배치하세요'; return; }
  const j = await api('/api/session', { map:S.mapName, mission:$('mission').value,
    blue_positions:S.place.blue, red_positions:S.place.red, objective:S.place.obj });
  S.session = j.session; S.view = j; $('hint').style.display='none';
  setPhase(true);
  renderTree();
  await loadRecommendation();
  await loadCandidates();
};

$('commit').onclick = async () => {
  if (S.sel==null) return;
  const c = S.cells[S.sel];
  S.view = await api('/api/select', {session:S.session, engage_bin:c.engage_bin, spread_bin:c.spread_bin});
  S.cells=[]; S.sel=null; stopPlay(); renderTree();
  await loadRecommendation(); await loadCandidates();
};

$('map-select').onchange = () => { S.mapName = $('map-select').value;
  S.place={blue:[],red:[],obj:null}; S.session=null; S.view=null; initMap(); };

setPhase(false);

api('/api/maps').then(j => {
  j.maps.forEach(m => S.maps[m.name] = m);
  S.mapName = j.maps[0].name;
  $('map-select').innerHTML = j.maps.map(m=>`<option>${m.name}</option>`).join('');
  const script = document.createElement('script');
  script.src = 'https://oapi.map.naver.com/openapi/v3/maps.js?ncpKeyId=' + encodeURIComponent(j.naver_client_id);
  script.onload = initMap;
  script.onerror = () => $('err').textContent = '네이버 지도 SDK 로드 실패 — 클라이언트 ID와 Web 서비스 URL 등록을 확인하세요';
  document.head.appendChild(script);
});
</script>
</body>
</html>
"""
