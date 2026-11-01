#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""web_ui.py — 板端轻量可视化服务（Flask + MJPEG）

面试现场最怕"只能靠我说"。这个项目跑起来是一条 RTSP 流，非要装播放器才能看，
不如再叠一层 HTTP 服务：浏览器打开就能看到实时带框画面 + IN/OUT 计数 +
抓拍事件列表 + 抓拍原图，不需要任何客户端软件。

设计要点(面试可讲):
  1. **零额外采集**: 不做第二路摄像头。主循环每帧把已经画好框的画面交给本
     模块(引用传递), Web 端只读不停采集。
  2. **按需编码**: JPEG 编码在 RK3568 上约 8~15ms/帧(视分辨率), 如果每来一个
     浏览者就按 30fps 编码, 主线程直接被拖垮。所以做法是——只有**有客户端在
     看**的时候, 主线程才按最多 10fps 的频率刷一张 JPEG 缓存; 没人看就整帧
     跳过, 对推流主链路零开销。这是 MJPEG 服务的正确姿势: 观察者开销可控。
  3. **读画面与看画面解耦**: 浏览器拉的是 /mjpeg 长连接 multipart 流, 它只是
     不断读那份 JPEG 缓存, 不碰推理/编码线程。
  4. **事件列表直接读 SQLite**: 和 event_store.py 落的是同一个库, 刷新即可看到
     最新抓拍, 含原图链接。
  5. **无 Flask 也能跑**: import 失败只是不开 Web, 主流水线照常工作。

用法:
    # 独立验证(不需要摄像头, 用黑白渐变假画面):
    python3 web_ui.py --port 8080

    # 在 rtsp_stream.py 里用(推荐):
    python3 rtsp_stream.py --cpp --tcp --web
"""
import os
import io
import json
import time
import queue
import argparse
import threading

__all__ = ['WebUI']

# ---------- ROI 配置的默认值与落盘位置 ------------------------------------
# 全部用归一化坐标(0~1): 采集分辨率与推流分辨率是解耦的, 存像素坐标的话
# 换一档分辨率线就跑偏, 存比例则永远对。
DEFAULT_ROI = {
    'version': 2,
    # line.poly: 折线(曲线)计数线的点列; 直线时为空, 用 p1/p2
    'line': {'p1': [0.05, 0.52], 'p2': [0.95, 0.52], 'poly': [],
             'in_side': 1, 'enabled': False},
    'roi': {'poly': [], 'mode': 'include', 'enabled': False},
}

# ---------- 页面 ---------------------------------------------------------
# 直接内联字符串, 不依赖 templates 目录 —— 板子上少一个文件少一处路径问题
PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RK3568 实时检测</title>
<style>
  :root{ --bg:#0f1115; --card:#171a21; --line:#252a33; --txt:#e6e9ef; --dim:#8b93a3;
         --acc:#4da3ff; --up:#ff4d4f; --down:#19c37d; }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--txt);
       font:14px/1.6 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif}
  header{padding:14px 20px;border-bottom:1px solid var(--line);
         display:flex;align-items:baseline;gap:14px;flex-wrap:wrap}
  h1{font-size:17px;margin:0;font-weight:600}
  .sub{color:var(--dim);font-size:12px}
  .wrap{padding:16px 20px;display:grid;gap:16px;
        grid-template-columns:minmax(0,2fr) minmax(300px,1fr)}
  @media(max-width:900px){.wrap{grid-template-columns:1fr}}
  .card{background:var(--card);border:1px solid var(--line);border-radius:10px;
        padding:12px 14px}
  .card h2{margin:0 0 10px;font-size:13px;color:var(--dim);font-weight:600;
           letter-spacing:.5px}
  img.live{width:100%;display:block;border-radius:8px;background:#000}
  .kpi{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:12px}
  .kpi div{background:var(--card);border:1px solid var(--line);border-radius:10px;
           padding:10px 12px}
  .kpi b{display:block;font-size:22px;font-weight:600;line-height:1.2}
  .kpi span{color:var(--dim);font-size:12px}
  table{width:100%;border-collapse:collapse;font-size:13px}
  th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left}
  th{color:var(--dim);font-weight:500}
  .in{color:var(--up);font-weight:600}
  .out{color:var(--down);font-weight:600}
  td img{height:46px;border-radius:4px;border:1px solid var(--line)}
  .stat{display:flex;justify-content:space-between;padding:3px 0;
        border-bottom:1px dashed var(--line)}
  .stat span:last-child{color:var(--acc);font-variant-numeric:tabular-nums}
  .dot{display:inline-block;width:8px;height:8px;border-radius:50%;
       background:var(--down);margin-right:6px}
  #stage{position:relative;display:block;width:100%;line-height:0}
  #ov{position:absolute;left:0;top:0;cursor:crosshair}
  .toolbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
  .toolbar button{background:#222734;color:var(--txt);border:1px solid var(--line);
    border-radius:6px;padding:5px 10px;font-size:12px;cursor:pointer}
  .toolbar button.on{background:var(--acc);color:#08121f;border-color:var(--acc);
    font-weight:600}
</style>
</head>
<body>
<header>
  <h1>RK3568 实时目标检测 · 跟踪 · 越线计数</h1>
  <span class="sub">YOLOv8n int8 | RKNN NPU | ByteTrack | GStreamer+MPP 硬编</span>
  <span class="sub" id="link"></span>
</header>
<div class="wrap">
  <div>
    <div class="kpi">
      <div><b class="in" id="kin">0</b><span>进入 (IN)</span></div>
      <div><b class="out" id="kout">0</b><span>离开 (OUT)</span></div>
      <div><b id="ktrack">0</b><span>当前目标数</span></div>
      <div><b id="ksnap">0</b><span>抓拍总数</span></div>
    </div>
    <div class="card">
      <h2>实时画面（带检测框 / ID / 计数线）</h2>
      <div class="toolbar">
        <button onclick="setMode('line',this)">画计数线</button>
        <button onclick="setMode('roi',this)">画区域</button>
        <button onclick="flipDir()">反转进出方向</button>
        <button onclick="toggleMode()">区域：<span id="rmode">只看区域内</span></button>
        <button onclick="clearAll()">清除</button>
        <span class="sub" id="rst"></span>
      </div>
      <div id="stage">
        <img class="live" id="live" src="/mjpeg">
        <canvas id="ov"></canvas>
      </div>
      <div class="sub" style="margin-top:6px">
        计数线：<b>按住拖动</b>即可画线，拖出弧度就是曲线（拖太短自动当直线）。
        区域：拖动连画松手自动闭合，或单击加点后<b>双击</b>闭合。
      </div>
    </div>
  </div>
  <div>
    <div class="card">
      <h2>运行状态</h2>
      <div id="stats"></div>
    </div>
    <div class="card" style="margin-top:16px">
      <h2>抓拍事件（最近 30 条）</h2>
      <div style="max-height:460px;overflow:auto">
        <table><thead><tr><th>时间</th><th>目标</th><th>方向</th><th>图</th></tr></thead>
        <tbody id="evs"></tbody></table>
      </div>
    </div>
  </div>
</div>
<script>
function fmt(n){return (n===undefined||n===null)?'-':(Math.round(n*10)/10)}
async function poll(){
  try{
    const s = await (await fetch('/api/stats')).json();
    document.getElementById('kin').textContent = s.in;
    document.getElementById('kout').textContent = s.out;
    document.getElementById('ktrack').textContent = s.tracks;
    document.getElementById('ksnap').textContent = s.snap;
    const rows = [
      ['画面/推理帧率', fmt(s.stream)+' / '+fmt(s.infer)+' fps'],
      ['推理延迟 P95', fmt(s.lat_p95)+' ms'],
      ['板内端到端', fmt(s.lat_mean)+' ms'],
      ['分辨率', s.res],
      ['后处理', s.post],
      ['正在拉流', s.clients+' 个客户端'],
    ];
    document.getElementById('stats').innerHTML = rows.map(r=>
      '<div class="stat"><span>'+r[0]+'</span><span>'+r[1]+'</span></div>').join('');
    const e = await (await fetch('/api/events')).json();
    document.getElementById('evs').innerHTML = e.map(x=>
      '<tr><td>'+x.ts.slice(11)+'</td><td>'+x.cls+' #'+x.track_id+
      '</td><td class="'+(x.is_in?'in':'out')+'">'+(x.is_in?'进':'出')+
      '</td><td><img src="/img/'+x.img+'"></td></tr>').join('');
    document.getElementById('link').innerHTML =
      '<span class="dot"></span>已连接';
  }catch(err){
    document.getElementById('link').innerHTML =
      '<span class="dot" style="background:#ff4d4f"></span>断线重连中';
  }
}
setInterval(poll, 1000); poll();
setInterval(()=>{const i=document.getElementById('live');
  i.src='/mjpeg?'+Date.now();}, 30000);   // 掉线自愈: 每 30s 重连一次

// ---- ROI 画线: canvas 覆盖层 ----
const img=document.getElementById('live');
const ov=document.getElementById('ov'),ctx=ov.getContext('2d');
let cfg=null,mode=null,pts=[],drag=false;
// 出厂默认配置: "清除"必须回到它, 而不是页面打开那一刻的配置。
// (旧版把页面加载时的配置存成 DEF, 结果保存过的区域永远清不掉)
const DEF0={version:2,
  line:{p1:[0.05,0.52],p2:[0.95,0.52],poly:[],in_side:1,enabled:false},
  roi:{poly:[],mode:'include',enabled:false}};
function px(p){return [p[0]*ov.clientWidth,p[1]*ov.clientHeight];}
function fit(){const r=img.getBoundingClientRect();
  if(!r.width)return;
  const d=window.devicePixelRatio||1;
  ov.width=Math.round(r.width*d);ov.height=Math.round(r.height*d);
  ov.style.width=r.width+'px';ov.style.height=r.height+'px';
  ctx.setTransform(d,0,0,d,0,0);draw();}
function draw(){const W=ov.clientWidth,H=ov.clientHeight;
  ctx.clearRect(0,0,W,H);if(!cfg)return;
  const R=cfg.roi,L=cfg.line;
  if(R&&R.enabled&&R.poly.length>1){
    ctx.beginPath();const a=px(R.poly[0]);ctx.moveTo(a[0],a[1]);
    for(let i=1;i<R.poly.length;i++){const b=px(R.poly[i]);ctx.lineTo(b[0],b[1]);}
    ctx.closePath();
    if(R.mode==='include'){ctx.fillStyle='rgba(25,195,125,.12)';ctx.strokeStyle='#19c37d';}
    else{ctx.fillStyle='rgba(255,77,79,.12)';ctx.strokeStyle='#ff4d4f';}
    ctx.fill();ctx.lineWidth=2;ctx.stroke();}
  if(L&&L.enabled){
    const P=(L.poly&&L.poly.length>1)?L.poly:[L.p1,L.p2];
    if(P&&P.length>1){
      ctx.strokeStyle='#ff4d4f';ctx.lineWidth=3;ctx.beginPath();
      const a=px(P[0]);ctx.moveTo(a[0],a[1]);
      for(let i=1;i<P.length;i++){const b=px(P[i]);ctx.lineTo(b[0],b[1]);}
      ctx.stroke();
      // IN 方向提示: 画在首段中点的法向 34px 处
      const m=px(P[0]),n2=px(P[1]);
      let nx=-(n2[1]-m[1]),ny=(n2[0]-m[0]);const nl=Math.hypot(nx,ny)||1;
      nx=nx/nl*34*L.in_side;ny=ny/nl*34*L.in_side;
      ctx.fillStyle='#19c37d';ctx.font='600 13px sans-serif';
      ctx.fillText('IN',(m[0]+n2[0])/2+nx,(m[1]+n2[1])/2+ny);}}
  if(mode==='line'&&pts.length>1){        // 拖动中的实时预览
    ctx.strokeStyle='rgba(255,77,79,.75)';ctx.lineWidth=2;ctx.beginPath();
    const s0=px(pts[0]);ctx.moveTo(s0[0],s0[1]);
    for(let i=1;i<pts.length;i++){const b=px(pts[i]);ctx.lineTo(b[0],b[1]);}
    ctx.stroke();}
  ctx.fillStyle='#4da3ff';
  for(const p of pts){const z=px(p);ctx.beginPath();ctx.arc(z[0],z[1],4,0,6.3);ctx.fill();}}
function nrm(e){const r=ov.getBoundingClientRect();
  return [(e.clientX-r.left)/r.width,(e.clientY-r.top)/r.height];}
function setMode(m,el){mode=(mode===m)?null:m;pts=[];drag=false;
  document.querySelectorAll('.toolbar button').forEach(b=>b.classList.remove('on'));
  if(mode&&el)el.classList.add('on');draw();}
function status(t){document.getElementById('rst').textContent=t||'';}
function refresh(){const s=document.getElementById('rmode');
  if(cfg&&s)s.textContent=(cfg.roi.mode==='include')?'只看区域内':'排除该区域';}
function post(){status('保存中…');
  fetch('/api/roi',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(cfg)}).then(r=>r.json()).then(j=>{
    status(j.ok?'已保存 ✓':'失败: '+j.err);refresh();draw();})
    .catch(e=>status('失败: '+e));}
// ---- 拖动画线: 按住鼠标拖 -> 采点 -> 松手生效 ----
// 采点不是每个 mousemove 都要: 拖得再快, 间距小于画面宽 1.2% 的点直接丢,
// 既让曲线平滑, 又不会把 roi.json 撑出一堆没用的点。
ov.addEventListener('pointerdown',e=>{
  if(!mode)return;
  ov.setPointerCapture(e.pointerId);
  if(mode==='line') pts=[nrm(e)];         // 画线: 每次一笔重新开始
  else pts.push(nrm(e));                  // 区域: 单击累加 / 拖动的第一个点
  drag=true;draw();});
ov.addEventListener('pointermove',e=>{
  if(!drag)return;const p=nrm(e);const l=pts[pts.length-1];
  const dx=(p[0]-l[0])*ov.clientWidth,dy=(p[1]-l[1])*ov.clientHeight;
  const min=Math.max(6,ov.clientWidth*0.012);
  if(dx*dx+dy*dy<min*min)return;
  pts.push(p);draw();});
ov.addEventListener('pointerup',e=>{
  if(!drag)return;drag=false;
  const a=pts[0],b=pts[pts.length-1];
  const moved=Math.hypot((b[0]-a[0])*ov.clientWidth,(b[1]-a[1])*ov.clientHeight);
  if(mode==='line'){
    if(moved<12){pts=[];mode=null;draw();return;}       // 原地轻点 = 误触, 取消
    if(pts.length>=3)  // 拖出了一串点 = 曲线
      cfg.line={poly:pts.slice(),in_side:cfg.line.in_side||1,enabled:true};
    else               // 只有两个点 = 直线
      cfg.line={p1:a,p2:b,poly:[],in_side:cfg.line.in_side||1,enabled:true};
    pts=[];mode=null;post();}
  else if(moved>=12&&pts.length>=3){   // 区域: 拖动松手自动闭合
    cfg.roi.poly=pts.slice();cfg.roi.enabled=true;pts=[];mode=null;post();}
  draw();});
ov.addEventListener('dblclick',e=>{
  if(mode!=='roi')return;
  if(pts.length>=3){cfg.roi.poly=pts.slice();cfg.roi.enabled=true;
    pts=[];mode=null;post();}
  else status('至少 3 个点');});
function flipDir(){if(!cfg)return;cfg.line.in_side=-cfg.line.in_side;post();}
function toggleMode(){if(!cfg)return;
  cfg.roi.mode=(cfg.roi.mode==='include')?'exclude':'include';post();}
function clearAll(){cfg=JSON.parse(JSON.stringify(DEF0));post();}
img.addEventListener('load',fit);
window.addEventListener('resize',fit);
fetch('/api/roi').then(r=>r.json()).then(c=>{
  cfg=c;refresh();fit();});
</script>
</body>
</html>
"""


class WebUI(object):
    """把最新一帧画好的画面 + 统计 + 事件表用 HTTP 暴露出去。

    主线程每帧调用 update(frame); Web 端按需取 JPEG 缓存。
    """

    def __init__(self, port=8080, host='0.0.0.0', snap_root=None,
                 quality=72, interval=0.1, verbose=True, roi_path=None):
        try:
            from flask import Flask, Response, jsonify, request, send_file
        except ImportError:
            raise ImportError('需要 Flask: pip3 install flask')
        self.port = int(port)
        self.host = host
        self.snap_root = os.path.abspath(os.path.expanduser(snap_root or '~/snapshots'))
        self.quality = int(quality)
        self.interval = float(interval)     # 最少间隔多久重编一次 JPEG = 最高刷新率
        self.verbose = bool(verbose)

        self._lock = threading.Lock()
        self._frame = None                  # 主循环给的最新帧(引用)
        self._jpg = None                    # JPEG 缓存
        self._enc_at = 0.0
        self._stats = {}
        self._clients = 0
        self._last_events = [{}]            # 变化才重查, 省 SQLite IO

        # ---- ROI(网页画线 / 画区域) ----
        self.roi_path = os.path.expanduser(roi_path or '~/roi.json')
        self._roi_lock = threading.Lock()   # 网页线程写, 主循环读, 必须加锁
        self._roi = self._roi_load()
        self._on_roi = None                 # 配置变化时回调主循环热更新

        self.app = Flask(__name__)
        self.app.add_url_rule('/', 'index', self._index)
        self.app.add_url_rule('/mjpeg', 'mjpeg', self._mjpeg)
        self.app.add_url_rule('/api/stats', 'api_stats', self._api_stats)
        self.app.add_url_rule('/api/events', 'api_events', self._api_events)
        self.app.add_url_rule('/img/<path:sub>', 'img', self._img)
        self.app.add_url_rule('/api/roi', 'api_roi', self._api_roi,
                              methods=['GET', 'POST'])
        self._mod = {'Flask': Flask, 'Response': Response, 'jsonify': jsonify,
                     'request': request, 'send_file': send_file}
        self.thread = None

    # -- 启动 ------------------------------------------------------------
    def start(self):
        if self.thread is not None:
            return
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()
        if self.verbose:
            print('可视化服务: http://<板子IP>:%d  (另一台机器用浏览器打开即可)'
                  % self.port)

    def _serve(self):
        # threaded=True 必须有: MJPEG 是长连接, 单线程会把事件查询请求堵死
        self.app.run(host=self.host, port=self.port, threaded=True,
                     use_reloader=False, debug=False)

    # -- 主循环侧接口 ----------------------------------------------------
    def update(self, frame):
        """主线程每帧调用一次, 只存引用(零拷贝)。真正的 JPEG 编码按需做。"""
        now = time.time()
        with self._lock:
            self._frame = frame
            busy = False
            if self._clients > 0 and frame is not None:
                busy = (now - self._enc_at) >= self.interval
        if busy:
            try:
                import cv2
                ok, buf = cv2.imencode('.jpg', frame,
                                       [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
                if ok:
                    with self._lock:
                        self._jpg = buf.tobytes()
                        self._enc_at = time.time()
            except Exception as e:
                if self.verbose:
                    print('[web] 编码失败:', e)

    def set_stats(self, **kw):
        # 'in' 是 Python 关键字, 调用方只能写 in_=x, 这里统一还原给前端
        if 'in_' in kw:
            kw['in'] = kw.pop('in_')
        with self._lock:
            self._stats.update(kw)

    # -- ROI: 网页画线 / 画区域 -------------------------------------------
    def _roi_load(self):
        """启动时读上次保存的配置(没有就用默认值, 不报错)。"""
        try:
            with open(self.roi_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            ok, err, cfg = self._roi_validate(cfg)
            if ok:
                return cfg
            print('[ROI] 配置不合法(%s), 用默认值' % err)
        except FileNotFoundError:
            pass
        except Exception as e:
            print('[ROI] 读取失败(%s), 用默认值' % e)
        return json.loads(json.dumps(DEFAULT_ROI))

    def _roi_validate(self, cfg):
        """校验 + 归一化。返回 (是否合法, 错误说明, 清理后的配置)。

        为什么要校验: 这是唯一一个"外部输入直接改运行逻辑"的接口。
        坐标越界会让线画到画面外导致永远不计数; 点数过多是手抖;
        线段太短等于没画。宁可拒绝, 也不要让一份脏配置把计数搞成 0。
        """
        if not isinstance(cfg, dict):
            return False, '格式不是对象', None
        out = json.loads(json.dumps(DEFAULT_ROI))

        def _pt(v):
            return (isinstance(v, (list, tuple)) and len(v) == 2
                    and all(isinstance(x, (int, float)) for x in v))

        line = cfg.get('line') or {}
        p1, p2 = line.get('p1'), line.get('p2')
        if _pt(p1) and _pt(p2):
            p1 = [min(1.0, max(0.0, float(p1[0]))), min(1.0, max(0.0, float(p1[1])))]
            p2 = [min(1.0, max(0.0, float(p2[0]))), min(1.0, max(0.0, float(p2[1])))]
            ln = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
            if ln < 0.05:
                return False, '线段太短(<5% 画面), 拒绝', None
            out['line']['p1'], out['line']['p2'] = p1, p2
        # 折线(曲线): 拖动画出来的点列。给了折线就以折线为准
        lpoly = line.get('poly') or []
        if isinstance(lpoly, list) and 2 <= len(lpoly) <= 64 \
                and all(_pt(p) for p in lpoly):
            cl = [[min(1.0, max(0.0, float(p[0]))),
                   min(1.0, max(0.0, float(p[1])))] for p in lpoly]
            total = sum(((cl[i + 1][0] - cl[i][0]) ** 2
                         + (cl[i + 1][1] - cl[i][1]) ** 2) ** 0.5
                        for i in range(len(cl) - 1))
            if total < 0.15:                 # 折线全长太短 = 手抖, 拒绝
                return False, '曲线太短(<15% 画面), 拒绝', None
            out['line']['poly'] = cl
        out['line']['in_side'] = -1 if int(line.get('in_side', 1)) < 0 else 1
        out['line']['enabled'] = bool(line.get('enabled', False)) and bool(
            out['line']['poly'] or (_pt(out['line']['p1']) and _pt(out['line']['p2'])))

        roi = cfg.get('roi') or {}
        poly = roi.get('poly') or []
        if isinstance(poly, list) and 3 <= len(poly) <= 16 and all(_pt(p) for p in poly):
            out['roi']['poly'] = [[min(1.0, max(0.0, float(p[0]))),
                                   min(1.0, max(0.0, float(p[1])))] for p in poly]
        out['roi']['mode'] = 'exclude' if roi.get('mode') == 'exclude' else 'include'
        out['roi']['enabled'] = bool(roi.get('enabled', False)) and bool(out['roi']['poly'])
        return True, '', out

    def set_roi_callback(self, fn):
        """注册回调: 网页改了配置之后立刻生效(不用重启流水线)。

        fn(cfg) 会在 Flask 线程里被调用, 里面只做"塞参数", 不要做重活。
        """
        self._on_roi = fn

    def get_roi(self):
        """主循环每帧调用: 返回配置的**浅拷贝快照**。

        为什么是快照: 直接读 self._roi 的话, 网页线程可能在主循环遍历到一半
        的时候把它换掉, 读到"半个新半个旧"的配置。拷贝一份就避免了加锁竞争。
        每帧一次拷贝的开销远小于加锁。
        """
        with self._roi_lock:
            return json.loads(json.dumps(self._roi))

    def _api_roi(self):
        request = self._mod['request']
        if request.method == 'GET':
            return self._mod['jsonify'](self.get_roi())
        try:
            cfg = request.get_json(force=True, silent=False)
        except Exception as e:
            return self._mod['jsonify']({'ok': False, 'err': '解析失败: %s' % e})
        ok, err, cfg = self._roi_validate(cfg)
        if not ok:
            return self._mod['jsonify']({'ok': False, 'err': err})
        with self._roi_lock:
            self._roi = cfg
        try:
            # 原子写: 先写临时文件再 rename, 防止写到一半断电留下半截 JSON
            tmp = self.roi_path + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(cfg, f, ensure_ascii=False)
            os.replace(tmp, self.roi_path)
        except Exception as e:
            print('[ROI] 保存失败:', e)
        if self._on_roi is not None:
            try:
                self._on_roi(cfg)
            except Exception as e:
                print('[ROI] 回调异常:', e)
        return self._mod['jsonify']({'ok': True})

    # -- HTTP 路由 -------------------------------------------------------
    def _index(self):
        return self._mod['Response'](PAGE, mimetype='text/html')

    def _mjpeg(self):
        def gen():
            with self._lock:
                self._clients += 1
            try:
                idle = 0
                while True:
                    with self._lock:
                        jpg = self._jpg
                    if jpg:
                        yield (b'--frame\r\nContent-Type: image/jpeg\r\n'
                               b'Content-Length: ' + str(len(jpg)).encode() +
                               b'\r\n\r\n' + jpg + b'\r\n')
                        idle = 0
                    else:
                        idle += 1
                        if idle > 200:          # ~10 秒没画面就断, 让浏览器重连
                            return
                    time.sleep(0.05)            # 最高 20fps 推送, 够用且省 CPU
            finally:
                with self._lock:
                    self._clients -= 1

        return self._mod['Response'](
            gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def _api_stats(self):
        with self._lock:
            s = dict(self._stats)
            s['clients'] = self._clients
        return self._mod['jsonify'](s)

    def _api_events(self):
        """直接读 event_store 落的那份 SQLite。库的路径跟 --snap-dir 一致。"""
        db = os.path.join(self.snap_root, 'events.db')
        rows = []
        if os.path.exists(db):
            import sqlite3
            try:
                con = sqlite3.connect(db, timeout=1.0)
                cur = con.execute(
                    'SELECT ts,track_id,cls,score,direction,is_in,n_in,n_out,image '
                    'FROM events ORDER BY id DESC LIMIT 30')
                cols = [c[0] for c in cur.description]
                for r in cur.fetchall():
                    d = dict(zip(cols, r))
                    # 库里存的是绝对路径, 前端只需要相对 ReviewsRoot 的部分
                    try:
                        d['img'] = os.path.relpath(d['image'], self.snap_root)
                    except Exception:
                        d['img'] = ''
                    rows.append(d)
                con.close()
            except Exception:
                pass
        return self._mod['jsonify'](rows)

    def _img(self, sub):
        send_file = self._mod['send_file']
        # 目录穿越防护: 只允许访问抓拍根目录内部的文件
        real_root = os.path.realpath(self.snap_root)
        real = os.path.realpath(os.path.join(real_root, sub))
        if not real.startswith(real_root + os.sep):
            return 'forbidden', 403
        return send_file(real, mimetype='image/jpeg')


def _demo(args):
    """没有摄像头也能起页面: 用合成画面验证整套链路。"""
    import numpy as np
    import cv2
    ui = WebUI(port=args.port, snap_root=args.snap_root)
    ui.start()
    print('演示模式: 合成画面。浏览器打开 http://localhost:%d' % args.port)
    n = 0
    while True:
        f = np.zeros((480, 640, 3), np.uint8)
        cv2.putText(f, 'RK3568 live frame %d' % n, (30, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.line(f, (0, 240), (640, 240), (0, 0, 255), 2)
        ui.update(f)
        ui.set_stats(stream=30.0, infer=14.7, lat_mean=29.1, lat_p95=58.0,
                     res='1024x768', post='C++', in_=4, out=3, tracks=2, snap=5)
        n += 1
        time.sleep(0.033)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Web 可视化自测页面')
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--snap-root', default='~/snapshots')
    _demo(ap.parse_args())
