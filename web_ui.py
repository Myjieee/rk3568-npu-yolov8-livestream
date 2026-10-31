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
import time
import queue
import argparse
import threading

__all__ = ['WebUI']

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
      <img class="live" id="live" src="/mjpeg">
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
</script>
</body>
</html>
"""


class WebUI(object):
    """把最新一帧画好的画面 + 统计 + 事件表用 HTTP 暴露出去。

    主线程每帧调用 update(frame); Web 端按需取 JPEG 缓存。
    """

    def __init__(self, port=8080, host='0.0.0.0', snap_root=None,
                 quality=72, interval=0.1, verbose=True):
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

        self.app = Flask(__name__)
        self.app.add_url_rule('/', 'index', self._index)
        self.app.add_url_rule('/mjpeg', 'mjpeg', self._mjpeg)
        self.app.add_url_rule('/api/stats', 'api_stats', self._api_stats)
        self.app.add_url_rule('/api/events', 'api_events', self._api_events)
        self.app.add_url_rule('/img/<path:sub>', 'img', self._img)
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
