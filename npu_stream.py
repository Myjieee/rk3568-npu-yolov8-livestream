#!/usr/bin/env python3
"""npu_stream.py v3 — 采集 / 推理 / 推流 三线程全并行流水线

v2.1 诊断结论(2026-10-28 实测):
  采集+推流可满帧 30fps; 瓶颈是推理在主循环内同步执行,
  单次(letterbox + NPU + numpy后处理)约 100ms, 把主循环卡到 18fps;
  降到 640x480 无改善 => 瓶颈是固定推理耗时, 不是像素搬运量。
v3 改法: 采集线程 / 推理线程 / 主循环(画框+推流) 全并行,
  主循环永不等待推理 => 画面满帧 30fps, 检测框按推理实际速度(约10fps)刷新。

用法(板子上):
    python3 npu_stream.py --host 192.168.100.2 --port 1234
PC 端: ffplay -fflags nobuffer -flags low_delay -framedrop \
    -probesize 32 -analyzeduration 0 -protocol_whitelist "file,udp,rtp" -i stream.sdp
"""
import os
import sys
import time
import threading
import argparse
import numpy as np
import cv2

ZOO_PY = os.path.expanduser('~/rknn_model_zoo/examples/yolov8/python')
sys.path.insert(0, ZOO_PY)
try:
    from yolov8 import post_process, CLASSES
    from py_utils.coco_utils import COCO_test_helper
except ImportError as e:
    sys.exit('找不到官方例程模块: %s' % e)

ap = argparse.ArgumentParser()
ap.add_argument('--host',   default='192.168.100.2')
ap.add_argument('--port',   type=int, default=1234)
ap.add_argument('--model',  default=os.path.expanduser(
    '~/rknn_model_zoo/examples/yolov8/model/yolov8n.rknn'))
ap.add_argument('--width',  type=int, default=1024)
ap.add_argument('--height', type=int, default=768)
ap.add_argument('--fps',    type=int, default=30)
ap.add_argument('--bps',    type=int, default=2500000)
ap.add_argument('--no-infer',  action='store_true',
                help='诊断: 关闭 NPU, 只测 采集+推流 上限')
ap.add_argument('--no-stream', action='store_true',
                help='诊断: 关闭推流, 只测 采集+推理 上限')
ap.add_argument('--bench', type=int, default=0,
                help='只跑 N 帧后自动退出(测速用), 0=不限')
ap.add_argument('--lang', choices=['zh', 'en'], default='zh',
                help='标签语言。中文需板端安装 fonts-noto-cjk 与 python3-pil')
ap.add_argument('--classes', default='',
                help='自定义类别文件(一行一个名字), 换自训练模型时用')
args = ap.parse_args()

# COCO 80 类的中文名(顺序与官方 CLASSES 严格一致)
CLASSES_ZH = [
    '人', '自行车', '汽车', '摩托车', '飞机', '公交车', '火车', '卡车', '船', '红绿灯',
    '消防栓', '停止标志', '停车计时器', '长椅', '鸟', '猫', '狗', '马', '羊', '牛',
    '大象', '熊', '斑马', '长颈鹿', '背包', '雨伞', '手提包', '领带', '行李箱', '飞盘',
    '双板滑雪', '单板滑雪', '运动球', '风筝', '棒球棒', '棒球手套', '滑板', '冲浪板',
    '网球拍', '瓶子', '酒杯', '杯子', '叉子', '刀', '勺子', '碗', '香蕉', '苹果',
    '三明治', '橙子', '西兰花', '胡萝卜', '热狗', '披萨', '甜甜圈', '蛋糕', '椅子',
    '沙发', '盆栽', '床', '餐桌', '马桶', '电视', '笔记本电脑', '鼠标', '遥控器',
    '键盘', '手机', '微波炉', '烤箱', '烤面包机', '水槽', '冰箱', '书', '时钟',
    '花瓶', '剪刀', '泰迪熊', '吹风机', '牙刷',
]

# 换自训练模型时, 用 --classes 覆盖类别名(中英文同时覆盖)
if args.classes:
    with open(args.classes, encoding='utf-8') as fh:
        _names = [s.strip() for s in fh.read().splitlines() if s.strip()]
    CLASSES[:] = _names
    CLASSES_ZH = _names
    print('自定义类别 %d 个: %s' % (len(_names), ' / '.join(_names[:5])))

# ---------- 1. NPU ----------
rknn = None
if not args.no_infer:
    from rknnlite.api import RKNNLite
    print('加载模型:', args.model)
    rknn = RKNNLite()
    if rknn.load_rknn(args.model) != 0:
        sys.exit('load_rknn failed')
    if rknn.init_runtime() != 0:
        sys.exit('init_runtime failed')
    print('NPU 就绪')
else:
    print('[诊断] NPU 已关闭, 只测 采集+推流')
co_helper = COCO_test_helper(enable_letter_box=True)

# ---------- 2. 采集线程 ----------
cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
cap.set(cv2.CAP_PROP_FPS, args.fps)
if not cap.isOpened():
    sys.exit('摄像头打开失败')
print('摄像头: %.0fx%.0f' % (cap.get(3), cap.get(4)))

_stop = False
_lock = threading.Lock()
_latest   = [None]   # 采集线程 -> 主循环(最新帧)
_infer_in = [None]   # 主循环 -> 推理线程(待推理帧, 空闲才投放)
_det      = [None]   # 推理线程 -> 主循环(最近一次检测结果)
infer_n = 0

def capture_loop():
    while not _stop:
        ok, f = cap.read()
        if ok:
            with _lock:
                _latest[0] = f

def infer_loop():
    """背靠背推理: 有帧就推理, 没帧就等。永不阻塞主循环。"""
    global infer_n
    while not _stop:
        with _lock:
            f = _infer_in[0]
            _infer_in[0] = None
        if f is None:
            time.sleep(0.002)
            continue
        img = co_helper.letter_box(im=f, new_shape=(640, 640),
                                   pad_color=(0, 0, 0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        outputs = rknn.inference(inputs=[np.expand_dims(img, 0)])
        boxes, classes, scores = post_process(outputs)
        if boxes is not None:
            with _lock:
                _det[0] = (co_helper.get_real_box(boxes), scores, classes)
        infer_n += 1

cap_thread = threading.Thread(target=capture_loop, daemon=True)
cap_thread.start()
if rknn is not None:
    infer_thread = threading.Thread(target=infer_loop, daemon=True)
    infer_thread.start()

# ---------- 3. GStreamer 硬编推流 ----------
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
Gst.init(None)

pipeline_str = (
    'appsrc name=src is-live=true do-timestamp=true format=time block=false '
    'caps=video/x-raw,format=BGR,width=%d,height=%d,framerate=%d/1 ! '
    'videoconvert ! video/x-raw,format=NV12 ! '
    'mpph264enc bps=%d gop=30 rc-mode=1 ! '
    'h264parse ! '
    'rtph264pay config-interval=1 pt=96 mtu=1200 aggregate-mode=zero-latency ! '
    'udpsink host=%s port=%d sync=false async=false'
    % (args.width, args.height, args.fps, args.bps, args.host, args.port)
)
appsrc = None
if not args.no_stream:
    pipeline = Gst.parse_launch(pipeline_str)
    appsrc = pipeline.get_by_name('src')
    pipeline.set_state(Gst.State.PLAYING)
    print('推流 -> %s:%d   (Ctrl+C 停止)' % (args.host, args.port))
else:
    print('[诊断] 推流已关闭, 只跑 采集+推理')


_zh_font = None
_zh_checked = False

def load_zh_font(size=22):
    """加载 CJK 字体。OpenCV 的 putText 不支持中文, 必须走 PIL 渲染。
    板端装字体: sudo apt install -y fonts-noto-cjk python3-pil"""
    global _zh_font, _zh_checked
    if _zh_checked:
        return _zh_font
    _zh_checked = True
    try:
        from PIL import ImageFont
    except ImportError:
        print('未装 python3-pil, 回落英文标签')
        return None
    for p in ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
              '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',
              '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
              '/usr/share/fonts/truetype/arphic/uming.ttc'):
        if os.path.exists(p):
            _zh_font = ImageFont.truetype(p, size)
            print('中文字体:', p)
            return _zh_font
    print('未找到中文字体, 回落英文标签 (sudo apt install fonts-noto-cjk)')
    return None


def _text_bbox(font, text):
    try:
        return font.getbbox(text)       # Pillow >= 9.2
    except AttributeError:
        w, h = font.getsize(text)       # 老版本
        return (0, 0, w, h)


# 文字贴图缓存: 检测结果在下一次推理出来前完全不变, 没必要每帧重新渲染。
# 缓存后 PIL 调用频率从 30fps 降到推理频率(约5~10fps), 主循环每帧只剩
# numpy 切片赋值, 与推理线程的 GIL 争抢基本消除。
_label_cache = {'key': None, 'items': []}


def _render_label_items(det, font):
    """检测结果更新时调用一次: 把每条文字渲染成带黑底的小 patch(BGR)。"""
    from PIL import Image, ImageDraw
    boxes, scores, classes = det
    items = []
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = [int(v) for v in box]
        name = CLASSES_ZH[cl] if cl < len(CLASSES_ZH) else CLASSES[cl]
        text = '%s %.2f' % (name, score)
        b = _text_bbox(font, text)
        w, h = b[2] - b[0] + 8, b[3] - b[1] + 8
        img = Image.new('RGB', (w, h), (0, 0, 0))
        ImageDraw.Draw(img).text((4 - b[0], 4 - b[1]), text,
                                 font=font, fill=(0, 255, 0))
        patch = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        items.append((top, max(0, left - h - 2), patch))
    return items


def draw_boxes(frame, det, zh=False):
    boxes, scores, classes = det
    # 矩形框用 OpenCV 画(C 实现, 释放 GIL, 很便宜)
    for box in boxes:
        top, left, right, bottom = [int(v) for v in box]
        cv2.rectangle(frame, (top, left), (right, bottom), (0, 255, 0), 2)

    font = load_zh_font() if zh else None
    if font is None:            # 字体不可用 -> 回落英文, putText 很便宜
        for box, score, cl in zip(boxes, scores, classes):
            top, left, right, bottom = [int(v) for v in box]
            cv2.putText(frame, '%s %.2f' % (CLASSES[cl], score),
                        (top, max(20, left - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return

    # 用 det 对象身份做缓存 key(缓存持有引用, 不会出现 id 复用误命中)
    if _label_cache['key'] is not det:
        _label_cache['key'] = det
        _label_cache['items'] = _render_label_items(det, font)

    H, W = frame.shape[:2]
    for x, y, patch in _label_cache['items']:
        ph, pw = patch.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x0 + pw), min(H, y0 + ph)
        if x1 > x0 and y1 > y0:
            frame[y0:y1, x0:x1] = patch[:y1 - y0, :x1 - x0]


# ---------- 4. 主循环: 每帧画框+推流, 绝不等推理 ----------
t0 = time.time()
n = 0
try:
    while True:
        with _lock:
            frame = _latest[0]
            _latest[0] = None
        if frame is None:
            time.sleep(0.002)
            continue

        # 投放最新帧给推理线程(仅当其空闲, 避免积压旧帧)
        if rknn is not None:
            with _lock:
                if _infer_in[0] is None:
                    _infer_in[0] = frame.copy()

        # 画最近一次检测结果(取引用, 不等推理)
        with _lock:
            det = _det[0]
        if det is not None:
            draw_boxes(frame, det, zh=(args.lang == 'zh'))

        # 推流(每帧都推)
        if appsrc is not None:
            buf = Gst.Buffer.new_wrapped(frame.tobytes())
            appsrc.emit('push-buffer', buf)

        n += 1
        if args.bench and n >= args.bench:
            break
        if n % 150 == 0:
            el = time.time() - t0
            print('stream fps: %.1f | infer fps: %.1f'
                  % (n / el, infer_n / el))
except KeyboardInterrupt:
    pass
finally:
    _stop = True
    time.sleep(0.1)
    if appsrc is not None:
        pipeline.set_state(Gst.State.NULL)
    cap.release()
    if rknn is not None:
        rknn.release()
    el = time.time() - t0
    print('stopped | stream %.1f fps | infer %.1f fps' % (n / el, infer_n / el))
