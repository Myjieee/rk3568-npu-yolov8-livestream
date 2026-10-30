#!/usr/bin/env python3
"""rtsp_stream.py — RK3568 三线程流水线 + 板端 RTSP 服务器

替代裸 UDP: 板子本身就是 RTSP 服务器, VLC / 手机 / ffplay 主动来拉流,
PC 端不需要先开接收程序; 多客户端共享同一条编码管线(只编码一次)。

板子端:
    python3 rtsp_stream.py
拉流地址:
    rtsp://192.168.100.1:8554/live
PC 端:
    ffplay -fflags nobuffer -flags low_delay -framedrop rtsp://192.168.100.1:8554/live
板端依赖:
    sudo apt install -y gstreamer1.0-rtsp gir1.2-gst-rtsp-server-1.0 python3-gi
"""
import os
import re
import sys
import time
import threading
import subprocess
import argparse
import numpy as np
import cv2

# OpenCV 多线程: cvtColor / resize / letterbox 在高分辨率下是纯 CPU 大头,
# 默认有时只跑 1 个线程, 4 个 A55 全开能省掉一半以上的 CPU 时间
cv2.setNumThreads(min(4, os.cpu_count() or 4))

ZOO_PY = os.path.expanduser('~/rknn_model_zoo/examples/yolov8/python')
sys.path.insert(0, ZOO_PY)
try:
    from yolov8 import post_process, CLASSES
    from py_utils.coco_utils import COCO_test_helper
except ImportError as e:
    sys.exit('找不到官方例程模块: %s' % e)

# 跟踪/计数模块: 与本文件同目录。没传上来也能跑(自动降级为纯检测)
try:
    from tracker import ByteTracker, LineCounter
    TRACK_AVAILABLE = True
except ImportError as e:
    TRACK_AVAILABLE = False
    print('[提示] 未找到 tracker.py, 退回纯检测模式: %s' % e)

# 事件抓拍/落库/上报模块, 同样可选
try:
    from event_store import EventStore
    STORE_AVAILABLE = True
except ImportError as e:
    STORE_AVAILABLE = False
    print('[提示] 未找到 event_store.py, 不做抓拍: %s' % e)

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GLib
Gst.init(None)

ap = argparse.ArgumentParser()
ap.add_argument('--port',   type=int, default=8554)
ap.add_argument('--path',   default='/live')
ap.add_argument('--model',  default=os.path.expanduser(
    '~/rknn_model_zoo/examples/yolov8/model/yolov8n.rknn'))
ap.add_argument('--width',  type=int, default=None,
                help='采集分辨率宽(同时也是 AI 看到的原始画面尺寸)')
ap.add_argument('--height', type=int, default=None, help='采集分辨率高')
ap.add_argument('--out-width',  type=int, default=None,
                help='推流/编码输出宽。默认跟随采集; 采集>1920 时自动等比缩到 '
                     '1920(VPU + 客户端解码都轻松)。画框与计数在**采集原图**'
                     '坐标上算, 缩放只发生在最后一步, 不影响精度')
ap.add_argument('--out-height', type=int, default=None, help='推流/编码输出高')
ap.add_argument('--fps',    type=int, default=None)
ap.add_argument('--bps',    type=int, default=None,
                help='码率。不填时按输出分辨率自动算(约 0.13 bpp)')
ap.add_argument('--max',    action='store_true',
                help='一键最高档: 自动探测摄像头支持的最大分辨率, 按像素数自动'
                     '选帧率(>2.4MP 用 15, 否则 30), 码率自动给够, '
                     '编码器能支持的质量开关全开')
ap.add_argument('--lang',   choices=['zh', 'en'], default='zh')
ap.add_argument('--classes', default='')
ap.add_argument('--gop',    type=int, default=None,
                help='GOP 长度, 越小首帧越快(码率略高), WiFi 场景建议 10~15')
ap.add_argument('--tcp',    action='store_true',
                help='强制 RTSP over TCP, WiFi 丢包场景更稳(代价: 延迟略高)')
ap.add_argument('--wifi',   action='store_true',
                help='手机热点/WiFi 一键预设(可被显式参数覆盖): '
                     '640x480@20 1.2M GOP=15 + TCP + 关 WiFi 省电')
ap.add_argument('--no-infer', action='store_true',
                help='关掉 NPU 推理, 用于判断卡顿是网络还是 CPU 抢跑')
ap.add_argument('--cpp', action='store_true',
                help='用 C++ 后处理代替 numpy 版(需先编译 libpostprocess.so), '
                     '同一条命令加/不加此参数即可 A/B 对比帧率')
ap.add_argument('--track-max-lost', type=int, default=45,
                help='目标丢失多少轮后才回收 ID(约 14fps -> 45 轮≈3.2 秒)。'
                     '调大更抗断续检测, 代价是 ID 回收慢')
ap.add_argument('--track-rebirth', type=int, default=45,
                help='短期重识别窗口(轮): 目标被判定消失后, 若在这个窗口内'
                     '又在附近出现则复用原来的 ID, 防止同一个人/物体 ID 跳变。'
                     '设 0 关闭')
ap.add_argument('--no-track', action='store_true',
                help='关闭 ByteTrack 跟踪与计数, 退回纯检测(画框不带 ID)')
ap.add_argument('--no-count', action='store_true',
                help='只跟踪画 ID, 不画计数线、不统计进出')
ap.add_argument('--line-axis', choices=['h', 'v'], default='h',
                help="计数线方向: h 水平线(上下穿越) / v 垂直线(左右穿越)")
ap.add_argument('--line-pos', type=float, default=0.5,
                help='计数线在画面中的位置 0~1, 默认 0.5(正中间)')
ap.add_argument('--line-dir', choices=['down', 'up', 'left', 'right'],
                default=None, help='计入 IN 的方向, 默认 h->down / v->right')
ap.add_argument('--count-classes', default='39',
                help='参与计数的类别 id, 逗号分隔。默认 39(瓶子); '
                     '0=人, 2=汽车, 5=公交车, 7=卡车, 1=自行车')
ap.add_argument('--no-snap', action='store_true',
                help='关闭事件抓拍(仍然计数, 但不存图不落库)')
ap.add_argument('--snap-dir', default='~/snapshots',
                help='抓拍图片与 events.db 存放目录, 默认 ~/snapshots')
ap.add_argument('--snap-interval', type=float, default=2.0,
                help='同一目标两次抓拍的最小间隔(秒), 防反复横跳写满磁盘')
ap.add_argument('--mqtt-host', default=None,
                help='MQTT broker 地址, 不填则只本地落盘不上报')
ap.add_argument('--mqtt-port', type=int, default=1883)
ap.add_argument('--mqtt-topic', default='rk3568/events')
ap.add_argument('--count-anchor', default='top',
                choices=['top', 'bottom', 'center'],
                help='越线判定点: top=框顶边(头顶/物体最上端, 目标部分入镜时用, '
                     '默认); bottom=框底边(脚点, 全身入镜时用); center=框中心')
ap.add_argument('--debug-count', action='store_true',
                help='打印每个 ID 的跨线坐标与判定过程, 用于排查"不计数"')
ap.add_argument('--lowlat', action='store_true',
                help='低延迟档: GOP 压到 ~1/6 秒(关键帧更密, 客户端更快出画), '
                     '码率按分辨率重算。码率/画质换延迟')
args = ap.parse_args()

# ---------- 最高档: 自动探测/自动配置 ----------
def probe_cam_max_res(dev='/dev/video0'):
    """枚举 v4l2 支持的离散分辨率, 取像素数最大的那个。
    rkisp 常常不上报帧率档位(之前 grep interval 为空属正常), 所以只取分辨率。"""
    try:
        out = subprocess.check_output(
            ['v4l2-ctl', '-d', dev, '--list-formats-ext'],
            stderr=subprocess.DEVNULL, timeout=10).decode(errors='ignore')
    except Exception:
        return None
    sizes = set()
    for line in out.splitlines():
        m = re.search(r'Size:\s*Discrete\s+(\d+)x(\d+)', line)
        if m:
            sizes.add((int(m.group(1)), int(m.group(2))))
    if not sizes:
        return None
    return max(sizes, key=lambda wh: wh[0] * wh[1])


WIFI_PRESET = dict(width=640, height=480, fps=20, bps=1200000, gop=15)
DEFAULT = dict(width=1024, height=768, fps=30, bps=2500000, gop=30)

MAX_CHOICE = None
if args.max:
    MAX_CHOICE = probe_cam_max_res()
    if MAX_CHOICE is None:
        MAX_CHOICE = (3264, 2448)
        print('[最高档] v4l2 没枚举出尺寸, 兜底用 overlay 已知最高档 3264x2448')
    else:
        print('[最高档] 摄像头支持的最大分辨率: %dx%d' % MAX_CHOICE)
    args.width  = args.width  or MAX_CHOICE[0]
    args.height = args.height or MAX_CHOICE[1]
    if args.fps is None:
        # 像素越多, 单帧读出+处理越贵: 8MP 只能到 15fps, ~2MP 还能到 30fps
        args.fps = 15 if MAX_CHOICE[0] * MAX_CHOICE[1] > 2400000 else 30

preset = WIFI_PRESET if args.wifi else DEFAULT
_skip = ('width', 'height') if MAX_CHOICE else ()
for k, v in preset.items():
    if k in _skip:
        continue
    if getattr(args, k) is None:
        setattr(args, k, v)

# 采集(AI)尺寸 vs 推流(编码)尺寸分离:
#   检测/跟踪/计数全程在采集原图上做 -> 分辨率越高, 远处小目标像素越多, 检测越准;
#   只在最后一步缩到输出尺寸再喂给编码器 -> 省 VPU 算力、省带宽、客户端不卡。
CAM_W, CAM_H = args.width, args.height


def calc_out(cam_w, cam_h):
    """推流输出尺寸: 采集(AI)尺寸与编码尺寸分离。缺省规则 -> 超过 1920 宽就
    等比压到 1920(16 对齐), 因为再大的码流 VPU 与客户端解码都不划算。"""
    if args.out_width and args.out_height:
        return args.out_width, args.out_height
    if args.out_width or args.out_height or cam_w > 1920:
        w = (args.out_width or min(cam_w, 1920)) & ~15     # 编码器要求 16 对齐
        h = (args.out_height or int(round(w * cam_h / float(cam_w)))) & ~15
        return w, h
    return cam_w, cam_h


OUT_W, OUT_H = calc_out(CAM_W, CAM_H)
OUT_SCALE = (OUT_W, OUT_H) != (CAM_W, CAM_H)
if MAX_CHOICE:
    print('[最高档] 采集/AI %dx%d@%d -> 推流 %dx%d (自动降采样, 可用 '
          '--out-width/--out-height 覆盖)' % (CAM_W, CAM_H, args.fps, OUT_W, OUT_H))

# 码率自动给够: ~0.13 bit/像素(H.264 常规画质即可), 2M 下限 / 20M 上限
if args.bps is None:
    args.bps = int(max(2.0e6, min(2.0e7, OUT_W * OUT_H * args.fps * 0.13)))

if args.lowlat:
    # 低延迟三件套: 关键帧密 -> 客户端随时能起画; 码率抬一点保持画质;
    # GOP 取 ~1/6 秒(兼顾带宽), 且不超过 15
    args.gop = max(2, min(int(args.fps) // 6, 15))
    print('[低延迟档] GOP=%d (%.0f ms/关键帧)' % (args.gop, 1000.0 * args.gop / args.fps))

if args.wifi:
    # rtl8188f 省电模式会造成周期性丢包/延迟抖动, 必须关
    subprocess.call('iw dev wlan0 set power_save off', shell=True)
    print('[wifi 模式] %dx%d@%dfps %.1fMbps GOP=%d %s, WiFi 省电已关'
          % (args.width, args.height, args.fps, args.bps / 1e6,
             args.gop, 'TCP' if args.tcp else 'UDP'))


def dev_ip(dev):
    """取网卡 IPv4, 方便直接打印可拉的 rtsp 地址"""
    try:
        out = subprocess.check_output(['ip', '-4', 'addr', 'show', dev],
                                      stderr=subprocess.DEVNULL).decode()
        for line in out.splitlines():
            line = line.strip()
            if line.startswith('inet '):
                return line.split()[1].split('/')[0]
    except Exception:
        pass
    return None

# COCO 80 类中文名(顺序与官方 CLASSES 严格一致)
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
if args.classes:
    with open(args.classes, encoding='utf-8') as fh:
        _names = [s.strip() for s in fh.read().splitlines() if s.strip()]
    CLASSES[:] = _names
    CLASSES_ZH = _names

# ---------- 1. NPU ----------
if args.no_infer:
    rknn = None
    print('NPU 推理: 已关闭(--no-infer)')
else:
    from rknnlite.api import RKNNLite
    print('加载模型:', args.model)
    rknn = RKNNLite()
    if rknn.load_rknn(args.model) != 0:
        sys.exit('load_rknn failed')
    if rknn.init_runtime() != 0:
        sys.exit('init_runtime failed')
    print('NPU 就绪')
co_helper = COCO_test_helper(enable_letter_box=True)

# ---------- 1.2 后处理实现选择 ----------
USE_CPP = False
if args.cpp:
    try:
        from postprocess_cpp import post_process_cpp
        USE_CPP = True
        print('后处理: C++ (libpostprocess.so)')
    except Exception as e:
        print('C++ 后处理不可用, 回落 numpy: %s' % e)
if not USE_CPP:
    print('后处理: numpy (加 --cpp 可切到 C++ 版)')

# ---------- 1.5 跟踪 / 越线计数 ----------
# 跟踪与推理绑死: 没有检测结果就没有轨迹, 所以 --no-infer 时自动关闭
tracker = None
counter = None
store = None
USE_TRACK = TRACK_AVAILABLE and (not args.no_track) and (rknn is not None)
if USE_TRACK:
    # max_lost/rebirth 按推理帧率折算: 约 14fps -> 45 轮≈3.2 秒。
    # rebirth = 短期重识别: 断续检测(或小目标置信度掉到阈值下)时同一个人会被
    # 当成新目标 -> ID 跳变。记住已删除轨迹的位置与 ID, 短期内原地复现就复用。
    tracker = ByteTracker(track_thresh=0.45, low_thresh=0.12,
                          match_thresh=0.3, max_lost=args.track_max_lost,
                          n_init=3, rebirth=args.track_rebirth)
    if not args.no_count:
        cls_ids = tuple(int(x) for x in args.count_classes.split(',') if x.strip())
        counter = LineCounter(axis=args.line_axis, pos=args.line_pos,
                              in_dir=args.line_dir, cls_ids=cls_ids,
                              anchor=args.count_anchor,
                              debug=args.debug_count)
        # configure() 放到摄像头打开之后: 要用驱动给的**实际**分辨率,
        # 不能直接用 args 的设定值(驱动可能拒绝并给另一个, 坐标空间错位会永远不计数)
        names = ', '.join(CLASSES_ZH[c] if c < len(CLASSES_ZH) else str(c)
                          for c in cls_ids)
        print('越线计数: %s线 @%.0f%% | 计入方向=%s | 类别=%s'
              % ('水平' if args.line_axis == 'h' else '垂直',
                 args.line_pos * 100,
                 counter.in_dir, names))
        if STORE_AVAILABLE and not args.no_snap:
            store = EventStore(root=args.snap_dir, mqtt_host=args.mqtt_host,
                               mqtt_port=args.mqtt_port,
                               mqtt_topic=args.mqtt_topic,
                               min_interval=args.snap_interval)
        else:
            print('事件抓拍: 关')
    else:
        print('目标跟踪: 开(仅 ID, 不计数)')
elif args.no_track:
    print('目标跟踪: 关(--no-track)')

# ---------- 2. 采集线程 ----------
cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)


def try_set_res(w, h):
    """申请一个分辨率, 返回驱动实际给的。rkisp 常常「不听话」:
    要 8MP 可能给不了, 要 1224 高会自动裁到 1216(16 对齐要求)。"""
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(w))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(h))
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    return int(cap.get(3)), int(cap.get(4))


_real_w, _real_h = try_set_res(CAM_W, CAM_H)
if not cap.isOpened():
    sys.exit('摄像头打开失败')

if (_real_w, _real_h) != (CAM_W, CAM_H):
    # 驱动不给就往下找它肯给的最高档。顺序即优先级(由大到小)。
    # 注意 ISP 步进限制: 有些高度会被裁到 16 的倍数(1224 -> 1216), 属正常。
    FALLBACKS = [(3264, 2448), (2592, 1944), (2048, 1536), (1920, 1080),
                 (1632, 1224), (1600, 1200), (1280, 960), (1024, 768)]
    print('  ⚠ 驱动不支持 %dx%d(实际给了 %dx%d), 自动往下找最高可用档:'
          % (CAM_W, CAM_H, _real_w, _real_h))
    for fw, fh in FALLBACKS:
        if fw * fh >= CAM_W * CAM_H:
            continue
        rw, rh = try_set_res(fw, fh)
        if rw >= fw - 16 and abs(rh - fh) <= 16:     # 允许 16 对齐的小偏差
            print('    -> 采用 %dx%d (驱动实际 %dx%d)' % (fw, fh, rw, rh))
            CAM_W, CAM_H = rw, rh
            _real_w, _real_h = rw, rh
            args.width, args.height = CAM_W, CAM_H
            OUT_W, OUT_H = calc_out(CAM_W, CAM_H)
            OUT_SCALE = (OUT_W, OUT_H) != (CAM_W, CAM_H)
            # 输出尺寸变了, 码率也按新的像素数重算一次
            args.bps = int(max(2.0e6, min(2.0e7, OUT_W * OUT_H * args.fps * 0.13)))
            print('    推流 %dx%d | %.1f Mbps' % (OUT_W, OUT_H, args.bps / 1e6))
            break
    else:
        print('    -> 全部候选都被拒, 沿用驱动给的 %dx%d' % (_real_w, _real_h))
_cam_fps = cap.get(cv2.CAP_PROP_FPS)
if _cam_fps and _cam_fps > 0:
    print('摄像头: %dx%d @ %.1f fps (程序设定 %dx%d@%d)'
          % (_real_w, _real_h, _cam_fps, args.width, args.height, args.fps))
    if _cam_fps + 1 < args.fps:
        print('  ⚠ 摄像头实际只能到 %.1f fps, 达不到设定的 %d fps'
              '\n    查看它支持哪些帧率: v4l2-ctl -d /dev/video0 --list-formats-ext'
              % (_cam_fps, args.fps))
else:
    # OpenCV 的 V4L2 后端在 rkisp 上读不到 CAP_PROP_FPS, 返回 -1, 属正常现象
    print('摄像头: %dx%d (程序设定 %dx%d@%d)'
          % (_real_w, _real_h, args.width, args.height, args.fps))
    print('  (OpenCV 读不到 V4L2 帧率属性, 显示 -1 是正常现象, 不影响采集;'
          '\n   实际帧率用 v4l2-ctl -d /dev/video0 --list-formats-ext 查)')
if counter is not None:
    counter.configure(_real_w, _real_h)     # 计数必须与实际画面同一坐标系
    print('   判定点=%s | 计数坐标空间=%dx%d | 线在 y=%d'
          % (args.count_anchor, _real_w, _real_h, int(args.line_pos * _real_h)))

_stop = False
_lock = threading.Lock()
_latest   = [None]
_lat_ts   = [0.0]      # 本帧被采集线程读到的时刻(用于量端到端延迟)
_infer_in = [None]
_det      = [None]
infer_n = 0
_lat_samples = []      # 最近一段时间的 取帧->推帧完成 耗时(毫秒)


def capture_loop():
    while not _stop:
        ok, f = cap.read()
        if ok:
            with _lock:
                _latest[0] = f
                _lat_ts[0] = time.time()


def infer_loop():
    global infer_n
    while not _stop:
        if rknn is None:
            time.sleep(0.05)
            continue
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
        if USE_CPP:
            boxes, classes, scores = post_process_cpp(outputs)
        else:
            boxes, classes, scores = post_process(outputs)
        if boxes is not None:
            rb = co_helper.get_real_box(boxes)      # 还原到原图坐标
            ids = None
            if tracker is not None:
                # 跟踪放在推理线程: 计数只需要 ID, 没必要每帧都重算
                rb, scores, classes, ids = tracker.update(rb, scores, classes)
                if counter is not None and len(ids):
                    counter.update(rb, classes, ids, tracker.frame_no)
            with _lock:
                _det[0] = (rb, scores, classes, ids)
        infer_n += 1


# ---------- 3. 中文标签渲染(贴图缓存, 避免与推理线程抢 GIL) ----------
_zh_font = None
_zh_checked = False


def load_zh_font(size=22):
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
    print('未找到中文字体, 回落英文标签')
    return None


def _text_bbox(font, text):
    try:
        return font.getbbox(text)
    except AttributeError:
        w, h = font.getsize(text)
        return (0, 0, w, h)


_label_cache = {'key': None, 'items': []}


def _label_text(det, i):
    """统一标签文案: 开启跟踪时带 ID, 否则只有类别+置信度"""
    boxes, scores, classes, ids = det
    cl = int(classes[i])
    name = CLASSES_ZH[cl] if cl < len(CLASSES_ZH) else CLASSES[cl]
    if ids is not None and i < len(ids):
        return 'ID%d %s %.2f' % (int(ids[i]), name, scores[i])
    return '%s %.2f' % (name, scores[i])


def _render_label_items(det, font):
    from PIL import Image, ImageDraw
    boxes, scores, classes, ids = det
    items = []
    for i, box in enumerate(boxes):
        top, left, right, bottom = [int(v) for v in box]
        text = _label_text(det, i)
        b = _text_bbox(font, text)
        w, h = b[2] - b[0] + 8, b[3] - b[1] + 8
        img = Image.new('RGB', (w, h), (0, 0, 0))
        ImageDraw.Draw(img).text((4 - b[0], 4 - b[1]), text,
                                 font=font, fill=(0, 255, 0))
        patch = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        items.append((top, max(0, left - h - 2), patch))
    return items


def draw_boxes(frame, det, zh=False):
    boxes, scores, classes, ids = det
    _lw = max(2, int(round(frame.shape[1] / 640.0 * 2)))   # 分辨率越高线越粗
    for box in boxes:
        top, left, right, bottom = [int(v) for v in box]
        cv2.rectangle(frame, (top, left), (right, bottom), (0, 255, 0), _lw)

    # 大图上 22px 的字几乎看不见, 按画面宽度等比放大
    _fs = max(10, int(22 * frame.shape[1] / 1024.0))
    font = load_zh_font(size=_fs) if zh else None
    if font is None:
        for i, box in enumerate(boxes):
            top, left, right, bottom = [int(v) for v in box]
            cv2.putText(frame, _label_text(det, i),
                        (top, max(20, left - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return

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


def draw_counter(frame, cnt):
    """画计数线 + 左上角 HUD(IN/OUT)。没开计数时直接跳过"""
    if cnt is None:
        return
    sc = max(1.0, frame.shape[1] / 1024.0)
    lw = max(2, int(round(2 * sc)))
    (x1, y1), (x2, y2) = cnt.endpoints()
    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), lw)     # BGR 红
    cv2.putText(frame, cnt.hud(), (int(12 * sc), int(34 * sc)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9 * sc, (0, 0, 255), lw)


# ---------- 4. RTSP 服务器 ----------
def _elem_props(factory):
    """取某元素的属性名集合。mpph264enc / videoconvert 的可用属性在不同版本的
    mpp 包里不完全一致, 硬写属性会让整条管道起不来, 所以统一「有才加」。"""
    try:
        el = Gst.ElementFactory.make(factory, None)
        if el is None:
            return set()
        return set(p.name for p in el.list_properties())
    except Exception:
        return set()


def build_launch():
    # 注意: RTSP factory 的管道必须以 payloader 结尾并命名为 pay0
    # videoconvert 多线程: 高分辨率下 BGR->NV12 是纯 CPU 活, 4 个 A55 并行能省不少
    vc = 'videoconvert'
    if 'threads' in _elem_props('videoconvert'):
        vc += ' threads=%d' % max(1, min(4, os.cpu_count() or 1))
    enc = 'mpph264enc bps=%d gop=%d rc-mode=1' % (args.bps, args.gop)
    _ep = _elem_props('mpph264enc')
    # 画质开关: 放宽 QP 下限, 复杂画面(GOP 里的 P 帧)才舍得给码率
    for name, val in (('min-qp', 8), ('max-qp', 45), ('qp-init', 22)):
        if name in _ep:
            enc += ' %s=%d' % (name, val)
    return (
        '( appsrc name=src is-live=true do-timestamp=true format=time block=false '
        'caps=video/x-raw,format=BGR,width=%d,height=%d,framerate=%d/1 ! '
        '%s ! video/x-raw,format=NV12 ! '
        '%s ! '
        'h264parse ! '
        'rtph264pay name=pay0 pt=96 mtu=1200 config-interval=1 )'
        % (OUT_W, OUT_H, args.fps, vc, enc))


class RtspServer:
    def __init__(self):
        self.appsrc = None
        self.need_data = True       # 背压标志: appsrc 队列满时暂停投喂
        self.last_push = 0.0
        self.server = GstRtspServer.RTSPServer()
        self.server.props.address = '0.0.0.0'   # 监听所有网卡: 有线/WiFi 都能拉
        self.server.props.service = str(args.port)
        if args.tcp:
            try:
                self.server.props.protocols = GstRtspServer.RTSPLowerTrans.TCP
                print('RTSP 传输: 强制 TCP')
            except Exception as e:
                print('强制 TCP 失败(忽略):', e)
        factory = GstRtspServer.RTSPMediaFactory()
        factory.set_launch(build_launch())
        factory.set_shared(True)          # 多客户端共享同一条编码管线
        factory.connect('media-configure', self.on_media_configure)
        self.server.get_mount_points().add_factory(args.path, factory)
        if self.server.attach(None) == 0:
            sys.exit('RTSP 服务器启动失败')
        print('RTSP 就绪:')
        for dev in ('eth0', 'wlan0'):
            ip = dev_ip(dev)
            if ip:
                print('  %s -> rtsp://%s:%d%s' % (dev, ip, args.port, args.path))

    def on_media_configure(self, factory, media):
        """客户端接入时回调: 从 media 的 pipeline 里拿到 appsrc"""
        pipe = media.get_element()
        self.appsrc = pipe.get_by_name('src')
        if self.appsrc is None:
            print('客户端接入, 但未找到 appsrc')
            return
        # 背压: appsrc 内部队列最多留 ~2 帧。默认无上限时, 客户端消费慢
        # (WiFi 抖动) 会让旧帧无限堆积 -> 播放十几秒前的画面。
        self.appsrc.set_property('max-bytes', OUT_W * OUT_H * 3 * 2)
        self.appsrc.connect('need-data', self._on_need_data)
        self.appsrc.connect('enough-data', self._on_enough_data)
        self.need_data = True
        print('客户端接入, appsrc 就绪(背压模式, 队列上限 2 帧)')

    def _on_need_data(self, src, length):
        self.need_data = True

    def _on_enough_data(self, src):
        self.need_data = False


rtsp = RtspServer()
# RTSP 服务需要 GLib 主循环; 放到后台线程跑, 主循环继续负责采帧推帧
loop = GLib.MainLoop()
threading.Thread(target=loop.run, daemon=True).start()

cap_thread = threading.Thread(target=capture_loop, daemon=True)
cap_thread.start()
infer_thread = threading.Thread(target=infer_loop, daemon=True)
infer_thread.start()

# ---------- 5. 主循环: 每帧画框 + 喂给 appsrc ----------
t0 = time.time()
n = 0
n_push = 0                      # 实际推进 appsrc 的帧数(有客户端拉流才>0)
_snap_idx = 0                  # 已处理到第几个越线事件(counter.events 只增不减)
interval = 1.0 / args.fps      # 限速: 只按目标帧率取帧, 多余的直接丢
last_emit = 0.0
try:
    while True:
        now = time.time()
        if now - last_emit < interval:
            time.sleep(0.001)
            continue
        with _lock:
            frame = _latest[0]
            _latest[0] = None
            t_cap = _lat_ts[0]
        if frame is None:
            time.sleep(0.002)
            continue

        if rknn is not None:
            with _lock:
                if _infer_in[0] is None:
                    _infer_in[0] = frame.copy()

        with _lock:
            det = _det[0]
        if det is not None:
            draw_boxes(frame, det, zh=(args.lang == 'zh'))
        draw_counter(frame, counter)

        # 事件抓拍: 越线事件由推理线程产生, 主循环拿「已画好框的当前帧」落盘。
        # 只把帧的引用交给后台线程(零拷贝, 此帧已推给编码器不再被改), 主循环不等待。
        if store is not None and counter is not None:
            if len(counter.events) > _snap_idx:
                id2cls = {}
                if det is not None and det[3] is not None:
                    for i, tid in enumerate(det[3]):
                        id2cls[int(tid)] = (int(det[2][i]), float(det[1][i]))
                for ev in counter.events[_snap_idx:]:
                    c, s = id2cls.get(int(ev['id']), (-1, 0.0))
                    nm = CLASSES_ZH[c] if 0 <= c < len(CLASSES_ZH) else str(c)
                    store.on_event(ev, frame, nm, s)
                _snap_idx = len(counter.events)

        # 推帧三条件: 有 appsrc + 媒体 PLAYING + 背压允许。
        # 1) 客户端切换期间 media 会短暂 PAUSE, 此时推帧会堆旧帧 -> 黑屏;
        # 2) 队列满(enough-data)时继续投喂 -> 延迟越拉越大 -> 十几秒延迟。
        # 最后一步才缩放: 画框/计数已在采集原图坐标系完成, 这里只影响投喂给
        # 编码器的那一帧(省 VPU 算力、省带宽、客户端更好解, 检测精度不受影响)
        out_frame = frame
        if OUT_SCALE:
            out_frame = cv2.resize(frame, (OUT_W, OUT_H),
                                   interpolation=cv2.INTER_AREA)

        appsrc = rtsp.appsrc
        if appsrc is not None:
            # 兜底: 万一 enough-data 后再没收到 need-data, 1 秒后强制恢复投喂
            if not rtsp.need_data and time.time() - rtsp.last_push > 1.0:
                rtsp.need_data = True
            if rtsp.need_data:
                ok, state, _pend = appsrc.get_state(0)
                if ok == Gst.StateChangeReturn.SUCCESS and state == Gst.State.PLAYING:
                    try:
                        buf = Gst.Buffer.new_wrapped(out_frame.tobytes())
                        appsrc.emit('push-buffer', buf)
                        rtsp.last_push = time.time()
                        n_push += 1
                        # 板内延迟 = 这帧从 cam.read() 到推给编码器的耗时
                        if t_cap:
                            _lat_samples.append((time.time() - t_cap) * 1000.0)
                    except Exception:
                        rtsp.appsrc = None

        last_emit = now
        n += 1
        # 每 5 秒打印一次实际帧率(低帧率下按帧数打印间隔太长)
        if now - t0 >= 5.0:
            msg = ('stream fps: %.1f | infer fps: %.1f | push %d'
                   % (n / (now - t0), infer_n / (now - t0), n_push))
            if counter is not None:
                msg += ' | %s' % counter.hud()
            if store is not None:
                msg += ' | snap %d' % store.n_snap
            if _lat_samples:
                # 单独一行: 拼在 fps 行后面会超出终端宽度被截断
                print('   板内延迟(cam.read -> 推给编码器) 均值 %.1f ms | P95 %.1f ms'
                      % (np.mean(_lat_samples), np.percentile(_lat_samples, 95)))
            elif n_push == 0:
                print('   (push=0: 没有客户端在拉流或媒体未 PLAYING, 板内延迟无从统计;'
                      ' 用 ffplay 连上再看)')
                del _lat_samples[:]
            print(msg)
            t0 = now
            n = 0
            n_push = 0
            infer_n = 0
except KeyboardInterrupt:
    pass
finally:
    _stop = True
    loop.quit()
    time.sleep(0.1)
    cap.release()
    if store is not None:
        store.close()
    if rknn is not None:
        rknn.release()
    el = time.time() - t0
    print('stopped | stream %.1f fps | infer %.1f fps' % (n / el, infer_n / el))
