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
import json
import time
import threading
import subprocess
import argparse
import numpy as np
import cv2

# OpenCV 多线程: cvtColor / resize / letterbox 在高分辨率下是纯 CPU 大头,
# 默认有时只跑 1 个线程, 4 个 A55 全开能省掉一半以上的 CPU 时间。
# 但注意这是**全局**的: 采集主线程和推理线程都要用 OpenCV, 两边同时跑的时候
# 线程多了反而互相抢核。所以留了 CV_THREADS 环境变量做 A/B:
#   CV_THREADS=4 python3 rtsp_stream.py --profile   (默认, 单线程任务最快)
#   CV_THREADS=2 python3 rtsp_stream.py --profile   (双线程并发时常常更快)
CV_THREADS = int(os.environ.get('CV_THREADS', '') or 0)
if CV_THREADS <= 0:
    CV_THREADS = min(4, os.cpu_count() or 4)
cv2.setNumThreads(CV_THREADS)

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

# Web 可视化模块(可选): 只在加了 --web 时才真正启动
WEB_AVAILABLE = True
WEB_ERR = ''
try:
    from web_ui import WebUI
except ImportError as e:
    WEB_AVAILABLE = False
    WEB_ERR = str(e)

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
ap.add_argument('--classes', default=os.path.expanduser('~/labels.txt'),
                help='类别名清单文件, 一行一个。默认 ~/labels.txt: 文件存在就自动'
                     '加载(单类模型只需写一行, 如: 烟盒); 不存在则沿用 COCO 80 类')
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
ap.add_argument('--conf', type=float, default=0.25,
                help='检测置信度阈值(默认 0.25)。小目标/远距离目标置信度本来就低, '
                     '调到 0.15 能明显多检出框(代价是偶尔误检); '
                     '注意: 提高分辨率并不能改善检出 —— 输入固定 640, '
                     'letterbox 等比缩放后物体占比与原始分辨率无关')
ap.add_argument('--infer-stride', type=int, default=1,
                help='推理抽帧: 每 N 帧才跑一次 NPU, 推流仍保持满帧率。'
                     '画面上的框沿用上一次推理结果(ID 由 tracker 保持连续), '
                     'NPU/CPU 负载降到 1/N —— 裸板长时间跑不烫、不降频的关键手段。'
                     'N=2: 30fps 推流 + 15fps 检测(框延迟 ~33ms, 观感无损); '
                     'N=3: 检测 10fps, 更适合长时间挂机')
ap.add_argument('--track-max-lost', type=int, default=45,
                help='目标丢失多少轮后才回收 ID(约 14fps -> 45 轮≈3.2 秒)。'
                     '调大更抗断续检测, 代价是 ID 回收慢')
ap.add_argument('--track-rebirth', type=int, default=45,
                help='短期重识别窗口(轮): 目标被判定消失后, 若在这个窗口内'
                     '又在附近出现则复用原来的 ID, 防止同一个人/物体 ID 跳变。'
                     '设 0 关闭')
ap.add_argument('--track-thresh', type=float, default=0.45,
                help='跟踪高分阈值: 高于它才配 ID、参与计数(默认 0.45)。'
                     '远距离/小目标场景请和 --conf 一起降(如 --conf 0.15 '
                     '--track-thresh 0.2), 否则检测出来了也不给 ID、不计数')
ap.add_argument('--track-cross-iou', type=float, default=0.5,
                help='类别抖动容忍: YOLO 常把同一个瓶子一会儿判 bottle 一会儿'
                     '判 cup, 类别不同时只有重叠 >= 该值才允许续用老 ID(默认 0.5)。'
                     'ID 频繁跳变时可降到 0.3')
ap.add_argument('--track-reid-thresh', type=float, default=0.62,
                help='出画重识别的"长相"门槛(0~1, 默认 0.62): 目标离开画面/被挡住'
                     '几秒后再回来, 位置和类别都不可信, 此时只比颜色直方图。'
                     '同一物体通常 0.75~0.95, 不同物体 0.3~0.6。'
                     '两个相似物体互相抢 ID 就调高(0.70), 明明同一个却认不出就调低(0.55)。'
                     '设 1.1 等于关闭外观匹配')
ap.add_argument('--track-n-init', type=int, default=3,
                help='新轨迹需要连续命中多少帧才算确认(默认 3)。'
                     '调小=出现即给 ID(闪现也能数), 调大=更抗误检')
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
ap.add_argument('--count-classes', default='0',
                help='参与计数的类别 id, 逗号分隔。默认 0(第一个类 —— 单类模型'
                     '(如烟盒)就是它); COCO 80 类时: 0=人, 39=瓶子, 2=汽车')
ap.add_argument('--profile', action='store_true',
                help='每 5 秒打印一次各段耗时(ms): resize/copy/draw/tobytes 是'
                     '主线程, letterbox/cvt/npu/post/track 是推理线程。'
                     '帧率不对时先看这个, 别猜。skip = 推理没跟上被丢掉的帧数')
ap.add_argument('--web', action='store_true',
                help='额外起一个 HTTP 可视化服务: 浏览器直接看带框实时画面 + '
                     'IN/OUT 计数 + 抓拍事件列表, 不用装播放器。'
                     '没人访问时零开销(按需编码), 有人访问才按 --web-fps 编码')
ap.add_argument('--web-port', type=int, default=8080,
                help='Web 可视化端口, 默认 8080')
ap.add_argument('--roi-json', default='~/roi.json',
                help='网页画线/画区域的配置落盘位置(归一化坐标, 重启后自动恢复)。'
                     '只在 --web 时生效。')
ap.add_argument('--stream-json', default='~/stream.json',
                help='网页「分辨率/帧率」档位的落盘位置, 启动时自动应用上次的'
                     '档位。只在 --web 时生效。')
ap.add_argument('--web-fps', type=float, default=10.0,
                help='Web 页面最高刷新率(默认 10fps)。JPEG 编码在主线程做, '
                     '调低可进一步减少对推流主链路的影响')
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
ap.add_argument('--mt-vc',  action='store_true',
                help='videoconvert 多线程(实验性, 默认关): GStreamer 1.14 的'
                     '多线程格式转换按 Y 分片, 实测会产生水平条带状花屏, '
                     '单线程对 2MP@30 也够用, 不建议开')
ap.add_argument('--hq',     action='store_true',
                help='给编码器注入 min-qp=8/max-qp=45/qp-init=22(实验性画质开关, '
                     '默认关: 调参未经验证前不往编码链路里塞东西)')
ap.add_argument('--cool',   action='store_true',
                help='长时间挂机预设(可被显式参数覆盖): 推理抽帧 N=2 + RTSP over '
                     'TCP + GOP=15。解决「跑久了画面花 + CPU 烫」: TCP 排除 UDP '
                     '丢包, 抽帧把 NPU/CPU 负载砍半, 短 GOP 让丢包后更快恢复')
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


# 注意: DEFAULT 不写 bps —— bps 统一交给后面的「按输出像素自适应」公式,
# 否则 preset 先把 bps 填了, 自动公式永远轮不到执行(实测导致 8MP 档只有 2.5M)
WIFI_PRESET = dict(width=640, height=480, fps=20, bps=1200000, gop=15)
DEFAULT = dict(width=1024, height=768, fps=30, gop=30)

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

# ---------- 网页保存的档位(分辨率/帧率): 启动时自动应用 --------------------
# 白名单必须与 web_ui.py 的 STREAM_SIZES/STREAM_FPSES 一致:
# 都是 rkisp 实测能开的档(16 对齐 + ISP 步进), 白名单之外直接拒绝。
STREAM_SIZES = ((1632, 1216), (1280, 960), (1024, 768), (640, 480))
STREAM_FPSES = (30, 25, 20, 15)


def load_stream_cfg(path):
    """读网页上次保存的档位。没有/不合法返回 None(沿用命令行参数)。
    这是「网页改配置」能跨重启生效的另一半: web 端负责写, 这里负责读。"""
    try:
        with open(os.path.expanduser(path), 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        w, h, fps = int(cfg['width']), int(cfg['height']), int(cfg['fps'])
        if (w, h) in STREAM_SIZES and fps in STREAM_FPSES:
            return w, h, fps
        print('[档位] stream.json 的档位不在白名单(%dx%d@%d), 忽略'
              % (w, h, fps))
    except FileNotFoundError:
        pass
    except Exception as e:
        print('[档位] stream.json 读取失败(%s), 用命令行参数' % e)
    return None


_saved = load_stream_cfg(args.stream_json)
if _saved:
    args.width, args.height, args.fps = _saved
    print('[档位] 应用网页保存的档位: %dx%d @ %d fps (stream.json)'
          % _saved)

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

if args.cool:
    # 三件套: 抽帧降温 + TCP 抗丢包 + 短 GOP 快速恢复
    if args.infer_stride == 1:
        args.infer_stride = 2
    args.tcp = True
    if args.gop is None:
        args.gop = max(2, min(int(args.fps) // 2, 30))
    print('[挂机档] 抽帧 N=%d | RTSP over TCP | GOP=%d'
          % (args.infer_stride, args.gop))

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
elif not args.tcp:
    # 为什么不默认 TCP: UDP 延迟更低。但它的代价是丢包, 而丢包的症状很有迷惑性 ——
    # **画面静止时才出现错块, 一动反而干净**。原因见 README: P 帧残差只在有变化的
    # 宏块上写数据, 静止区域是"跳过"模式直接沿用参考帧, 参考帧坏掉就一直坏着。
    print('[提示] 当前为 UDP 传输: 丢包会在静止区域留下残留错块, '
          '若画面出现条纹请加 --tcp, 或客户端用 -rtsp_transport tcp')

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
    # 单类模型(如 smk)的类别 id 是 0, 不换名单的话框上会显示 COCO 的 "人"。
    # 默认路径(~/labels.txt)不存在只提示不报错(还能继续跑 80 类旧模型);
    # 手动指定的路径打错则直接退出, 免得拿错误名单跑半天。
    try:
        with open(args.classes, encoding='utf-8') as fh:
            _names = [s.strip() for s in fh.read().splitlines() if s.strip()]
    except FileNotFoundError:
        if args.classes != os.path.expanduser('~/labels.txt'):
            sys.exit('类别文件不存在: %s' % args.classes)
        print('[类别] 未找到 %s, 沿用 COCO 80 类名单' % args.classes)
        _names = []
    if _names:
        # 注意: CLASSES 是从 yolov8 模块 import 进来的 tuple(不可变),
        # 不能用切片赋值 CLASSES[:]=..., 必须整体替换成新的 list。
        CLASSES = list(_names)
        CLASSES_ZH = list(_names)
        print('[类别] 从 %s 加载 %d 个类名: %s'
              % (args.classes, len(_names), ', '.join(_names[:5])))

INFER_STRIDE = max(1, args.infer_stride)
if INFER_STRIDE > 1 and not args.no_infer:
    print('推理抽帧: 每 %d 帧跑一次 NPU -> 推流 %d fps / 检测约 %.1f fps'
          '(NPU 与 CPU 负载降到 1/%d, 长时间挂机不烫的关键)'
          % (INFER_STRIDE, args.fps, args.fps / float(INFER_STRIDE), INFER_STRIDE))

# ---------- 1. NPU ----------
# SINGLE_OUT: True = ultralytics 原生导出的单输出模型 (1, 4+nc, 8400),
# 框已解码、分数已过 sigmoid, 用下面的 post_process_single 解码;
# False = rknn_model_zoo 原版模型(三张特征图, 后处理里做 DFL 解码), 用 zoo 的
# post_process。两种格式互不兼容 —— 换自己训练的模型必须分清是哪种, 所以
# 启动时自动检测并打印。
SINGLE_OUT = None    # None=还没判定(启动自检失败的话, 首帧真实推理时再定)
ZOO_NC = None        # zoo/airockchip 六输出模型的类别数(从 cls 输出通道数读出)
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
    # 输出格式自检: 拿一张全零图跑一遍 NPU, 只为看输出形状。
    # 输入形状必须和正式推理完全一致(带 batch 维的 (1,640,640,3)),
    # 否则有的运行时会因形状不匹配抛异常 -> 自检失败 -> 走错后处理。
    try:
        _outs = rknn.inference(inputs=[np.zeros((1, 640, 640, 3), np.uint8)])
        OUT_SHAPES = [list(np.asarray(o).shape) for o in _outs]
        print('模型输出形状:', OUT_SHAPES)
        SINGLE_OUT = (len(OUT_SHAPES) == 1 and len(OUT_SHAPES[0]) == 3
                      and 4 < OUT_SHAPES[0][1] < 32)
        if SINGLE_OUT:
            print('后处理: 单输出解码(nc=%d, 自训练 ultralytics 模型)'
                  % (OUT_SHAPES[0][1] - 4))
        elif len(OUT_SHAPES) == 6:
            # airockchip 导出布局: 3 个尺度 x (box 64ch, cls nc ch) 成对出现,
            # 索引 0/2/4 是 box, 1/3/5 是 cls -> nc 就是 cls 输出的通道数。
            # (nc=1 时官方代码里的 cls_sum 和 cls 数学等价, 被 ONNX 导出器
            #  合并掉了, 所以是 6 个输出而不是 9 个 —— 不是模型坏了)
            ZOO_NC = OUT_SHAPES[1][1]
            print('后处理: zoo 解码(3 尺度 x [box64 + cls%d], airockchip 导出)'
                  % ZOO_NC)
    except Exception as e:
        print('⚠ 输出自检失败(%s), 首帧真实推理时再自动判定' % e)
co_helper = COCO_test_helper(enable_letter_box=True)

# 检测阈值: numpy 版是模块级常量, 直接改它即可生效
import yolov8
yolov8.OBJ_THRESH = args.conf

# zoo 的 dfl() 内部 import torch —— 板子没装 torch 会直接崩, 装了也很慢。
# 它做的事是"16 个 bins 做 softmax, 再按 bin 序号加权求和"(DFL 解码),
# numpy 写出来数学完全等价且更快, 直接换掉。
def _dfl_np(position):
    n, c, h, w = position.shape
    mc = c // 4                            # 每个坐标 16 个 bins
    y = position.reshape(n, 4, mc, h, w)
    e = np.exp(y - y.max(axis=2, keepdims=True))   # 数值稳定的 softmax
    y = e / e.sum(axis=2, keepdims=True)
    acc = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
    return (y * acc).sum(axis=2)
yolov8.dfl = _dfl_np

if rknn is not None:
    print('检测阈值 --conf = %.2f' % args.conf)

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

# C++ 版(libpostprocess.so)是按 zoo 原版模型的三输出格式编译的,
# 单输出的自训练模型喂给它会解出乱框 —— 直接拦下, 不给静默出错的机会。
if USE_CPP and SINGLE_OUT:
    print('C++ 后处理不支持单输出模型(它是按 zoo 三输出格式编译的), '
          '自动切回 python 解码')
    USE_CPP = False
# C++ 版还是按 80 类编译的: 它硬编码 cls 通道 = 80, nc=1 的模型只有 1 个
# cls 通道, 喂进去通道数对不上会解出乱框 —— 同样直接拦下。
if USE_CPP and ZOO_NC is not None and ZOO_NC != 80:
    print('C++ 后处理按 80 类编译(cls 通道 80 != %d), 自动切回 python 解码'
          % ZOO_NC)
    USE_CPP = False

# ---------- 1.3 单输出模型后处理(自训练 ultralytics 模型用) ----------
def nms_greedy(boxes, scores, iou_thresh=0.45):
    """贪心 NMS: 按分数从高到低逐个"录取", 每录取一个就删掉与它 IoU 超过
    阈值的剩余框。IoU 的数学与 tracker.py 的匹配完全是同一个 —— 那里挑
    "最像的"配对, 这里删"太像的"去重。(即 21 天复写计划里的 _greedy,
    写在这里当教材: 同一个 IoU, 两种相反的用法)"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    order = scores.argsort()[::-1]            # 按分数从高到低的下标
    keep = []
    while order.size:
        i = order[0]                           # 当前最高分, 录取
        keep.append(i)
        rest = order[1:]
        if rest.size == 0:
            break
        # 剩余框各自与"录取框"的 IoU: 重叠区是"左边界取大者、右边界取小者"
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(xx2 - xx1, 0) * np.maximum(yy2 - yy1, 0)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thresh]        # 只留"不像"的, 进入下一轮
    return np.array(keep, dtype=np.int64)


_det_dbg_ts = 0.0    # 检测调试打印节流(每秒最多一次, 防刷屏)


def post_process_single(outputs, conf_thresh=0.25):
    """解码 ultralytics 原生导出的单输出 (1, 4+nc, 8400):
      前 4 行 = 框中心x / 中心y / 宽 / 高(像素, 640 letterbox 坐标系, 已解码);
      后 nc 行 = 各类别分数(已过 sigmoid)。
    输出 (x1,y1,x2,y2) + 类别 id + 分数, 坐标约定与 zoo 版完全一致,
    后面照旧交给 get_real_box 还原到原图坐标。"""
    global _det_dbg_ts
    out = np.asarray(outputs[0])
    if out.ndim == 3 and out.shape[0] == 1:
        out = out[0]                           # (4+nc, 8400)
    pred = out.T                               # (8400, 4+nc) 一行一个候选框
    cls_scores = pred[:, 4:]                   # (8400, nc)
    cls_ids = cls_scores.argmax(axis=1)
    conf = cls_scores.max(axis=1)
    keep = conf >= conf_thresh                 # 置信度过滤(8400 个先筛掉绝大多数)
    _now = time.time()
    if _now - _det_dbg_ts > 1.0:               # 每秒一次体检, "没框"时先看这行
        _det_dbg_ts = _now
        top = np.sort(conf)[::-1][:5]
        print('[检测调试] conf>%.2f 的候选 %d 个 | top5 分数 %s | 坐标范围 x[%.1f~%.1f] y[%.1f~%.1f]'
              % (conf_thresh, int(keep.sum()), np.round(top, 3),
                 pred[:, 0].min(), pred[:, 0].max(),
                 pred[:, 1].min(), pred[:, 1].max()))
    if not keep.any():
        return None, None, None
    cx, cy, w, h = (pred[:, i][keep] for i in range(4))
    boxes = np.stack([cx - w * 0.5, cy - h * 0.5,    # 中心+宽高 -> 左上右下
                      cx + w * 0.5, cy + h * 0.5], axis=1)
    cls_ids, conf = cls_ids[keep], conf[keep]
    keep2 = nms_greedy(boxes, conf, 0.45)      # 去重
    if keep2.size == 0:
        return None, None, None
    return boxes[keep2], cls_ids[keep2], conf[keep2]

# ---------- 1.5 跟踪 / 越线计数 ----------
# 跟踪与推理绑死: 没有检测结果就没有轨迹, 所以 --no-infer 时自动关闭
tracker = None
counter = None
store = None
USE_TRACK = TRACK_AVAILABLE and (not args.no_track) and (rknn is not None)
if USE_TRACK:
    # max_lost/rebirth 按推理帧率折算: 约 14fps -> 45 轮≈3.2 秒。
    # rebirth = 短期重识别: 断续检测(或小目标置信度掉到阈值下)时同一个人会被
    # 当成新目标 -> ID 跳变。记住已删除轨迹的位置、ID 和颜色直方图, 短期内
    # 原地复现就复用; 出画几秒再进来则靠颜色(位置和类别都已经不可信)。
    tracker = ByteTracker(track_thresh=args.track_thresh,
                          low_thresh=min(0.12, args.track_thresh * 0.4),
                          match_thresh=0.3, max_lost=args.track_max_lost,
                          n_init=args.track_n_init, rebirth=args.track_rebirth,
                          cross_iou=args.track_cross_iou,
                          reid_thresh=args.track_reid_thresh)
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

# ---------- 1.6 Web 可视化(可选) + ROI 画线 ----------
web = None

# ROI 当前配置的快照。列表包一层是为了让回调里赋值能穿透到全局(和 _det 同理)。
ROI_CFG = [None]

# 分辨率/帧率切换不能热改: GStreamer caps 与 V4L2 格式都锁死在流水线里,
# 原地改是空谈。做法: 网页写下 stream.json + 塞标志 -> 主循环看到 ->
# os.execv 整进程重启(所有句柄随进程映像一起干净释放, 比原地 teardown
# 重建稳得多 —— 这块驱动实测就怕频密重开)。
RESTART_REQ = [None]


def on_stream_change(cfg):
    """网页点了分辨率/帧率档位(Flask 线程里调用): 只塞标志, 不做重活。"""
    if (cfg['width'], cfg['height']) == (CAM_W, CAM_H) and \
            cfg['fps'] == args.fps:
        print('[档位] 已是 %dx%d@%d, 忽略重复点击' % (CAM_W, CAM_H, args.fps))
        return '当前已是该档位'
    RESTART_REQ[0] = dict(cfg)
    print('[档位] 网页请求切换到 %dx%d@%d, 主循环将重启流水线'
          % (cfg['width'], cfg['height'], cfg['fps']))
    return ''


def on_roi_change(cfg, initial=False):
    """网页改了线/区域 -> 立刻热更新, 不重启流水线。

    在 Flask 线程里被调用: 只做"塞参数"这种轻活, 不做重活、不碰摄像头。
    initial=True 表示"启动时加载上次保存的配置": 这时如果线是关的,
    保留命令行 --line-axis/--line-pos 的默认线(不然一启动线就消失)。
    网页上点"清除"(initial=False)则是彻底关掉计数线。
    """
    ROI_CFG[0] = cfg
    ln = (cfg or {}).get('line', {})
    if counter is not None and ln.get('enabled'):
        if ln.get('poly'):
            counter.set_polyline(ln['poly'], ln.get('in_side', 1))
            print('[ROI] 曲线计数线已更新: %d 个点, in_side=%+d'
                  % (len(ln['poly']), ln.get('in_side', 1)))
        else:
            counter.set_line(ln['p1'], ln['p2'], ln.get('in_side', 1))
            print('[ROI] 计数线已更新: (%.2f,%.2f)->(%.2f,%.2f) in_side=%+d'
                  % (ln['p1'][0], ln['p1'][1], ln['p2'][0], ln['p2'][1],
                     ln.get('in_side', 1)))
    elif counter is not None and not initial and counter.poly is not None:
        counter.disable_line()
        print('[ROI] 计数线已彻底关闭(网页清除), 不再显示默认线')
    r = (cfg or {}).get('roi', {})
    n_poly = len(r.get('poly') or [])
    if n_poly < 3:
        # 这一行是专门用来排"画了区域却什么都没滤"的: 0 点 = 区域根本没存进去,
        # 千万别再去猜阈值和判定规则(那都是 3 点以上才轮得到的事)
        print('[ROI] 区域未生效: 0 点 —— 配置里没有多边形, 等于不筛。'
              '请在网页上重新画一次区域(画完看按钮旁是否出现"已保存 ✓")')
    else:
        print('[ROI] 区域: %s (%d 点, 判定=%s)'
              % ('只看区域内' if r.get('mode') == 'include' else '排除该区域',
                 n_poly, '完全在内' if r.get('rule') == 'all' else '有交集'))


def apply_roi(boxes, scores, classes):
    """按多边形 ROI 过滤检测结果。**必须在 tracker.update 之前调用**。

    顺序为什么不能反: 先跟踪再过滤的话, 目标走到 ROI 边界会"凭空消失",
    tracker 判它 lost, 走出 ROI 再回来就是新 ID —— 同一个目标被数两次。
    """
    cfg = ROI_CFG[0]
    r = (cfg or {}).get('roi', {})
    if not r.get('enabled') or not r.get('poly'):
        return boxes, scores, classes       # 没配区域: 原样返回

    poly = np.array([[int(x * OUT_W), int(y * OUT_H)] for x, y in r['poly']],
                    dtype=np.int32)
    if len(poly) < 3:                       # 两点不成区域, 直接不过滤
        return boxes, scores, classes

    inc = (r.get('mode') == 'include')      # True=只看区域内, False=排除该区域
    # rule: 'any' = 框跟区域有交集就算(默认, 符合直觉: 踩进来一只脚就要看到)
    #       'all' = 框必须整个落在区域内(严格)
    rule = r.get('rule', 'any')

    keep = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = [float(v) for v in boxes[i][:4]]
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        pts = [(cx, y2), (cx, cy),                      # 底边中点 / 中心
               (x1, y1), (x2, y1), (x1, y2), (x2, y2)]  # 四个角
        n_in = sum(1 for p in pts
                   if cv2.pointPolygonTest(poly, p, False) >= 0)
        inside = (n_in > 0) if rule == 'any' else (n_in == len(pts))
        if inside == inc:
            keep.append(i)

    # 节流打印: 直接回答"区域到底有没有在过滤"这个疑问
    now = time.time()
    if now - getattr(apply_roi, '_log_at', 0) > 2.0:
        apply_roi._log_at = now
        print('[ROI] 过滤 %s(%s): %d -> %d'
              % ('只看区域内' if inc else '排除该区域', rule,
                 len(boxes), len(keep)))

    if len(keep) == len(boxes):
        return boxes, scores, classes
    return boxes[keep], scores[keep], classes[keep]


def draw_roi(frame):
    """把区域多边形画到画面上(推流端也能看到, 不只是网页)。"""
    cfg = ROI_CFG[0]
    r = (cfg or {}).get('roi', {})
    if not r.get('enabled') or not r.get('poly'):
        return
    pts = np.array([[int(x * OUT_W), int(y * OUT_H)] for x, y in r['poly']],
                   dtype=np.int32)
    color = (0, 200, 0) if r.get('mode') == 'include' else (0, 0, 255)
    cv2.polylines(frame, [pts], True, color, 2, cv2.LINE_AA)
    cv2.putText(frame, 'ROI' if r.get('mode') == 'include' else 'MASK',
                (int(pts[0][0]) + 6, int(pts[0][1]) + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


if args.web:
    if WEB_AVAILABLE:
        web = WebUI(port=args.web_port, snap_root=args.snap_dir,
                    interval=1.0 / max(1.0, args.web_fps),
                    roi_path=args.roi_json,
                    stream_path=args.stream_json,
                    stream_cfg=dict(width=CAM_W, height=CAM_H, fps=args.fps))
        web.set_roi_callback(on_roi_change)
        web.set_stream_callback(on_stream_change)
        ROI_CFG[0] = web.get_roi()          # 启动时先加载上次保存的配置
        on_roi_change(ROI_CFG[0], initial=True)
        web.start()
    else:
        print('[提示] 未启用 Web 可视化: %s (pip3 install flask)' % WEB_ERR)

# ---------- 2. 采集线程 ----------
# 重试 3 次: exec 重启后驱动释放设备可能有一瞬延迟, 一次打不开别直接死
cap = None
for _try in range(3):
    cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)
    if cap.isOpened():
        break
    print('  ⚠ 摄像头打开失败(第 %d/3 次), 2 秒后重试...' % (_try + 1))
    time.sleep(2.0)


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
    # 计数/跟踪/画框的坐标系 = 推流输出尺寸(下游统一坐标系, 见主循环)。
    # 不能用采集实际尺寸: 8MP 采集+1920 推流时框坐标是 1920 坐标,
    # 若按 3264 配置, 线位置/越线判定会整体错位
    counter.configure(OUT_W, OUT_H)
    print('   判定点=%s | 计数坐标空间=%dx%d | 线在 y=%d'
          % (args.count_anchor, OUT_W, OUT_H, int(args.line_pos * OUT_H)))

_stop = False
_lock = threading.Lock()
_latest   = [None]
_lat_ts   = [0.0]      # 本帧被采集线程读到的时刻(用于量端到端延迟)
_infer_in = [None]
_det      = [None]
_det_ts   = [0.0]      # _det 里这份结果的时间戳(运动补偿用)
_det_dt   = [0.09]     # 推理间隔 EMA(秒/轮): infer 11fps 时约 0.09, 外推的"每轮"单位
infer_n = 0
_infer_err = [0]           # 推理线程被捕获的异常次数(>0 说明有异常但线程还活着)
_infer_err_msg = ['']


# ---------- 分段计时器(--profile) ----------
# 帧率低的时候最忌讳瞎猜。NPU 单帧 54ms 是硬件常数, 如果实测推理只有 5fps
# (200ms/帧), 那 146ms 一定花在别处: resize / 整帧 copy / 画框 / tobytes /
# letterbox / 后处理 / 跟踪 —— 到底哪一段, 靠 --profile 打印出来说话。
_prof = {}


def _bucket(name):
    b = _prof.get(name)
    if b is None:
        b = _prof[name] = [0.0, 0]
    return b


def _pr(name):
    if not getattr(args, 'profile', False):
        return None
    return time.time(), _bucket(name)


def _pr_end(ctx):
    if ctx is None:
        return
    t0, b = ctx
    b[0] += time.time() - t0
    b[1] += 1


def prof_report():
    if not _prof:
        return ''
    parts = []
    for name in ('resize', 'copy', 'draw', 'tobytes', 'letterbox', 'cvt',
                 'npu', 'post', 'track'):
        b = _prof.get(name)
        if b and b[1]:
            parts.append('%s %.1fms' % (name, b[0] * 1000.0 / b[1]))
    _sk = int(_prof.get('skip', [0, 0])[0])
    for name, b in _prof.items():
        b[0] = 0.0
        b[1] = 0
    if _sk:
        parts.append('skip %d帧' % _sk)
    return '   | '.join(parts)
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
        try:
            infer_once()
        except Exception as e:
            # 绝不让线程静默死亡: 打印一次 traceback, 记录计数后继续跑
            _infer_err[0] += 1
            msg = '%s: %s' % (type(e).__name__, e)
            if msg != _infer_err_msg[0] or _infer_err[0] <= 3:
                _infer_err_msg[0] = msg
                import traceback
                tb = traceback.format_exc().strip().splitlines()
                print('[推理线程异常] 第 %d 次(已捕获, 继续运行):' % _infer_err[0])
                # 只打最深 5 行: 含具体文件名+行号, 一张截图就能定位,
                # 打全栈会把有用信息顶出屏幕(踩过一次)
                for ln in tb[-5:]:
                    print('   ' + ln.strip())
            time.sleep(0.01)


def infer_once():
    """跑一帧推理。单独抽出来是为了让 infer_loop 能整体包 try/except ——
    推理线程必须绝对健壮: 它一死, 画面还在推(stream fps 正常)但 infer fps
    恒为 0 且再也回不来, 现象极具误导性。"""
    global infer_n, SINGLE_OUT, USE_CPP, ZOO_NC   # 首帧兜底会改写这些全局开关
    with _lock:
        f = _infer_in[0]
        _infer_in[0] = None
    if f is None:
        time.sleep(0.002)
        return
    _t = _pr('letterbox')
    img = co_helper.letter_box(im=f, new_shape=(640, 640),
                               pad_color=(0, 0, 0))
    _pr_end(_t)
    _t = _pr('cvt')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    _pr_end(_t)
    _t = _pr('npu')
    outputs = rknn.inference(inputs=[np.expand_dims(img, 0)])
    _pr_end(_t)
    _t = _pr('post')
    # 首帧兜底: 启动自检若失败, SINGLE_OUT 还是 None, 这里用真实输出补判一次。
    # 只在第一帧执行(判定后就锁死), 不增加每帧开销。
    if SINGLE_OUT is None:
        OUT_SHAPES = [list(np.asarray(o).shape) for o in outputs]
        SINGLE_OUT = (len(OUT_SHAPES) == 1 and len(OUT_SHAPES[0]) == 3
                      and 4 < OUT_SHAPES[0][1] < 32)
        if SINGLE_OUT and USE_CPP:
            USE_CPP = False        # C++ 解码器按 zoo 三输出格式写的, 拦下
            print('首帧自检: 单输出模型, C++ 后处理已关闭, 改用 python 解码')
        elif len(OUT_SHAPES) == 6:
            ZOO_NC = OUT_SHAPES[1][1]
            if USE_CPP and ZOO_NC != 80:
                USE_CPP = False    # C++ 按 80 类编译, cls 通道数对不上
                print('首帧自检: zoo nc=%d, C++ 后处理已关闭, 改用 python 解码'
                      % ZOO_NC)
        print('首帧自检 | 输出形状:', OUT_SHAPES,
              '| 后处理:', '单输出解码' if SINGLE_OUT
              else ('C++' if USE_CPP else 'zoo numpy'))
    if SINGLE_OUT:
        boxes, classes, scores = post_process_single(outputs, conf_thresh=args.conf)
    elif USE_CPP:
        boxes, classes, scores = post_process_cpp(outputs, conf_thresh=args.conf)
    else:
        boxes, classes, scores = post_process(outputs)
    _pr_end(_t)
    if boxes is not None:
        _t = _pr('track')
        rb = co_helper.get_real_box(boxes)      # 还原到原图坐标
        ids = None
        # ROI 过滤必须在跟踪之前: 反过来的话目标走到区域边界会被判 lost,
        # 出去再回来就是新 ID —— 同一个目标被数两次(和 10.30 的 ID 跳变同源)
        rb, scores, classes = apply_roi(rb, scores, classes)
        if tracker is not None:
            # 跟踪放在推理线程: 计数只需要 ID, 没必要每帧都重算。
            # 帧也一起传进去(rb 与 f 同一坐标系): tracker 要拿它抽颜色直方图,
            # 目标出画几秒再进来时靠这个认出老朋友。
            rb, scores, classes, ids = tracker.update(rb, scores, classes, f)
            if counter is not None and len(ids):
                counter.update(rb, classes, ids, tracker.frame_no)
        _pr_end(_t)
        with _lock:
            _now = time.time()
            if _det_ts[0] > 0:      # 推理间隔的滑动平均: 外推要拿它当"每轮多少秒"
                _det_dt[0] = np.float32(0.7) * _det_dt[0] \
                    + np.float32(0.3) * (_now - _det_ts[0])
            _det_ts[0] = _now
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
    """画计数线(支持直线/曲线折线) + 左上角 HUD(IN/OUT)。没开计数时不画线"""
    if cnt is None:
        return
    sc = max(1.0, frame.shape[1] / 1024.0)
    lw = max(2, int(round(2 * sc)))
    pts = cnt.polyline_px()
    if len(pts) >= 2:                       # 折线逐段画: 直线就是 2 个点的特例
        for i in range(len(pts) - 1):
            cv2.line(frame, pts[i], pts[i + 1], (0, 0, 255), lw)   # BGR 红
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
    # videoconvert 多线程默认关: GStreamer 1.14 多线程 Y 分片实测会花屏
    vc = 'videoconvert'
    if args.mt_vc and 'threads' in _elem_props('videoconvert'):
        vc += ' threads=%d' % max(1, min(4, os.cpu_count() or 1))
    enc = 'mpph264enc bps=%d gop=%d rc-mode=1' % (args.bps, args.gop)
    if args.hq:
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
        if web is not None:
            print('Web 可视化:')
            for dev in ('eth0', 'wlan0'):
                ip = dev_ip(dev)
                if ip:
                    print('  %s -> http://%s:%d' % (dev, ip, args.web_port))

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
_frame_no = 0                   # 全局帧序号(不受 5 秒统计窗口重置影响, 抽帧用)
n_push = 0                      # 实际推进 appsrc 的帧数(有客户端拉流才>0)
interval = 1.0 / args.fps      # 限速: 只按目标帧率取帧, 多余的直接丢
last_emit = 0.0
try:
    while True:
        # 网页请求切换分辨率/帧率 -> 整进程重启(os.execv):
        # 新档位已经在 stream.json 里, 重启后启动段会自动读回并应用。
        if RESTART_REQ[0] is not None:
            _c = RESTART_REQ[0]
            RESTART_REQ[0] = None
            print('[档位] 切换到 %dx%d@%d, 重启流水线(约 3~5 秒)...'
                  % (_c['width'], _c['height'], _c['fps']))
            time.sleep(1.0)             # 先让网页的 HTTP 响应送达浏览器
            # 关键坑: os.execv 是「同进程换映像」, fd 表原样继承。
            # OpenCV 打开的 V4L2 fd 没有 FD_CLOEXEC 标记, 会被原封不动
            # 带进「新进程」, /dev/video0 仍被占用 -> 摄像头打开失败。
            # 所以 exec 前把所有继承来的 fd(>2) 标记为执行时自动关闭。
            import fcntl
            for _fd in os.listdir('/proc/self/fd'):
                try:
                    _fd = int(_fd)
                    if _fd > 2:
                        fcntl.fcntl(_fd, fcntl.F_SETFD, fcntl.FD_CLOEXEC)
                except Exception:
                    pass
            os.execv(sys.executable, [sys.executable] + sys.argv)
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

        # 先统一坐标系: 采集原帧(可能 8MP)缩到推流输出尺寸, 之后的
        # 推理输入/检测框/跟踪/计数/画框/抓拍全在这个尺寸上做。
        # 8MP 档收益: 给推理线程的整帧 copy 从 24MB -> 8.3MB,
        # infer 线程 letterbox 的输入也小了 2.9 倍。
        _t = _pr('resize')
        if OUT_SCALE:
            frame = cv2.resize(frame, (OUT_W, OUT_H),
                               interpolation=cv2.INTER_AREA)
        _pr_end(_t)

        # 推理抽帧: 推流保持满帧, 但 NPU 每 N 帧才跑一次。非推理帧沿用上一次
        # 的检测框(tracker 负责 ID 连续), 所以画面上框会有 ~N/fps 秒的滞后,
        # 换来的是 NPU/CPU 负载降到 1/N —— 裸板长时间跑不烫、不降频的正解
        if rknn is not None and (_frame_no % INFER_STRIDE == 0):
            _t = _pr('copy')
            with _lock:
                if _infer_in[0] is None:
                    # copy 一份给推理线程(OUT 尺寸): 主线程马上要在 frame 上
                    # 画框, 不 copy 的话 infer 读到的就是「半张画了框的图」
                    _infer_in[0] = frame.copy()
                else:
                    # 上一帧还没推理完 -> 这一帧直接丢弃(宁可不提精度, 也不能
                    # 让队列越堆越多, 那才是"框完全跟不上"的真凶)
                    _skip = _bucket('skip')
                    _skip[0] += 1
            _pr_end(_t)

        _t = _pr('draw')
        with _lock:
            det = _det[0]
            det_ts = _det_ts[0]
            det_dt = _det_dt[0]
        if det is not None:
            # ==== TODO(你写): 帧间运动补偿 ================================
            # 症状: NPU 11fps / 推流 30fps, 两次推理之间框钉在上一次的坐标上,
            #       物体快时"框卡在原地, 物体走了它还在那"。
            # 已有的料(上面 with _lock 里已经取好):
            #   det      = (rb, scores, classes, ids)   rb 是 (N,4): x1,y1,x2,y2
            #   det_ts   = 这份推理结果诞生的时间戳 (time.time())
            #   det_dt   = 推理间隔的滑动平均, 单位 秒/轮 (11fps 时 ≈ 0.09)
            #   now      = 当前时刻 (循环开头取好的, 直接用)
            #   tracker.vel_of(ids) -> (N,4) 每条轨迹的速度, 单位是"每轮"的增量
            # 要做的事: 算出已流逝轮数 k, 把 rb 沿速度外推 k 轮, 画补偿后的框。
            #
            #   1) 流逝了多少秒?  流逝了多少"轮"?  (轮 = 秒 / det_dt)
            #   2) rb + 速度 × 轮数 = 预测框  (注意 (N,4) + (N,4) 逐元素加)
            #   3) 组装 det_draw = (rb_pred, scores, classes, ids)
            #   4) 最后 draw_boxes 画 det_draw, 不再画 det
            #
            # 边界(自己想清楚再写):
            #   - ids 是 None (没开跟踪) 或长度 0 -> 没速度可查, 不补偿
            #   - det 太陈旧时 k 会很大, 框会画飞 -> 给 k 封顶, 上限你定
            #   - 进阶(选做): det 诞生在"采集那帧"之后约一轮(推理本身耗时),
            #     想更准可以把轮数多加 1, 想想为什么
            #
            # rb_pred = ...
            # det_draw = ...
            # ==============================================================
            draw_boxes(frame, det, zh=(args.lang == 'zh'))
        draw_counter(frame, counter)
        if web is not None:
            draw_roi(frame)                     # 区域也画进推流画面里
        _pr_end(_t)

        # Web 可视化: 把「已画好框的这一帧」交给 HTTP 服务(只存引用)。
        # 真正的 JPEG 编码由 WebUI 按需做(有人看才编, 且不超过 --web-fps),
        # 没人访问时这里几乎零开销, 不影响推流主链路。
        if web is not None:
            web.update(frame)
            if det is not None and det[3] is not None:
                web_tracks = len(det[3])
            else:
                web_tracks = 0
            web.set_stats(in_=counter.n_in if counter is not None else 0,
                          out=counter.n_out if counter is not None else 0,
                          tracks=web_tracks,
                          snap=store.n_snap if store is not None else 0,
                          res='%dx%d' % (OUT_W, OUT_H),
                          post=('C++' if USE_CPP else
                                ('single' if SINGLE_OUT else 'numpy')))

        # 事件抓拍: 越线事件由推理线程产生, 主循环拿「已画好框的当前帧」落盘。
        # 只把帧的引用交给后台线程(零拷贝, 此帧已推给编码器不再被改), 主循环不等待。
        if store is not None and counter is not None:
            new_ev = counter.take_events()      # 取走即清空, events 不会再无限增长
            if new_ev:
                id2cls = {}
                if det is not None and det[3] is not None:
                    for i, tid in enumerate(det[3]):
                        id2cls[int(tid)] = (int(det[2][i]), float(det[1][i]))
                for ev in new_ev:
                    c, s = id2cls.get(int(ev['id']), (-1, 0.0))
                    nm = CLASSES_ZH[c] if 0 <= c < len(CLASSES_ZH) else str(c)
                    store.on_event(ev, frame, nm, s)

        # 推帧三条件: 有 appsrc + 媒体 PLAYING + 背压允许。
        # 1) 客户端切换期间 media 会短暂 PAUSE, 此时推帧会堆旧帧 -> 黑屏;
        # 2) 队列满(enough-data)时继续投喂 -> 延迟越拉越大 -> 十几秒延迟。
        appsrc = rtsp.appsrc
        if appsrc is not None:
            # 兜底: 万一 enough-data 后再没收到 need-data, 1 秒后强制恢复投喂
            if not rtsp.need_data and time.time() - rtsp.last_push > 1.0:
                rtsp.need_data = True
            if rtsp.need_data:
                ok, state, _pend = appsrc.get_state(0)
                if ok == Gst.StateChangeReturn.SUCCESS and state == Gst.State.PLAYING:
                    try:
                        _t = _pr('tobytes')
                        buf = Gst.Buffer.new_wrapped(frame.tobytes())
                        _pr_end(_t)
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
        _frame_no += 1
        # 每 5 秒打印一次实际帧率(低帧率下按帧数打印间隔太长)
        if now - t0 >= 5.0:
            msg = ('stream fps: %.1f | infer fps: %.1f | push %d'
                   % (n / (now - t0), infer_n / (now - t0), n_push))
            if counter is not None:
                msg += ' | %s' % counter.hud()
            if store is not None:
                msg += ' | snap %d' % store.n_snap
            if web is not None:
                # 帧率/延迟类指标每 5 秒刷新一次就够, 前端 1 秒轮询一次接口
                web.set_stats(stream=n / (now - t0), infer=infer_n / (now - t0),
                              lat_mean=float(np.mean(_lat_samples)) if _lat_samples else 0.0,
                              lat_p95=float(np.percentile(_lat_samples, 95)) if _lat_samples else 0.0)
            rep = prof_report()
            if rep:
                print('   [profile] ' + rep)
            if _lat_samples:
                # 单独一行: 拼在 fps 行后面会超出终端宽度被截断
                print('   板内延迟(cam.read -> 推给编码器) 均值 %.1f ms | P95 %.1f ms'
                      % (np.mean(_lat_samples), np.percentile(_lat_samples, 95)))
            elif n_push == 0:
                print('   (push=0: 没有客户端在拉流或媒体未 PLAYING, 板内延迟无从统计;'
                      ' 用 ffplay 连上再看)')
                del _lat_samples[:]
            if _infer_err[0]:
                print('   ⚠ 推理线程异常 %d 次(已捕获并继续): %s'
                      % (_infer_err[0], _infer_err_msg[0]))
                _infer_err[0] = 0
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
