#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_rtsp_stream.py —— rtsp_stream.py 的「骨架版」，只用来读懂流程，不是拿来跑的。

【一句话讲清整个项目】
摄像头不停拍照片 -> 芯片(NPU)看照片里有什么东西 -> 在照片上画框和编号
-> 把画好的照片压成视频 -> 通过网络发出去 -> 手机/电脑上的播放器能看。

【为什么要用三条线程(三个"工人")】
如果一个人干所有活：拍一张 -> 想半天(AI 慢) -> 再拍下一张，画面就卡成幻灯片。
所以分成三个人同时干：
    工人A(采集)：只管不停拍照片，拍完往桌上一放就走，继续拍下一张。
    工人B(推理)：桌上有新照片就拿去给 AI 看，看完把"有什么东西、在哪"写纸上。
    工人C(主循环)：拿最新的照片，照着工人B写的纸画框，然后交给快递(编码器)发走。
三个人各干各的，谁慢都不拖死别人 —— 这就是"流水线"。

【为什么 AI 慢也不会让画面卡】
工人B 看一张要 68ms，工人A 每秒拍 30 张。B 跟不上就跳过几张，
C 照样每秒发 30 张画面(框是上一帧的结果，人眼看不出来)。
所以画面永远是流畅的 30fps，AI 只有 14fps —— 这个数字差是设计出来的，不是 bug。
"""

import time
import threading
import argparse

import cv2
import numpy as np
import gi

gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GLib

# ==========================================================================
# 第 0 步：看命令行给了什么参数(比如分辨率、端口、要不要开网页)
# ==========================================================================
# 类比：开工前先看一眼订单要求 —— 客户要多大的图？送到哪个端口？
ap = argparse.ArgumentParser()
ap.add_argument('--width',  type=int, default=1024)   # 画面宽
ap.add_argument('--height', type=int, default=768)    # 画面高
ap.add_argument('--fps',    type=int, default=30)     # 每秒几张
ap.add_argument('--port',   type=int, default=8554)   # RTSP 端口号
args = ap.parse_args()

W, H, FPS = args.width, args.height, args.fps

Gst.init(None)

# ==========================================================================
# 第 1 步：打开摄像头
# ==========================================================================
# 类比：把摄像头当成一台不停拍照的相机，我们只管从里面"取下一张"。
cap = cv2.VideoCapture('/dev/video0')
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
print('摄像头打开：%dx%d' % (W, H))

# ==========================================================================
# 第 2 步：把 AI 模型交给 NPU(芯片里专门跑 AI 的那块硬件)
# ==========================================================================
# 类比：NPU 是厂里唯一一台会"认东西"的机器。它只吃 640x640 的方形图，
# 所以不管你采集多大，喂给它之前都要先"缩一缩、补个边"(这叫 letterbox)。
from rknnlite.api import RKNNLite
rknn = RKNNLite()
rknn.load_rknn('./yolov8n.rknn')     # 把模型文件读进来
rknn.init_runtime()                  # 交给 NPU 准备开工
print('NPU 就绪')

# ==========================================================================
# 第 3 步：搭一条"快递通道"(GStreamer)，把画好的帧压成 H.264 再发到网上
# ==========================================================================
# 这一串字符串就是"流水线图纸"，从左到右一个接一个：
#   appsrc        = 入口，我们往里塞画好的图
#   videoconvert  = 颜色格式转换(YUV/RGB 互转)
#   mpph264enc    = 硬编：用芯片里的 VPU 把图压成 H.264(不占 CPU)
#   rtph264pay    = 打包：H.264 装进网络小包裹(RTP)
#   udpsink       = 发货：把包裹扔到网络上
# 真实项目里这一段有 60 多行(要判断硬件支持哪些参数)，这里全简化。
LAUNCH = (
    '( appsrc name=src is-live=true do-timestamp=true format=time block=false '
    '  caps=video/x-raw,format=BGR,width=%d,height=%d,framerate=%d/1 ! '
    '  videoconvert ! video/x-raw,format=NV12 ! '
    '  mpph264enc ! rtph264pay name=pay0 )' % (W, H, FPS)
)


class RtspServer(object):
    """一个超简版的 RTSP 服务器：有人来拉流，就按上面的图纸搭一条管道。"""

    def __init__(self, launch, port):
        self.launch = launch
        self.appsrc = None                    # 入口，等客户端来了才能拿到
        server = GstRtspServer.RTSPServer()
        server.set_service(str(port))
        factory = GstRtspServer.RTSPMediaFactory()
        factory.set_launch(launch)
        factory.connect('media-configure', self.on_connect)  # 有人连上时回调
        server.get_mount_points().add_factory('/test', factory)
        server.attach(None)
        print('RTSP 服务已启动，播放地址：rtsp://<板子IP>:%d/test' % port)

    def on_connect(self, factory, media):
        """客户端接入：从管道里找到入口 appsrc，之后就能往里塞图了。"""
        pipe = media.get_element().get_by_name('bin') or media.get_element()
        self.appsrc = media.get_element().get_by_name('src')
        print('有人来拉流了，appsrc 就绪')


rtsp = RtspServer(LAUNCH, args.port)

# ==========================================================================
# 第 4 步：三个"共享的信箱"
# ==========================================================================
# 线程之间不能直接喊话，只能靠"往同一个全局变量里放东西"。
# 为什么每个都是只装一个元素的列表([None] 而不是 None)？
#   因为 Python 里给普通变量赋值(= xxx)在函数内不会改动外面的变量，
#   但改列表里的内容(_latest[0] = f)可以。这是多线程传值最省事的办法。
# 真实项目里这些都要加锁(threading.Lock)，这里为了好读先省略。
_latest   = [None]     # 工人A 放的：最新拍到的原图
_infer_in = [None]     # 工人C 放的：交给 AI 的那张图(拷贝出来的一份)
_det      = [None]     # 工人B 放的：AI 看完的结果(框在哪、是什么、分数)
_stop     = False      # 让三个工人下班(Ctrl+C)


def capture_loop():
    """工人A：只管拍照片，拍完往桌上一放，立刻去拍下一张。"""
    while not _stop:
        ok, frame = cap.read()
        if ok:
            _latest[0] = frame          # 桌上有新照片了(旧的会被盖掉，不积压)


def infer_once():
    """工人B 的一轮活：拿一张图给 AI 看，把结果写进 _det。"""
    frame = _infer_in[0]
    _infer_in[0] = None                # 拿走就清空，表示"这张我看过了"
    if frame is None:
        time.sleep(0.002)
        return

    # ① 把图缩成 NPU 要的 640x640 方形(缺的地方补黑边)
    small = cv2.resize(frame, (640, 640))
    small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    # ② 交给 NPU(这一步最慢，实测约 54ms，硬件决定的，优化不掉)
    outputs = rknn.inference(inputs=[np.expand_dims(small, 0)])

    # ③ 后处理：NPU 吐出来的是一堆数字，要翻译成"几个框 + 类别 + 分数"
    #    真实项目里这一步用 C++ 写的(5ms)，比 numpy 版(40ms)快 8 倍
    boxes, classes, scores = postprocess(outputs)

    # ④ 跟踪：给每个框编个号(ID)，让同一个物体前后帧是同一个号
    #    真实项目里这是 ByteTrack，demo 里直接编号 1、2、3...
    ids = list(range(1, len(boxes) + 1))

    _det[0] = (boxes, scores, classes, ids)


def infer_loop():
    """工人B：不停地问"桌上有我要看的图吗"，有就看，没有就歇 2 毫秒再看。"""
    while not _stop:
        infer_once()


def postprocess(outputs):
    """假的后处理：真实版本在 postprocess.cpp 里(dfl 解码 + 阈值过滤 + NMS)。

    输入是 NPU 吐的 8400 个格子的猜测(约 70 万个数字)，
    输出是几个干净的框。demo 里直接返回空，让你看清流程就行。
    """
    return [], [], []


def draw_boxes(frame, det):
    """在画面上画框和编号。注意：画的是"原始照片"，还没压缩。"""
    boxes, scores, classes, ids = det
    for box, tid in zip(boxes, ids):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)   # 绿框
        cv2.putText(frame, 'ID%d' % tid, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


# ==========================================================================
# 第 5 步：主循环(工人C)—— 整个程序的灵魂，只有十几行
# ==========================================================================
def main_loop():
    while not _stop:
        # ① 取最新一帧(没有就等 2 毫秒再来)
        frame = _latest[0]
        _latest[0] = None
        if frame is None:
            time.sleep(0.002)
            continue

        # ② 给 AI 留一份拷贝(因为下一步我要在原图上画框，
        #    不拷贝的话 AI 看到的就是"画了一半框的图")
        if _infer_in[0] is None:
            _infer_in[0] = frame.copy()

        # ③ 看一眼 AI 最新的结果，有就画上去
        det = _det[0]
        if det is not None:
            draw_boxes(frame, det)

        # ④ 有人在看网页的话，把这张画好的图也给它一份
        #    (真实项目：WebUI 只在有人访问时才压 JPEG，没人看就是零开销)
        # web.update(frame)

        # ⑤ 交给快递：把图塞进 appsrc，后面的硬编+打包+发货全自动
        if rtsp.appsrc is not None:
            buf = Gst.Buffer.new_wrapped(frame.tobytes())
            rtsp.appsrc.emit('push-buffer', buf)

        time.sleep(1.0 / FPS)      # 控制节奏，别塞太快


# ==========================================================================
# 第 6 步：开工 —— 启动三个工人，然后等他们干完(其实是等到 Ctrl+C)
# ==========================================================================
if __name__ == '__main__':
    t1 = threading.Thread(target=capture_loop, daemon=True)   # 工人A
    t2 = threading.Thread(target=infer_loop,   daemon=True)   # 工人B
    t1.start()
    t2.start()
    print('开始推流，Ctrl+C 停止')
    try:
        main_loop()                                           # 工人C 就是主线程自己
    except KeyboardInterrupt:
        _stop = True
        time.sleep(0.3)
        print('已停止')

# ==========================================================================
# 【读完之后你要能讲清楚的 5 件事】
# 1. 为什么要三条线程？        —— AI 慢，不能让它拖住"拍"和"发"
# 2. 为什么画面 30fps、AI 只有 14fps 却看不出卡？
#                              —— 画框用的是上一帧结果，画面照发
# 3. 为什么给 AI 的图要 copy？  —— 主线程马上要在原图上画框
# 4. 编码在哪发生？            —— 画完框之后，交给 mpph264enc(芯片硬编，不吃 CPU)
# 5. 网页画面是哪来的？        —— 就是主循环画好框的同一帧，按需压 JPEG
# ==========================================================================
