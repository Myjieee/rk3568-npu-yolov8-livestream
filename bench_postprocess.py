#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bench_postprocess.py — 板端实测：numpy 后处理 vs C++ 后处理

跑法（板子端）：
    python3 bench_postprocess.py              # 用摄像头抓一帧
    python3 bench_postprocess.py --source ~/bus.jpg --loop 100

输出三件事：
  1. NPU 纯推理耗时
  2. 两种后处理各自的耗时与理论帧率上限
  3. 两者结果是否一致（防止 C++ 版有 bug 悄悄改了检测框）
"""
import os
import sys
import time
import argparse
import numpy as np
import cv2

ZOO_PY = os.path.expanduser('~/rknn_model_zoo/examples/yolov8/python')
sys.path.insert(0, ZOO_PY)
try:
    from yolov8 import post_process
    from py_utils.coco_utils import COCO_test_helper
except ImportError as e:
    sys.exit('找不到官方例程模块: %s' % e)

ap = argparse.ArgumentParser()
ap.add_argument('--model', default=os.path.expanduser(
    '~/rknn_model_zoo/examples/yolov8/model/yolov8n.rknn'))
ap.add_argument('--source', default='/dev/video0',
                help='摄像头设备或图片路径')
ap.add_argument('--loop', type=int, default=50, help='每种后处理重复几次')
args = ap.parse_args()

try:
    from postprocess_cpp import post_process_cpp
except Exception as e:
    sys.exit('C++ 后处理不可用: %s\n'
             '先编译: g++ -O3 -ffast-math -shared -fPIC -o libpostprocess.so '
             'postprocess.cpp' % e)

# ---------- 取一帧 ----------
if os.path.exists(args.source):
    frame = cv2.imread(args.source)
    if frame is None:
        sys.exit('读图失败: %s' % args.source)
    print('输入: 图片 %s %s' % (args.source, frame.shape))
else:
    cap = cv2.VideoCapture(args.source, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
    if not cap.isOpened():
        sys.exit('摄像头打开失败: %s' % args.source)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit('取帧失败')
    print('输入: 摄像头 %s %s' % (args.source, frame.shape))

# ---------- 加载模型 ----------
from rknnlite.api import RKNNLite
print('加载模型:', args.model)
rknn = RKNNLite()
if rknn.load_rknn(args.model) != 0:
    sys.exit('load_rknn failed')
if rknn.init_runtime() != 0:
    sys.exit('init_runtime failed')

co_helper = COCO_test_helper(enable_letter_box=True)
img = co_helper.letter_box(im=frame.copy(), new_shape=(640, 640), pad_color=(0, 0, 0))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ---------- NPU 纯推理耗时 ----------
N_INFER = 20
t0 = time.perf_counter()
for _ in range(N_INFER):
    outputs = rknn.inference(inputs=[np.expand_dims(img, 0)])
infer_ms = (time.perf_counter() - t0) / N_INFER * 1000.0
print('输出张量:', [o.shape for o in outputs])
print()
print('NPU 纯推理:      %6.2f ms' % infer_ms)

# ---------- 两种后处理 ----------
N = args.loop
t0 = time.perf_counter()
for _ in range(N):
    a = post_process(outputs)
numpy_ms = (time.perf_counter() - t0) / N * 1000.0

t0 = time.perf_counter()
for _ in range(N):
    b = post_process_cpp(outputs)
cpp_ms = (time.perf_counter() - t0) / N * 1000.0

print('后处理 numpy:    %6.2f ms' % numpy_ms)
print('后处理 C++:      %6.2f ms' % cpp_ms)
print('加速比:          %6.1f x' % (numpy_ms / max(cpp_ms, 1e-6)))
print()
print('单帧总耗时 numpy: %6.2f ms -> 理论上限 %5.1f fps'
      % (infer_ms + numpy_ms, 1000.0 / (infer_ms + numpy_ms)))
print('单帧总耗时 C++:   %6.2f ms -> 理论上限 %5.1f fps'
      % (infer_ms + cpp_ms, 1000.0 / (infer_ms + cpp_ms)))

# ---------- 一致性校验 ----------
print()
if a[0] is None and b[0] is None:
    print('一致性: 两者都没检出目标（换个场景再测）')
elif a[0] is None or b[0] is None:
    print('一致性: ❌ 一方有结果一方没有 -> numpy=%s cpp=%s'
          % ('None' if a[0] is None else len(a[0]),
             'None' if b[0] is None else len(b[0])))
else:
    oa = np.argsort(-a[2])
    ob = np.argsort(-b[2])
    same_n = len(a[0]) == len(b[0])
    dc = int(np.abs(a[1][oa] - b[1][ob]).max()) if same_n else -1
    db = float(np.abs(a[0][oa] - b[0][ob]).max()) if same_n else -1
    ok = same_n and dc == 0 and db < 0.01
    print('检出框数: numpy=%d  C++=%d' % (len(a[0]), len(b[0])))
    print('坐标最大误差: %.5f 像素 | 类别最大差异: %d' % (db, dc))
    print('一致性: %s' % ('✅ 结果一致' if ok else '❌ 不一致，检查 C++ 实现'))

rknn.release()
