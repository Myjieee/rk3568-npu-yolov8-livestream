#!/usr/bin/env python3
# 板端冒烟测试：绕过摄像头和 rtsp_stream.py，直接喂一张图给 rknn 模型
# 用法: python3 test_img.py [模型路径] [图片路径]
import sys
import cv2
import numpy as np
from rknnlite.api import RKNNLite

MODEL = sys.argv[1] if len(sys.argv) > 1 else '/home/cat/rknn_model_zoo/examples/yolov8/model/yolov8n.rknn'
IMG = sys.argv[2] if len(sys.argv) > 2 else '/home/cat/004.jpg'


def letterbox(img, new=640):
    """和 ultralytics 一致的 letterbox：等比缩放 + 灰边填充"""
    h, w = img.shape[:2]
    r = min(new / h, new / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((new, new, 3), 114, np.uint8)
    top, left = (new - nh) // 2, (new - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, left, top


rknn = RKNNLite()
print('== 加载模型:', MODEL)
ret = rknn.load_rknn(MODEL)
print('load_rknn ret =', ret)
ret = rknn.init_runtime()
print('init_runtime ret =', ret)
if ret != 0:
    sys.exit(1)

img = cv2.imread(IMG)
if img is None:
    print('图片读取失败:', IMG)
    sys.exit(1)
print('== 原图尺寸:', img.shape)

lb, r, dx, dy = letterbox(img)
lb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)  # cv2 是 BGR，模型要 RGB
inp = np.expand_dims(lb, 0)  # (1,640,640,3) NHWC，0~255，与 config(mean=0,std=255) 匹配
outputs = rknn.inference(inputs=[inp])

print('== 输出数量:', len(outputs))
for i, o in enumerate(outputs):
    print(f'输出[{i}] shape={o.shape} dtype={o.dtype}')

out = np.array(outputs[0])
print('== 输出[0] shape =', out.shape)

# 统一整理成 (8400, 5)
if out.ndim == 3 and out.shape[1] == 5 and out.shape[2] != 5:
    dets = out[0].T          # (1,5,8400) -> (8400,5)
elif out.ndim == 3 and out.shape[2] == 5:
    dets = out[0]            # (1,8400,5)
else:
    print('!! 未知输出格式，无法解析')
    sys.exit(1)

print('== 各通道最大值 (x, y, w, h, conf):')
for c in range(dets.shape[1]):
    print(f'  通道{c} max = {dets[:, c].max():.4f}')

confs = dets[:, 4]
print('== conf 最大值:', confs.max())
idx = np.where(confs > 0.25)[0]
print('== conf>0.25 候选数:', len(idx))
top5 = np.argsort(confs)[::-1][:5]
for i in top5:
    x, y, w_, h_, c = dets[i]
    print(f'  conf={c:.4f}  box(cx,cy,w,h)=({x:.1f},{y:.1f},{w_:.1f},{h_:.1f})')

rknn.release()
print('== 测试完成')
