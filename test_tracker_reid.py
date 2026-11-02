# -*- coding: utf-8 -*-
"""本地验证: 出画 -> 换个位置/换个类别回来, 还能不能复用原 ID。
本机没 cv2, 这里用 numpy 版"长相"替代(逻辑等价: 颜色分布 -> 归一化向量)。
板子上跑的是 tracker.appearance(), cv2 版。
跑法: python _t_reid.py
"""
import numpy as np
import tracker


def app_np(frame, box, bins=8):
    """numpy 版颜色直方图(替代 cv2 版, 只为本地验证逻辑)。"""
    if frame is None:
        return None
    h, w = frame.shape[:2]
    x1, y1 = int(max(0, box[0])), int(max(0, box[1]))
    x2, y2 = int(min(w, box[2])), int(min(h, box[3]))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    crop = frame[y1:y2, x1:x2].astype(np.float32)
    # 用 G/B 两个通道当"色相+饱和度"的替代
    gi = np.clip((crop[..., 1] / 32).astype(int), 0, bins - 1)
    bi = np.clip((crop[..., 0] / 32).astype(int), 0, bins - 1)
    hist = np.zeros((bins, bins), np.float32)
    np.add.at(hist, (gi.ravel(), bi.ravel()), 1.0)
    hist = hist.ravel()
    hist /= hist.sum()
    np.sqrt(hist, out=hist)
    hist /= np.linalg.norm(hist)
    return hist


tracker.appearance = app_np        # 猴子补丁: 只影响本次测试

H, W = 480, 640
# 第 1 帧: 绿瓶在左边; 第 2 帧: 绿瓶被挪到右边(同一只瓶子, 位置全变)
frame = np.full((H, W, 3), 200, np.uint8)
frame[300:400, 60:110] = (40, 90, 40)          # 深绿瓶子(要追踪的那个)
frame[300:400, 500:550] = (30, 30, 200)        # 红色干扰瓶
frame2 = np.full((H, W, 3), 200, np.uint8)
frame2[300:400, 500:550] = (40, 90, 40)        # 绿瓶挪到右边
frame2[80:180, 60:110] = (30, 30, 200)         # 红瓶被挪到别处

EMPTY = np.zeros((0, 4), np.float32)


def one(trk, box, cls, fr, sc=0.7):
    return trk.update(np.asarray([box], np.float32),
                      np.asarray([sc], np.float32),
                      np.asarray([cls], np.float32), fr)


trk = tracker.ByteTracker(track_thresh=0.45, max_lost=45, rebirth=150,
                          reid_thresh=0.62)
box = np.array([60, 300, 110, 400], np.float32)
ids = None
for _ in range(5):
    _, _, _, ids = one(trk, box, 39, frame)
first = int(ids[0])
print('第一次出现 -> ID', first)

for _ in range(50):          # 出画 50 帧(> max_lost=45): 轨迹删除, 进墓地
    trk.update(EMPTY, np.zeros(0, np.float32), np.zeros(0, np.float32), frame)

box2 = np.array([500, 300, 550, 400], np.float32)   # 挪到画面另一侧
_, _, _, ids2 = one(trk, box2, 57, frame2)          # 类别还被判成 57
print('出画再进来 -> ID', ids2, '(原 ID = %d)' % first)
print('结果:', 'PASS 复用原 ID' if len(ids2) and int(ids2[0]) == first
      else 'FAIL 换了新 ID')

# 对照: 不传图像(老行为, 纯几何) -> 认不出来是正常的
trk2 = tracker.ByteTracker(track_thresh=0.45, max_lost=45, rebirth=150)
for _ in range(5):
    trk2.update(np.asarray([box], np.float32), np.asarray([0.7], np.float32),
                np.asarray([39], np.float32))
for _ in range(50):
    trk2.update(EMPTY, np.zeros(0, np.float32), np.zeros(0, np.float32))
_, _, _, i2 = trk2.update(np.asarray([box2], np.float32),
                          np.asarray([0.7], np.float32),
                          np.asarray([57], np.float32))
print('对照(不传图像) -> ID', i2, '(换新 ID 才符合预期)')

c1 = app_np(frame, np.array([60, 300, 110, 400]))
c2 = app_np(frame, np.array([500, 300, 550, 400]))
print('长相相似度: 绿瓶自身 %.3f / 绿瓶-红瓶 %.3f'
      % (float(np.dot(c1, c1)), float(np.dot(c1, c2))))