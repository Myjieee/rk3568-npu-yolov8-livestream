#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_tracker.py — 不接摄像头、不接 NPU, 用假框验证 tracker 逻辑

板子不在线时也能自测。三个场景:
  1. 单人从上往下穿越中线 -> IN 应该 +1, 且 ID 全程不变
  2. 两人并行同向走 -> 各自 ID 稳定, 不互换, IN 应该 +2
  3. 目标在计数线附近来回抖动 -> 不能重复计数(margin + cooldown 生效)
"""
import numpy as np
from tracker import ByteTracker, LineCounter

W, H = 1024, 768


def new():
    trk = ByteTracker()
    cnt = LineCounter(axis='h', pos=0.5, in_dir='down', cls_ids=(0,))
    cnt.configure(W, H)
    return trk, cnt


def step(trk, cnt, boxes, scores, classes, tag=''):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    classes = np.asarray(classes, dtype=np.float32).reshape(-1)
    b, s, c, ids = trk.update(boxes, scores, classes)
    ev = cnt.update(b, c, ids, trk.frame_no)
    if tag or ev:
        print('  f%-3d %-28s ids=%-12s %s'
              % (trk.frame_no, tag, list(ids),
                 ['%s#%d(%s)' % ('IN' if e['is_in'] else 'OUT',
                                 e['id'], e['direction']) for e in ev]))
    return b, s, c, ids, ev


print('=' * 78)
print('场景1: 单人从上往下穿越中线(y=80 -> 745, 每帧 +35)')
trk, cnt = new()
for i in range(20):
    y = 80 + i * 35
    step(trk, cnt, [[400, y - 40, 500, y + 40]], [0.85], [0],
         tag='y=%d' % y)
print('  结果 HUD =', cnt.hud(), '   (期望 IN 1   OUT 0)')
assert cnt.n_in == 1 and cnt.n_out == 0, '场景1 计数错误'

print('=' * 78)
print('场景2: 两人并行同向下走(A 在 x=300, B 在 x=700)')
trk, cnt = new()
seen_a, seen_b = set(), set()
for i in range(20):
    y = 80 + i * 35
    b, s, c, ids, ev = step(
        trk, cnt,
        [[260, y - 40, 360, y + 40], [660, y - 40, 760, y + 40]],
        [0.9, 0.88], [0, 0], tag='y=%d' % y)
    if len(ids) == 2:
        # 按 x 中心区分左右两人, 检查各自 ID 是否稳定
        cx = (b[:, 0] + b[:, 2]) * 0.5
        order = np.argsort(cx)
        seen_a.add(int(ids[order[0]]))
        seen_b.add(int(ids[order[1]]))
print('  左侧目标用过的 ID =', seen_a, '  右侧目标用过的 ID =', seen_b)
print('  结果 HUD =', cnt.hud(), '   (期望 IN 2   OUT 0)')
assert len(seen_a) == 1 and len(seen_b) == 1, '场景2 ID 发生跳变'
assert cnt.n_in == 2, '场景2 计数错误'

print('=' * 78)
print('场景3: 目标缓慢接近计数线, 线附近小幅抖动后穿过(线 y=384)')
trk, cnt = new()
# 真实抖动是几像素级的, 不是瞬移; 380/388/382/390 都在缓冲带±11.5附近晃
ys = [200, 260, 320, 360, 380, 388, 382, 390, 386, 400, 440, 520, 600]
for y in ys:
    step(trk, cnt, [[400, y - 40, 500, y + 40]], [0.85], [0], tag='y=%d' % y)
print('  结果 HUD =', cnt.hud(), '   (期望 IN 1 OUT 0: 抖动不该重复计数)')
assert cnt.n_in == 1 and cnt.n_out == 0, '场景3 计数错误: %s' % cnt.hud()

print('=' * 78)
print('场景5: 快速运动(每帧下移 90 像素, 两帧框几乎不重叠) -> ID 必须稳定')
trk, cnt = new()
ids_seen = set()
for i in range(12):
    y = 100 + i * 90
    b, s, c, ids, ev = step(trk, cnt, [[400, y - 40, 500, y + 40]], [0.85], [0],
                            tag='y=%d' % y)
    ids_seen |= set(int(x) for x in ids)
print('  全程出现过的 ID =', ids_seen, '  (应为单一 ID)')
print('  结果 HUD =', cnt.hud(), '   (期望 IN 1   OUT 0)')
assert len(ids_seen) == 1, '场景5 快速运动导致 ID 断裂: %s' % ids_seen
assert cnt.n_in == 1, '场景5 计数错误: %s' % cnt.hud()

print('=' * 78)
print('场景4: 置信度掉到 0.2(遮挡/模糊), 验证 ID 不被打断')
trk, cnt = new()
ids_seen = set()
for i in range(16):
    y = 100 + i * 40
    sc = 0.85 if i % 4 != 2 else 0.20      # 每 4 帧掉一次低分
    b, s, c, ids, ev = step(trk, cnt, [[400, y - 40, 500, y + 40]], [sc], [0],
                            tag='y=%d sc=%.2f' % (y, sc))
    ids_seen |= set(int(x) for x in ids)
print('  全程出现过的 ID =', ids_seen, '  (应为单一 ID, 说明低分二次匹配生效)')
print('  结果 HUD =', cnt.hud(), '   (期望 IN 1   OUT 0)')
assert len(ids_seen) == 1, '场景4 ID 被打断: %s' % ids_seen

print('=' * 78)
print('全部场景通过 ✅')
