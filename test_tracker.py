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
print('场景6: 人走近拢线——框中心始终不越线, 只有脚(框底边)越线  <== 实战最常见')
print('      模拟: 人走到线前站住——框中心最终 y=318(线在384, 始终没越线),')
print('            但框底边(脚点)到了 y=508(越过 384)')
trk = ByteTracker()
cnt_bottom = LineCounter(axis='h', pos=0.5, in_dir='down', cls_ids=(0,),
                         anchor='bottom')
cnt_center = LineCounter(axis='h', pos=0.5, in_dir='down', cls_ids=(0,),
                         anchor='center')
cnt_bottom.configure(W, H)
cnt_center.configure(W, H)
for i in range(12):
    y = 120 + i * 18                     # 框中心 120 -> 318 (线=384, 始终未越线)
    box = [[400, y - 190, 500, y + 190]]  # 框高 380, 底边 = y+190 (末帧 508, 已越线)
    ev_b, ev_c = [], []
    boxes = np.asarray(box, dtype=np.float32)
    b, s, c, ids = trk.update(boxes, np.array([0.85]), np.array([0.0]))
    if len(ids):
        ev_b = cnt_bottom.update(b, c, ids, trk.frame_no)
        ev_c = cnt_center.update(b, c, ids, trk.frame_no)
    print('  f%-3d 中心y=%-6.1f 底边y=%-6.1f  脚点判定:%-10s 中心判定:%-10s'
          % (trk.frame_no, y, y + 190,
             cnt_bottom.hud(), cnt_center.hud()))
print('  脚点(anchor=bottom) =', cnt_bottom.hud(), ' <- 期望 IN 1   OUT 0')
print('  中心(anchor=center) =', cnt_center.hud(), ' <- 说明旧行为会漏数')
assert cnt_bottom.n_in == 1, '场景6 脚点判定漏数: %s' % cnt_bottom.hud()

print('=' * 78)
print('场景7: 目标大部分在线下方, 只有**顶边**越线  <== 半身入镜 / 瓶子贴线的情形')
print('      模拟: 框顶 300->450(跨过线 384), 框高 300 -> 中心 450->600, 底边 600->750')
print('            中心和底边全程在线下方 -> 用它们判定永远为 0; 只有顶边能看见变化')
trk = ByteTracker()
anchors = {}
for a in ('top', 'center', 'bottom'):
    c = LineCounter(axis='h', pos=0.5, in_dir='down', cls_ids=(0,), anchor=a)
    c.configure(W, H)
    anchors[a] = c
for i in range(8):
    top = 300 + i * 22                    # 顶边 300 -> 454 (跨过 384)
    boxes = np.asarray([[400, top, 500, top + 300]], dtype=np.float32)
    b, s, c, ids = trk.update(boxes, np.array([0.85]), np.array([0.0]))
    if len(ids):
        for cc in anchors.values():
            cc.update(b, c, ids, trk.frame_no)
    print('  f%-3d 顶边y=%-6.1f 中心y=%-6.1f 底边y=%-6.1f  | top:%-10s center:%-10s bottom:%-10s'
          % (trk.frame_no, top, top + 150, top + 300,
             anchors['top'].hud(), anchors['center'].hud(),
             anchors['bottom'].hud()))
print('  top    =', anchors['top'].hud(), ' <- 期望 IN 1   OUT 0')
print('  center =', anchors['center'].hud(), ' <- 旧行为: 全程 0')
print('  bottom =', anchors['bottom'].hud(), ' <- 旧行为: 全程 0')
assert anchors['top'].n_in == 1, '场景7 top 判据漏数: %s' % anchors['top'].hud()
assert anchors['center'].n_in == 0, '场景7 center 不该计数'

print('=' * 78)
print('全部场景通过 ✅')
