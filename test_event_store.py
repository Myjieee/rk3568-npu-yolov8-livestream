#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_event_store.py — 不接摄像头验证抓拍落库逻辑

本地没有 cv2 时用假模块顶替(只验证流程: 限流 / 落盘 / 建库), 板子上是真 JPEG。
"""
import os
import sys
import time
import types
import shutil
import sqlite3
import tempfile
import numpy as np

if 'cv2' not in sys.modules:
    try:
        import cv2                                     # 板子上直接用真 cv2
    except ImportError:
        fake = types.ModuleType('cv2')
        fake.imwrite = lambda path, img: open(path, 'wb').write(b'\xff\xd8\xff\xd9') or True
        sys.modules['cv2'] = fake
        print('[本地模式] cv2 缺失, 用假 imwrite 顶替(只验证流程)')

from event_store import EventStore

ROOT = tempfile.mkdtemp(prefix='snap_test_')
print('临时目录:', ROOT)


def make_ev(tid, is_in, n_in, n_out):
    return {'id': tid, 'direction': 'down' if is_in else 'up',
            'is_in': is_in, 'n_in': n_in, 'n_out': n_out,
            'box': np.array([100., 200., 300., 400.], dtype=np.float32),
            'frame': 1}


frame = np.zeros((768, 1024, 3), dtype=np.uint8)

print('=' * 74)
print('场景1: 同一目标 2 秒内连触发 5 次 -> 限流只应抓 1 张')
st = EventStore(root=ROOT, min_interval=2.0, max_files=3000, verbose=False)
for i in range(5):
    st.on_event(make_ev(1, True, i + 1, 0), frame, '人', 0.87)
time.sleep(1.2)
day = time.strftime('%Y%m%d')
d = os.path.join(ROOT, day)
n1 = len([f for f in os.listdir(d) if f.endswith('.jpg')]) if os.path.isdir(d) else 0
print('  实际抓拍张数 =', n1, '  (期望 1)')
assert n1 == 1, '限流失效, 抓了 %d 张' % n1

print('=' * 74)
print('场景2: 3 个不同目标各触发 1 次 -> 应抓 3 张, 文件名带 ID')
for tid in (2, 3, 4):
    st.on_event(make_ev(tid, True, tid, 0), frame, '人', 0.9)
time.sleep(1.2)
jpgs = sorted(f for f in os.listdir(d) if f.endswith('.jpg'))
print('  抓拍文件:', jpgs)
assert len(jpgs) == 4, '应为 4 张(1 张来自场景1), 实际 %d' % len(jpgs)
assert any('ID2' in f for f in jpgs), '文件名缺 track id'

print('=' * 74)
print('场景3: 反向穿越应标记为 OUT')
st.on_event(make_ev(9, False, 4, 1), frame, '人', 0.8)
time.sleep(1.0)
outs = [f for f in os.listdir(d) if f.startswith('OUT')]
print('  OUT 文件:', outs)
assert len(outs) == 1, 'OUT 命名错误'

print('=' * 74)
print('场景4: SQLite 记录内容正确')
st.db.commit()
rows = st.db.execute('SELECT ts,track_id,cls,score,direction,is_in,n_in,n_out,box '
                     'FROM events ORDER BY id').fetchall()
for r in rows:
    print('  ', r)
assert len(rows) == 5, '库里应有 5 条, 实际 %d' % len(rows)
assert rows[0][1] == 1 and rows[0][2] == '人', '字段写入错误'
assert rows[0][5] == 1, 'is_in 应为 1'
assert rows[-1][5] == 0, '最后一条应为 OUT(is_in=0)'
print('  box 字段 =', repr(rows[0][8]))

print('=' * 74)
print('场景5: 磁盘保护 —— max_files=2 时不再继续抓')
st2 = EventStore(root=os.path.join(ROOT, 'cap'), max_files=2, min_interval=0.0,
                 verbose=False)
for tid in (11, 12, 13, 14):
    st2.on_event(make_ev(tid, True, tid, 0), frame, '人', 0.9)
time.sleep(1.2)
d2 = os.path.join(ROOT, 'cap', day)
n2 = len([f for f in os.listdir(d2) if f.endswith('.jpg')]) if os.path.isdir(d2) else 0
print('  max_files=2, 触发 4 次 -> 实际抓', n2, '张  (期望 2)')
assert n2 == 2, '磁盘保护失效'

print('=' * 74)
print('场景6: MQTT 未配置时不能报错(板端没装 paho 也要能跑)')
print('  已隐含验证: 全程 mqtt_host=None, 无异常抛出 ✅')

st.close()
st2.close()
shutil.rmtree(ROOT, ignore_errors=True)
print('=' * 74)
print('全部场景通过 ✅   (临时目录已清理)')
