#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""event_store.py — 事件抓拍 / SQLite 落库 / MQTT 上报

设计要点(面试可讲):
  1. **抓拍不阻塞主循环**: 主循环只把帧的引用塞进队列(零拷贝, 此时帧已推给
     编码器、不会再被修改), 落盘/建库/上报全部由后台线程做。若在 33ms 的推帧
     周期里做 JPEG 编码(1024x768 约 15ms), 每来一个事件就要掉一帧。
  2. **限流**: 同一 track_id 在 min_interval 秒内只抓一张, 防止一个人在计数线
     上反复横跳把磁盘写满。
  3. **磁盘保护**: max_files 上限 + 启动时按 keep_days 清理过期目录。
  4. **MQTT 可选**: 板端没装 paho-mqtt 也能跑, 只是不上报(import 失败自动降级)。

对外接口:
    store = EventStore(root='~/snapshots', mqtt_host='192.168.1.10')
    store.on_event(ev, frame, cls_name='人', score=0.87)   # 非阻塞
"""
import os
import time
import json
import queue
import shutil
import sqlite3
import datetime
import threading

__all__ = ['EventStore']

SCHEMA = """CREATE TABLE IF NOT EXISTS events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        TEXT,
    track_id  INTEGER,
    cls       TEXT,
    score     REAL,
    direction TEXT,
    is_in     INTEGER,
    n_in      INTEGER,
    n_out     INTEGER,
    box       TEXT,
    image     TEXT
)"""


class EventStore(object):
    def __init__(self, root='~/snapshots', mqtt_host=None, mqtt_port=1883,
                 mqtt_topic='rk3568/events', min_interval=2.0,
                 keep_days=7, max_files=3000, verbose=True):
        self.root = os.path.expanduser(root)
        os.makedirs(self.root, exist_ok=True)
        self.min_interval = float(min_interval)
        self.max_files = int(max_files)
        self.verbose = bool(verbose)
        self.n_snap = 0
        self._last = {}                       # track_id -> 上次抓拍时刻
        self._q = queue.Queue(maxsize=32)

        self.db = sqlite3.connect(os.path.join(self.root, 'events.db'),
                                  check_same_thread=False)
        self.db.execute(SCHEMA)
        self.db.execute('CREATE INDEX IF NOT EXISTS idx_ts ON events(ts)')
        self.db.commit()

        self.mqtt = self._init_mqtt(mqtt_host, mqtt_port, mqtt_topic)
        threading.Thread(target=self._worker, daemon=True).start()
        self._cleanup(keep_days)
        print('抓拍存储: %s | 限流 %.1fs/目标 | 上限 %d 张 | MQTT %s'
              % (self.root, self.min_interval, self.max_files,
                 'on -> %s' % mqtt_topic if self.mqtt else 'off'))

    # -- MQTT: 装不上就降级, 不影响主流程 ---------------------------------
    def _init_mqtt(self, host, port, topic):
        if not host:
            return None
        try:
            import paho.mqtt.client as mqtt
            # paho-mqtt 2.x 要求显式指定 callback API 版本, 1.x 没有这个参数
            try:
                c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
            except AttributeError:
                c = mqtt.Client()
            c.connect(host, int(port), 60)
            c.loop_start()
            print('MQTT 已连接: %s:%d' % (host, port))
            return (c, topic)
        except Exception as e:
            print('MQTT 未启用(仅本地落盘): %s' % e)
            return None

    # -- 清理过期目录 ----------------------------------------------------
    def _cleanup(self, keep_days):
        cut = (datetime.date.today()
               - datetime.timedelta(days=int(keep_days))).strftime('%Y%m%d')
        for name in sorted(os.listdir(self.root)):
            p = os.path.join(self.root, name)
            if os.path.isdir(p) and name < cut:
                shutil.rmtree(p, ignore_errors=True)
                print('  清理过期抓拍目录:', name)

    # -- 主入口: 非阻塞, 只入队 ------------------------------------------
    def on_event(self, ev, frame, cls_name='', score=0.0):
        tid = int(ev['id'])
        now = time.time()
        if now - self._last.get(tid, -1e9) < self.min_interval:
            return                            # 同一目标限流
        if self.n_snap >= self.max_files:
            return                            # 磁盘保护
        self._last[tid] = now
        self.n_snap += 1
        try:
            self._q.put_nowait((dict(ev), frame, cls_name, float(score)))
        except queue.Full:
            pass                              # 队列满就丢弃, 绝不能反压主循环

    # -- 后台线程: 落盘 + 建库 + 上报 -------------------------------------
    def _worker(self):
        import cv2
        while True:
            ev, frame, cls_name, score = self._q.get()
            ts = datetime.datetime.now()
            day = ts.strftime('%Y%m%d')
            d = os.path.join(self.root, day)
            try:
                os.makedirs(d, exist_ok=True)
                name = '%s_ID%d_%s.jpg' % ('IN' if ev.get('is_in') else 'OUT',
                                           int(ev['id']), ts.strftime('%H%M%S'))
                path = os.path.join(d, name)
                if not cv2.imwrite(path, frame):
                    continue
            except Exception as e:
                print('抓拍失败:', e)
                continue

            box = ev.get('box')
            box_s = '' if box is None else ','.join('%.0f' % v for v in box[:4])
            row = (ts.strftime('%Y-%m-%d %H:%M:%S'), int(ev['id']), cls_name,
                   score, ev.get('direction', ''), 1 if ev.get('is_in') else 0,
                   int(ev.get('n_in', 0)), int(ev.get('n_out', 0)), box_s, path)
            try:
                self.db.execute('INSERT INTO events (ts,track_id,cls,score,'
                                'direction,is_in,n_in,n_out,box,image) '
                                'VALUES (?,?,?,?,?,?,?,?,?,?)', row)
                self.db.commit()
            except Exception as e:
                print('落库失败:', e)

            if self.verbose:
                print('  [抓拍] %s ID%d %s -> %s'
                      % ('IN ' if ev.get('is_in') else 'OUT', int(ev['id']),
                         cls_name, os.path.relpath(path, self.root)))

            if self.mqtt is not None:
                c, topic = self.mqtt
                payload = {'ts': row[0], 'track_id': int(ev['id']),
                           'cls': cls_name, 'score': round(score, 3),
                           'direction': ev.get('direction', ''),
                           'in': int(ev.get('n_in', 0)),
                           'out': int(ev.get('n_out', 0)), 'image': path}
                try:
                    c.publish(topic, json.dumps(payload, ensure_ascii=False),
                              qos=0)
                except Exception:
                    pass

    def close(self):
        try:
            self.db.close()
        except Exception:
            pass
        if self.mqtt is not None:
            try:
                self.mqtt[0].loop_stop()
            except Exception:
                pass
