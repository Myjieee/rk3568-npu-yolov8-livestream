"""
三个线程 + 三个信箱 —— 极简 demo（对应三道题）

只依赖 numpy, Windows / Linux / 板子都能跑:
    python3 demo_three_threads.py

三道题在代码里的位置, 搜 "第1题" "第2题" "第3题" 就能找到。

布局:
    采集线程 ──(图)──> _latest ──> 主循环 ──(图的副本)──> _infer_in ──> 推理线程
                                     ^                                      │
                                     └──────────── _det (只有数字) ─────────┘
"""
import threading
import time

import numpy as np

# ============================ 三个信箱 ============================
# 为什么每个都是 [None] 而不是 None?
#   Python 里, 线程函数内给普通变量赋值改不到外面(会变成局部变量)。
#   用列表包一层, 改 _xx[0] 就能穿透到全局。这是最省事的写法。
_latest = [None]      # 采集 -> 主循环: 只留最新一帧, 新的直接覆盖旧的
_infer_in = [None]    # 主循环 -> 推理: 只能装一张, 满了就扔(第2题)
_det = [None]         # 推理 -> 主循环: 取了不清空, 所以永远有上一次的框
_lock = threading.Lock()
_skip = [0]           # 丢帧计数

RUN = [True]
H, W = 48, 64
MARK = (0, 0)         # 实验用: 主循环"画框"时会把这个像素涂成 255
COPY = [True]         # 第1题开关: True=正常拍照, False=直接把原件递过去
INFER_STRIDE = 2      # 每 2 帧送一张给推理(模拟推流 30fps / 推理 14fps)

_dirty = [0]          # 推理读到"被涂改过的图"的次数
_infer_done = [0]


# ============================ 线程 1: 采集 ============================
def capture_loop():
    """摄像头: 每 33ms 一帧。它不管别人要不要, 只管往信箱里塞最新的一帧。"""
    while RUN[0]:
        img = np.full((H, W), 30, np.uint8)
        with _lock:
            _latest[0] = img        # 覆盖: 上一帧没被取走就直接扔掉
        time.sleep(1.0 / 30.0)


# ============================ 线程 2: 主循环 ============================
def main_loop():
    """推流 + 画框的老大。它永远不等推理。"""
    n = 0
    while RUN[0]:
        # ① 取帧: 拿走就清空信箱
        with _lock:
            frame = _latest[0]
            _latest[0] = None
        if frame is None:
            time.sleep(0.002)
            continue
        n += 1

        # ② 投帧给推理(第1题 + 第2题 + 第3题全在这一段)
        if n % INFER_STRIDE == 0:
            with _lock:                          # <-- 第3题: 锁
                if _infer_in[0] is None:         #     第1步: 看一眼
                    # 第1题: 为什么必须 copy?
                    #   主循环紧接着就要在这张图上画框/写字, 推理要 70ms 后才读。
                    #   不 copy 的话推理读到的是"已经画了框的脏图"。
                    _infer_in[0] = frame.copy() if COPY[0] else frame
                else:
                    # 第2题: 为什么丢帧不排队?
                    #   生产 30fps > 消费 14fps, 排队每秒净积 16 帧,
                    #   延迟和内存一路涨到爆。宁可少算几帧, 也要"永远最新"。
                    _skip[0] += 1

        # ③ 取结果: 看一眼有没有(注意: 取了不清空!)
        with _lock:
            det = _det[0]

        # ④ 画框: det 可能是上一次的结果, 甚至第一次是 None —— 都不耽误推流
        frame[MARK] = 255           # 用"涂一个像素"模拟画框写字
        if det is not None:
            pass                    # 真实代码里这里画矩形 + ID

        # ⑤ 推流(这里省略)
        time.sleep(1.0 / 30.0)


# ============================ 线程 3: 推理 ============================
def infer_loop():
    """NPU: 慢, 一帧 70ms(约 14fps)。它只回传数字, 不回传图片。"""
    while RUN[0]:
        with _lock:                 # <-- 第3题: 取 + 清空也是两步, 也要锁
            f = _infer_in[0]
            _infer_in[0] = None     # 清空 = 写操作, 所以两边都在写同一个槽
        if f is None:
            time.sleep(0.002)
            continue

        time.sleep(0.070)           # 模拟 NPU 跑一帧

        # 第1题的验证点: 如果主循环没 copy, 这里读到的 MARK 已经被涂成 255 了
        if f[MARK] == 255:
            _dirty[0] += 1
        _infer_done[0] += 1

        # 只回传数字: 框坐标 + 分数 + 类别 + ID(真实场景是 4 个 numpy 数组)
        fake_boxes = np.array([[10, 10, 30, 30]], np.float32)
        with _lock:
            _det[0] = (fake_boxes,)


# ==================== 第3题单独验证: 不加锁会发生什么 ====================
def race_demo(use_lock):
    """两个人同时"看一眼再动手", 中间人为插一个 0.5ms 的空隙放大竞态。

    槽里只能放一张。正确结果 = 线程数 x 每个线程的次数。
    不加锁 -> 一人刚放进槽, 另一人已经看完了"是空的"也往里放, 互相覆盖,
              实际干成的次数对不上(既可能重复干, 也可能活被吞掉)。
    加锁   -> 严格等于, 一次不差。
    """
    slot = [None]
    done = [0]
    TOTAL = 200          # 每个线程尝试 200 次
    NTHREAD = 2

    def worker():
        for _ in range(TOTAL):
            if use_lock:
                with _lock:
                    if slot[0] is None:
                        slot[0] = 1
                        time.sleep(0.0005)
                        done[0] += 1
                        slot[0] = None
            else:
                if slot[0] is None:          # 第1步: 看
                    slot[0] = 1
                    time.sleep(0.0005)       # 中间被切走的概率被放大
                    done[0] += 1             # 第2步: 动手
                    slot[0] = None
            time.sleep(0.0005)

    ts = [threading.Thread(target=worker) for _ in range(NTHREAD)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return done[0], TOTAL * NTHREAD


# ============================ 跑起来 ============================
def main():
    print('=' * 62)
    print('第1题 + 第2题: 跑 2 秒真流水线')
    print('=' * 62)
    ts = [threading.Thread(target=capture_loop, daemon=True),
          threading.Thread(target=main_loop, daemon=True),
          threading.Thread(target=infer_loop, daemon=True)]
    for t in ts:
        t.start()

    time.sleep(1.0)
    d0 = _dirty[0]           # 前半段(开着 copy)的脏图次数
    COPY[0] = False          # 后半段故意不 copy, 看会发生什么
    time.sleep(1.0)
    RUN[0] = False
    time.sleep(0.3)
    d1 = _dirty[0] - d0      # 后半段(关掉 copy)的脏图次数

    print('推理完成帧数        : %d' % _infer_done[0])
    print('丢掉的帧数(槽满)    : %d   <- 第2题: 不排队, 满了就扔' % _skip[0])
    print('第1题 开着 copy 时读到脏图: %d 次' % d0)
    print('第1题 关掉 copy 后读到脏图: %d 次   <- 立刻暴涨, copy 不可省' % d1)

    print()
    print('=' * 62)
    print('第3题: 不加锁 vs 加锁 (两个人抢一个只能装一张的槽)')
    print('=' * 62)
    bad, total = race_demo(use_lock=False)
    good, _ = race_demo(use_lock=True)
    print('两个人各干 200 次活, 正确结果 = %d 次' % total)
    print('不加锁: 只干成 %d 次   <- 少了 %d 次: 一人刚放进槽, 另一人看见'
          '"有人"就跳过 -> 活被吞了' % (bad, total - bad))
    print('加  锁: 干成 %d 次     <- 严格等于, 一次不差' % good)
    print()
    print('结论: 锁锁的是「看一眼 + 动手」这整套动作, 不是锁那个变量本身。')


if __name__ == '__main__':
    main()
