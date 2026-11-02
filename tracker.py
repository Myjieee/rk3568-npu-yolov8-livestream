#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tracker.py — 零依赖(仅 numpy)的轻量 ByteTrack + 越线计数

为什么自己写而不用官方 ByteTrack:
  - 官方实现依赖 filterpy(Kalman) + cython 编译的 lapjv/cython_bbox, 板端
    Debian12 aarch64 装不上(没外网、无 gcc 交叉编译环境), 也不值得为 ≤30 个目标
    引入这么多依赖;
  - 我们的场景(YOLOv8n + RK3568, 检测约 10fps, 目标数 ≤ 30)用「匀速外推 + EMA 平滑
    + 贪心 IoU 匹配」就够, 与匈牙利解结果几乎一致。

设计要点(面试可讲):
  1. 两阶段关联(ByteTrack 核心): 高分框(≥0.45)先与已有轨迹匹配; 剩下的轨迹再用
     低分框(0.12~0.45)二次匹配。YOLO 在遮挡/运动模糊时置信度会掉到 0.2 左右,
     若只用 0.45 阈值一刀切, 轨迹会被打断 → ID 跳变 → 计数翻倍。
  2. 匀速预测: 用上一轮的速度外推框位置, 10fps 下相邻帧位移大, 预测后 IoU 才够匹配。
  3. n_init 确认: 连续命中 3 次才对外暴露 ID, 抑制误检产生的"假轨迹"和 ID 抖动。
  4. 类别互斥匹配: 只允许同类目标互相匹配, 避免"人"和"公交车"重叠时 ID 互换。

对外接口:
    trk = ByteTracker(track_thresh=0.45, low_thresh=0.12, match_thresh=0.3,
                      max_lost=30, n_init=3)
    boxes, scores, classes, ids = trk.update(boxes_xyxy, scores, classes)
    # 返回的只有「已确认且本帧命中」的轨迹, 顺序与输入无关

    cnt = LineCounter(axis='h', pos=0.5, in_dir='down', cls_ids=(0,))
    cnt.configure(width, height)
    events = cnt.update(boxes, classes, ids, frame_no)
    # events: [{'id':3,'direction':'down','is_in':True,'n_in':1,'n_out':0,'box':...}]
"""
import numpy as np

__all__ = ['ByteTracker', 'LineCounter', 'iou_matrix', 'sim_matrix']


# ---------------------------------------------------------------- IoU / 匹配
def iou_matrix(a, b):
    """a:(N,4) b:(M,4), 格式 xyxy -> (N,M) IoU 矩阵"""
    n, m = len(a), len(b)
    out = np.zeros((n, m), dtype=np.float32)
    if n == 0 or m == 0:
        return out
    ax1, ay1 = a[:, 0][:, None], a[:, 1][:, None]
    ax2, ay2 = a[:, 2][:, None], a[:, 3][:, None]
    bx1, by1 = b[:, 0][None, :], b[:, 1][None, :]
    bx2, by2 = b[:, 2][None, :], b[:, 3][None, :]
    iw = np.minimum(ax2, bx2) - np.maximum(ax1, bx1)
    ih = np.minimum(ay2, by2) - np.maximum(ay1, by1)
    np.clip(iw, 0, None, out=iw)
    np.clip(ih, 0, None, out=ih)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    np.divide(inter, union, out=out, where=union > 0)
    return out


def sim_matrix(a, b, dist_ratio=1.2, size_ratio=0.5):
    """匹配相似度矩阵: IoU 为主, 不重叠时用「中心距离 + 尺寸接近度」兜底

    为什么需要兜底(这是实测出来的坑):
      推理只有约 10fps, 人正常走一帧就能挪 50~90 像素。两帧框几乎不重叠时
      IoU 直接掉到 0.1 以下, 纯 IoU 匹配会判定为"新目标" -> 轨迹断裂 ->
      ID 跳变 -> 同一个人被数两次。
      所以不重叠时改看: 中心点距离是否在 1.2 倍框对角线以内, 且尺寸是否
      接近(防大框吞小框)。距离匹配给分上限 0.85, 永远低于强 IoU, 保证
      重叠良好的配对优先。
    """
    iou = iou_matrix(a, b)
    n, m = iou.shape
    if n == 0 or m == 0:
        return iou
    ac = np.stack([(a[:, 0] + a[:, 2]) * 0.5, (a[:, 1] + a[:, 3]) * 0.5], 1)
    bc = np.stack([(b[:, 0] + b[:, 2]) * 0.5, (b[:, 1] + b[:, 3]) * 0.5], 1)
    dist = np.sqrt(((ac[:, None, :] - bc[None, :, :]) ** 2).sum(-1))
    da = np.sqrt((a[:, 2] - a[:, 0]) ** 2 + (a[:, 3] - a[:, 1]) ** 2)[:, None]
    db = np.sqrt((b[:, 2] - b[:, 0]) ** 2 + (b[:, 3] - b[:, 1]) ** 2)[None, :]
    diag = (da + db) * 0.5 + 1e-6
    sim_d = 1.0 - dist / (diag * float(dist_ratio))
    np.clip(sim_d, 0.0, 1.0, out=sim_d)
    sim_d *= 0.85                                    # 距离匹配的天花板
    # 尺寸约束: 相差超过 2 倍不接受, 避免远处大框把近处小框吸走
    sim_d *= (np.minimum(da, db) / np.maximum(da, db)) > float(size_ratio)
    return np.maximum(iou, sim_d)


def _greedy(iou, thresh):
    """贪心最大 IoU 匹配。目标数少时与匈牙利解等价, 但零依赖、纯 numpy/C 排序。"""
    rows, cols = list(range(iou.shape[0])), list(range(iou.shape[1]))
    pairs = []
    if not rows or not cols:
        return pairs, rows, cols
    ur, uc = set(rows), set(cols)
    flat = iou.ravel()
    for k in np.argsort(-flat):
        if flat[k] < thresh:
            break
        r, c = divmod(int(k), iou.shape[1])
        if r in ur and c in uc:
            pairs.append((r, c))
            ur.discard(r)
            uc.discard(c)
    return pairs, sorted(ur), sorted(uc)


# ---------------------------------------------------------------- 单条轨迹
class Track(object):
    """一条轨迹。box 存 xyxy 图像坐标。"""

    __slots__ = ('id', 'box', 'vel', 'score', 'cls', 'hits', 'age', 'since',
                 'confirmed', 'n_init', 'smooth')

    def __init__(self, box, score, cls, tid, n_init=3, smooth=0.5):
        self.id = int(tid)
        self.box = np.asarray(box, dtype=np.float32).copy()
        self.vel = np.zeros(4, dtype=np.float32)
        self.score = float(score)
        self.cls = int(cls)
        self.hits = 1              # 累计命中次数
        self.age = 0               # 存活轮次
        self.since = 0             # 距上次命中的轮次(0=本帧命中)
        self.confirmed = False
        self.n_init = int(n_init)
        self.smooth = float(smooth)

    def predict(self):
        """匀速外推: 用上一轮速度把框往前推一格"""
        self.box = self.box + self.vel
        self.age += 1
        self.since += 1

    def update(self, box, score, cls):
        meas = np.asarray(box, dtype=np.float32)
        delta = meas - self.box
        # 速度平滑(0.6 旧 + 0.4 新), 避免单帧抖动带偏外推
        self.vel = np.float32(0.6) * self.vel + np.float32(0.4) * delta
        # 位置 EMA 平滑: 抑制检测框逐帧跳动, 画面更稳
        self.box = self.box + np.float32(self.smooth) * delta
        self.score = float(score)
        self.cls = int(cls)
        self.hits += 1
        self.since = 0
        if not self.confirmed and self.hits >= self.n_init:
            self.confirmed = True

    # 便捷属性
    @property
    def center(self):
        return ((self.box[0] + self.box[2]) * 0.5,
                (self.box[1] + self.box[3]) * 0.5)


# ---------------------------------------------------------------- 跟踪器
class ByteTracker(object):
    def __init__(self, track_thresh=0.45, low_thresh=0.12, match_thresh=0.3,
                 match_thresh_low=0.25, max_lost=45, n_init=3, smooth=0.5,
                 match_class=True, max_tracks=64,
                 dist_ratio=1.2, size_ratio=0.5, rebirth=45, cross_iou=0.5):
        self.track_thresh = float(track_thresh)
        self.low_thresh = float(low_thresh)
        self.match_thresh = float(match_thresh)
        self.match_thresh_low = float(match_thresh_low)
        self.max_lost = int(max_lost)     # 丢失多少轮后删除(14fps 下 45 轮≈3.2 秒)
        self.rebirth = int(rebirth)       # 删除后多少轮内可用原 ID 复活
        self.n_init = int(n_init)
        self.smooth = float(smooth)
        self.match_class = bool(match_class)
        self.max_tracks = int(max_tracks)
        self.dist_ratio = float(dist_ratio)   # 距离兜底: 1.2 倍框对角线以内
        self.size_ratio = float(size_ratio)   # 尺寸约束: 相差不超过 2 倍
        self.cross_iou = float(cross_iou)     # 类别抖动时允许跨类匹配的 IoU 门槛
        self.tracks = []
        self._graveyard = []              # 近期删除的已确认轨迹 [box, cls, id, frame]
        self.rebirth_n = 0                # 累计复活次数(调试用)
        self._next_id = 1
        self.frame_no = 0
        self.id_log_n = 0                 # ID 生命周期日志已打印条数(防刷屏)
        self.id_log_max = 400             # 上限: 超过就不再打, 避免持 GIL 拖慢推理

    def _idlog(self, msg):
        """打印 ID 生命周期事件(新建/入殓/复活)。

        为什么需要: "ID 跳变"是最难靠读代码定位的问题——到底是漏检、
        类别抖动、还是移动太快超出匹配范围, 光看画面分不出来。
        把每次新建 ID 的**失败原因**打出来, 一次运行就能定死病因。
        正常场景下每秒最多几条, 不会像逐帧 debug 那样把线程拖死。
        """
        if self.id_log_n >= self.id_log_max:
            return
        self.id_log_n += 1
        if self.id_log_n == self.id_log_max:
            print('[ID] 日志已满 %d 条, 后续不再打印' % self.id_log_max)
        print('[ID] ' + msg)

    # -- 短期重识别: 让刚丢不久的目标复用原 ID ---------------------------------
    def _rebirth(self, box, cls):
        """断续检测时同一个目标会被当成新目标 -> ID 跳变。

        这里保留已删除轨迹的最后位置和 ID, 若新目标落在它的位置附近且在
        rebirth 轮内 -> 复用老 ID。类别不同但重叠很高(IoU>=0.5)也允许复用,
        用来兜住 YOLO 把同一个瓶子一会儿判成 bottle 一会儿判成 cup 的情况。
        """
        if not self._graveyard:
            return None, None, '墓地空'
        gb = np.asarray([g[0] for g in self._graveyard], dtype=np.float32)
        b = np.asarray(box, dtype=np.float32).reshape(1, 4)
        iou = iou_matrix(gb, b).ravel()
        sim = sim_matrix(gb, b, self.dist_ratio, self.size_ratio).ravel()
        best, best_sim = None, 0.0
        for k, g in enumerate(self._graveyard):
            same_cls = (int(cls) == g[1])
            if not same_cls and iou[k] < 0.5:
                continue                       # 类别变了且重叠不高 -> 不复用
            s = max(float(iou[k]), float(sim[k]))
            if s > best_sim:
                best, best_sim = g, s
        if best is None:
            return None, None, ('墓地有 %d 个目标但类别都对不上'
                                % len(self._graveyard))
        if best_sim < self.match_thresh:
            return None, None, ('最近的老目标只像 %.2f 分(要 >=%.2f), 判定为新目标'
                                % (best_sim, self.match_thresh))
        return best[2], best, '像 %.2f 分' % best_sim

    # -- 内部: 一轮匹配 -------------------------------------------------
    def _match(self, tr_indices, dets, thresh):
        """dets: (M,6) = x1y1x2y2,score,cls ; 返回 (pairs, 未匹配轨迹, 未匹配检测)"""
        if not tr_indices or len(dets) == 0:
            return [], list(tr_indices), list(range(len(dets)))
        tb = np.array([self.tracks[i].box for i in tr_indices], dtype=np.float32)
        iou = sim_matrix(tb, dets[:, :4], self.dist_ratio, self.size_ratio)
        if self.match_class:
            tc = np.array([self.tracks[i].cls for i in tr_indices], dtype=np.float32)
            same = (tc[:, None] == dets[:, 5][None, :])
            # 类别不同 -> 原则上禁止匹配; 但 YOLO 常把同一个物体一会儿判成
            # bottle 一会儿判成 cup/phone, 此时旧轨迹与新框永远配不上 -> 当场
            # 分裂成两个 ID。这里留个口子: 重叠够高(>=cross_iou)时按同类别处理。
            raw = iou_matrix(tb, dets[:, :4])
            iou = iou * (same | (raw >= self.cross_iou))
        pairs, ur, uc = _greedy(iou, thresh)
        return ([(tr_indices[r], c) for r, c in pairs],
                [tr_indices[r] for r in ur], uc)

    # -- 主入口 ---------------------------------------------------------
    def update(self, boxes, scores, classes):
        """boxes: (N,4) xyxy 图像坐标; scores:(N,); classes:(N,)
        返回 (boxes, scores, classes, ids) —— 仅已确认且本帧命中的轨迹"""
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        classes = np.asarray(classes, dtype=np.float32).reshape(-1)
        self.frame_no += 1

        for t in self.tracks:
            t.predict()

        hi = scores >= self.track_thresh
        lo = (scores >= self.low_thresh) & (~hi)
        det_hi = (np.concatenate([boxes[hi], scores[hi, None],
                                  classes[hi, None]], axis=1)
                  if hi.any() else np.zeros((0, 6), np.float32))
        det_lo = (np.concatenate([boxes[lo], scores[lo, None],
                                  classes[lo, None]], axis=1)
                  if lo.any() else np.zeros((0, 6), np.float32))

        # 阶段 1: 全部轨迹 vs 高分框
        pairs, rest, new_det = self._match(list(range(len(self.tracks))),
                                           det_hi, self.match_thresh)
        for ti, di in pairs:
            self.tracks[ti].update(det_hi[di, :4], det_hi[di, 4], det_hi[di, 5])

        # 阶段 2: 阶段 1 没匹配上的轨迹 vs 低分框(ByteTrack 的关键一步)
        pairs2, rest2, _ = self._match(rest, det_lo, self.match_thresh_low)
        for ti, di in pairs2:
            self.tracks[ti].update(det_lo[di, :4], det_lo[di, 4], det_lo[di, 5])

        # 未匹配的高分框 -> 新建轨迹(低分框不新建, 避免误检建轨迹)
        # 先问一句: 它是不是刚"丢"掉的那个目标? 是的话复用老 ID
        for di in new_det:
            if len(self.tracks) >= self.max_tracks:
                break
            bx = det_hi[di, :4]
            tid, entry, why = self._rebirth(bx, det_hi[di, 5])
            if tid is None:
                tid = self._next_id
                self._next_id += 1
                self._idlog('新建 id=%d  cls=%d 分%.2f  位置(%.0f,%.0f)  复活失败: %s'
                            % (tid, int(det_hi[di, 5]), det_hi[di, 4],
                               (bx[0] + bx[2]) * 0.5, (bx[1] + bx[3]) * 0.5, why))
            t = Track(bx, det_hi[di, 4], det_hi[di, 5],
                      tid, n_init=self.n_init, smooth=self.smooth)
            if entry is not None:            # 复活: 直接恢复"已确认"状态
                self._graveyard.remove(entry)
                t.confirmed = True
                t.hits = max(t.hits, self.n_init)
                self.rebirth_n += 1
                self._idlog('id=%d 复活成功(%s)  累计复活 %d 次'
                            % (tid, why, self.rebirth_n))
            self.tracks.append(t)

        # 清理: 已确认轨迹允许丢失 max_lost 轮; 未确认轨迹只容 1 轮
        alive = []
        for t in self.tracks:
            if t.since == 0:
                alive.append(t)
            elif t.confirmed:
                if t.since <= self.max_lost:
                    alive.append(t)
                elif self.rebirth > 0:
                    # 入殓: 记下最后位置与 ID, 短期内可被复用(治 ID 跳变)
                    self._graveyard.append([t.box.copy(), int(t.cls),
                                            int(t.id), self.frame_no])
                    self._idlog('id=%d 连续 %d 帧没检测到, 入墓地(cls=%d)'
                                % (t.id, t.since, int(t.cls)))
            elif t.since <= 1:
                alive.append(t)
            # 未确认且丢帧的轨迹直接丢弃(本来就可能是一次误检, 不值得复活)
        self.tracks = alive
        if self._graveyard:                  # 过期清理
            self._graveyard = [g for g in self._graveyard
                               if self.frame_no - g[3] <= self.rebirth]

        # 输出: 只给本帧命中且已确认的轨迹
        ob, os_, oc, oi = [], [], [], []
        for t in self.tracks:
            if not t.confirmed or t.since != 0:
                continue
            ob.append(t.box)
            os_.append(t.score)
            oc.append(t.cls)
            oi.append(t.id)
        if not ob:
            return (np.zeros((0, 4), np.float32), np.zeros(0, np.float32),
                    np.zeros(0, np.float32), np.zeros(0, np.int32))
        return (np.asarray(ob, dtype=np.float32),
                np.asarray(os_, dtype=np.float32),
                np.asarray(oc, dtype=np.float32),
                np.asarray(oi, dtype=np.int32))

    # -- 只拿坐标(画轨迹用) ---------------------------------------------
    def active_boxes(self):
        return [(t.id, t.box.copy(), t.cls) for t in self.tracks if t.confirmed]


# ---------------------------------------------------------------- 越线计数
class LineCounter(object):
    """越线计数: 目标中心点跨过一条直线就记一次。

    抗抖动设计: 直线两侧各留 margin(默认 1.5% 画面高)的"缓冲区",
    中心点落在缓冲区内不改变状态 —— 人站在线上左右晃动不会反复计数。
    同一 ID 两次计数之间强制冷却 cooldown 轮, 防丢帧造成的重复触发。
    """

    def __init__(self, axis='h', pos=0.5, in_dir=None, cls_ids=(0,),
                 margin=0.015, cooldown=3, anchor='bottom', debug=False):
        self.axis = axis                    # 'h' 水平线(上下走) / 'v' 垂直线(左右走)
        self.pos = float(pos)               # 线在画面中的位置比例 0~1
        self.in_dir = in_dir or ('down' if axis == 'h' else 'right')
        self.cls_ids = set(int(c) for c in cls_ids)
        self.margin = float(margin)
        self.cooldown = int(cooldown)
        # anchor: 用哪个点判定越线
        #   'top'    = 框顶边(头顶/物体最上端)  <- 目标部分入镜、下半部分出画时用
        #   'bottom' = 框底边(脚点)            <- 全身入镜、人横穿地面线时用(标准做法)
        #   'center' = 框中心                  <- 俯拍车流、目标整体较小时用
        # 判据选错的表现始终是同一个: ID 正常增加, IN/OUT 恒为 0。
        # 因为参考点一直停在线的同一侧, side 恒不变 -> 永远判不到"跨过"。
        # 快速判断法: 人走到线前站住, 看画面里"哪个部位越过去了"就用哪个判据。
        self.anchor = str(anchor)
        self.debug = bool(debug)
        # 任意角度计数线(网页画线用): 两个端点, 归一化 0~1。
        # 为 None 时走旧的 axis/pos 模式(命令行 --line-axis/--line-pos)。
        # 为什么归一化: 采集分辨率和推流分辨率是解耦的(1024/1632/1920 都可能),
        # 存像素坐标换个分辨率线就跑偏了, 存比例则永远对。
        self.p1 = None
        self.p2 = None
        self.in_side = 1              # +1: 从负侧跨到正侧算进入; -1: 反过来
        # 折线(曲线)计数线: 网页拖动画出来的点列, 直线是它 2 个点的特例。
        # enabled=False 表示用户明确关掉了计数(点过"清除"), 这时不画线也不计数。
        self.poly = None
        self.enabled = True
        # 端点容差: 目标贴着线端点经过时, anchor 投影会瞬间跳出 [0,1] 而漏计,
        # 允许 t 超出线段两端各 6% 线长(实测"半根线"场景的主要漏计来源之一)。
        self.end_tol = 0.06
        self.w = self.h = None
        self.n_in = 0
        self.n_out = 0
        self.events = []
        self._state = {}                    # tid -> [side, last_count_frame]
        self._pruned_at = 0
        # 调试打印节流: 每个 ID 至少间隔这么久才打印一次"状态未变"。
        # 不节流的后果很严重 —— print 在推理线程里是持锁 IO, 画面里每多一个
        # 目标就多一行: 14fps × 15 个 ID = 210 行/秒, 足以把推理帧率拖到 0。
        self._dbg_gap = 0.5
        self._dbg_last = {}

    def _dbg(self, tid, text):
        now = time.time()
        if now - self._dbg_last.get(tid, -1e9) < self._dbg_gap:
            return
        self._dbg_last[tid] = now
        print(text)

    def configure(self, width, height):
        self.w, self.h = int(width), int(height)

    # -- 网页画线: 用两个端点替换 axis/pos -------------------------------
    def set_line(self, p1, p2, in_side=1):
        """设定任意角度的计数线。p1/p2 为归一化坐标 [0~1, 0~1]。

        in_side 表示"线哪一侧算进入": 目标最终落在 d 的正侧时,
        in_side=+1 记 IN; 反过来设 -1 就把进出方向对调。
        """
        self.p1 = (float(p1[0]), float(p1[1]))
        self.p2 = (float(p2[0]), float(p2[1]))
        self.poly = [list(self.p1), list(self.p2)]
        self.enabled = True
        self.in_side = 1 if int(in_side) >= 0 else -1
        # 换线之后旧的侧别状态全部作废, 否则会拿旧线的 side 跟新线比, 乱计数
        self._state = {}

    # -- 折线(曲线)计数线: 网页拖动画出来的 --------------------------------
    def set_polyline(self, pts, in_side=1):
        """pts: 归一化坐标列表 [[x,y],...], 至少 2 个点。

        直线只是折线的特例(2 个点), 所以内部统一存成 self.poly,
        self.p1/self.p2 保留为首尾点, 兼容旧代码里对它们的判断。
        """
        pts = [[float(x), float(y)] for x, y in pts]
        if len(pts) < 2:
            return
        self.poly = pts
        self.p1 = pts[0]
        self.p2 = pts[-1]
        self.enabled = True
        self.in_side = 1 if int(in_side) >= 0 else -1
        self._state = {}

    def disable_line(self):
        """网页点了"清除": 计数线彻底关掉, 不再回落命令行默认线。"""
        self.enabled = False
        self.poly = None
        self.p1 = None
        self.p2 = None
        self._state = {}

    def _seg_metrics(self, boxes):
        """折线模式: 返回 (d, ok)。

        d[k]  = 第 k 个目标 anchor 点到折线的**最小有符号**距离(像素)。
                曲线是很多小线段连成的, 对每段分别算距离, 取 |d| 最小的那段 ——
                "离目标最近的那段"才代表目标相对曲线的位置。
        ok[k] = 是否至少落在某一段的投影范围内(允许端点容差 end_tol)

        为什么 ok 必须判断: 线只画了画面上半段时, 从下半段走过去的人不该被算。
        只判距离会把他也算进去 —— 这是"斜线画一半"场景下最常见的误计来源。
        """
        w, h = float(self.w), float(self.h)

        # 1) 每个目标取 anchor 点(整列运算, 不用 for 循环)
        cx = (boxes[:, 0] + boxes[:, 2]) * 0.5       # 框水平中心
        if self.anchor == 'top':
            px, py = cx, boxes[:, 1]
        elif self.anchor == 'bottom':
            px, py = cx, boxes[:, 3]
        else:
            px, py = cx, (boxes[:, 1] + boxes[:, 3]) * 0.5

        # 2) 逐段算距离/投影, 取 |d| 最小且投影在段内的那段
        n = len(boxes)
        best_d = np.full(n, np.inf, dtype=np.float32)
        best_ok = np.zeros(n, dtype=bool)
        tol = float(self.end_tol)
        for i in range(len(self.poly) - 1):
            Ax, Ay = self.poly[i][0] * w, self.poly[i][1] * h
            Bx, By = self.poly[i + 1][0] * w, self.poly[i + 1][1] * h
            ex, ey = Bx - Ax, By - Ay
            L2 = ex * ex + ey * ey
            if L2 < 1e-6:                            # 相邻两点重合, 跳过该段
                continue
            # 叉积 / |AB| = 带符号的距离, 正负即线两侧
            d = ((ex * (py - Ay) - ey * (px - Ax)) / np.sqrt(L2)).astype(np.float32)
            # 投影比例 t: 落在 [−tol, 1+tol] 才算"在这段上"
            t = ((px - Ax) * ex + (py - Ay) * ey) / L2
            ok = (t >= -tol) & (t <= 1.0 + tol)
            take = ok & (np.abs(d) < np.abs(best_d))
            best_d[take] = d[take]
            best_ok[take] = True
        return best_d, best_ok

    def polyline_px(self):
        """折线顶点(像素坐标), 画到推流画面用。没开计数线时返回空列表。"""
        if not self.enabled or self.poly is None or self.w is None:
            return []
        return [(int(x * self.w), int(y * self.h)) for x, y in self.poly]

    # -- 线的两个端点(画图用) -------------------------------------------
    def endpoints(self):
        if self.w is None:
            return (0, 0), (0, 0)
        if self.p1 is not None:
            return ((int(self.p1[0] * self.w), int(self.p1[1] * self.h)),
                    (int(self.p2[0] * self.w), int(self.p2[1] * self.h)))
        if self.axis == 'h':
            y = int(self.pos * self.h)
            return (0, y), (self.w, y)
        x = int(self.pos * self.w)
        return (x, 0), (x, self.h)

    def hud(self):
        return 'IN %d   OUT %d' % (self.n_in, self.n_out)

    # -- 主入口 ---------------------------------------------------------
    def update(self, boxes, classes, ids, frame_no):
        """返回本帧发生越线的事件列表"""
        if self.w is None or len(ids) == 0 or not self.enabled:
            return []

        if self.poly is not None:
            # ---- 网页画的任意角度线段 / 曲线(折线) ----
            d_arr, ok = self._seg_metrics(boxes)
            band = self.margin * max(self.w, self.h)
            pos_name, neg_name = 'in', 'out'
        else:
            # ---- 命令行 --line-axis/--line-pos ----
            if self.axis == 'h':
                line = self.pos * self.h
                band = self.margin * self.h
                along = (boxes[:, 0] + boxes[:, 2]) * 0.5  # 沿线方向坐标
                # 跨线方向坐标: 按 anchor 取框的顶边 / 底边 / 中心
                if self.anchor == 'top':
                    across = boxes[:, 1]
                elif self.anchor == 'bottom':
                    across = boxes[:, 3]
                else:
                    across = (boxes[:, 1] + boxes[:, 3]) * 0.5
                span = (0, self.w)
                pos_name, neg_name = 'down', 'up'
            else:
                line = self.pos * self.w
                band = self.margin * self.w
                along = (boxes[:, 1] + boxes[:, 3]) * 0.5
                if self.anchor == 'top':                    # 垂直线: 左边缘
                    across = boxes[:, 0]
                elif self.anchor == 'bottom':               # 垂直线: 右边缘
                    across = boxes[:, 2]
                else:
                    across = (boxes[:, 0] + boxes[:, 2]) * 0.5
                span = (0, self.h)
                pos_name, neg_name = 'right', 'left'
            d_arr = across - line
            ok = (along >= span[0]) & (along <= span[1])

        out = []
        for k, tid in enumerate(ids):
            tid = int(tid)
            if self.cls_ids and int(classes[k]) not in self.cls_ids:
                continue
            if not ok[k]:
                continue
            d = d_arr[k]
            side = 1 if d > band else (-1 if d < -band else 0)
            if side == 0:                     # 缓冲区里
                if tid not in self._state:
                    # 首见即在缓冲区: 记 side=0。这样它走出缓冲区时 prev=0
                    # 会判成"跨线"并计数 —— 否则短线/半根线场景下, 目标常常
                    # 是先在线附近入镜的, 第一次登记侧别就发生在过线之后,
                    # 那次过线永远漏计("画半根线不灵敏"的主因)。
                    self._state[tid] = [0, -10 ** 6]
                if self.debug:
                    self._dbg(tid, '   [cnt] id%-3d d=%6.1f -> 缓冲区 维持'
                              % (tid, d))
                continue
            st = self._state.get(tid)
            if st is None:                    # 首次出现: 只记侧别, 不计数
                self._state[tid] = [side, -10 ** 6]
                if self.debug:
                    self._dbg(tid, '   [cnt] id%-3d d=%6.1f -> 首见 side=%+d'
                              % (tid, d, side))
                continue
            if side == st[0]:
                if self.debug:
                    self._dbg(tid, '   [cnt] id%-3d d=%6.1f side=%+d 未变'
                              % (tid, d, side))
                continue
            prev = st[0]
            st[0] = side
            if frame_no - st[1] < self.cooldown:
                if self.debug:
                    print('   [cnt] id%-3d 冷却中 跳过' % tid)
                continue
            st[1] = frame_no
            if self.poly is not None:
                # 折线模式: 方向由"最终落在哪一侧"决定, in_side 由网页点选
                is_in = (side == self.in_side)
                direction = 'in' if is_in else 'out'
            else:
                direction = pos_name if side > 0 else neg_name
                is_in = (direction == self.in_dir)
            if is_in:
                self.n_in += 1
            else:
                self.n_out += 1
            if self.debug:
                print('   [cnt] id%-3d d=%6.1f  %+d -> %+d  %s  IN=%d OUT=%d'
                      % (tid, d, prev, side,
                         'IN' if is_in else 'OUT', self.n_in, self.n_out))
            ev = {'id': tid, 'direction': direction, 'is_in': is_in,
                  'from': prev, 'frame': frame_no,
                  'n_in': self.n_in, 'n_out': self.n_out,
                  'box': np.asarray(boxes[k], dtype=np.float32).copy()}
            self.events.append(ev)
            out.append(ev)

        # 状态表防膨胀: 每 500 轮清一次长期没出现的 ID
        if frame_no - self._pruned_at > 500:
            self._pruned_at = frame_no
            keep = {t: s for t, s in self._state.items()
                    if frame_no - s[1] < 3000}
            self._state = keep
            if self.debug:
                self._dbg_last.clear()
        # 事件表防膨胀: 主循环不取(比如关了抓拍)也不能让它无限长
        if len(self.events) > 200:
            self.events = self.events[-50:]
        return out

    def take_events(self):
        """取走自上次调用以来的全部新事件并清空内部表(主循环是唯一消费者)。

        用「取走」而不是「按下标增量读」的原因: 让 events 不随运行时间无限增长,
        长时间挂机就不会慢慢吃掉内存。
        """
        if not self.events:
            return []
        out = self.events
        self.events = []
        return out
