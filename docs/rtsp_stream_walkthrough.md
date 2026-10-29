# rtsp_stream.py 逐段讲解（面试复习用）

> 目标：让你对这份代码**讲得出每一块在干什么、为什么这么写、踩过什么坑**。
> 行号对应 `rtsp_stream.py` 当前版本，代码更新后行号可能偏移，按段落标题找即可。

---

## 0. 先建立整体印象：三个线程 + 一条管道

```
                 ┌──────────── 采集线程 capture_loop ────────────┐
摄像头(/dev/video0) ──cap.read()──> _latest（只留最新一帧）
                 └───────────────────────────────────────────────┘
                              │ 主循环取走最新帧
                              ▼
                 ┌──────────── 主循环（每帧必做）────────────┐
                 │  1. 按目标帧率取帧（限速）                 │
                 │  2. 顺手把帧 copy 一份丢给推理队列          │
                 │  3. 画框（用最近一次检测结果）              │
                 │  4. push 到 appsrc → 编码 → RTSP           │
                 └───────────────────────────────────────┘
                              ▲
                 ┌──────────── 推理线程 infer_loop ────────────┐
                 │  取 _infer_in → letterbox 640 → RKNN → NMS  │
                 │  结果写 _det（主循环自己来取，绝不阻塞）      │
                 └─────────────────────────────────────────┘
```

**关键设计一句话**：主循环**永远不等推理**。推理慢就慢，画面照常 30fps 走，检测框按推理实际速度刷新。这就是从 19fps 提到 30fps 的全部秘密。

---

## 1. 参数与预设（L39–88）

```python
WIFI_PRESET = dict(width=640, height=480, fps=20, bps=1200000, gop=15)
DEFAULT     = dict(width=1024, height=768, fps=30, bps=2500000, gop=30)
preset = WIFI_PRESET if args.wifi else DEFAULT
for k, v in preset.items():
    if getattr(args, k) is None:      # 只在用户没显式指定时才填
        setattr(args, k, v)
```

**为什么这么写**：最早版本是 `--wifi` 无条件覆盖所有参数，导致用户写 `--wifi --fps 24` 完全无效（静默失效，很难查）。改成 `default=None` + "只填空" 之后，显式参数永远优先。

- `iw dev wlan0 set power_save off`（L71）：rtl8188f 的省电模式会让 WiFi 周期性休眠，表现为**规律性卡顿**。这是实测出来的，不是猜的。
- `dev_ip()`（L77）：启动时直接打印 eth0/wlan0 的可拉流地址，省得每次手敲 IP。

**面试点**：`--wifi` 预设为什么这么配？→ 降负载（CPU）、降码率（带宽）、短 GOP（快速出画 + 抗丢包）、关省电（防抖动）。

---

## 2. NPU 初始化（L108–121）

```python
rknn = RKNNLite()
rknn.load_rknn(args.model)     # 加载 .rknn（PC 端转换好的，板端不能转）
rknn.init_runtime()            # 板端不带 target 参数
co_helper = COCO_test_helper(enable_letter_box=True)
```

**三个必须记住的坑**：

1. **板端只能跑 lite2**（`rknnlite`），转换必须在 PC 的 `rknn-toolkit2` 上做，aarch64 转不了；
2. **`init_runtime()` 不能带 target**，官方 `rknn_executor.py` 里的 `target='rk3568'` 分支是给 PC 连板调试用的；
3. **版本三方对齐**：PC toolkit2 / 板端 lite2 / `librknnrt.so` 必须同版本（本项目 2.3.0），否则 `load_rknn` 直接失败。

`--no-infer` 时 `rknn = None`，整个推理线程空转——**这是最重要的诊断开关**：关掉 NPU 还卡，就说明瓶颈不在算力。

---

## 3. 采集线程（L123–146）

```python
def capture_loop():
    while not _stop:
        ok, f = cap.read()
        if ok:
            with _lock:
                _latest[0] = f        # 只保留最新一帧，旧的直接丢
```

**为什么用线程**：`cap.read()` 是阻塞调用。v1 版本在主循环里串行读，被 100ms 的推理卡住 → 摄像头缓冲区超时（`cap_v4l select() timeout`）→ 9.9fps。

**为什么"只留最新一帧"**：实时流场景，**旧帧没有价值**。宁可丢帧也不能让延迟累积——这条原则贯穿整个项目（背压、限速都是同一思想）。

**共享数据用 `_lock` + 单元素列表**：`_latest = [None]` 是 Python 里最简单的"可变引用盒"，配合 Lock 保证线程安全。

---

## 4. 推理线程（L149–169）

```python
img = co_helper.letter_box(im=f, new_shape=(640, 640), pad_color=(0,0,0))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)            # 模型要 RGB
outputs = rknn.inference(inputs=[np.expand_dims(img, 0)])   # 必须 4 维 NHWC
boxes, classes, scores = post_process(outputs)
```

**四个必踩的坑**（官方例程全中）：

1. `letter_box` 保持宽高比 + 黑边填充，避免拉伸变形；推理完要用 `get_real_box()` 把坐标**映射回原图尺寸**；
2. OpenCV 是 BGR，模型要 RGB，必须转；
3. `np.expand_dims(img, 0)` —— RKNNLite 要求 NHWC 4 维，3 维报 `need 4dims input`；
4. `post_process()` 里的 `dfl()` 官方用 torch 实现，板端无 torch，改成了**纯 numpy**（减最大值 softmax + 位置索引加权和）。

**为什么 `_infer_in` 只在为空时才投放**（L346–349）：推理线程忙的时候直接跳过，避免帧堆积在推理队列里变成"处理几秒前的画面"。

---

## 5. 中文标签渲染（L172–253）

OpenCV 的 `cv2.putText` **不支持中文**（会画出问号），必须走 PIL + CJK 字体。

**两次优化（这是本项目性能故事的高潮）**：

| 版本 | 做法 | 问题 |
|---|---|---|
| v3.1 | 整帧 `BGR→RGB→PIL→画完→RGB→BGR` | 主循环每帧搬运 78 万像素，占满 GIL |
| v3.2 | 只转换文字所在的小矩形 | 改善但仍每帧调用 PIL |
| **v3.3** | **文字渲染成小 patch，只在检测结果变化时重渲染** | PIL 调用频率 30fps → 推理频率（~10fps） |

```python
if _label_cache['key'] is not det:            # 用对象身份做缓存 key
    _label_cache['key'] = det
    _label_cache['items'] = _render_label_items(det, font)
for x, y, patch in _label_cache['items']:      # 每帧只做 numpy 切片赋值
    frame[y0:y1, x0:x1] = patch[:y1-y0, :x1-x0]
```

**为什么必须这么做**：Python 的 **GIL** 只允许一个线程执行字节码。主循环做重 CPU 活 → 推理线程（numpy 计算密集）拿不到 GIL → infer fps 从 10 掉到 5.6。而画框明明在主循环、根本不计入 infer 统计——**这个反直觉的现象就是 GIL 争抢的铁证**。

---

## 6. RTSP 服务器（L256–315）

### 管道串（L257–266）

```
appsrc(BGR) → videoconvert→NV12 → mpph264enc(硬编) → h264parse
            → rtph264pay name=pay0
```

**硬性规则**：factory 的 launch 串**必须以 payloader 结尾并命名为 `pay0`**，整体括号包裹。否则客户端接入时拿不到 payload，直接失败。

### 四个关键点

| 点 | 代码 | 为什么 |
|---|---|---|
| 监听所有网卡 | `props.address = '0.0.0.0'` | 有线、WiFi 都能拉流 |
| 共享管线 | `factory.set_shared(True)` | 多客户端只编码一次 |
| 拿 appsrc | `media-configure` 回调里 `get_by_name('src')` | **客户端接入才创建管道**，启动阶段拿不到 |
| GLib 主循环 | 后台线程 `loop.run` | RTSP 服务需要事件循环，放主线程会阻塞采帧 |

**背压（L303–309）**：

```python
self.appsrc.set_property('max-bytes', width*height*3*2)   # 队列最多 2 帧
self.appsrc.connect('need-data',  ...)   # 队列空了 → 允许投喂
self.appsrc.connect('enough-data', ...)  # 队列满了 → 暂停投喂
```

`max-bytes` 默认是 0（**无上限**）。WiFi 客户端消费慢时，旧帧无限堆积 → 你看到的是十几秒前的画面。这是"高延迟"的真正原因，跟网络快慢无关。

---

## 7. 主循环（L328–382）

```python
interval = 1.0 / args.fps
if now - last_emit < interval:     # 限速：不到时间不取帧
    time.sleep(0.001); continue
```

**限速的意义**：不加限速时主循环会全速 30fps 推帧 + 全速渲染，WiFi 场景下 CPU 被打满，板子整机卡死（ssh 都断）。按目标帧率取帧，多余的丢弃。

**推帧三条件（L356–372）**——这三条每一条都对应一个真实 bug：

```python
if appsrc is not None:                                    # ① 有客户端接入
    if not rtsp.need_data and time.time()-last_push > 1.0: #   兜底防死锁
        rtsp.need_data = True
    if rtsp.need_data:                                    # ② 背压允许
        ok, state, _ = appsrc.get_state(0)
        if ok == SUCCESS and state == Gst.State.PLAYING:   # ③ 媒体在播放
            appsrc.emit('push-buffer', buf)
```

| 条件 | 缺了会怎样 |
|---|---|
| ① appsrc 存在 | 客户端断开后 push 抛异常 |
| ② 背压允许 | 旧帧堆积 → 十几秒延迟 |
| ③ PLAYING | 客户端切换时 media 短暂 PAUSE，此时推帧 → 旧帧乱序 → **解码器全部丢弃 → 黑屏** |

---

## 8. 收尾（L383–393）

`_stop = True` 通知两个后台线程退出，`loop.quit()` 停掉 GLib，`cap.release()` / `rknn.release()` 释放资源。**daemon=True 的线程**保证 Ctrl+C 时进程能干净退出。

---

## 9. 三圈知识对照表（照着准备面试）

### 🟢 第一圈：必须讲透（你亲手踩出来的，别人抄不走）

1. 9.9fps → 30fps：同步推理阻塞 → 三线程并行
2. 降分辨率零改善 → 瓶颈是固定推理成本，不是像素量
3. GIL 争抢：主循环重活反噬推理线程（10 → 5.6），靠渲染缓存修复
4. appsrc 队列无上限 → 十几秒延迟，靠背压修复
5. 客户端切换 PAUSE → 黑屏，靠 PLAYING 检查修复
6. RKNN 版本三方对齐 + DRM 节点是 renderD129（6.1 内核无 /dev/rknpu）
7. int8 量化校准必须用自有数据，否则精度暴跌

### 🟡 第二圈：说得出思路即可

- letterbox 为什么要做、坐标怎么映射回原图
- NMS 在干什么（去重重叠框）
- GOP / CBR 的含义与取舍（GOP 小 = 出画快、抗丢包，但码率高）
- RTSP 与裸 UDP 的区别（客户端主动拉、支持多客户端、可走 TCP）
- V4L2 采集 + YUYV→BGR 的转换开销在哪

### 🔴 第三圈：可以坦白"还在深入"

- numpy dfl 的具体数学推导
- ByteTrack 匈牙利匹配的完整实现
- rkisp / MPP 编码器内部流水线

**被问到第三圈的标准答法**：
> "这块是我调通并验证过效果的，内部实现细节还在深入。不过我可以讲讲当时是怎么定位到它、又是怎么验证修复有效的。"

然后主动把话题拉回第一圈——**你亲手解决的问题才是你的主场**。

---

## 10. 自测题（能答上 8 条就可以去面试了）

1. 为什么一开始只有 9.9fps？改了什么结构？
2. 为什么把分辨率从 1024×768 降到 640×480 帧率几乎不变？
3. 推理线程独立后，画面 30fps 但检测框只有 10fps，为什么不冲突？
4. GIL 争抢的表现是什么？怎么定位到的？怎么修的？
5. appsrc 的 `max-bytes` 默认是多少？不设会怎样？
6. 多客户端切换为什么会黑屏？修复手段是什么？
7. RTSP factory 的管道串必须以什么结尾、命名成什么？
8. 为什么 appsrc 只能在 media-configure 回调里拿？
9. int8 量化后模型从多大变多小？换自定义目标时校准要注意什么？
10. 手机热点下卡顿，你怎么一步步排除是板子 CPU 的问题？

---

## 附：排障方法论（比代码本身更值钱）

> **现象 → 分段测量 → 定位 → 修复 → 复测**

本项目三次性能问题全靠这个闭环解决，而且**两次直觉判断都被实测推翻**（"肯定是 NPU 慢"→ 不是；"肯定是分辨率太高"→ 不是）。

可复用的测量手段：
- `--no-infer` / `--no-stream`：切断某一段，测另一段上限
- 板端 `stream fps | infer fps`：看发送侧是否满速
- `top`：系统负载，判断 CPU 是否瓶颈
- `iw dev wlan0 link`：信号强度与速率，判断无线链路
- A/B 对照：改一个变量，看现象是否变化
