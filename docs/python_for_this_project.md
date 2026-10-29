# 为读懂本项目而学的 Python（最小集）

> 不是通用教程。每个语法点都配 **rtsp_stream.py 里的真实代码**，学完就能读懂这份文件。
> 建议：每天 1 小时，2 周内过完前 10 课；第 11 课开始动手改。

---

## 第 1 课：变量与打印

```python
width = 1024                 # 数字
name  = 'person'             # 字符串
ok    = True                 # 布尔值
print('摄像头: %dx%d' % (width, 768))     # 输出：摄像头: 1024x768
```

- `%d` 放数字、`%s` 放字符串、`%.2f` 保留两位小数，后面 `%(...)` 按顺序填。
- 本项目里的 `print('stream fps: %.1f | infer fps: %.1f' % (a, b))` 就是这个。

---

## 第 2 课：列表 list 与字典 dict

```python
CLASSES_ZH = ['人', '自行车', '汽车', ...]     # 列表：按顺序存，用下标取
CLASSES_ZH[0]        # → '人'（第 0 个，编程从 0 开始数）
len(CLASSES_ZH)      # → 80

_label_cache = {'key': None, 'items': []}      # 字典：用名字取
_label_cache['key']                            # → None
_label_cache['key'] = det                      # 赋值
```

**本项目里**：`_latest = [None]` 是个"只装一个东西的盒子"——因为函数里改外层变量需要可变对象，这是 Python 的小技巧。

---

## 第 3 课：判断与循环

```python
if args.wifi:                     # 如果为真
    print('wifi 模式')
else:
    print('默认模式')

while not _stop:                  # 只要条件成立就一直循环
    ok, f = cap.read()            # 读一帧
    if ok:
        _latest[0] = f            # 存最新帧

for dev in ('eth0', 'wlan0'):     # 依次取出每个元素
    print(dev)
```

**本项目里**：`capture_loop`、`infer_loop`、主循环都是 `while True/while not _stop` —— 线程就是"一直在转的循环"。

---

## 第 4 课：函数

```python
def dev_ip(dev):                  # def 定义函数，dev 是参数
    ...
    return None                   # return 返回结果；没写就返回 None

ip = dev_ip('eth0')               # 调用函数
```

**本项目里**：`build_launch()` 负责拼 GStreamer 管道串，`draw_boxes(frame, det)` 负责画框。函数就是把一段活打包、起个名字。

---

## 第 5 课：import（拿别人的工具）

```python
import os, sys, time              # 整包导入
import numpy as np                # 起别名 np，后面写 np.xxx
import cv2
from yolov8 import post_process   # 只拿里面的一个东西
from gi.repository import Gst     # 从包里拿模块
```

**本项目里**：`cv2` 是 OpenCV（图像处理）、`np` 是 numpy（数组运算）、`Gst` 是 GStreamer 的 Python 绑定、`RKNNLite` 是瑞芯微 NPU 的推理接口。

**`sys.path.insert(0, ZOO_PY)` 的意思**：把官方例程目录加进搜索路径，这样 `from yolov8 import ...` 才找得到。

---

## 第 6 课：类与对象（最关键的一课）

```python
rknn = RKNNLite()          # 类名() = 造一个对象（实例化）
rknn.load_rknn(model)      # 对象.方法() —— 让这个对象干活
rknn.init_runtime()

server = GstRtspServer.RTSPServer()   # 造一个 RTSP 服务器对象
server.props.address = '0.0.0.0'      # 对象.属性 = 值（配置它）
server.attach(None)                   # 让它开始监听
```

**一句话理解**：类 = 图纸，对象 = 按图纸造出来的实物。`rknn` 就是一个"NPU 推理器"，`server` 就是一台"RTSP 服务器"。

自己写类：

```python
class RtspServer:                     # 定义一个类
    def __init__(self):               # 造对象时自动执行（初始化）
        self.appsrc = None            # self.变量名 = 这个对象自己的属性
    def on_media_configure(self, factory, media):   # 这个对象的方法
        ...
```

`self` 就是"这个对象自己"，`self.appsrc` 是它记住的一个值。

---

## 第 7 课：numpy 数组与切片（图像就是数组）

```python
frame                # 一张 1024x768 的图，本质是 numpy 数组
frame.shape          # → (768, 1024, 3)  高、宽、通道(BGR)
frame.tobytes()      # 转成字节流，喂给 GStreamer

frame[y0:y1, x0:x1] = patch[:h, :w]     # 切片：把 patch 贴到图的某个矩形区域
```

**本项目里**：中文标签就是这么贴上去的——先渲染一小块文字图（patch），再用切片赋值贴到画面上。**切片赋值是 numpy 的 C 实现，很快，这是渲染优化的关键。**

---

## 第 8 课：线程与锁

```python
import threading

def capture_loop():          # 这个函数将在一个独立线程里跑
    while not _stop:
        ...

t = threading.Thread(target=capture_loop, daemon=True)
t.start()                    # 启动后，它会和主循环"同时"运行

_lock = threading.Lock()
with _lock:                  # 拿锁：同一时刻只有一个线程能进
    _latest[0] = f           # 改共享数据
                             # 出 with 自动放锁
```

**为什么要用锁**：多个线程同时改同一个变量会乱套。`with _lock:` 保证改的时候别人不能插手。

**GIL（必背）**：Python 有一个全局锁，**同一时刻只有一个线程能执行 Python 字节码**。所以：
- 主线程干重活 → 推理线程拿不到 GIL → 推理变慢
- 本项目真实案例：主循环每帧做整帧色彩转换，infer fps 从 10 掉到 5.6
- 解决办法：让主线程只做极轻的活（numpy 切片、C 实现的 cv2 调用会释放 GIL）

---

## 第 9 课：异常处理

```python
try:
    appsrc.emit('push-buffer', buf)     # 试着做
except Exception:
    rtsp.appsrc = None                  # 出错就走这里（客户端已断开）
finally:
    cap.release()                       # 无论对错最后都执行（Ctrl+C 也会走）
```

**本项目里**：客户端断开后 appsrc 失效，push 会抛异常，捕获后把 appsrc 置空，等下一个客户端接入。这就是"程序不会崩"的原因。

---

## 第 10 课：命令行参数

```python
ap = argparse.ArgumentParser()
ap.add_argument('--fps', type=int, default=30)     # 定义 --fps，默认 30
args = ap.parse_args()                             # 解析命令行

print(args.fps)      # 使用时：python3 rtsp_stream.py --fps 20 → 输出 20
```

**本项目里**：`--wifi`、`--fps`、`--bps`、`--no-infer` 全是这么来的。

---

## 第 11 课：读懂一份代码的顺序（心法）

不要从上往下逐行读。按这个顺序：

1. **先看 import 段**——知道用了哪些库（OpenCV / numpy / GStreamer / RKNN）
2. **跳到最下面的主循环**——知道程序的主干在干什么
3. **再看线程函数**——每个线程负责哪一段
4. **最后看辅助函数**——画框、拼管道串这些细节

**本项目主干只有 5 步**：取帧 → 送推理 → 画框 → 推给编码器 → 打印帧率。其余全是细节。

---

## 学习路径（每天 1 小时，两周）

| 天 | 内容 | 检验方式 |
|---|---|---|
| 1-2 | 第 1-4 课（变量/列表/判断/循环/函数） | 能说出 `_latest[0] = f` 在干什么 |
| 3-4 | 第 5-6 课（import / 类与对象） | 能说出 `rknn = RKNNLite()` 是什么 |
| 5-6 | 第 7-8 课（numpy 切片 / 线程与 GIL） | 能解释 GIL 为什么让推理变慢 |
| 7-8 | 第 9-10 课 + 通读 rtsp_stream.py 主干 | 能说出主循环 5 个步骤 |
| 9-10 | **动手改**：把框改成红色、把标签改成"目标:分数"格式 | 改完跑起来看效果 |
| 11-12 | **动手加**：在画面左上角画当前 fps | 独立完成一个小改动 |
| 13-14 | 对着讲解文档讲一遍给我听 | 能讲清 🟢 第一圈 7 条 |

---

## 最短自测：这 8 行能读懂，就算入门了

```python
cap = cv2.VideoCapture('/dev/video0')     # 打开摄像头
ok, frame = cap.read()                    # 读一帧
if ok:                                    # 读到了
    small = frame[0:100, 0:100]           # 切左上角 100x100
    cv2.rectangle(frame, (0,0), (100,100), (0,255,0), 2)   # 画绿框
    print('fps: %.1f' % (30.0))           # 打印
```

对照：这就是 `rtsp_stream.py` 主循环在做的事（读帧 → 画 → 送出去 → 统计）。整份代码只是它外面多包了线程和 RTSP。
