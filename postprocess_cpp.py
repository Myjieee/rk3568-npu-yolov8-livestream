#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""postprocess_cpp.py — C++ 后处理的 Python 封装（ctypes）

只做三件事：加载 .so、把 numpy 数组的首地址交给 C++、把结果取回来。
**没有任何数据拷贝** —— ctypes 传的是指针，C++ 直接读写 numpy 的缓冲区。

用法（与官方 numpy 版 post_process 完全同签名，可直接替换）：
    from postprocess_cpp import post_process_cpp
    boxes, classes, scores = post_process_cpp(outputs)
    # 返回 None, None, None 表示这一帧没检出目标（与 numpy 版一致）

如果 .so 不存在会自动抛出清晰错误，方便在板子上定位。
"""
import os
import ctypes
import numpy as np

__all__ = ['post_process_cpp', 'load_lib', 'CPP_AVAILABLE']

# 一次后处理最多保留多少个候选框（进入 NMS 之前）。8400 个网格点全过阈值的
# 极端情况也够用；超出部分直接丢弃，避免写越界。
MAX_DET = 4096

_LIB = None
_ptr_f = ctypes.POINTER(ctypes.c_float)
_ptr_i = ctypes.POINTER(ctypes.c_int)
CPP_AVAILABLE = False


def load_lib(path=None):
    """加载 libpostprocess.so，并声明每个 C 函数的参数类型"""
    global _LIB, CPP_AVAILABLE
    if _LIB is not None:
        return _LIB
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'libpostprocess.so')
    if not os.path.exists(path):
        raise RuntimeError(
            '找不到 %s。先在板子上编译：\n'
            '  g++ -O3 -ffast-math -shared -fPIC -o libpostprocess.so '
            'postprocess.cpp' % path)

    lib = ctypes.CDLL(path)
    lib.decode_branch.restype = ctypes.c_int
    lib.decode_branch.argtypes = [
        _ptr_f, _ptr_f,                       # box, cls
        ctypes.c_int, ctypes.c_int, ctypes.c_int,   # h, w, nc
        ctypes.c_float, ctypes.c_float, ctypes.c_float,  # conf, img_w, img_h
        _ptr_f, _ptr_f, _ptr_i,               # out_box, out_score, out_cls
        ctypes.c_int,                         # max_det
    ]
    lib.nms_boxes.restype = ctypes.c_int
    lib.nms_boxes.argtypes = [
        _ptr_f, _ptr_f, _ptr_i,               # boxes, scores, cls
        ctypes.c_int, ctypes.c_float, _ptr_i,  # n, nms_thresh, keep
    ]
    _LIB = lib
    CPP_AVAILABLE = True
    return lib


# 复用的输出缓冲：避免每帧重新分配内存（numpy 分配也是要花时间的）
_buf_box = np.zeros((MAX_DET, 4), dtype=np.float32)
_buf_score = np.zeros(MAX_DET, dtype=np.float32)
_buf_cls = np.zeros(MAX_DET, dtype=np.int32)
_buf_keep = np.zeros(MAX_DET, dtype=np.int32)


def post_process_cpp(input_data, conf_thresh=0.25, nms_thresh=0.45,
                     img_size=(640, 640)):
    """YOLOv8 后处理（C++ 版）。返回 (boxes, classes, scores)，无目标时全 None

    input_data: rknn.inference 的输出列表，每个分支 2 个输出 [box, cls]
                box: [1][64][h][w]   cls: [1][80][h][w]
    """
    lib = load_lib()
    n_branch = 3
    pair = len(input_data) // n_branch      # 每个分支占几个输出（兼容 score_sum）

    total = 0
    for i in range(n_branch):
        box = input_data[pair * i]
        cls = input_data[pair * i + 1]
        # C++ 要的是连续内存的 float32，rknn 输出偶尔不是，这里兜一下
        if not box.flags['C_CONTIGUOUS']:
            box = np.ascontiguousarray(box)
        if not cls.flags['C_CONTIGUOUS']:
            cls = np.ascontiguousarray(cls)
        h, w = int(box.shape[2]), int(box.shape[3])
        nc = int(cls.shape[1])
        room = MAX_DET - total
        if room <= 0:
            break
        # 从第 total 个位置开始的一段视图(不是单个元素!)。切片才是数组,
        # 单个下标取出来是标量, 标量没有 .ctypes 属性 —— 之前就栽在这
        out_box = _buf_box[total:]
        out_score = _buf_score[total:]
        out_cls = _buf_cls[total:]
        k = lib.decode_branch(
            box.ctypes.data_as(_ptr_f), cls.ctypes.data_as(_ptr_f),
            ctypes.c_int(h), ctypes.c_int(w), ctypes.c_int(nc),
            ctypes.c_float(conf_thresh),
            ctypes.c_float(img_size[0]), ctypes.c_float(img_size[1]),
            out_box.ctypes.data_as(_ptr_f),
            out_score.ctypes.data_as(_ptr_f),
            out_cls.ctypes.data_as(_ptr_i),
            ctypes.c_int(room))
        total += k

    if total == 0:
        return None, None, None

    k = lib.nms_boxes(
        _buf_box.ctypes.data_as(_ptr_f),
        _buf_score.ctypes.data_as(_ptr_f),
        _buf_cls.ctypes.data_as(_ptr_i),
        ctypes.c_int(total), ctypes.c_float(nms_thresh),
        _buf_keep.ctypes.data_as(_ptr_i))

    if k <= 0:
        return None, None, None

    idx = _buf_keep[:k]
    # 拷贝出来：缓冲区下一帧会被覆盖，必须复制
    return (_buf_box[idx].copy(),
            _buf_cls[idx].astype(np.float32),
            _buf_score[idx].copy())


def available():
    """只探测不抛异常，供主程序决定要不要降级到 numpy 版"""
    try:
        load_lib()
        return True
    except Exception:
        return False
