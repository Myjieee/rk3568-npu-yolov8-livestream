// postprocess.cpp — YOLOv8 后处理 C++ 实现（dfl 解码 + 阈值过滤 + 按类 NMS）
//
// 为什么用 C++ 重写（这是本项目最有价值的一次优化）：
//   numpy 版每帧约 60~70ms，其中：
//     1) dfl() 对全部 8400 个网格点(80x80+40x40+20x20)无差别做 softmax，
//        但其中 95% 以上根本过不了置信度阈值 —— 纯浪费；
//     2) nms_boxes() 是 Python while 循环，每轮几十次 numpy 调用，
//        光函数调用开销就几毫秒；
//     3) transpose/concat/reshape 造了一堆临时数组。
//
// C++ 版的关键改变（顺序调整，不是单纯把代码翻译成 C++）：
//     **先算类别得分、过滤，只对通过的候选做 dfl 解码**
//   dfl 计算量因此降到原来的 5% 以下，这是最大的一块收益。
//
// 给 C 语言基础的同学：
//   - 下面全是普通 C 语法，只用到 vector/sort 两个 C++ 特性；
//   - `box[(p*16+m)*hw + i]` 就是二维数组下标，模型输出是 NCHW 排布：
//     通道优先，同一像素位置 i 的不同通道相隔 hw 个元素；
//   - `extern "C"` 表示按 C 规则导出函数名，Python ctypes 才能按名字找到它。
//
// 编译（板子端）：
//   g++ -O3 -ffast-math -shared -fPIC -o libpostprocess.so postprocess.cpp

#include <cmath>
#include <vector>
#include <algorithm>

// dfl 每组通道数：box 分支通道数 64 = 4 条边 × 16 个分箱
static const int DFL_MC = 16;

// ---------------------------------------------------------------------------
// 单个分支解码：拿到 box 分支(dfl 前的 64 通道) 和 cls 分支(80 通道)，
// 输出通过阈值的候选框（xyxy，640x640 输入坐标系）。
//
// 参数:
//   box        [1][64][h][w]  NCHW，dfl 前的边框分布
//   cls        [1][nc][h][w]  NCHW，各类别得分（模型已含 sigmoid，直接用）
//   conf_thresh 置信度阈值
//   img_w/h     输入尺寸（640x640），用于把网格坐标乘回像素坐标
//   out_box/out_score/out_cls  调用方预分配的输出缓冲
//   max_det     输出缓冲最多能装几个，防止写越界
// 返回: 本分支实际写入的候选框个数
// ---------------------------------------------------------------------------
extern "C" int decode_branch(
    const float* box, const float* cls,
    int h, int w, int nc,
    float conf_thresh, float img_w, float img_h,
    float* out_box, float* out_score, int* out_cls,
    int max_det)
{
    const int  hw = h * w;
    const float sx = img_w / (float)w;      // 网格 -> 像素的放大倍数(stride)
    const float sy = img_h / (float)h;
    int count = 0;

    // ---- 第 1 步：逐通道扫出每个点的最大类别得分 ----
    // 内存访问顺序是关键：NCHW 布局下同一通道的像素是连续的。
    // 逐点跳通道(外层点/内层通道)每次读都隔 hw 个元素, CPU 缓存全部失效,
    // 实测 8400 点 x 80 通道要 25ms；反过来(外层通道/内层点)顺序读只要 1~2ms。
    // 结果与 numpy 的 max/argmax 完全一致(并列时取先遇到的通道)。
    std::vector<float> best_s(hw, -1e9f);
    std::vector<int>   best_c(hw, 0);
    for (int c = 0; c < nc; ++c) {
        const float* p = cls + c * hw;      // 本通道连续铺开的所有点
        for (int i = 0; i < hw; ++i) {
            if (p[i] > best_s[i]) { best_s[i] = p[i]; best_c[i] = c; }
        }
    }

    // ---- 第 2 步：只对过阈值的点做 dfl 解码 ----
    for (int i = 0; i < hw; ++i) {
        if (best_s[i] < conf_thresh) continue;
        if (count >= max_det) break;

        // dfl：每条边 16 个分箱做 softmax，再按 0..15 加权求和，得到边的距离
        const int y = i / w;
        const int x = i - y * w;
        float dist[4];
        for (int p = 0; p < 4; ++p) {       // p: 0=left 1=top 2=right 3=bottom
            const float* src = box + (p * DFL_MC) * hw + i;
            // 先减最大值再 exp，防止溢出（softmax 标准写法）
            // 注：这里跨步读是故意的——只有过阈值的少数候选才走到这，
            //     次数太少，不值得为它再开一块临时数组换顺序
            float mx = src[0];
            for (int m = 1; m < DFL_MC; ++m) {
                float v = src[m * hw];
                if (v > mx) mx = v;
            }
            float sum = 0.f, acc = 0.f;
            for (int m = 0; m < DFL_MC; ++m) {
                float e = expf(src[m * hw] - mx);
                sum += e;
                acc += e * (float)m;
            }
            dist[p] = acc / sum;
        }

        const float cx = (float)x + 0.5f;
        const float cy = (float)y + 0.5f;
        out_box[count * 4 + 0] = (cx - dist[0]) * sx;   // x1
        out_box[count * 4 + 1] = (cy - dist[1]) * sy;   // y1
        out_box[count * 4 + 2] = (cx + dist[2]) * sx;   // x2
        out_box[count * 4 + 3] = (cy + dist[3]) * sy;   // y2
        out_score[count] = best_s[i];       // 该点最大的类别得分(数组)
        out_cls[count]   = best_c[i];       // 该点得分最高的类别编号(数组)
        ++count;
    }
    return count;
}

// ---------------------------------------------------------------------------
// 按类别分别做 NMS（与 numpy 版行为一致：不同类别之间互不抑制）
//
// 参数:
//   boxes/score/cls  候选框数组，长度 n
//   nms_thresh       IoU 超过这个值就抑制
//   keep             输出：保留下来的下标
// 返回: keep 的有效长度
// ---------------------------------------------------------------------------
extern "C" int nms_boxes(
    const float* boxes, const float* scores, const int* cls, int n,
    float nms_thresh, int* keep)
{
    if (n <= 0) return 0;

    // 按得分从高到低排序（C++ 的 sort，比 numpy argsort 快得多）
    std::vector<int> order(n);
    for (int i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [scores](int a, int b) {
        return scores[a] > scores[b];
    });

    std::vector<unsigned char> dead(n, 0);
    int k = 0;
    for (int ii = 0; ii < n; ++ii) {
        int i = order[ii];
        if (dead[i]) continue;
        keep[k++] = i;

        const float* bi = boxes + i * 4;
        float ax1 = bi[0], ay1 = bi[1], ax2 = bi[2], ay2 = bi[3];
        float area_i = (ax2 - ax1) * (ay2 - ay1);

        for (int jj = ii + 1; jj < n; ++jj) {
            int j = order[jj];
            if (dead[j]) continue;
            if (cls[j] != cls[i]) continue;      // 不同类不互相抑制

            const float* bj = boxes + j * 4;
            float ix1 = ax1 > bj[0] ? ax1 : bj[0];
            float iy1 = ay1 > bj[1] ? ay1 : bj[1];
            float ix2 = ax2 < bj[2] ? ax2 : bj[2];
            float iy2 = ay2 < bj[3] ? ay2 : bj[3];
            float iw = ix2 - ix1 + 0.00001f;     // 与 numpy 版一致，防止除 0
            float ih = iy2 - iy1 + 0.00001f;
            if (iw <= 0.f || ih <= 0.f) continue;

            float inter  = iw * ih;
            float area_j = (bj[2] - bj[0]) * (bj[3] - bj[1]);
            float ovr    = inter / (area_i + area_j - inter);
            if (ovr > nms_thresh) dead[j] = 1;
        }
    }
    return k;
}
