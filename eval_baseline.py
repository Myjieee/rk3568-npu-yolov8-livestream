"""精度基线：把人工数出来的三列数字算成 precision / recall / F1。

零依赖，Python 标准库即可，Ubuntu / 板子 / Windows 都能跑。

用法:
    python3 eval_baseline.py --template > ~/eval/counts.csv    # 生成空白表
    python3 eval_baseline.py ~/eval/counts.csv                 # 算指标
    python3 eval_baseline.py --total 107 42 60                 # 只有合计三个数
    python3 eval_baseline.py --demo                            # 看一遍示例

CSV 每张图一行, 三列(表头必须有):
    frame,P,TP,G
    0001,2,2,2

    P  = 画面上一共有几个检测框(含框错的)
    TP = 其中确实框住瓶子的有几个
    G  = 画面里真实有几个瓶子(人眼数, 不看模型)

由此还能推出:
    FP = P  - TP   误检(乱框)
    FN = G  - TP   漏检(没看见)
"""
import argparse
import csv
import sys

HEADER = ['frame', 'P', 'TP', 'G']


def compute(rows, total_mode=False):
    """rows: [(frame, P, TP, G), ...] -> dict 指标 + 逐帧明细"""
    detail = []
    sP = sTP = sG = 0
    for i, r in enumerate(rows, start=2):   # 从第 2 行开始(表头是第 1 行)
        frame = r.get('frame', '?') or str(i)
        try:
            P, TP, G = int(r['P']), int(r['TP']), int(r['G'])
        except (KeyError, ValueError):
            raise ValueError('第 %d 行(%s) 的 P/TP/G 必须是整数' % (i, frame))
        if min(P, TP, G) < 0:
            raise ValueError('第 %d 行(%s) 出现负数' % (i, frame))
        if TP > P:
            raise ValueError('第 %d 行(%s): TP=%d 比 P=%d 还大, 数错了'
                             % (i, frame, TP, P))
        if TP > G:
            raise ValueError('第 %d 行(%s): TP=%d 比真实瓶数 G=%d 还大, 数错了'
                             % (i, frame, TP, G))
        sP += P
        sTP += TP
        sG += G
        detail.append((frame, P, TP, G, P - TP, G - TP))

    if not detail:
        raise ValueError('表里一帧都没有')
    precision = sTP / sP if sP else 0.0
    recall = sTP / sG if sG else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {
        'n': len(detail), 'P': sP, 'TP': sTP, 'G': sG,
        'FP': sP - sTP, 'FN': sG - sTP,
        'precision': precision, 'recall': recall, 'f1': f1,
        'detail': detail, 'total_mode': total_mode,
    }


def report(m):
    print('=' * 52)
    if m['total_mode']:
        print('精度基线  |  合计输入')
    else:
        print('精度基线  |  %d 帧' % m['n'])
    print('=' * 52)
    print('  ΣP  = %3d   模型一共报了几个框' % m['P'])
    print('  ΣTP = %3d   其中框对的' % m['TP'])
    print('  ΣG  = %3d   真实一共有几个瓶子' % m['G'])
    print('-' * 52)
    print('  precision 精确率 = %5.1f%%   (误检 FP=%d)'
          % (m['precision'] * 100, m['FP']))
    print('  recall    召回率 = %5.1f%%   (漏检 FN=%d)'
          % (m['recall'] * 100, m['FN']))
    print('  F1              = %5.1f%%' % (m['f1'] * 100))
    print('-' * 52)

    if not m['total_mode']:
        bad = sorted(m['detail'], key=lambda d: (d[4] + d[5]), reverse=True)
        bad = [d for d in bad if (d[4] + d[5]) > 0][:5]
        if bad:
            print('  错得最多的帧(回看这几张, 通常是快速移动/遮挡/光照突变):')
            for frame, P, TP, G, FP, FN in bad:
                print('    %s  误检%d 漏检%d   (框%d / 对%d / 真%d)'
                      % (frame, FP, FN, P, TP, G))

    # 谁是大头, 先修哪个
    if m['FP'] > m['FN'] * 1.5:
        print('  诊断: 误检(%d)远多于漏检(%d) -> 框太松, 先把 --conf 调高'
              % (m['FP'], m['FN']))
    elif m['FN'] > m['FP'] * 1.5:
        if m['precision'] >= 0.85:
            print('  诊断: 漏检(%d)多、误检(%d)极少 -> 不缺"框得准", 缺"看得见"。'
                  % (m['FN'], m['FP']))
            print('        再调 --conf 只会更漏; 提召回只能走: 提高输入分辨率'
                  '(640->960)、只检测 ROI 区域、或换更大模型(n->s)。')
        else:
            print('  诊断: 漏检(%d)远多于误检(%d) -> 看得不够, 先查小目标/光照/'
                  '运动模糊, 或把 --conf 调低' % (m['FN'], m['FP']))
    else:
        print('  诊断: 误检(%d) 与漏检(%d) 相当 -> 模型本身到了能力边界'
              % (m['FP'], m['FN']))
    print('=' * 52)
    if m['total_mode']:
        print('提示: 这条只是个数字, 要看出改进必须"同一段素材 + 同一口径"')
        print('      前后各测一次; 口径不一致(比如 P 数了人、G 只数瓶子)')
        print('      算出来的 precision 是假的。')
    else:
        print('提示: 30 帧只是基线, 不是结论。前后对比必须回放同一段 A.mp4,')
        print('      否则素材变了, 数字没有可比性。')


def main():
    ap = argparse.ArgumentParser(description='精度基线指标计算')
    ap.add_argument('csv', nargs='?', help='手工填好的 counts.csv')
    ap.add_argument('--template', action='store_true', help='只打印空白 CSV 表头')
    ap.add_argument('--demo', action='store_true', help='用内置示例跑一遍')
    ap.add_argument('--total', nargs=3, type=int, metavar=('P', 'TP', 'G'),
                    help='直接给三个合计: --total 107 42 60')
    a = ap.parse_args()

    if a.template:
        print(','.join(HEADER))
        for i in range(1, 31):
            print('%04d,,,' % i)
        return

    if a.total is not None:
        P, TP, G = a.total
        rows = [{'frame': '合计', 'P': str(P), 'TP': str(TP), 'G': str(G)}]
        report(compute(rows, total_mode=True))
        return

    if a.demo:
        rows = [
            {'frame': '0001', 'P': '2', 'TP': '2', 'G': '2'},
            {'frame': '0002', 'P': '3', 'TP': '2', 'G': '3'},
            {'frame': '0003', 'P': '2', 'TP': '1', 'G': '3'},
        ]
    elif a.csv:
        with open(a.csv, newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
    else:
        ap.print_help()
        return

    report(compute(rows))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as e:
        print('出错: %s' % e, file=sys.stderr)
        sys.exit(1)
