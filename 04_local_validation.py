"""
CS321M Predictive Evaluation Challenge
Step 4: 本地验证 — K-Fold Cold-Start 交叉验证 + Adaptive Labeling 模拟

关键改进 (相比单次 split):
  1. 5-Fold 交叉验证: 按 item 划分，每折20%的 items 作为 cold-start 测试集
  2. 真实 Adaptive Labeling 模拟: 用 acquisition_function 从测试集选 items → 揭示 label → predict
  3. 统计显著性: 报告每个指标的均值 ± 标准差

竞赛规则回顾:
  - 测试集的 items 全新 (cold-start)，subjects 在训练集中见过
  - 每 round: 平台给 N 个候选 → acquisition_function 排序 → 选 top-K 揭示 label
  - predict(input, labeled) 用已揭示的 labels 做在线校准

运行方法:
    python 04_local_validation.py [--folds 5] [--n-labeled 25] [--n-eval 500]

前置条件:
    先运行 02_train_model.py 生成模型文件
"""

import sys
import os
import json
import time
import math
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ── 参数解析 ─────────────────────────────────────────────
parser = argparse.ArgumentParser(description="CS321M 本地验证")
parser.add_argument('--folds', type=int, default=5, help='K-Fold 折数 (default: 5)')
parser.add_argument('--n-labeled', type=int, default=25, help='每 round 揭示的 labeled 数量 (default: 25)')
parser.add_argument('--n-eval', type=int, default=500, help='每折评估的样本数 (default: 500, 0=全部)')
parser.add_argument('--seed', type=int, default=42, help='随机种子')
args = parser.parse_args()

OUTPUT_DIR = Path(__file__).parent
DATA_DIR = OUTPUT_DIR / "data"
MODEL_DIR = OUTPUT_DIR / "models"
SUBMIT_DIR = OUTPUT_DIR / "submission"

np.random.seed(args.seed)

# ══════════════════════════════════════════════════════════
#  1. 加载数据 & 文本内容
# ══════════════════════════════════════════════════════════
print("=" * 65)
print("  加载数据")
print("=" * 65)

df = pd.read_parquet(DATA_DIR / "train.parquet")
df = df.dropna(subset=['response'])

# ── 关键: 过滤非二值 response (与训练一致) ──
original_len = len(df)
df = df[df['response'].isin([0, 1])]
df['response'] = df['response'].astype(int)
print(f"总数据: {original_len:,} 行 → 过滤非二值后 {len(df):,} 行")
print(f"  {df['item_id'].nunique():,} items, {df['subject_id'].nunique():,} subjects")

# 加载文本内容
item_text_map = {}
subject_text_map = {}

try:
    from torch_measure.datasets import load, list_datasets
    available = list_datasets()
    benchmarks_in_data = df['benchmark_id'].unique()
    for bname in benchmarks_in_data:
        candidates = [bname, bname.replace('_test', ''), bname.replace('_val', '')]
        for cand in candidates:
            if cand in available:
                try:
                    rm = load(cand)
                    if hasattr(rm, 'item_contents') and rm.item_contents:
                        for iid, text in zip(rm.item_ids, rm.item_contents):
                            item_text_map[iid] = text
                    if hasattr(rm, 'subject_ids'):
                        for sid in rm.subject_ids:
                            subject_text_map[sid] = sid
                except Exception:
                    pass
                break
    print(f"torch_measure: {len(item_text_map):,} item texts, {len(subject_text_map):,} subject names")
except ImportError:
    print("torch_measure 未安装，使用 fallback 文本")

if not item_text_map:
    item_pass = df.groupby('item_id').agg(
        benchmark=('benchmark_id', 'first'),
        correct_answer=('correct_answer', 'first'),
        pass_rate=('response', 'mean')
    )
    for item_id, row in item_pass.iterrows():
        ans = row['correct_answer'] if pd.notna(row['correct_answer']) else 'N/A'
        item_text_map[item_id] = f"Benchmark: {row['benchmark']}. Answer: {ans}. Difficulty: {1-row['pass_rate']:.2f}"

    subject_agg = df.groupby('subject_id').agg(
        pass_rate=('response', 'mean'),
        n_benchmarks=('benchmark_id', 'nunique')
    )
    for sid, row in subject_agg.iterrows():
        subject_text_map[sid] = f"AI model with accuracy {row['pass_rate']:.3f} across {row['n_benchmarks']} benchmarks"

condition_col = 'test_condition' if 'test_condition' in df.columns else 'condition'


def make_input_dict(row):
    """将 dataframe 行转为竞赛 input dict 格式"""
    cond = row[condition_col] if pd.notna(row.get(condition_col)) else "none"
    return {
        "benchmark": str(row['benchmark_id']),
        "condition": str(cond),
        "subject_content": subject_text_map.get(row['subject_id'], f"model {row['subject_id'][:8]}"),
        "item_content": item_text_map.get(row['item_id'], f"item {row['item_id'][:8]}"),
    }


def compute_metrics(predictions, labels):
    """计算竞赛指标: NLL, Accuracy, AUC"""
    p_clip = np.clip(predictions, 0.01, 0.99)
    nll = -np.mean(labels * np.log(p_clip) + (1 - labels) * np.log(1 - p_clip))
    neg_nll = -nll
    acc = np.mean((predictions > 0.5).astype(int) == labels)

    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(labels, predictions)
    except Exception:
        auc = float('nan')

    # Baseline: always guess mean
    base_rate = labels.mean()
    base_p = np.clip(np.full_like(predictions, base_rate), 0.01, 0.99)
    base_nll = -np.mean(labels * np.log(base_p) + (1 - labels) * np.log(1 - base_p))

    return {
        'neg_nll': neg_nll,
        'nll': nll,
        'accuracy': acc,
        'auc': auc,
        'base_nll': base_nll,
        'improvement_pct': (nll - base_nll) / base_nll * 100,
    }


# ══════════════════════════════════════════════════════════
#  2. 导入模型 & 准备环境
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  导入 submission 模块")
print("=" * 65)

sys.path.insert(0, str(SUBMIT_DIR))

import shutil
for fname in ['ncf_head.pt', 'condition_map.json']:
    src = MODEL_DIR / fname
    dst = SUBMIT_DIR / fname
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)

original_cwd = os.getcwd()
os.chdir(str(SUBMIT_DIR))

import model
import labeling
print("  ✓ model.py 和 labeling.py 导入成功")


# ══════════════════════════════════════════════════════════
#  3. K-Fold Cold-Start 交叉验证
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(f"  {args.folds}-Fold Cold-Start 交叉验证")
print("=" * 65)

all_items = df['item_id'].unique()
np.random.shuffle(all_items)

# 将 items 分成 K 折
fold_size = len(all_items) // args.folds
folds = []
for k in range(args.folds):
    start = k * fold_size
    end = start + fold_size if k < args.folds - 1 else len(all_items)
    folds.append(set(all_items[start:end]))

print(f"  总 items: {len(all_items):,}")
print(f"  每折约 {fold_size:,} items 作为 cold-start 测试集")

fold_results = []
fold_benchmark_results = defaultdict(list)

for fold_idx in range(args.folds):
    print(f"\n{'─' * 50}")
    print(f"  Fold {fold_idx + 1}/{args.folds}")
    print(f"{'─' * 50}")

    test_items = folds[fold_idx]
    df_fold_train = df[~df['item_id'].isin(test_items)]
    df_fold_test = df[df['item_id'].isin(test_items)]

    print(f"  训练集: {len(df_fold_train):,} 行 ({df_fold_train['item_id'].nunique():,} items)")
    print(f"  测试集: {len(df_fold_test):,} 行 ({df_fold_test['item_id'].nunique():,} items)")

    # ── 3a. Adaptive Labeling 模拟 ──
    # 竞赛真实流程:
    #   1. 平台给一批候选 (测试集的 items)
    #   2. acquisition_function 对每个候选打分
    #   3. 平台选 top-K 揭示 ground-truth label
    #   4. 下一 round 的 predict() 可以用这些 labeled 数据

    # Fix #14: 使用 reset_state() 正确重置 acquisition 状态
    labeling.reset_state()

    # 从测试集取候选 (模拟平台提供的候选池)
    n_candidates = min(200, len(df_fold_test))
    candidate_pool = df_fold_test.sample(n_candidates, random_state=fold_idx)

    print(f"  Adaptive labeling: 从 {n_candidates} 个候选中选 {args.n_labeled} 个")

    # 用 acquisition_function 给每个候选打分
    acq_scores = []
    acq_errors = 0
    for _, row in candidate_pool.iterrows():
        inp = make_input_dict(row)
        try:
            score = labeling.acquisition_function(inp)
            acq_scores.append(score)
        except Exception as e:
            acq_errors += 1
            if acq_errors <= 2:
                print(f"    ⚠ acquisition 错误: {type(e).__name__}: {e}")
            acq_scores.append(0.0)
    if acq_errors > 0:
        print(f"    acquisition 共 {acq_errors} 错误")

    # 选 top-K (最高分) 揭示 label
    acq_scores = np.array(acq_scores)
    top_k_idx = np.argsort(acq_scores)[-args.n_labeled:]

    labeled_list = []
    for idx in top_k_idx:
        row = candidate_pool.iloc[idx]
        d = make_input_dict(row)
        d["label"] = int(row['response'])
        labeled_list.append(d)

    # 检查 labeled 的多样性
    labeled_benchmarks = set(candidate_pool.iloc[i]['benchmark_id'] for i in top_k_idx)
    print(f"  Labeled 覆盖 {len(labeled_benchmarks)} 个 benchmarks: {sorted(labeled_benchmarks)}")

    # ── 3b. Cold-Start 预测 ──
    # 重置 Platt scaling (模拟新 round)
    model._calibrator = None

    # 评估样本
    n_eval = args.n_eval if args.n_eval > 0 else len(df_fold_test)
    n_eval = min(n_eval, len(df_fold_test))
    eval_sample = df_fold_test.sample(n_eval, random_state=fold_idx + 100)

    predictions = []
    labels = []
    errors = 0
    t0 = time.time()

    for idx, (_, row) in enumerate(eval_sample.iterrows()):
        inp = make_input_dict(row)
        try:
            p = model.predict(inp, labeled=labeled_list)
            assert isinstance(p, float) and 0 <= p <= 1 and not math.isnan(p)
            predictions.append(p)
            labels.append(int(row['response']))
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"    ⚠ 预测错误 #{errors}: {type(e).__name__}: {e}")

        if (idx + 1) % 200 == 0:
            print(f"    已完成 {idx+1}/{n_eval}...")

    elapsed = time.time() - t0
    predictions = np.array(predictions)
    labels = np.array(labels)

    print(f"  预测完成: {len(predictions)}/{n_eval} 成功 ({errors} 错误), 耗时 {elapsed:.1f}s")

    # ── 3c. 计算指标 ──
    if len(predictions) == 0:
        print(f"  ❌ 所有预测失败! 跳过此折")
        fold_results.append({'neg_nll': float('nan'), 'nll': float('nan'),
                             'accuracy': float('nan'), 'auc': float('nan'),
                             'base_nll': float('nan'), 'improvement_pct': float('nan')})
        continue

    metrics = compute_metrics(predictions, labels)
    fold_results.append(metrics)

    print(f"  NLL: {metrics['neg_nll']:+.4f} | Acc: {metrics['accuracy']:.4f} | "
          f"AUC: {metrics['auc']:.4f} | vs baseline: {metrics['improvement_pct']:+.1f}%")

    # 按 benchmark 分组
    eval_df = eval_sample.iloc[:len(predictions)].copy()
    eval_df['pred'] = predictions
    eval_df['label'] = labels

    for bname, grp in eval_df.groupby('benchmark_id'):
        if len(grp) >= 5:
            bm_metrics = compute_metrics(grp['pred'].values, grp['label'].values)
            fold_benchmark_results[bname].append(bm_metrics)


# ══════════════════════════════════════════════════════════
#  4. 汇总结果
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  交叉验证汇总结果")
print("=" * 65)

metric_names = ['neg_nll', 'nll', 'accuracy', 'auc']
metric_labels = {
    'neg_nll': 'Neg Log-Loss (主指标, ↑越高越好)',
    'nll': 'Log-Loss (↓越低越好)',
    'accuracy': 'Accuracy',
    'auc': 'AUC-ROC',
}

print(f"\n  ┌────────────────────────────────────────────────────────────┐")
print(f"  │  {args.folds}-Fold Cold-Start 交叉验证结果 (mean ± std)               │")
print(f"  ├────────────────────────────────────────────────────────────┤")

for m in metric_names:
    vals = [r[m] for r in fold_results]
    mean_v = np.mean(vals)
    std_v = np.std(vals)
    label = metric_labels[m]
    print(f"  │  {label:35s}: {mean_v:>7.4f} ± {std_v:.4f}  │")

# Baseline comparison
imp_vals = [r['improvement_pct'] for r in fold_results]
print(f"  │  {'vs Baseline':35s}: {np.mean(imp_vals):>+6.1f}% ± {np.std(imp_vals):.1f}%    │")
print(f"  └────────────────────────────────────────────────────────────┘")

# 每折详情
print(f"\n  每折详情:")
print(f"  {'Fold':>6s} | {'NLL':>8s} | {'Acc':>8s} | {'AUC':>8s} | {'vs Base':>8s}")
print(f"  {'─'*6} | {'─'*8} | {'─'*8} | {'─'*8} | {'─'*8}")
for k, r in enumerate(fold_results):
    print(f"  {k+1:>6d} | {r['neg_nll']:>+8.4f} | {r['accuracy']:>8.4f} | {r['auc']:>8.4f} | {r['improvement_pct']:>+7.1f}%")

# 按 Benchmark 汇总
print(f"\n  按 Benchmark 汇总 (mean ± std across folds):")
print(f"  {'Benchmark':>25s} | {'NLL':>15s} | {'Acc':>15s} | {'AUC':>15s}")
print(f"  {'─'*25} | {'─'*15} | {'─'*15} | {'─'*15}")
for bname in sorted(fold_benchmark_results.keys()):
    bm_list = fold_benchmark_results[bname]
    if len(bm_list) >= 2:
        nll_vals = [r['neg_nll'] for r in bm_list]
        acc_vals = [r['accuracy'] for r in bm_list]
        auc_vals = [r['auc'] for r in bm_list]
        print(f"  {bname:>25s} | {np.mean(nll_vals):+.4f}±{np.std(nll_vals):.3f} | "
              f"{np.mean(acc_vals):.4f}±{np.std(acc_vals):.3f} | "
              f"{np.mean(auc_vals):.4f}±{np.std(auc_vals):.3f}")

# 稳定性分析
print(f"\n  模型稳定性分析:")
nll_vals = [r['nll'] for r in fold_results]
nll_cv = np.std(nll_vals) / np.mean(nll_vals) * 100
acc_vals = [r['accuracy'] for r in fold_results]
auc_vals = [r['auc'] for r in fold_results]

print(f"    NLL 变异系数 (CV): {nll_cv:.1f}% {'← 稳定 (<5%)' if nll_cv < 5 else '← 波动较大 (>5%)'}")
print(f"    Accuracy 范围: [{min(acc_vals):.4f}, {max(acc_vals):.4f}]")
print(f"    AUC 范围:      [{min(auc_vals):.4f}, {max(auc_vals):.4f}]")


# ══════════════════════════════════════════════════════════
#  5. Acquisition Function 质量评估
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  Acquisition Function 质量评估")
print("=" * 65)

# 对比: adaptive labeling vs random labeling
print(f"  对比实验: Adaptive (acquisition) vs Random labeling")

# 用最后一折做对比实验
test_items_last = folds[-1]
df_last_test = df[df['item_id'].isin(test_items_last)]

# Random labeling baseline — sample labeled set first, then exclude from eval
random_labeled_idx = np.random.choice(len(df_last_test), args.n_labeled, replace=False)
random_labeled = df_last_test.iloc[random_labeled_idx]
random_labeled_list = []
for _, row in random_labeled.iterrows():
    d = make_input_dict(row)
    d["label"] = int(row['response'])
    random_labeled_list.append(d)

# Exclude labeled items from eval set to prevent label leak
labeled_iloc_set = set(random_labeled_idx)
df_eval_pool = df_last_test.iloc[[i for i in range(len(df_last_test)) if i not in labeled_iloc_set]]
n_eval_cmp = min(300, len(df_eval_pool))
eval_cmp = df_eval_pool.sample(n_eval_cmp, random_state=999)

# Predict with random labeled
model._calibrator = None
preds_random = []
labels_cmp = []
cmp_errors = 0
for _, row in eval_cmp.iterrows():
    inp = make_input_dict(row)
    try:
        p = model.predict(inp, labeled=random_labeled_list)
        preds_random.append(p)
        labels_cmp.append(int(row['response']))
    except Exception as e:
        cmp_errors += 1
        if cmp_errors <= 2:
            print(f"    ⚠ 对比预测错误: {type(e).__name__}: {e}")
if cmp_errors > 0:
    print(f"  对比实验共 {cmp_errors} 错误")

# Predict with adaptive labeled (reuse last fold's labeled_list)
model._calibrator = None
preds_adaptive = []
for _, row in eval_cmp.iloc[:len(preds_random)].iterrows():
    inp = make_input_dict(row)
    try:
        p = model.predict(inp, labeled=labeled_list)
        preds_adaptive.append(p)
    except Exception:
        preds_adaptive.append(0.5)

preds_random = np.array(preds_random)
preds_adaptive = np.array(preds_adaptive[:len(preds_random)])
labels_cmp = np.array(labels_cmp)

m_random = compute_metrics(preds_random, labels_cmp)
m_adaptive = compute_metrics(preds_adaptive, labels_cmp)

print(f"\n  {'策略':>15s} | {'NLL':>8s} | {'Acc':>8s} | {'AUC':>8s}")
print(f"  {'─'*15} | {'─'*8} | {'─'*8} | {'─'*8}")
print(f"  {'Random':>15s} | {m_random['neg_nll']:>+8.4f} | {m_random['accuracy']:>8.4f} | {m_random['auc']:>8.4f}")
print(f"  {'Adaptive':>15s} | {m_adaptive['neg_nll']:>+8.4f} | {m_adaptive['accuracy']:>8.4f} | {m_adaptive['auc']:>8.4f}")

nll_diff = m_random['nll'] - m_adaptive['nll']
print(f"\n  Adaptive vs Random NLL 差异: {nll_diff:+.4f} "
      f"{'← Adaptive 更好' if nll_diff > 0 else '← Random 更好 (acquisition 需要改进)'}")


# ══════════════════════════════════════════════════════════
#  6. 提交文件完整性检查
# ══════════════════════════════════════════════════════════
os.chdir(original_cwd)

print("\n" + "=" * 65)
print("  提交文件完整性检查")
print("=" * 65)

required_files = {
    "model.py": "必需 — predict() 函数",
    "labeling.py": "可选 — acquisition_function()",
    "models.txt": "必需 — 声明 HF 模型",
    "requirements.txt": "可选 — 依赖",
    "ncf_head.pt": "必需 — 模型权重",
    "condition_map.json": "必需 — condition 映射",
}

all_ok = True
for fname, desc in required_files.items():
    path = SUBMIT_DIR / fname
    if path.exists():
        size = path.stat().st_size
        print(f"  ✓ {fname:25s} ({size:>10,} bytes) — {desc}")
    else:
        if "必需" in desc:
            print(f"  ❌ {fname:25s} 缺少! — {desc}")
            all_ok = False
        else:
            print(f"  ⚠ {fname:25s} 缺少 — {desc}")

if all_ok:
    print(f"\n✅ 所有必需文件就绪! 运行 03_create_submission.py 打包提交")
else:
    print(f"\n❌ 有必需文件缺失，请先运行 02_train_model.py")

print(f"\n{'=' * 65}")
print(f"  本地验证完成!")
print(f"  交叉验证 {args.folds} 折, 每折 cold-start {fold_size} items")
print(f"  Adaptive labeling: {args.n_labeled} labeled / round")
print(f"{'=' * 65}")
