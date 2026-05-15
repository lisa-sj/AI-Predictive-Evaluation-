"""
CS321M Predictive Evaluation Challenge
Step 1: 下载数据 + 探索性分析 (EDA)

运行方法:
    pip install datasets pandas matplotlib seaborn numpy
    python 01_download_and_eda.py

输出:
    - data/train.parquet  (本地缓存)
    - eda_report.txt      (统计摘要)
    - plots/              (可视化图表)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datasets import load_dataset

# ── 配置 ──────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent
DATA_DIR = OUTPUT_DIR / "data"
PLOT_DIR = OUTPUT_DIR / "plots"
DATA_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

# ── 1. 下载数据 ──────────────────────────────────────────
print("=" * 60)
print("Step 1: 下载 aims-foundations/measurement-db 数据集")
print("=" * 60)

cache_path = DATA_DIR / "train.parquet"

if cache_path.exists():
    print(f"从本地缓存加载: {cache_path}")
    df = pd.read_parquet(cache_path)
else:
    print("从 HuggingFace 下载中...")
    df = None

    # 方法1: 用 huggingface_hub 列出仓库文件, 找到 parquet, 直接用 pandas 读取
    # (完全绕过 datasets 库的 arrow 类型问题)
    try:
        print("  方法1: 用 huggingface_hub API 查找 parquet 文件...")
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi()
        repo_files = api.list_repo_files("aims-foundations/measurement-db", repo_type="dataset")
        parquet_files = [f for f in repo_files if f.endswith('.parquet')]
        print(f"    找到 {len(parquet_files)} 个 parquet 文件: {parquet_files}")

        if parquet_files:
            dfs = []
            for pf in parquet_files:
                print(f"    下载: {pf}")
                local_path = hf_hub_download(
                    repo_id="aims-foundations/measurement-db",
                    filename=pf,
                    repo_type="dataset"
                )
                dfs.append(pd.read_parquet(local_path))
            df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
            print(f"  ✓ 方法1 成功! 加载 {len(df):,} 行")
    except Exception as e1:
        print(f"  方法1 失败: {e1}")

    # 方法2: 直接用 pandas 从 HF URL 读取 parquet
    if df is None:
        try:
            print("  方法2: 用 pandas 直接从 HF URL 读取 parquet...")
            url = "https://huggingface.co/datasets/aims-foundations/measurement-db/resolve/main/measurement-db.parquet"
            df = pd.read_parquet(url)
            print(f"  ✓ 方法2 成功! 加载 {len(df):,} 行")
        except Exception as e2:
            print(f"  方法2 失败: {e2}")

    # 方法3: 用 datasets 库加载 (可能有 arrow 类型问题)
    if df is None:
        try:
            print("  方法3: 用 datasets 库直接加载...")
            ds = load_dataset("aims-foundations/measurement-db", split="test")
            df = pd.DataFrame(ds)
            print(f"  ✓ 方法3 成功! 加载 {len(df):,} 行")
        except Exception as e3:
            print(f"  方法3 失败: {e3}")

    # 方法4: 用 datasets 库 streaming 模式
    if df is None:
        try:
            print("  方法4: 用 datasets streaming 模式加载...")
            ds = load_dataset("aims-foundations/measurement-db", split="test", streaming=True)
            rows = []
            for i, row in enumerate(ds):
                rows.append(row)
                if (i + 1) % 100000 == 0:
                    print(f"    已加载 {i+1:,} 行...")
            df = pd.DataFrame(rows)
            print(f"  ✓ 方法4 成功! 加载 {len(df):,} 行")
        except Exception as e4:
            print(f"  方法4 也失败: {e4}")
            raise RuntimeError("所有下载方法都失败了，请检查网络连接或手动下载数据")
    df.to_parquet(cache_path)
    print(f"已保存到: {cache_path}")

print(f"\n数据集大小: {len(df):,} 行")
print(f"列名: {list(df.columns)}")
print(f"\n前5行:")
print(df.head())

# ── 2. 基本统计 ──────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 2: 基本统计")
print("=" * 60)

report = []

def log(msg):
    print(msg)
    report.append(msg)

log(f"总行数: {len(df):,}")
log(f"列: {list(df.columns)}")
log(f"\n数据类型:")
for col in df.columns:
    log(f"  {col}: {df[col].dtype}, 非空: {df[col].notna().sum():,}, 空: {df[col].isna().sum():,}")

# response 分布
log(f"\nResponse 分布:")
log(f"  均值 (总体通过率): {df['response'].mean():.4f}")
log(f"  0 (fail): {(df['response'] == 0).sum():,}")
log(f"  1 (pass): {(df['response'] == 1).sum():,}")
log(f"  NaN: {df['response'].isna().sum():,}")

# Benchmark 统计
n_benchmarks = df['benchmark_id'].nunique()
log(f"\nBenchmark 数量: {n_benchmarks}")
benchmark_stats = df.groupby('benchmark_id').agg(
    n_rows=('response', 'size'),
    n_subjects=('subject_id', 'nunique'),
    n_items=('item_id', 'nunique'),
    pass_rate=('response', 'mean')
).sort_values('n_rows', ascending=False)

log(f"\n前20大 Benchmarks:")
log(benchmark_stats.head(20).to_string())

# Subject 统计
n_subjects = df['subject_id'].nunique()
log(f"\n总 Subject (模型) 数量: {n_subjects}")
subject_stats = df.groupby('subject_id').agg(
    n_responses=('response', 'size'),
    n_benchmarks=('benchmark_id', 'nunique'),
    pass_rate=('response', 'mean')
).sort_values('n_responses', ascending=False)

log(f"\n前20个 Subjects:")
log(subject_stats.head(20).to_string())

# Item 统计
n_items = df['item_id'].nunique()
log(f"\n总 Item 数量: {n_items}")

# Test condition
log(f"\nTest Condition 分布:")
log(df['test_condition'].value_counts(dropna=False).head(20).to_string())

# ── 3. 可视化 ──────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 3: 生成可视化")
print("=" * 60)

# 3a. Benchmark 按行数排序
fig, ax = plt.subplots(figsize=(14, 8))
top_benchmarks = benchmark_stats.head(30)
ax.barh(range(len(top_benchmarks)), top_benchmarks['n_rows'])
ax.set_yticks(range(len(top_benchmarks)))
ax.set_yticklabels(top_benchmarks.index, fontsize=8)
ax.set_xlabel('Number of Rows')
ax.set_title('Top 30 Benchmarks by Row Count')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(PLOT_DIR / "benchmark_sizes.png", dpi=150)
plt.close()
print("  ✓ benchmark_sizes.png")

# 3b. 通过率分布 (按benchmark)
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(benchmark_stats['pass_rate'].dropna(), bins=30, edgecolor='black', alpha=0.7)
ax.set_xlabel('Pass Rate')
ax.set_ylabel('Number of Benchmarks')
ax.set_title('Distribution of Pass Rates across Benchmarks')
ax.axvline(x=benchmark_stats['pass_rate'].mean(), color='red', linestyle='--', label=f'Mean: {benchmark_stats["pass_rate"].mean():.3f}')
ax.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR / "benchmark_pass_rates.png", dpi=150)
plt.close()
print("  ✓ benchmark_pass_rates.png")

# 3c. Subject 通过率分布
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(subject_stats['pass_rate'].dropna(), bins=30, edgecolor='black', alpha=0.7, color='green')
ax.set_xlabel('Pass Rate')
ax.set_ylabel('Number of Subjects (Models)')
ax.set_title('Distribution of Pass Rates across Subjects')
ax.axvline(x=subject_stats['pass_rate'].mean(), color='red', linestyle='--', label=f'Mean: {subject_stats["pass_rate"].mean():.3f}')
ax.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR / "subject_pass_rates.png", dpi=150)
plt.close()
print("  ✓ subject_pass_rates.png")

# 3d. 稀疏性分析: 每个subject参与多少个benchmark
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(subject_stats['n_benchmarks'], bins=30, edgecolor='black', alpha=0.7, color='orange')
ax.set_xlabel('Number of Benchmarks per Subject')
ax.set_ylabel('Count')
ax.set_title('How Many Benchmarks Does Each Subject Participate In?')
plt.tight_layout()
plt.savefig(PLOT_DIR / "subject_benchmark_coverage.png", dpi=150)
plt.close()
print("  ✓ subject_benchmark_coverage.png")

# 3e. Response矩阵稀疏性
total_cells = n_subjects * n_items
filled_cells = len(df)
sparsity = 1 - filled_cells / total_cells
log(f"\nResponse Matrix 稀疏性:")
log(f"  总 subjects: {n_subjects}")
log(f"  总 items: {n_items}")
log(f"  理论矩阵大小: {total_cells:,}")
log(f"  实际填充: {filled_cells:,}")
log(f"  稀疏率: {sparsity:.4%}")

# ── 4. IRT 预分析 ──────────────────────────────────────
print("\n" + "=" * 60)
print("Step 4: IRT 预分析")
print("=" * 60)

# 每个item的难度 (1 - pass_rate 作为难度代理)
item_stats = df.groupby('item_id').agg(
    n_subjects=('subject_id', 'nunique'),
    pass_rate=('response', 'mean'),
    benchmark=('benchmark_id', 'first')
).reset_index()

log(f"\nItem 难度分布 (用 1 - pass_rate 近似):")
log(f"  最简单 items (pass_rate ≈ 1.0): {(item_stats['pass_rate'] > 0.95).sum():,}")
log(f"  最难 items (pass_rate < 0.05): {(item_stats['pass_rate'] < 0.05).sum():,}")
log(f"  中等难度 (0.3-0.7): {((item_stats['pass_rate'] >= 0.3) & (item_stats['pass_rate'] <= 0.7)).sum():,}")

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(item_stats['pass_rate'].dropna(), bins=50, edgecolor='black', alpha=0.7, color='purple')
ax.set_xlabel('Item Pass Rate')
ax.set_ylabel('Count')
ax.set_title('Distribution of Item Difficulty (Pass Rate)')
plt.tight_layout()
plt.savefig(PLOT_DIR / "item_difficulty_distribution.png", dpi=150)
plt.close()
print("  ✓ item_difficulty_distribution.png")

# ── 5. 保存报告 ──────────────────────────────────────────
report_path = OUTPUT_DIR / "eda_report.txt"
with open(report_path, "w") as f:
    f.write("\n".join(report))
print(f"\n报告已保存到: {report_path}")

# ── 6. 保存处理后的统计数据 ──────────────────────────────
benchmark_stats.to_csv(DATA_DIR / "benchmark_stats.csv")
subject_stats.to_csv(DATA_DIR / "subject_stats.csv")
item_stats.to_csv(DATA_DIR / "item_stats.csv", index=False)
print(f"统计数据已保存到 {DATA_DIR}/")

print("\n" + "=" * 60)
print("EDA 完成！")
print("=" * 60)
print(f"\n下一步: 运行 02_rasch_baseline.py 训练 Rasch 模型")
