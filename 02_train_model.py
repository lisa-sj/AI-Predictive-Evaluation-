"""
CS321M Predictive Evaluation Challenge
Step 2: 训练模型 (Rasch IRT + NCF with Condition Embedding)

重要: 这个脚本需要先用 torch_measure 获取真实文本内容，
      不能只用 subject_id/item_id 哈希值做 embedding。

运行方法:
    pip install torch numpy pandas scikit-learn sentence-transformers
    pip install torch-measure   # AIMS 提供的测量工具包
    python 02_train_model.py

输出:
    - models/rasch_params.npz       (IRT参数: θ和β)
    - models/ncf_head.pt            (NCF MLP权重)
    - models/condition_map.json     (condition -> index映射)
    - models/config.json            (模型配置)
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from collections import defaultdict

# ── 配置 ──────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent
DATA_DIR = OUTPUT_DIR / "data"
MODEL_DIR = OUTPUT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── 1. 加载数据 ──────────────────────────────────────────
print("=" * 60)
print("Step 1: 加载数据")
print("=" * 60)

df = pd.read_parquet(DATA_DIR / "train.parquet")
print(f"加载 {len(df):,} 行数据")
print(f"列名: {list(df.columns)}")

# Fix #9: 先过滤非二值 response，再 astype(int)
# 原来的顺序 (dropna → astype(int) → 再过滤) 会导致 float 值
# 如 0.7 被 astype(int) 截断为 0，造成标签错误
df = df.dropna(subset=['response'])
original_len = len(df)
df = df[df['response'].isin([0, 1])]  # 先过滤，确保只有 0.0 和 1.0
df['response'] = df['response'].astype(int)  # 再安全转换
print(f"有效数据: {original_len:,} 行 → 过滤非二值后 {len(df):,} 行 (去除 {original_len - len(df):,} 行)")
print(f"Response 分布: 0={( df['response']==0).sum():,}, 1={(df['response']==1).sum():,}")

# ── 1b. 获取真实文本内容 ─────────────────────────────────
# 关键: measurement-db 的 HF 数据只有 subject_id/item_id (哈希值)
# 真实文本在 torch-measure 的 .pt 文件里 (item_contents, subject_ids=模型名)
# 或者在各 benchmark 的 item_content.csv 里

print("\n尝试用 torch_measure 加载真实文本...")

subject_text_map = {}  # subject_id -> 文本描述
item_text_map = {}     # item_id -> 文本内容

try:
    from torch_measure.datasets import load, list_datasets
    available = list_datasets()
    print(f"torch_measure 有 {len(available)} 个可用 benchmark")

    benchmarks_in_data = df['benchmark_id'].unique()
    for bname in benchmarks_in_data:
        # 尝试匹配 benchmark 名（可能需要去掉后缀如 _test）
        candidates = [bname, bname.replace('_test', ''), bname.replace('_val', '')]
        for cand in candidates:
            if cand in available:
                try:
                    rm = load(cand)
                    # 填充 item 文本
                    if hasattr(rm, 'item_contents') and rm.item_contents:
                        for iid, text in zip(rm.item_ids, rm.item_contents):
                            item_text_map[iid] = text
                    # 填充 subject 文本 (subject_ids 通常是模型名)
                    if hasattr(rm, 'subject_ids'):
                        for sid in rm.subject_ids:
                            # subject_id 在 dataset 里是哈希
                            # 需要匹配——用 subject 在该 benchmark 的性能
                            subject_text_map[sid] = sid  # 暂用模型名
                    print(f"  ✓ {cand}: {len(rm.item_ids)} items, {len(rm.subject_ids)} subjects")
                except Exception as e:
                    print(f"  ✗ {cand}: {e}")
                break

    print(f"\n获得 {len(item_text_map):,} 个 item 文本, {len(subject_text_map):,} 个 subject 名称")

except ImportError:
    print("⚠ torch_measure 未安装，使用 benchmark 元数据作为替代特征")
    print("  安装方法: pip install torch-measure")

# 如果 torch_measure 不可用，用 benchmark 级别的统计特征作为替代
# (这是一个 fallback，不如真实文本好)
if not item_text_map:
    print("\n使用 fallback: benchmark 名 + 统计特征作为文本")

    # 用 groupby 一次性计算所有 item 的统计特征 (比逐行过滤快几百倍)
    print("  计算 item 统计特征 (groupby)...")
    item_agg = df.groupby('item_id').agg(
        benchmark=('benchmark_id', 'first'),
        correct_answer=('correct_answer', 'first'),
        pass_rate=('response', 'mean')
    )
    for item_id, row in item_agg.iterrows():
        ans = row['correct_answer'] if pd.notna(row['correct_answer']) else 'N/A'
        item_text_map[item_id] = f"Benchmark: {row['benchmark']}. Answer: {ans}. Difficulty: {1-row['pass_rate']:.2f}"
    print(f"  ✓ {len(item_text_map):,} items 完成")

    # 用 groupby 一次性计算所有 subject 的统计特征
    print("  计算 subject 统计特征 (groupby)...")
    subject_agg = df.groupby('subject_id').agg(
        pass_rate=('response', 'mean'),
        n_benchmarks=('benchmark_id', 'nunique')
    )
    for subject_id, row in subject_agg.iterrows():
        subject_text_map[subject_id] = (
            f"AI model with accuracy {row['pass_rate']:.3f} across {int(row['n_benchmarks'])} benchmarks"
        )
    print(f"  ✓ {len(subject_text_map):,} subjects 完成")

# ── 2. Stage 1: Rasch 模型参数估计 (JMLE) ────────────────
print("\n" + "=" * 60)
print("Step 2: Rasch 模型参数估计 (Joint MLE)")
print("=" * 60)

subject_ids = df['subject_id'].unique()
item_ids = df['item_id'].unique()
subject2idx = {s: i for i, s in enumerate(subject_ids)}
item2idx = {it: i for i, it in enumerate(item_ids)}

n_subjects = len(subject_ids)
n_items = len(item_ids)
print(f"Subjects: {n_subjects}, Items: {n_items}")

# PyTorch JMLE
print("开始 JMLE 优化...")

s_idx = torch.tensor([subject2idx[s] for s in df['subject_id']], dtype=torch.long)
i_idx = torch.tensor([item2idx[it] for it in df['item_id']], dtype=torch.long)
y = torch.tensor(df['response'].values, dtype=torch.float32)

theta = nn.Parameter(torch.zeros(n_subjects))
beta = nn.Parameter(torch.zeros(n_items))

# 用通过率初始化
item_pass_rates = df.groupby('item_id')['response'].mean()
for it_name, idx in item2idx.items():
    pr = np.clip(item_pass_rates.get(it_name, 0.5), 0.01, 0.99)
    beta.data[idx] = -np.log(pr / (1 - pr))

subject_pass_rates = df.groupby('subject_id')['response'].mean()
for s_name, idx in subject2idx.items():
    pr = np.clip(subject_pass_rates.get(s_name, 0.5), 0.01, 0.99)
    theta.data[idx] = np.log(pr / (1 - pr))

optimizer = torch.optim.Adam([theta, beta], lr=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, min_lr=1e-4)
bce = nn.BCEWithLogitsLoss()
batch_size = 65536
n_batches = (len(s_idx) + batch_size - 1) // batch_size
best_rasch_loss = float('inf')

for epoch in range(50):
    perm = torch.randperm(len(s_idx))
    total_loss = 0
    for b in range(n_batches):
        idx = perm[b * batch_size : (b + 1) * batch_size]
        logits = theta[s_idx[idx]] - beta[i_idx[idx]]
        loss = bce(logits, y[idx]) + 1e-4 * (theta.pow(2).mean() + beta.pow(2).mean())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / n_batches
    scheduler.step(avg_loss)

    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            all_logits = theta[s_idx] - beta[i_idx]
            acc = ((torch.sigmoid(all_logits) > 0.5).float() == y).float().mean()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | Acc: {acc:.4f} | LR: {current_lr:.5f}")

    # Early stopping for Rasch
    if avg_loss < best_rasch_loss:
        best_rasch_loss = avg_loss
        best_theta = theta.detach().clone()
        best_beta = beta.detach().clone()

# Use best parameters (not last epoch which may have degraded)
theta.data.copy_(best_theta)
beta.data.copy_(best_beta)

theta_np = theta.detach().numpy()
beta_np = beta.detach().numpy()
np.savez(MODEL_DIR / "rasch_params.npz", theta=theta_np, beta=beta_np,
         subject_ids=subject_ids, item_ids=item_ids)
print(f"Rasch 参数已保存: θ [{theta_np.min():.2f}, {theta_np.max():.2f}], β [{beta_np.min():.2f}, {beta_np.max():.2f}]")

# ── 3. 计算 Sentence Embeddings ──────────────────────────
print("\n" + "=" * 60)
print("Step 3: 计算 Sentence Embeddings")
print("=" * 60)

# Condition 映射 — 加入 "unknown" 作为 fallback
condition_col = 'test_condition' if 'test_condition' in df.columns else 'condition'
conditions = list(df[condition_col].fillna('none').unique())
if 'unknown' not in conditions:
    conditions.append('unknown')  # fallback for unseen conditions at test time
condition2idx = {c: i for i, c in enumerate(conditions)}
n_conditions = len(conditions)
print(f"Conditions: {n_conditions} 种 (含 unknown fallback)")

with open(MODEL_DIR / "condition_map.json", "w") as f:
    json.dump(condition2idx, f)

print("加载 Sentence Transformer...")
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("all-mpnet-base-v2")

# 编码 subjects
print(f"编码 {n_subjects} 个 subjects...")
subject_text_list = [subject_text_map.get(s, f"AI model {s[:8]}") for s in subject_ids]
subject_embs = encoder.encode(subject_text_list, batch_size=256, show_progress_bar=True)
np.save(MODEL_DIR / "subject_embeddings.npy", subject_embs)

# 编码 items
print(f"编码 {n_items} 个 items...")
item_text_list = [item_text_map.get(it, f"item {it[:8]}") for it in item_ids]
item_embs = encoder.encode(item_text_list, batch_size=256, show_progress_bar=True)
np.save(MODEL_DIR / "item_embeddings.npy", item_embs)

EMB_DIM = subject_embs.shape[1]
print(f"Embedding 维度: {EMB_DIM}")

# ── 4. 训练 NCF + Condition Embedding ────────────────────
print("\n" + "=" * 60)
print("Step 4: 训练 NCF + Condition Embedding (Stage 2)")
print("=" * 60)

COND_DIM = 32
HIDDEN = 256

# 架构与 submission/model.py 保持同步 — 使用 BatchNorm1d
# 当前已训练权重 (ncf_head.pt) 使用 BatchNorm，推理时 NCF.eval()
# 会使用 running statistics，batch_size=1 不影响正确性。
# 未来考虑切换 LayerNorm 时需同步两个文件并重新训练。
class NCFWithCondition(nn.Module):
    def __init__(self, emb_dim, n_conditions, cond_dim=32, hidden=256):
        super().__init__()
        self.condition_emb = nn.Embedding(n_conditions, cond_dim)
        input_dim = 2 * emb_dim + cond_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, u, v, c):
        c_emb = self.condition_emb(c)
        x = torch.cat([u, v, c_emb], dim=-1)
        return self.mlp(x).squeeze(-1)

# 准备训练数据 (内存优化: 只存索引, 训练时按 batch 查找 embedding)
print("准备 NCF 训练数据 (内存优化模式)...")

# 只存索引, 不展开 embedding (节省 ~30GB 内存)
# L2 归一化 embedding, 防止数值溢出
print("  L2 归一化 embeddings...")
from sklearn.preprocessing import normalize
subject_embs = normalize(subject_embs, norm='l2', axis=1)
item_embs = normalize(item_embs, norm='l2', axis=1)
# 重新保存归一化后的 embeddings
np.save(MODEL_DIR / "subject_embeddings.npy", subject_embs)
np.save(MODEL_DIR / "item_embeddings.npy", item_embs)

subject_embs_t = torch.tensor(subject_embs, dtype=torch.float32)  # [n_subjects, 768]
item_embs_t = torch.tensor(item_embs, dtype=torch.float32)        # [n_items, 768]
print(f"  Embedding 范围: subject [{subject_embs.min():.4f}, {subject_embs.max():.4f}], item [{item_embs.min():.4f}, {item_embs.max():.4f}]")

S_idx = torch.tensor([subject2idx[s] for s in df['subject_id']], dtype=torch.long)
I_idx = torch.tensor([item2idx[it] for it in df['item_id']], dtype=torch.long)
C = torch.tensor([condition2idx.get(c, condition2idx['unknown'])
                   for c in df[condition_col].fillna('none')], dtype=torch.long)
Y = torch.tensor(df['response'].values, dtype=torch.float32)

# 数据验证
print(f"  Y 范围: [{Y.min().item()}, {Y.max().item()}], 均值: {Y.mean().item():.4f}")
assert Y.min().item() >= 0 and Y.max().item() <= 1, f"Y 范围异常! [{Y.min()}, {Y.max()}]"

print(f"  Embedding 表: subjects {subject_embs_t.shape}, items {item_embs_t.shape}")
print(f"  索引数组: {len(S_idx):,} 行 (仅存 int64 索引, 不展开 embedding)")

# 80/20 split
n = len(Y)
perm = torch.randperm(n)
n_train = int(0.8 * n)
train_idx = perm[:n_train]
val_idx = perm[n_train:]
print(f"训练集: {n_train:,}, 验证集: {n - n_train:,}")

model = NCFWithCondition(EMB_DIM, n_conditions, COND_DIM, HIDDEN).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, min_lr=1e-5)
criterion = nn.BCEWithLogitsLoss()
MAX_GRAD_NORM = 1.0  # gradient clipping

batch_size = 4096
# Fix #7: 用 NLL (竞赛主指标) 选最佳模型，而非 accuracy
best_val_nll = float('inf')
best_val_acc = 0.0
patience_counter = 0

for epoch in range(50):
    model.train()
    epoch_loss = 0
    n_b = 0
    shuf = train_idx[torch.randperm(len(train_idx))]
    for b_start in range(0, len(shuf), batch_size):
        idx = shuf[b_start : b_start + batch_size]
        # 按 batch 查找 embedding (不预展开, 省内存)
        u_batch = subject_embs_t[S_idx[idx]].to(DEVICE)
        v_batch = item_embs_t[I_idx[idx]].to(DEVICE)
        logits = model(u_batch, v_batch, C[idx].to(DEVICE))
        loss = criterion(logits, Y[idx].to(DEVICE))

        # 诊断: 第一个 epoch 的第一个 batch 打印 loss 详情
        if epoch == 0 and n_b == 0:
            print(f"  [诊断] 第一个 batch:")
            print(f"    logits 范围: [{logits.min().item():.4f}, {logits.max().item():.4f}], 均值: {logits.mean().item():.4f}")
            print(f"    Y batch 范围: [{Y[idx].min().item()}, {Y[idx].max().item()}]")
            print(f"    loss.item(): {loss.item():.6f}")
            print(f"    loss 是否 NaN: {torch.isnan(loss).item()}, 是否 Inf: {torch.isinf(loss).item()}")

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        epoch_loss += loss.item()
        n_b += 1

    avg_train_loss = epoch_loss / n_b

    # 验证
    model.eval()
    with torch.no_grad():
        val_logits_all = []
        for b_start in range(0, len(val_idx), batch_size):
            idx = val_idx[b_start : b_start + batch_size]
            u_batch = subject_embs_t[S_idx[idx]].to(DEVICE)
            v_batch = item_embs_t[I_idx[idx]].to(DEVICE)
            val_logits_all.append(model(u_batch, v_batch, C[idx].to(DEVICE)).cpu())
        val_logits = torch.cat(val_logits_all)
        val_y = Y[val_idx]

        # 手动计算 log-loss (不依赖 criterion, 更可靠)
        val_probs = torch.sigmoid(val_logits).clamp(0.01, 0.99)
        log_loss = -(val_y * torch.log(val_probs) + (1 - val_y) * torch.log(1 - val_probs)).mean().item()
        val_acc = ((val_probs > 0.5).float() == val_y).float().mean().item()

    scheduler.step(log_loss)  # 用 NLL 让 ReduceLROnPlateau 在 NLL 不降时降 LR
    current_lr = optimizer.param_groups[0]['lr']

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:3d} | Train: {avg_train_loss:.4f} | "
              f"LogLoss: {log_loss:.4f} | Acc: {val_acc:.4f} | LR: {current_lr:.6f}")

    # Fix #7: 用 NLL 选最佳模型 (与竞赛评估指标对齐)
    if log_loss < best_val_nll:
        best_val_nll = log_loss
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_DIR / "ncf_head.pt")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= 8:
            print(f"  Early stopping at epoch {epoch+1}")
            break

print(f"\n最佳验证 NLL: {best_val_nll:.4f} (Accuracy: {best_val_acc:.4f})")

# 保存配置
config = {
    "n_subjects": n_subjects,
    "n_items": n_items,
    "n_conditions": n_conditions,
    "emb_dim": EMB_DIM,
    "cond_dim": COND_DIM,
    "hidden_dim": HIDDEN,
    "best_val_acc": float(best_val_acc),
    "best_val_nll": float(best_val_nll),  # Fix #7: 记录 NLL
}

# Fix #4: 保存 subject_text → θ 映射供推理使用
rasch_subject_map = {}
for sid, idx in subject2idx.items():
    text = subject_text_map.get(sid, f"AI model {sid[:8]}")
    rasch_subject_map[text] = float(theta_np[idx])
with open(MODEL_DIR / "rasch_subject_map.json", "w") as f:
    json.dump(rasch_subject_map, f)
print(f"Rasch subject map 已保存: {len(rasch_subject_map)} 个 subject")
with open(MODEL_DIR / "config.json", "w") as f:
    json.dump(config, f, indent=2)

print("\n" + "=" * 60)
print("训练完成！下一步: python 03_create_submission.py")
print("=" * 60)
