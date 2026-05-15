"""
CS321M Predictive Evaluation Challenge
Step 3: 打包提交文件

运行方法:
    python 03_create_submission.py

输出:
    - submission.zip (上传到 https://aimslab.stanford.edu/competition/submit)
"""

import os
import shutil
import zipfile
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent
MODEL_DIR = OUTPUT_DIR / "models"
SUBMIT_DIR = OUTPUT_DIR / "submission"
ZIP_PATH = OUTPUT_DIR / "submission.zip"

print("=" * 60)
print("打包提交文件")
print("=" * 60)

# 检查必要文件
required = ["model.py", "labeling.py", "models.txt", "requirements.txt"]
for f in required:
    p = SUBMIT_DIR / f
    if not p.exists():
        print(f"  ❌ 缺少: {p}")
    else:
        print(f"  ✓ {f}")

# 检查模型文件
# Rasch θ 融合已禁用 (text mismatch)，不再需要 rasch_subject_map.json
model_files = ["ncf_head.pt", "condition_map.json"]
for f in model_files:
    src = MODEL_DIR / f
    dst = SUBMIT_DIR / f
    if src.exists():
        shutil.copy2(src, dst)
        print(f"  ✓ 复制 {f} -> submission/")
    else:
        print(f"  ⚠ 模型文件不存在: {src}")
        print(f"    请先运行 02_train_model.py")

# 只打包需要的文件 (排除 rasch_subject_map.json 等无用文件)
ALLOWED_FILES = {
    "model.py", "labeling.py", "models.txt", "requirements.txt",
    "ncf_head.pt", "condition_map.json",
}

# 创建 ZIP
with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in SUBMIT_DIR.iterdir():
        if f.is_file() and f.name in ALLOWED_FILES:
            zf.write(f, f.name)
            size = f.stat().st_size
            print(f"  📦 {f.name} ({size:,} bytes)")
        elif f.is_file() and not f.name.startswith('.') and f.name not in ALLOWED_FILES:
            print(f"  ⏭ 跳过 {f.name} (不在白名单中)")

print(f"\n✅ 提交文件已创建: {ZIP_PATH}")
print(f"   大小: {ZIP_PATH.stat().st_size:,} bytes")
print(f"\n上传到: https://aimslab.stanford.edu/competition/submit")
