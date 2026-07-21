"""表7: 四个长度模型训练 + 按文本长度分组评估"""
import os, sys, subprocess, re, json, torch
import torch.nn.functional as F
from transformers import AutoTokenizer

BASE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = ["qwen_len77.py", "qwen_len128.py", "qwen_len256.py", "qwen_len512.py"]
TEST_PATH = "/root/autodl-tmp/test_processed.json"
IMG_ROOT  = "/root/autodl-tmp/"
DEVICE    = "cuda"
BINS      = [(0, 77), (78, 128), (129, 256), (257, 9999)]
BIN_NAMES = ["<=77", "78-128", "129-256", ">256"]


def extract_metrics(log_text):
    matches = re.findall(r"Avg R@1=([\d.]+)%\s*R@5=([\d.]+)%\s*R@10=([\d.]+)%", log_text)
    if matches:
        best = max(matches, key=lambda x: float(x[0]))
        return {"R@1": float(best[0]), "R@5": float(best[1]), "R@10": float(best[2])}
    return None


def train_one(script):
    print(f"\n{'='*55}\n  Train: {script}\n{'='*55}")
    proc = subprocess.Popen(
        [sys.executable, script], cwd=BASE,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    captured = []
    for line in proc.stdout:
        captured.append(line)
        s = line.rstrip()
        if "Epoch" in s and "|" in s: print(f"\r{s[:120]}", end='')
        elif "EVAL" in s or "best" in s or "完成" in s: print(f"\n{s}")
    print()
    proc.wait()
    return extract_metrics(''.join(captured))


def eval_by_length(tokenizer, trunc_len, test_path, img_root):
    """加载训练好的模型, 按文本长度分组评估"""
    from dataset import TestDataset
    from qwen_len77 import MHAWithLength  # 四个脚本用同一个模型类

    m = MHAWithLength("Qwen/Qwen2.5-1.5B", "OFA-Sys/chinese-clip-vit-base-patch16",
                       768, 8, trunc_len).to(DEVICE)
    ckpt = os.path.join(BASE, f"checkpoints_joint/exp1_qwen_len{trunc_len}/best.pt")
    if os.path.exists(ckpt):
        m.load_state_dict(torch.load(ckpt, map_location=DEVICE)["model_state"], strict=False)
    m.eval()

    dataset = TestDataset(test_path, img_root)
    # 按 token 数分组
    group_idx = {name: [] for name in BIN_NAMES}
    for idx in range(len(dataset)):
        text = dataset[idx][1]
        tok_len = len(tokenizer.encode(text, add_special_tokens=True))
        for (lo, hi), name in zip(BINS, BIN_NAMES):
            if lo <= tok_len <= hi:
                group_idx[name].append(idx); break

    results = {}
    for name, indices in group_idx.items():
        if not indices:
            results[name] = {"n": 0}
            continue
        imgs, txts = [], []
        for i in indices:
            img, txt = dataset[i]; imgs.append(img); txts.append(txt)

        all_img, all_txt = [], []
        for i in range(0, len(imgs), 16):
            batch_img = torch.stack(imgs[i:i+16]).to(DEVICE)
            batch_txt = txts[i:i+16]
            with torch.no_grad():
                all_img.append(F.normalize(m(images=batch_img), dim=-1).cpu())
                all_txt.append(F.normalize(m(texts=batch_txt), dim=-1).cpu())

        img_f = torch.cat(all_img, dim=0); txt_f = torch.cat(all_txt, dim=0)
        sim = img_f @ txt_f.T
        N = sim.size(0)
        r = {}
        for k in [1, 5, 10]:
            i2t = sum(1 for i in range(N) if i in sim[i].topk(min(k, N)).indices) / N
            t2i = sum(1 for i in range(N) if i in sim[:, i].topk(min(k, N)).indices) / N
            r[k] = round((i2t + t2i) / 2 * 100, 2)
        results[name] = {"n": N, "R@1": r[1], "R@5": r[5], "R@10": r[10]}
    return results


# ---- main ----
overall = {}
table7 = {}
tokenizer = AutoTokenizer.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")

for script in SCRIPTS:
    key = script.replace(".py", "")
    trunc = int(key.replace("qwen_len", ""))
    m = train_one(script)
    if m:
        overall[key] = m

    print(f"\n[INFO] 按长度评估 {key}...")
    grp = eval_by_length(tokenizer, trunc, TEST_PATH, IMG_ROOT)
    table7[key] = grp

# 打印表7
print(f"\n{'='*60}\n  表7: 按文本长度分组评估\n{'='*60}")
header = f"{'子集':<15} {'n':>4}"
keys = [s.replace('.py', '') for s in SCRIPTS]
for k in keys:
    header += f"  {k:>12}"
print(header)
print("-" * 60)
first_key = keys[0] if table7 and keys[0] in table7 else list(table7.keys())[0] if table7 else None
for name in BIN_NAMES:
    row = f"{name:<15}"
    n = table7.get(first_key, {}).get(name, {}).get('n', 0) if first_key else 0
    row += f" {n:>4}"
    for k in keys:
        v = table7.get(k, {}).get(name, {}).get('R@1', '-')
        row += f"  {str(v):>10}"
    print(row)

out = {"overall": overall, "table7": table7}
with open(os.path.join(BASE, "results_len.json"), "w") as f:
    json.dump(out, f, indent=2)
print(f"\n[DONE] -> results_len.json")
