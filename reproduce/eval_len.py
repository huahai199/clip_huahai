"""仅评估: 从已有 checkpoint 加载4个模型, 按长度分组输出表7"""
import os, sys, json, torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from qwen_len77 import MHAWithLength
from dataset import TestDataset

BASE = os.path.dirname(os.path.abspath(__file__))
TEST_PATH = "/root/autodl-tmp/test_processed.json"
IMG_ROOT  = "/root/autodl-tmp/"
DEVICE    = "cuda"
BINS      = [(0, 77), (78, 128), (129, 256), (257, 9999)]
BIN_NAMES = ["<=77", "78-128", "129-256", ">256"]
LENGTHS   = [77, 128, 256, 512]


def eval_one(trunc_len):
    m = MHAWithLength("Qwen/Qwen2.5-1.5B", "OFA-Sys/chinese-clip-vit-base-patch16",
                       768, 8, trunc_len).to(DEVICE)
    ckpt = os.path.join(BASE, f"checkpoints_joint/exp1_qwen_len{trunc_len}/best.pt")
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE)["model_state"], strict=False)
    m.eval()

    dataset = TestDataset(TEST_PATH, IMG_ROOT)
    tokenizer = AutoTokenizer.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")

    group_idx = {name: [] for name in BIN_NAMES}
    for i in range(len(dataset)):
        tok_len = len(tokenizer.encode(dataset[i][1], add_special_tokens=True))
        for (lo, hi), name in zip(BINS, BIN_NAMES):
            if lo <= tok_len <= hi: group_idx[name].append(i); break

    results = {}
    for name, indices in group_idx.items():
        if not indices: results[name] = {"n": 0}; continue
        imgs, txts = [], []
        for i in indices:
            img, txt = dataset[i]; imgs.append(img); txts.append(txt)

        all_img, all_txt = [], []
        for i in range(0, len(imgs), 16):
            bi = torch.stack(imgs[i:i+16]).to(DEVICE)
            bt = txts[i:i+16]
            with torch.no_grad():
                all_img.append(F.normalize(m(images=bi), dim=-1).cpu())
                all_txt.append(F.normalize(m(texts=bt), dim=-1).cpu())

        sim = torch.cat(all_img) @ torch.cat(all_txt).T
        N = sim.size(0)
        r = {}
        for k in [1, 5, 10]:
            i2t = sum(1 for i in range(N) if i in sim[i].topk(min(k, N)).indices) / N
            t2i = sum(1 for i in range(N) if i in sim[:, i].topk(min(k, N)).indices) / N
            r[k] = round((i2t + t2i) / 2 * 100, 2)
        results[name] = {"n": N, "R@1": r[1], "R@5": r[5], "R@10": r[10]}

    return results


table7 = {}
for L in LENGTHS:
    print(f"[INFO] 评估 qwen_len{L} ...", end=" ", flush=True)
    table7[f"qwen_len{L}"] = eval_one(L)
    print("OK")

# 打印表7
print(f"\n{'子集':<15} {'n':>4}  {'qwen_len77':>10}  {'qwen_len128':>10}  {'qwen_len256':>10}  {'qwen_len512':>10}")
print("-" * 70)
for name in BIN_NAMES:
    n = table7["qwen_len77"].get(name, {}).get("n", 0)
    row = f"{name:<15} {n:>4}"
    for L in LENGTHS:
        v = table7[f"qwen_len{L}"].get(name, {}).get("R@1", "-")
        row += f"  {str(v):>10}"
    print(row)

with open(os.path.join(BASE, "results_len.json"), "w") as f:
    json.dump(table7, f, indent=2)
print(f"\n[DONE] results_len.json")
