"""
按文本长度分组评估: 用 Chinese-CLIP tokenizer 计算真实 token 数, 分组报告 Recall
"""
import sys; sys.path.insert(0, "../../")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch, json
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, ChineseCLIPVisionModel
from dataset import TestDataset

BINS       = [(0, 77), (78, 128), (129, 256), (257, 9999)]
BIN_NAMES  = ["<=77", "78-128", "129-256", ">256"]
# 每个 bin 需要各自的模型 checkpoint (用不同 length 训练的)
CHECKPOINTS = {
    "./checkpoints_joint/exp1_qwen_len77/epoch20.pt":  "qwen_len77",
}
TEST_PATH = "/root/autodl-tmp/test_processed.json"
IMG_ROOT  = "/root/autodl-tmp/"
DEVICE    = "cuda"
CLIP_NAME = "OFA-Sys/chinese-clip-vit-base-patch16"
QWEN_NAME = "Qwen/Qwen2.5-1.5B"


def compute_recall(sim, ks=[1,5,10]):
    N = sim.size(0)
    r = {}
    for k in ks:
        topk = sim.topk(min(k, N), dim=1).indices
        r[k] = sum(1 for i in range(N) if i in topk[i]) / N
    return r


@torch.no_grad()
def eval_length_groups(model, tokenizer):
    """按文本真实 token 数分组评估"""
    dataset = TestDataset(TEST_PATH, IMG_ROOT)

    # 计算每条文本的真实 token 数
    group_indices = {name: [] for name in BIN_NAMES}
    for idx in range(len(dataset)):
        text = dataset[idx][1]
        tok_len = len(tokenizer.encode(text, add_special_tokens=True))
        for (lo, hi), name in zip(BINS, BIN_NAMES):
            if lo <= tok_len <= hi:
                group_indices[name].append(idx)
                break

    for name, indices in group_indices.items():
        if len(indices) == 0: continue
        imgs, txts = [], []
        for i in indices:
            img, txt = dataset[i]
            imgs.append(img); txts.append(txt)

        all_img, all_txt = [], []
        for i in range(0, len(imgs), 16):
            batch_img = torch.stack(imgs[i:i+16]).to(DEVICE)
            batch_txt = txts[i:i+16]
            all_img.append(model(images=batch_img).cpu())
            all_txt.append(model(texts=batch_txt).cpu())

        img_f = F.normalize(torch.cat(all_img, dim=0), dim=-1)
        txt_f = F.normalize(torch.cat(all_txt, dim=0), dim=-1)
        sim_i2t = img_f @ txt_f.T
        sim_t2i = txt_f @ img_f.T

        i2t = compute_recall(sim_i2t)
        t2i = compute_recall(sim_t2i)
        print(f"\n[{name}] n={len(indices)}")
        for k in [1, 5, 10]:
            avg = (i2t[k] + t2i[k]) / 2 * 100
            print(f"  Avg R@{k}: {avg:.2f}%")


if __name__ == "__main__":
    # 加载一个 checkpoint 做演示, 实际应加载对应的 checkpoint
    from qwen_len77 import MHAWithLength
    model = MHAWithLength(QWEN_NAME, CLIP_NAME, 768, 8, 77).to(DEVICE)
    ckpt = torch.load("./checkpoints_joint/exp1_qwen_len77/epoch20.pt", map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
    eval_length_groups(model, tokenizer)
