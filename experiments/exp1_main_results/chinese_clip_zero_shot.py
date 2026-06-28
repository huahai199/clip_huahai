"""
表1 零样本评估: Chinese-CLIP zero-shot 推理
"""
import sys; sys.path.insert(0, "../../")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import ChineseCLIPTextModel, ChineseCLIPVisionModel, AutoTokenizer
from tqdm import tqdm

from dataset import TestDataset

BATCH_SIZE    = 16
DATA_PATH     = "/root/autodl-tmp/test_processed.json"
IMAGE_ROOT    = "/root/autodl-tmp/"
DEVICE        = "cuda"
RECALL_KS     = [1, 5, 10]


def compute_recall(sim_matrix, ks):
    N = sim_matrix.size(0)
    recalls = {}
    for k in ks:
        topk_idx = torch.topk(sim_matrix, k=min(k, N), dim=1).indices
        correct = sum(1 for i in range(N) if i in topk_idx[i])
        recalls[k] = correct / N
    return recalls


@torch.no_grad()
def evaluate():
    model_name = "OFA-Sys/chinese-clip-vit-base-patch16"
    print(f"[INFO] 加载模型: {model_name}")
    img_model = ChineseCLIPVisionModel.from_pretrained(model_name).to(DEVICE).eval()
    txt_model = ChineseCLIPTextModel.from_pretrained(model_name).to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = TestDataset(DATA_PATH, IMAGE_ROOT)
    print(f"[INFO] 测试集: {len(dataset)} 条")

    all_img, all_txt = [], []
    for i in tqdm(range(0, len(dataset), BATCH_SIZE), desc="Encoding"):
        batch = [dataset[j] for j in range(i, min(i + BATCH_SIZE, len(dataset)))]
        images = torch.stack([b[0] for b in batch]).to(DEVICE)
        texts  = [b[1] for b in batch]

        img_out = img_model(pixel_values=images)
        all_img.append(img_out.pooler_output.cpu())

        inputs = tokenizer(texts, return_tensors="pt", padding=True,
                          truncation=True, max_length=52).to(DEVICE)
        txt_out = txt_model(**inputs)
        all_txt.append(txt_out.pooler_output.cpu())

    img_feats = F.normalize(torch.cat(all_img, dim=0), dim=-1)
    txt_feats = F.normalize(torch.cat(all_txt, dim=0), dim=-1)

    sim_i2t = img_feats @ txt_feats.T
    sim_t2i = txt_feats @ img_feats.T

    print("\nImage → Text Recall")
    for k, v in compute_recall(sim_i2t, RECALL_KS).items():
        print(f"  R@{k}: {v*100:.2f}%")
    print("\nText → Image Recall")
    for k, v in compute_recall(sim_t2i, RECALL_KS).items():
        print(f"  R@{k}: {v*100:.2f}%")
    print("\nAverage Recall")
    for k in RECALL_KS:
        avg = (compute_recall(sim_i2t, [k])[k] + compute_recall(sim_t2i, [k])[k]) / 2
        print(f"  R@{k}: {avg*100:.2f}%")


if __name__ == "__main__":
    evaluate()
