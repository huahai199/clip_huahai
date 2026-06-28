"""
评估脚本：在 test_processed.json 上测试图文检索 Recall@K
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm

from model_v2 import QwenCLIPRetriever
from dataset import TestDataset


CHECKPOINT    = "./checkpoints_joint/joint_epoch30.pt"  # ← 改成实际 epoch
DATA_PATH     = "/root/autodl-tmp/test_processed_v2.json"
IMAGE_ROOT    = "/root/autodl-tmp/"
BATCH_SIZE    = 16
RECALL_KS     = [1, 5, 10]

# 与训练时保持一致
USE_SOFT_PROMPT   = True
NUM_PROMPT_TOKENS = 16
USE_MULTISCALE    = True
CLIP_USE_LORA     = True


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
    device = "cuda"

    # ── 加载模型 ──
    model = QwenCLIPRetriever(
        qwen_name="Qwen/Qwen2.5-1.5B",
        clip_name="OFA-Sys/chinese-clip-vit-base-patch16",
        use_lora=False,
        clip_use_lora=CLIP_USE_LORA,
        use_soft_prompt=USE_SOFT_PROMPT,
        num_prompt_tokens=NUM_PROMPT_TOKENS,
        use_multiscale=USE_MULTISCALE,
    )
    ckpt = torch.load(CHECKPOINT, map_location="cpu")

    model.text_encoder.pooling.load_state_dict(ckpt["pooling"])
    model.text_encoder.proj.load_state_dict(ckpt["proj"])
    model.text_encoder.topic_proj_text.load_state_dict(ckpt["topic_proj"])
    if USE_SOFT_PROMPT and ckpt.get("prompt_embeddings") is not None:
        model.text_encoder.prompt_embeddings.data.copy_(ckpt["prompt_embeddings"])
    model.image_encoder.load_state_dict(ckpt["image_encoder"])

    model = model.to(device)
    model.eval()

    logit_scale = ckpt.get("logit_scale", torch.tensor(1.0 / 0.07))
    if isinstance(logit_scale, torch.Tensor):
        logit_scale = logit_scale.to(device)
    else:
        logit_scale = torch.tensor(logit_scale, device=device)
    print(f"[INFO] 已加载: {CHECKPOINT}")
    print(f"[INFO] logit_scale = {logit_scale.exp().item():.4f}")

    # ── 数据 ──
    dataset = TestDataset(DATA_PATH, IMAGE_ROOT)
    print(f"[INFO] 测试集: {len(dataset)} 条")

    # ── 编码 ──
    all_img_feats, all_text_feats = [], []
    for i in tqdm(range(0, len(dataset), BATCH_SIZE), desc="Encoding"):
        batch = [dataset[j] for j in range(i, min(i + BATCH_SIZE, len(dataset)))]
        images = torch.stack([b[0] for b in batch]).to(device)
        texts  = [b[1] for b in batch]

        img_f = F.normalize(model(images=images), dim=-1)
        txt_f = F.normalize(model(texts=texts), dim=-1)
        all_img_feats.append(img_f.cpu())
        all_text_feats.append(txt_f.cpu())

    img_feats = torch.cat(all_img_feats, dim=0)
    txt_feats = torch.cat(all_text_feats, dim=0)
    print(f"[INFO] 图像特征: {img_feats.shape}, 文本特征: {txt_feats.shape}")

    # ── 检索 ──
    scale = logit_scale.exp()
    img_feats = img_feats.to(device)
    txt_feats = txt_feats.to(device)
    sim_i2t = scale * (img_feats @ txt_feats.T)
    sim_t2i = scale * (txt_feats @ img_feats.T)

    print("\n" + "=" * 50)
    print("Image → Text Recall")
    print("=" * 50)
    i2t_recalls = compute_recall(sim_i2t, RECALL_KS)
    for k, v in i2t_recalls.items():
        print(f"  Recall@{k:2d}: {v*100:.2f}%")

    print("\n" + "=" * 50)
    print("Text → Image Recall")
    print("=" * 50)
    t2i_recalls = compute_recall(sim_t2i, RECALL_KS)
    for k, v in t2i_recalls.items():
        print(f"  Recall@{k:2d}: {v*100:.2f}%")

    print("\n" + "=" * 50)
    print("Average Recall")
    print("=" * 50)
    for k in RECALL_KS:
        avg = (i2t_recalls[k] + t2i_recalls[k]) / 2
        print(f"  R@{k:2d}: {avg*100:.2f}%")


if __name__ == "__main__":
    evaluate()
