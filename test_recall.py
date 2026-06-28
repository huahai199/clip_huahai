import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import TechManualDataset
from model import QwenCLIPRetriever

# ----------------------------
# 评测配置
# ----------------------------
BATCH_SIZE       = 16
MAX_TEXT_LENGTH  = 512
DATA_PATH        = "/root/autodl-tmp/test.json"      # 测试集
IMAGE_ROOT       = "./"
CHECKPOINT_PATH  = "/root/autodl-tmp/checkpoints/qwen_clip_epoch40.pt"  # 训练好的权重
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

# Recall@K 的 K 值
RECALL_KS = [1, 5, 10]


def load_model(checkpoint_path, device):
    """加载模型和 checkpoint"""
    model = QwenCLIPRetriever(
        qwen_name="Qwen/Qwen2.5-1.5B",
        clip_name="ViT-B/32",
        trainable_layers=0,
        dropout=0.0
    )
    model = model.to(device)
    model.eval()

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 兼容两种保存格式
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    elif "text_encoder_state" in checkpoint:
        model.text_encoder.load_state_dict(checkpoint["text_encoder_state"])
    else:
        model.load_state_dict(checkpoint)

    # 加载温度参数
    logit_scale = checkpoint.get("logit_scale", torch.tensor(1.0 / 0.07))
    if isinstance(logit_scale, torch.Tensor):
        logit_scale = logit_scale.to(device)
    else:
        logit_scale = torch.tensor(logit_scale, device=device)

    print(f"[INFO] 已加载 checkpoint: {checkpoint_path}")
    print(f"[INFO] 温度参数 logit_scale = {logit_scale.item():.4f}")
    return model, logit_scale


def encode_features(model, dataloader, device):
    """
    分别编码所有图像和文本特征
    返回: img_feats [N, D], txt_feats [N, D], img_paths, captions
    """
    all_img_feats = []
    all_txt_feats = []
    all_img_paths = []
    all_captions  = []

    with torch.no_grad():
        for images, texts in tqdm(dataloader, desc="Encoding"):
            images = images.to(device)

            img_feat = model(images=images)   # [B, D]
            txt_feat = model(texts=texts)     # [B, D]

            # L2 归一化
            img_feat = F.normalize(img_feat, dim=-1)
            txt_feat = F.normalize(txt_feat, dim=-1)

            all_img_feats.append(img_feat.cpu())
            all_txt_feats.append(txt_feat.cpu())
            all_captions.extend(texts)
            # 从 dataset 里取路径（DataLoader 不直接返回，这里简化处理）
            # 如果需要精确路径，可以在 Dataset __getitem__ 里返回

    img_feats = torch.cat(all_img_feats, dim=0)  # [N, D]
    txt_feats = torch.cat(all_txt_feats, dim=0)  # [N, D]
    return img_feats, txt_feats, all_captions


def compute_recall(sim_matrix, ks=RECALL_KS):
    """
    sim_matrix: [N_query, N_gallery]
    对角线为正样本（i-th query 对应 i-th gallery）
    返回各 K 下的 recall
    """
    N = sim_matrix.size(0)
    recalls = {}

    for k in ks:
        # 取 top-k 索引
        topk_idx = torch.topk(sim_matrix, k=min(k, sim_matrix.size(1)), dim=1).indices

        # 检查正样本（对角线 i）是否在 top-k 中
        correct = 0
        for i in range(N):
            if i in topk_idx[i]:
                correct += 1

        recalls[k] = correct / N

    return recalls


def evaluate(img_feats, txt_feats, logit_scale):
    """双向评测"""
    # 温度缩放后的相似度
    logit_scale = logit_scale.to(img_feats.device)
    sim_i2t = logit_scale.exp() * (img_feats @ txt_feats.T)  # [N, N]
    sim_t2i = logit_scale.exp() * (txt_feats @ img_feats.T)  # [N, N]

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
    print("Average Recall (mean of I2T & T2I)")
    print("=" * 50)
    for k in RECALL_KS:
        avg = (i2t_recalls[k] + t2i_recalls[k]) / 2
        print(f"  R@{k:2d}: {avg*100:.2f}%")

    return i2t_recalls, t2i_recalls


def test():
    # 1. 加载模型
    model, logit_scale = load_model(CHECKPOINT_PATH, DEVICE)

    # 2. 构建测试数据集
    dataset = TechManualDataset(DATA_PATH, IMAGE_ROOT, max_text_length=MAX_TEXT_LENGTH)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,          # 评测不能 shuffle！
        num_workers=4, 
        pin_memory=True
    )
    print(f"[INFO] 测试集样本数: {len(dataset)}")

    # 3. 编码特征
    img_feats, txt_feats, captions = encode_features(model, dataloader, DEVICE)
    print(f"[INFO] 图像特征: {img_feats.shape}, 文本特征: {txt_feats.shape}")

    # 4. 评测
    evaluate(img_feats, txt_feats, logit_scale)


if __name__ == "__main__":
    test()