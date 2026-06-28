import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import TechManualDataset
from model_v2 import QwenCLIPRetriever

# ---- 配置 ----
BATCH_SIZE       = 16
MAX_TEXT_LENGTH  = 512
DATA_PATH        = "/root/autodl-tmp/test.json"
IMAGE_ROOT       = "./"
CHECKPOINT_PATH  = "./checkpoints_v2/stage2_epoch30.pt"
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
RECALL_KS        = [1, 5, 10]


def load_model(checkpoint_path, device):
    # 训练时已 merge_and_unload，checkpoint 里是纯基座权重，直接用 use_lora=False
    model = QwenCLIPRetriever(
        qwen_name="Qwen/Qwen2.5-1.5B",
        clip_name="OFA-Sys/chinese-clip-vit-base-patch16",
        dropout=0.0,
        use_lora=False,
    )
    model = model.to(device)
    model.eval()

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.text_encoder.load_state_dict(checkpoint["text_encoder_state"], strict=False)
    model.text_encoder.requires_grad_(False)

    if "image_encoder_state" in checkpoint:
        model.image_encoder.load_state_dict(checkpoint["image_encoder_state"])

    logit_scale = checkpoint.get("logit_scale", torch.tensor(1.0 / 0.07))
    if isinstance(logit_scale, torch.Tensor):
        logit_scale = logit_scale.to(device)
    else:
        logit_scale = torch.tensor(logit_scale, device=device)

    print(f"[INFO] checkpoint: {checkpoint_path} (epoch={checkpoint['epoch']})")
    print(f"[INFO] logit_scale = {logit_scale.item():.4f}")
    return model, logit_scale


def encode_features(model, dataloader, device):
    all_img_feats = []
    all_txt_feats = []
    all_captions  = []

    with torch.no_grad():
        for images, texts in tqdm(dataloader, desc="Encoding"):
            images = images.to(device)
            img_feat = model(images=images)
            txt_feat = model(texts=texts)

            img_feat = F.normalize(img_feat, dim=-1)
            txt_feat = F.normalize(txt_feat, dim=-1)

            all_img_feats.append(img_feat.cpu())
            all_txt_feats.append(txt_feat.cpu())
            all_captions.extend(texts)

    img_feats = torch.cat(all_img_feats, dim=0)
    txt_feats = torch.cat(all_txt_feats, dim=0)
    return img_feats, txt_feats, all_captions


def compute_recall(sim_matrix, ks=RECALL_KS):
    N = sim_matrix.size(0)
    recalls = {}
    for k in ks:
        topk_idx = torch.topk(sim_matrix, k=min(k, sim_matrix.size(1)), dim=1).indices
        correct = sum(1 for i in range(N) if i in topk_idx[i])
        recalls[k] = correct / N
    return recalls


def evaluate(img_feats, txt_feats, logit_scale):
    logit_scale = logit_scale.to(img_feats.device)
    sim_i2t = logit_scale.exp() * (img_feats @ txt_feats.T)
    sim_t2i = logit_scale.exp() * (txt_feats @ img_feats.T)

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

    return i2t_recalls, t2i_recalls


def test():
    model, logit_scale = load_model(CHECKPOINT_PATH, DEVICE)

    dataset = TechManualDataset(DATA_PATH, IMAGE_ROOT, max_text_length=MAX_TEXT_LENGTH)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    print(f"[INFO] 测试集样本数: {len(dataset)}")

    img_feats, txt_feats, captions = encode_features(model, dataloader, DEVICE)
    print(f"[INFO] 图像特征: {img_feats.shape}, 文本特征: {txt_feats.shape}")

    evaluate(img_feats, txt_feats, logit_scale)


if __name__ == "__main__":
    test()
