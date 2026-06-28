import os
import glob
import re
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from dataset import TechManualDataset
from model import QwenCLIPRetriever

# ----------------------------
# 超参数
# ----------------------------
BATCH_SIZE      = 32
EPOCHS          = 40        # 跑40轮
LEARNING_RATE   = 1e-4
TEMPERATURE     = 0.07
TRAINABLE_QWEN_LAYERS = 0   # 0=只训练投影层，2=微调最后2层
MAX_TEXT_LENGTH = 512
DATA_PATH       = "/root/autodl-tmp/train.json"
IMAGE_ROOT      = "./"
SAVE_DIR        = "./checkpoints"


def remove_old_checkpoints(save_dir, keep=3):
    """只保留最新的 keep 个 checkpoint，删除更早的"""
    files = glob.glob(os.path.join(save_dir, "qwen_clip_epoch*.pt"))
    if len(files) <= keep:
        return
    def get_epoch(f):
        m = re.search(r'epoch(\d+)', os.path.basename(f))
        return int(m.group(1)) if m else 0
    files.sort(key=get_epoch)
    for f in files[:-keep]:
        os.remove(f)
        print(f"[INFO] 删除旧权重: {os.path.basename(f)}")


def train():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 数据集
    dataset = TechManualDataset(DATA_PATH, IMAGE_ROOT, max_text_length=MAX_TEXT_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=0, pin_memory=True, drop_last=True)

    # 模型
    model = QwenCLIPRetriever(qwen_name="Qwen/Qwen2.5-1.5B",
                              clip_name="ViT-B/32",
                              trainable_layers=TRAINABLE_QWEN_LAYERS,
                              dropout=0.1)
    model.image_encoder.model.to("cuda")       # CLIP 图像模型移到 GPU
    model.text_encoder.qwen.to("cuda")         # Qwen 主体移到 GPU
    model.text_encoder.pooling.to("cuda")      # 池化层移到 GPU
    model.text_encoder.proj.to("cuda")         # 投影层移到 GPU

    # 只训练需要梯度更新的参数
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=LEARNING_RATE)
    # 温度参数（可学习）
    logit_scale = torch.nn.Parameter(
        torch.log(torch.tensor(1.0 / TEMPERATURE, device="cuda"))
    )
    optimizer.add_param_group({"params": [logit_scale], "lr": LEARNING_RATE})

    # 训练循环
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, texts in pbar:
            images = images.cuda()
            img_feat = model(images=images)        # [B, D]
            txt_feat = model(texts=texts)          # [B, D]

            # 计算相似度矩阵
            logits = logit_scale.exp() * (img_feat @ txt_feat.T)
            labels = torch.arange(len(images)).cuda()

            # 双向交叉熵损失
            loss_i = torch.nn.functional.cross_entropy(logits, labels)
            loss_t = torch.nn.functional.cross_entropy(logits.T, labels)
            loss = (loss_i + loss_t) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} finished. Average loss: {avg_loss:.4f}")

        # 保存检查点
        checkpoint = {
            'epoch': epoch + 1,
            'text_encoder_state': model.text_encoder.state_dict(),
            'logit_scale': logit_scale.detach().cpu(),
            'optimizer': optimizer.state_dict(),
        }
        save_path = os.path.join(SAVE_DIR, f"qwen_clip_epoch{epoch+1}.pt")
        torch.save(checkpoint, save_path)
        print(f"[INFO] 已保存: {save_path}")

        # ✅ 只保留最后3个权重，自动删除更早的
        remove_old_checkpoints(SAVE_DIR, keep=7)

if __name__ == "__main__":
    train()