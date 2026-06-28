"""
联合训练：Soft Prompt + Multi-Scale + CLIP 图文对比
单阶段端到端，Qwen 冻结，ViT LoRA + 两个创新模块联训
"""
import os, glob, re
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from model_v2 import QwenCLIPRetriever
from dataset import TechManualDataset, TestDataset


# ══════════════════════════════
# 配置
# ══════════════════════════════
BATCH_SIZE        = 16
GRAD_ACCUM_STEPS  = 2              # 有效 batch = 32
EPOCHS            = 20
LEARNING_RATE     = 1e-4
TEMPERATURE       = 0.07
MAX_TEXT_LENGTH   = 256
DATA_PATH         = "train_processed_v2.json"
IMAGE_ROOT        = "/root/autodl-tmp/"

TEST_DATA_PATH    = "/root/autodl-tmp/test_processed_v2.json"
EVAL_START_LOSS   = 1.0             # loss 低于此值后每轮评估

# 创新模块开关（消融实验用）
USE_SOFT_PROMPT   = True
NUM_PROMPT_TOKENS = 16
USE_MULTISCALE    = True

# LoRA
QWEN_USE_LORA     = False
CLIP_USE_LORA     = True
CLIP_LORA_R       = 4

# 损失权重
TOPIC_ALPHA       = 0.3
KL_ALPHA          = 0.1

SAVE_DIR          = "./checkpoints_joint"
USE_AMP           = True


def remove_old_ckpt(save_dir, keep=3):
    files = glob.glob(os.path.join(save_dir, "joint_epoch*.pt"))
    if len(files) <= keep:
        return
    def get_epoch(f):
        m = re.search(r'epoch(\d+)', os.path.basename(f))
        return int(m.group(1)) if m else 0
    files.sort(key=get_epoch)
    for f in files[:-keep]:
        os.remove(f)


@torch.no_grad()
def eval_during_training(model, logit_scale, device):
    """训练中评估，返回 average R@1"""
    dataset = TestDataset(TEST_DATA_PATH, IMAGE_ROOT)
    all_img, all_txt = [], []

    for i in range(0, len(dataset), 16):
        batch = [dataset[j] for j in range(i, min(i + 16, len(dataset)))]
        images = torch.stack([b[0] for b in batch]).to(device)
        texts  = [b[1] for b in batch]
        all_img.append(F.normalize(model(images=images), dim=-1))
        all_txt.append(F.normalize(model(texts=texts), dim=-1))

    img_f = torch.cat(all_img, dim=0)
    txt_f = torch.cat(all_txt, dim=0)
    sim = logit_scale.exp() * (img_f @ txt_f.T)

    N = sim.size(0)
    i2t = sum(1 for i in range(N) if i in sim[i].topk(1).indices) / N
    t2i = sum(1 for i in range(N) if i in sim[:, i].topk(1).indices) / N
    avg_r1 = (i2t + t2i) / 2
    return avg_r1


def train():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("=" * 55)
    print("  联合训练: Soft Prompt + Multi-Scale + CLIP")
    print(f"  Soft Prompt: {USE_SOFT_PROMPT} ({NUM_PROMPT_TOKENS} tokens)")
    print(f"  Multi-Scale: {USE_MULTISCALE}")
    print(f"  ViT LoRA:    {CLIP_USE_LORA} (r={CLIP_LORA_R})")
    print("=" * 55)

    # ── 数据 ──
    dataset = TechManualDataset(DATA_PATH, IMAGE_ROOT, max_text_length=MAX_TEXT_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=4, pin_memory=True, drop_last=True)
    print(f"[INFO] 图文对: {len(dataset)}, batch/epoch: {len(dataloader)}")

    # ── 模型 ──
    model = QwenCLIPRetriever(
        qwen_name="Qwen/Qwen2.5-1.5B",
        clip_name="OFA-Sys/chinese-clip-vit-base-patch16",
        dropout=0.1,
        use_lora=False,
        clip_use_lora=CLIP_USE_LORA,
        clip_lora_r=CLIP_LORA_R,
        use_soft_prompt=USE_SOFT_PROMPT,
        num_prompt_tokens=NUM_PROMPT_TOKENS,
        use_multiscale=USE_MULTISCALE,
    )
    model.text_encoder.qwen.requires_grad_(False)
    model = model.to("cuda")

    total   = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] 参数: {total/1e6:.1f}M, 可训练: {trainable/1e6:.2f}M")

    # ── 优化器 ──
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    logit_scale = torch.nn.Parameter(
        torch.log(torch.tensor(1.0 / TEMPERATURE, device="cuda")))
    optimizer.add_param_group({"params": [logit_scale], "lr": LEARNING_RATE})
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    # ── 训练 ──
    best_r1, best_epoch = 0.0, 0
    for epoch in range(EPOCHS):
        model.train()
        model.text_encoder.qwen.eval()   # 关 Qwen dropout，LoRA 纯 Linear 不受影响
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Joint Epoch {epoch+1}/{EPOCHS}")
        optimizer.zero_grad()

        for step, (images, texts) in enumerate(pbar):
            images = images.cuda()
            B = len(images)
            scale = logit_scale.exp()

            with torch.cuda.amp.autocast(enabled=USE_AMP):
                img_feat, img_topic = model(images=images, return_topic=True)
                txt_feat, txt_topic = model(texts=texts, return_topic=True)

                labels = torch.arange(B, device="cuda")

                # ITC
                logits_i2t = scale * (img_feat @ txt_feat.T)   # [B, B]
                logits_t2i = scale * (txt_feat @ img_feat.T)   # [B, B]
                loss_itc = (F.cross_entropy(logits_i2t, labels) +
                            F.cross_entropy(logits_t2i, labels)) / 2

                # Topic
                logits_topic = scale * (img_topic @ txt_topic.T)
                loss_topic = (F.cross_entropy(logits_topic, labels) +
                              F.cross_entropy(logits_topic.T, labels)) / 2

                # KL
                with torch.no_grad():
                    teacher = F.softmax(scale * (txt_feat @ txt_feat.T), dim=-1)
                student = F.log_softmax(logits_i2t, dim=-1)
                loss_kl = F.kl_div(student, teacher, reduction='batchmean')

            loss = (loss_itc + TOPIC_ALPHA * loss_topic +
                    KL_ALPHA * loss_kl) / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * GRAD_ACCUM_STEPS
            pbar.set_postfix({
                "loss": f"{loss.item()*GRAD_ACCUM_STEPS:.4f}",
                "itc":  f"{loss_itc.item():.4f}",
                "tpc":  f"{loss_topic.item():.4f}",
                "kl":   f"{loss_kl.item():.4f}",
            })

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} | avg_loss: {avg_loss:.4f}")

        # 保存
        ckpt_path = os.path.join(SAVE_DIR, f"joint_epoch{epoch+1}.pt")
        torch.save({
            "epoch": epoch + 1,
            "pooling":        model.text_encoder.pooling.state_dict(),
            "proj":           model.text_encoder.proj.state_dict(),
            "prompt_embeddings": model.text_encoder.prompt_embeddings
                if USE_SOFT_PROMPT else None,
            "topic_proj":     model.text_encoder.topic_proj_text.state_dict(),
            "image_encoder":  model.image_encoder.state_dict(),
            "logit_scale":    logit_scale.detach().cpu(),
            "optimizer":      optimizer.state_dict(),
        }, ckpt_path)
        print(f"[INFO] 已保存: {ckpt_path}")
        remove_old_ckpt(SAVE_DIR, keep=3)

        # ── 评估 + 早停 ──
        if avg_loss < EVAL_START_LOSS:
            model.eval()
            r1 = eval_during_training(model, logit_scale, "cuda")
            model.train()
            model.text_encoder.qwen.eval()
            print(f"[EVAL] Epoch {epoch+1} | Avg R@1: {r1*100:.2f}%")

            if r1 > best_r1:
                best_r1 = r1
                best_epoch = epoch + 1
                print(f"[EVAL] ↑ 新高! best R@1: {r1*100:.2f}%")
            elif r1 < best_r1:
                print(f"[EVAL] ↓ 下降 (best: {best_r1*100:.2f}% @ epoch {best_epoch})")
                print(f"[INFO] 提前停止训练")
                break

    if best_r1 > 0:
        print(f"[INFO] 联合训练完成, best R@1: {best_r1*100:.2f}% @ epoch {best_epoch}")
    else:
        print("[INFO] 联合训练完成 (loss 未触发评估)")


if __name__ == "__main__":
    train()