"""
表1 对比实验: Chinese-CLIP (LoRA) — 双塔LoRA微调
"""
import sys; sys.path.insert(0, ".")
import os, glob, re
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from transformers import ChineseCLIPTextModel, ChineseCLIPVisionModel, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

from dataset import TechManualDataset, TestDataset


class Tee:
    """同时输出到控制台和日志文件"""
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()
# ── 配置 ──
BATCH_SIZE        = 16
GRAD_ACCUM_STEPS  = 2
EPOCHS            = 20
LEARNING_RATE     = 1e-4
TEMPERATURE       = 0.07
MAX_TEXT_LENGTH   = 52
DATA_PATH         = "/root/autodl-tmp/train_processed.json"
IMAGE_ROOT        = "/root/autodl-tmp/"
TEST_DATA_PATH    = "/root/autodl-tmp/test_processed.json"
EVAL_START_LOSS   = 1.0
CLIP_LORA_R       = 4
SAVE_DIR          = "./checkpoints_joint/exp1_chinese_clip_lora"
USE_AMP           = True


class ChineseCLIPLoRA(torch.nn.Module):
    """Chinese-CLIP 双塔 LoRA，图像侧 + 文本侧均应用 LoRA"""

    def __init__(self, clip_name="OFA-Sys/chinese-clip-vit-base-patch16",
                 lora_r=4, embed_dim=768):
        super().__init__()
        # ── 图像编码器 LoRA ──
        self.img_model = ChineseCLIPVisionModel.from_pretrained(clip_name)
        for p in self.img_model.parameters():
            p.requires_grad = False
        img_lora_config = LoraConfig(
            r=lora_r, lora_alpha=lora_r * 2,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1, bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        self.img_model = get_peft_model(self.img_model, img_lora_config)

        # ── 文本编码器 LoRA (BERT架构: query/value, 非ViT的q_proj/v_proj) ──
        self.txt_model = ChineseCLIPTextModel.from_pretrained(clip_name)
        self.tokenizer  = AutoTokenizer.from_pretrained(clip_name)
        for p in self.txt_model.parameters():
            p.requires_grad = False
        txt_lora_config = LoraConfig(
            r=lora_r, lora_alpha=lora_r * 2,
            target_modules=["query", "value"],      # BERT 架构的注意力模块名
            lora_dropout=0.1, bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        self.txt_model = get_peft_model(self.txt_model, txt_lora_config)

        # ── 图像投影层 ──
        self.img_proj = torch.nn.Sequential(
            torch.nn.Linear(768, 768 * 2), torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(768 * 2, embed_dim),
            torch.nn.LayerNorm(embed_dim),
        )
        # ── 文本投影层 ──
        self.txt_proj = torch.nn.Sequential(
            torch.nn.Linear(768, 768 * 2), torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(768 * 2, embed_dim),
            torch.nn.LayerNorm(embed_dim),
        )

    def encode_image(self, images):
        out = self.img_model.base_model(pixel_values=images)
        return self.img_proj(out.pooler_output)

    def encode_text(self, texts):
        device = next(self.txt_model.parameters()).device
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True,
                                truncation=True, max_length=52).to(device)
        out = self.txt_model.base_model(**inputs)
        return self.txt_proj(out.pooler_output)

    def forward(self, images=None, texts=None):
        if images is not None:
            return F.normalize(self.encode_image(images), dim=-1)
        return F.normalize(self.encode_text(texts), dim=-1)


def remove_old_ckpt(save_dir, keep=3):
    files = glob.glob(os.path.join(save_dir, "epoch*.pt"))
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
    """训练中评估，返回 average R@1，同时打印 R@1/5/10"""
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
    recalls = {}
    for k in [1, 5, 10]:
        i2t = sum(1 for i in range(N) if i in sim[i].topk(min(k, N)).indices) / N
        t2i = sum(1 for i in range(N) if i in sim[:, i].topk(min(k, N)).indices) / N
        recalls[k] = (i2t, t2i)
    print(f"[EVAL] Image->Text: R@1={recalls[1][0]*100:.2f}%%  R@5={recalls[5][0]*100:.2f}%%  R@10={recalls[10][0]*100:.2f}%%")
    print(f"[EVAL] Text->Image: R@1={recalls[1][1]*100:.2f}%%  R@5={recalls[5][1]*100:.2f}%%  R@10={recalls[10][1]*100:.2f}%%")
    avg_r1 = (recalls[1][0] + recalls[1][1]) / 2
    print(f"[EVAL] Avg R@1={avg_r1*100:.2f}%%  R@5={(recalls[5][0]+recalls[5][1])/2*100:.2f}%%  R@10={(recalls[10][0]+recalls[10][1])/2*100:.2f}%%")
    return avg_r1

def train():
    os.makedirs(SAVE_DIR, exist_ok=True)
    seed = int(os.environ.get("SEED", "0"))
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print(f"[SEED] {seed}")
    # 日志: 同时输出到控制台与文件
    log_f = open(os.path.join(SAVE_DIR, "train.log"), "w", encoding="utf-8")
    sys.__stdout__ = sys.stdout
    sys.stdout = Tee(sys.__stdout__, log_f)

    print("=" * 55)
    print("  对比实验: Chinese-CLIP (LoRA)")
    print(f"  LoRA r={CLIP_LORA_R}")
    print("=" * 55)

    dataset = TechManualDataset(DATA_PATH, IMAGE_ROOT, max_text_length=MAX_TEXT_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=4, pin_memory=True, drop_last=True)

    model = ChineseCLIPLoRA(lora_r=CLIP_LORA_R).to("cuda")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] 参数: {total/1e6:.1f}M, 可训练: {trainable/1e6:.2f}M")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    logit_scale = torch.nn.Parameter(
        torch.log(torch.tensor(1.0 / TEMPERATURE, device="cuda")))
    optimizer.add_param_group({"params": [logit_scale], "lr": LEARNING_RATE})
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    best_r1, best_epoch = 0.0, 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        optimizer.zero_grad()

        for step, (images, texts) in enumerate(pbar):
            images = images.cuda()
            B = len(images)
            scale = logit_scale.exp()

            with torch.cuda.amp.autocast(enabled=USE_AMP):
                img_feat = model(images=images)
                txt_feat = model(texts=texts)
                labels = torch.arange(B, device="cuda")
                logits_i2t = scale * (img_feat @ txt_feat.T)
                logits_t2i = scale * (txt_feat @ img_feat.T)
                loss = (F.cross_entropy(logits_i2t, labels) +
                        F.cross_entropy(logits_t2i, labels)) / 2
                loss = loss / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * GRAD_ACCUM_STEPS
            pbar.set_postfix({"loss": f"{loss.item()*GRAD_ACCUM_STEPS:.4f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} | avg_loss: {avg_loss:.4f}")

        ckpt_path = os.path.join(SAVE_DIR, f"epoch{epoch+1}.pt")
        torch.save({
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "logit_scale": logit_scale.detach().cpu(),
            "optimizer": optimizer.state_dict(),
        }, ckpt_path)
        remove_old_ckpt(SAVE_DIR, keep=3)

        if avg_loss < EVAL_START_LOSS:
            model.eval()
            r1 = eval_during_training(model, logit_scale, "cuda")
            model.train()
            print(f"[EVAL] Epoch {epoch+1} | Avg R@1: {r1*100:.2f}%")
            if r1 > best_r1:
                best_r1, best_epoch = r1, epoch + 1
                print(f"[EVAL] ↑ 新高! best R@1: {r1*100:.2f}%")

    print(f"[INFO] 完成, best R@1: {best_r1*100:.2f}% @ epoch {best_epoch}")



    sys.stdout = sys.__stdout__

if __name__ == "__main__":
    train()
