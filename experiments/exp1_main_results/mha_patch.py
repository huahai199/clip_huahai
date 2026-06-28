"""
MHA Pooling + MLP 投影头 + 图像投影残差 + 局部/全局 patch 融合
"""
import sys; sys.path.insert(0, "../../")
import os, glob, re
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel, ChineseCLIPVisionModel
from tqdm import tqdm

from dataset import TechManualDataset, TestDataset

BATCH_SIZE        = 16
GRAD_ACCUM_STEPS  = 2
EPOCHS            = 20
LEARNING_RATE     = 1e-4
TEMPERATURE       = 0.07
MAX_TEXT_LENGTH   = 256
DATA_PATH         = "/root/autodl-tmp/train_processed.json"
IMAGE_ROOT        = "/root/autodl-tmp/"
TEST_DATA_PATH    = "/root/autodl-tmp/test_processed.json"
EVAL_START_LOSS   = 1.0
SAVE_DIR          = "./checkpoints_joint/exp1_mha_patch"
USE_AMP           = True
QWEN_NAME         = "Qwen/Qwen2.5-1.5B"
CLIP_NAME         = "OFA-Sys/chinese-clip-vit-base-patch16"
EMBED_DIM         = 768
MHA_HEADS         = 8


class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files: f.write(obj); f.flush()
    def flush(self):
        for f in self.files: f.flush()


class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.agg_query = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, mask=None):
        B = hidden_states.size(0)
        Q = self.q_proj(self.agg_query)
        K = self.k_proj(hidden_states)
        V = self.v_proj(hidden_states)
        Q = Q.expand(B, -1, -1).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, 1, -1)
        out = self.out_proj(out)
        return self.norm(out.squeeze(1))


class ImageEncoderWithPatchFusion(nn.Module):
    """全局 [CLS] + 局部 patch 均值融合 + 残差投影"""

    def __init__(self, clip_name, embed_dim=768):
        super().__init__()
        self.vit = ChineseCLIPVisionModel.from_pretrained(clip_name)
        self.vit.requires_grad_(False)
        vit_dim = self.vit.config.hidden_size

        # 全局 [CLS] 投影
        self.global_proj = nn.Linear(vit_dim, embed_dim)

        # 局部 patch 均值投影
        self.local_proj = nn.Linear(vit_dim, embed_dim)

        # 融合全局+局部 → 最终嵌入
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.ReLU(),
            nn.LayerNorm(embed_dim),
        )

        # 残差投影: 将原始 [CLS] 映射到 embed_dim 作为 shortcut
        self.residual_proj = nn.Linear(vit_dim, embed_dim)

    def forward(self, images):
        with torch.no_grad():
            out = self.vit(pixel_values=images, output_hidden_states=True)
            cls_token = out.pooler_output                            # [B, D]
            patch_tokens = out.last_hidden_state[:, 1:, :]           # [B, N, D]

        # 全局分支
        global_feat = self.global_proj(cls_token)                    # [B, E]

        # 局部分支: patch 均值 → 投影
        local_mean  = patch_tokens.mean(dim=1)                       # [B, D]
        local_feat  = self.local_proj(local_mean)                    # [B, E]

        # 全局+局部融合
        concat = torch.cat([global_feat, local_feat], dim=-1)       # [B, 2E]
        fused  = self.fusion(concat)                                 # [B, E]

        # 残差: 原始 [CLS] 的线性投影
        residual = self.residual_proj(cls_token)                     # [B, E]

        return F.normalize(fused + residual, dim=-1)


class MHAPatch(nn.Module):
    """MHA Pooling + MLP 投影头 + 局部/全局 patch 融合 + 残差"""

    def __init__(self, qwen_name, clip_name, embed_dim=768, mha_heads=8):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_name, trust_remote_code=True)

        self.qwen = AutoModel.from_pretrained(qwen_name, trust_remote_code=True,
                                               torch_dtype=torch.float32)
        self.qwen.requires_grad_(False)
        qwen_dim = self.qwen.config.hidden_size

        self.pooling = MultiHeadAttentionPooling(qwen_dim, mha_heads)
        self.txt_proj = nn.Sequential(
            nn.Linear(qwen_dim, qwen_dim * 2), nn.ReLU(),
            nn.Linear(qwen_dim * 2, embed_dim), nn.LayerNorm(embed_dim),
        )

        self.img_encoder = ImageEncoderWithPatchFusion(clip_name, embed_dim)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / TEMPERATURE)))

    def forward(self, images=None, texts=None):
        if images is not None:
            return self.img_encoder(images)
        device = next(self.qwen.parameters()).device
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True,
                                truncation=True, max_length=512).to(device)
        with torch.no_grad():
            hidden = self.qwen(**inputs).last_hidden_state
        mask = inputs['attention_mask']
        pooled = self.pooling(hidden, mask)
        return F.normalize(self.txt_proj(pooled), dim=-1)


def remove_old_ckpt(save_dir, keep=3):
    files = glob.glob(os.path.join(save_dir, "epoch*.pt"))
    if len(files) <= keep: return
    def e(f):
        m = re.search(r'epoch(\d+)', os.path.basename(f))
        return int(m.group(1)) if m else 0
    files.sort(key=e)
    for f in files[:-keep]: os.remove(f)


@torch.no_grad()
def eval_during_training(model, logit_scale, device):
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
    print(f"[EVAL] I2T: R@1={recalls[1][0]*100:.2f}%  R@5={recalls[5][0]*100:.2f}%  R@10={recalls[10][0]*100:.2f}%")
    print(f"[EVAL] T2I: R@1={recalls[1][1]*100:.2f}%  R@5={recalls[5][1]*100:.2f}%  R@10={recalls[10][1]*100:.2f}%")
    avg_r1 = (recalls[1][0] + recalls[1][1]) / 2
    print(f"[EVAL] Avg R@1={avg_r1*100:.2f}%  R@5={(recalls[5][0]+recalls[5][1])/2*100:.2f}%  R@10={(recalls[10][0]+recalls[10][1])/2*100:.2f}%")
    return avg_r1


def train():
    os.makedirs(SAVE_DIR, exist_ok=True)
    log_f = open(os.path.join(SAVE_DIR, "train.log"), "w", encoding="utf-8")
    sys.__stdout__ = sys.stdout
    sys.stdout = Tee(sys.__stdout__, log_f)

    print("=" * 55)
    print("  MHA Pooling + Patch 融合 + 残差投影")
    print("=" * 55)

    dataset = TechManualDataset(DATA_PATH, IMAGE_ROOT, max_text_length=MAX_TEXT_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=4, pin_memory=True, drop_last=True)

    model = MHAPatch(QWEN_NAME, CLIP_NAME, EMBED_DIM, MHA_HEADS).to("cuda")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] 参数: {total/1e6:.1f}M, 可训练: {trainable/1e6:.2f}M")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
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
            scale = model.logit_scale.exp()

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
            "epoch": epoch + 1, "model_state": model.state_dict(),
            "logit_scale": model.logit_scale.detach().cpu(),
            "optimizer": optimizer.state_dict(),
        }, ckpt_path)
        remove_old_ckpt(SAVE_DIR, keep=3)

        if avg_loss < EVAL_START_LOSS:
            model.eval()
            r1 = eval_during_training(model, model.logit_scale, "cuda")
            model.train()
            if r1 > best_r1:
                best_r1, best_epoch = r1, epoch + 1
                print(f"[EVAL] ↑ 新高! best R@1: {best_r1*100:.2f}%")
            elif r1 < best_r1:
                print(f"[EVAL] ↓ 下降 (best: {best_r1*100:.2f}% @ epoch {best_epoch})")
                break

    print(f"[INFO] 完成, best R@1: {best_r1*100:.2f}% @ epoch {best_epoch}")
    sys.stdout = sys.__stdout__


if __name__ == "__main__":
    train()
