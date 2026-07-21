"""t-SNE: Docci Natural vs Automotive Maintenance Manual Images"""
import os, random
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np
from PIL import Image
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from transformers import ChineseCLIPVisionModel
from torchvision import transforms
from tqdm import tqdm

CLIP_NAME = "OFA-Sys/chinese-clip-vit-base-patch16"
DOC_DIR   = "/root/autodl-tmp/docci_images/images"
AUTO_DIR  = "/root/autodl-tmp/dataset/images"
DEVICE    = "cuda"
N         = 200  # 理想取数, 实际以少的一边为准

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711]),
])

print("[INFO] 加载 ViT ...")
vit = ChineseCLIPVisionModel.from_pretrained(CLIP_NAME).to(DEVICE).eval()

# ---- Docci ----
print("[INFO] 加载 Docci ...")
doc_files = [f for f in os.listdir(DOC_DIR)
             if f.lower().endswith(('.png','.jpg','.jpeg','.webp'))]
auto_files = [f for f in os.listdir(AUTO_DIR)
              if f.lower().endswith(('.png','.jpg','.jpeg','.webp'))]
M = min(len(doc_files), len(auto_files), N)
print(f"  Docci: {len(doc_files)} 张, 汽修: {len(auto_files)} 张, 各取: {M} 张")

doc_imgs = [TRANSFORM(Image.open(os.path.join(DOC_DIR, f)).convert('RGB'))
            for f in random.sample(doc_files, M)]
auto_imgs = [TRANSFORM(Image.open(os.path.join(AUTO_DIR, f)).convert('RGB'))
             for f in random.sample(auto_files, M)]

# ---- 编码 ----
af, df = [], []
with torch.no_grad():
    for img in tqdm(auto_imgs, desc="汽修"):
        af.append(vit(pixel_values=img.unsqueeze(0).to(DEVICE)).pooler_output.squeeze(0).cpu().numpy())
    for img in tqdm(doc_imgs, desc="Docci"):
        df.append(vit(pixel_values=img.unsqueeze(0).to(DEVICE)).pooler_output.squeeze(0).cpu().numpy())

# ---- t-SNE ----
all_f = np.concatenate([af, df], axis=0)
labels = ['Automotive Maintenance Manual Images'] * len(af) + ['Docci Natural'] * len(df)
emb = TSNE(n_components=2, perplexity=15, random_state=42).fit_transform(all_f)

fig, ax = plt.subplots(figsize=(8, 6))
for lbl, c in [('Automotive Maintenance Manual Images', '#A23B72'), ('Docci Natural', '#2E86AB')]:
    mask = [l == lbl for l in labels]
    ax.scatter(emb[mask,0], emb[mask,1], c=c, label=lbl,
               s=40, alpha=0.7, edgecolors='w', linewidth=0.5)
ax.legend(fontsize=14, loc='lower right', bbox_to_anchor=(1, 0))
ax.set_xlabel('t-SNE Dimension 1', fontsize=14)
ax.set_ylabel('t-SNE Dimension 2', fontsize=14)
ax.tick_params(labelsize=12)
plt.tight_layout()
plt.savefig('t-sne.png', dpi=200)
plt.close()
print("[DONE] t-sne.png")
