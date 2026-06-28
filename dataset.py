"""数据加载：Stage1 文本对 + Stage2 图文对"""
import csv, json, os, random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ── 图像预处理（Chinese-CLIP 标准） ──
IMG_SIZE = 224
IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711]),
])


# ═══════════════════════════════════════════════
#  Stage 2 图文数据集
# ═══════════════════════════════════════════════
class TechManualDataset(Dataset):
    """图文对数据集，Stage 2 对比学习用。

    每张图的所有 caption（original + 改写）展开为独立样本，数据量翻多倍。
    """

    def __init__(self, json_path, image_root, max_text_length=512):
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        self.image_root = image_root
        self.max_text_length = max_text_length

        # 展开：每张图的每条 caption 是一条独立记录
        self.samples = []
        for item in raw_data:
            candidates = [item['original']] + item['paraphrases']
            candidates = [c for c in candidates if c and c.strip()]
            for text in candidates:
                self.samples.append((item['img'], text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_rel, text = self.samples[idx]
        img_path = os.path.join(self.image_root, img_rel)
        image = Image.open(img_path).convert('RGB')
        image = IMG_TRANSFORM(image)
        return image, text


# ═══════════════════════════════════════════════
#  Stage 1 文本对数据集
# ═══════════════════════════════════════════════
class CaptionPairDataset(Dataset):
    """纯文本正例对，Stage 1 SimCSE 训练用。

    从 train_processed.json 构造：
    每张图有 original + 3 条改写 = 4 条正文本，
    两两配对产生 C(4,2) = 6 对正例。
    """
    def __init__(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.pairs = []
        for item in data:
            texts = [item['original']] + item['paraphrases']
            texts = [t for t in texts if t and t.strip()]
            n = len(texts)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        self.pairs.append((texts[i], texts[j]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


# ═══════════════════════════════════════════════
#  Stage 1b 汽车文本数据集（SupCon 格式）
# ═══════════════════════════════════════════════
class AutoTextGroupDataset(Dataset):
    """每张汽车图的所有 caption 为一组正例，用于 SupCon 损失。
    返回: [text1, text2, ...] 同图的所有有效文本。"""

    def __init__(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        self.groups = []
        for item in raw_data:
            texts = [item['original']] + item['paraphrases']
            texts = [t.strip() for t in texts if t and t.strip()]
            if len(texts) >= 2:
                self.groups.append(texts)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        return self.groups[idx]


# ═══════════════════════════════════════════════
#  Stage 1a CC3K 文本数据集（SupCon 格式）
# ═══════════════════════════════════════════════
class CC3KDataset(Dataset):
    """CC3K 监督对比学习数据集。
    每张图: 1 短描述(随机) + 3 长描述 → 4 条文本为正例组。
    collate_fn 需配合 train_stage1_cc3k.collate_fn 使用。
    """

    SHORT_COLS = ["shortIB_captions", "shortSV_captions", "shortLLA_captions"]
    LONG_COLS  = ["longIB_captions", "longSV_captions", "longLLA_captions"]

    def __init__(self, csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            self.data = list(csv.DictReader(f))

        self.data = [row for row in self.data
                     if any(row.get(c, "").strip() for c in self.SHORT_COLS)
                     and any(row.get(c, "").strip() for c in self.LONG_COLS)]
        print(f"[INFO] CC3K 有效样本: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]

        shorts = [row[c].strip() for c in self.SHORT_COLS if row.get(c, "").strip()]
        longs  = [row[c].strip() for c in self.LONG_COLS  if row.get(c, "").strip()]
        short = random.choice(shorts)

        texts = [short] + longs[:3]
        while len(texts) < 4:
            texts.append("")
        return texts


# ═══════════════════════════════════════════════
#  测试集（无改写增强，只用 original）
# ═══════════════════════════════════════════════
class TestDataset(Dataset):
    """测试用，图片配 original 文本，不加改写增强。"""
    def __init__(self, json_path, image_root):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.image_root = image_root

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.image_root, item['img'])
        image = Image.open(img_path).convert('RGB')
        image = IMG_TRANSFORM(image)
        return image, item['original']
