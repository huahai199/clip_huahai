"""
Qwen-CLIP 多模态检索模型
创新模块: (1) Soft Prompt 可学习文本提示  (2) 多尺度视觉特征融合
"""
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModel,
    ChineseCLIPTextModel, ChineseCLIPVisionModel,
)


# ── 注意力池化 ──
class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states, mask):
        scores = self.attn(hidden_states)
        scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
        attn_weights = torch.softmax(scores, dim=1)
        return (hidden_states * attn_weights).sum(dim=1)


# ── 多尺度视觉特征融合（创新模块 2） ──
class MultiScaleFusion(nn.Module):
    """从 ViT 多层提取 [CLS] token，拼接后投影到统一维度"""

    def __init__(self, hidden_size=768, embed_dim=768, num_layers=12,
                 num_selected_layers=4):
        super().__init__()
        # 均匀选取 K 层
        k = min(num_selected_layers, num_layers)
        step = max(1, num_layers // k)
        self.selected_layers = list(range(step - 1, num_layers, step))[-k:]
        in_dim = hidden_size * len(self.selected_layers)
        self.fusion = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, all_hidden_states):
        # all_hidden_states: tuple of [B, N, D], includes embedding layer at index 0
        # transformer layer outputs start at index 1
        selected = [all_hidden_states[i + 1][:, 0, :]    # [CLS] token
                    for i in self.selected_layers]
        concat = torch.cat(selected, dim=-1)              # [B, 4*D]
        return self.fusion(concat)


# ══════════════════════════════════════
#  Qwen 文本编码器（创新模块 1: Soft Prompt）
# ══════════════════════════════════════
class QwenTextEncoder(nn.Module):
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B", embed_dim=768,
                 max_length=512, trainable_layers=0, dropout=0.1,
                 use_lora=True, lora_r=4, lora_alpha=8,
                 use_bidirectional=True,
                 use_soft_prompt=False, num_prompt_tokens=8):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.qwen = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.float32)
        self.qwen.requires_grad_(False)
        self.use_soft_prompt = use_soft_prompt
        self.num_prompt_tokens = num_prompt_tokens

        hidden_size = self.qwen.config.hidden_size

        # Soft Prompt: 可学习前缀 token（创新 1）
        if use_soft_prompt:
            self.prompt_embeddings = nn.Parameter(
                torch.randn(num_prompt_tokens, hidden_size) * 0.02)
        else:
            self.prompt_embeddings = None

        # 双向注意力
        if use_bidirectional:
            self._make_bidirectional()

        # LoRA
        if use_lora:
            from peft import LoraConfig, get_peft_model, TaskType
            lora_config = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.1, bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            self.qwen = get_peft_model(self.qwen, lora_config)

        if trainable_layers > 0 and not use_lora:
            for layer in self.qwen.layers[-trainable_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        self.pooling = AttentionPooling(hidden_size)
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.topic_proj_text = nn.Linear(embed_dim, 64)
        self.max_length = max_length

    def _make_bidirectional(self):
        if hasattr(self.qwen, '_update_causal_mask'):
            def bidirectional_mask(self_, *args, **kwargs):
                return None
            self.qwen._update_causal_mask = types.MethodType(
                bidirectional_mask, self.qwen)

    def forward(self, texts, return_topic=False):
        device = next(self.qwen.parameters()).device
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=self.max_length
        ).to(device)

        if self.use_soft_prompt and self.prompt_embeddings is not None:
            # 获取文本 embedding
            word_embeds = self.qwen.get_input_embeddings()(inputs['input_ids'])   # [B, L, D]
            B = word_embeds.size(0)
            prompt = self.prompt_embeddings.unsqueeze(0).expand(B, -1, -1)       # [B, P, D]
            inputs_embeds = torch.cat([prompt, word_embeds], dim=1)               # [B, P+L, D]

            # 扩展 attention_mask：prompt token 全部可见
            prompt_mask = torch.ones(B, self.num_prompt_tokens, device=device,
                                     dtype=inputs['attention_mask'].dtype)
            attention_mask = torch.cat([prompt_mask, inputs['attention_mask']], dim=1)

            outputs = self.qwen(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        else:
            outputs = self.qwen(**inputs)
            attention_mask = inputs['attention_mask']

        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = self.pooling(hidden, mask)
        embeds = self.proj(pooled)
        embeds = F.normalize(embeds, dim=-1)
        if return_topic:
            topic = F.normalize(self.topic_proj_text(embeds.detach()), dim=-1)
            return embeds, topic
        return embeds


# ══════════════════════════════════════
#  Chinese-CLIP 图像编码器（创新模块 2: 多尺度特征）
# ══════════════════════════════════════
class CLIPImageEncoder(nn.Module):
    def __init__(self, model_name="OFA-Sys/chinese-clip-vit-base-patch16",
                 trainable_layers=0, embed_dim=768, dropout=0.1,
                 use_lora=False, lora_r=4, lora_alpha=8,
                 use_multiscale=False, num_multiscale_layers=4):
        super().__init__()
        self.model = ChineseCLIPVisionModel.from_pretrained(model_name)
        self.use_multiscale = use_multiscale

        if use_lora:
            from peft import LoraConfig, get_peft_model, TaskType
            for p in self.model.parameters():
                p.requires_grad = False
            lora_config = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1, bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            self.model = get_peft_model(self.model, lora_config)
            self._forward_model = self.model.base_model
        elif trainable_layers == -1:
            for p in self.model.parameters():
                p.requires_grad = True
        elif trainable_layers > 0:
            for p in self.model.parameters():
                p.requires_grad = False
            v = self.model
            if hasattr(v, 'vision_model'):
                layers = v.vision_model.encoder.layers
            elif hasattr(v, 'encoder'):
                layers = v.encoder.layers
            else:
                raise AttributeError("无法定位 ViT transformer layers")
            for layer in layers[-trainable_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True
        else:
            for p in self.model.parameters():
                p.requires_grad = False
        if not use_lora:
            self._forward_model = self.model

        hidden_size = self.model.config.hidden_size

        # 多尺度融合（创新 2）
        if use_multiscale:
            num_layers = self.model.config.num_hidden_layers if hasattr(
                self.model.config, 'num_hidden_layers') else 12
            self.multiscale_fusion = MultiScaleFusion(
                hidden_size, hidden_size, num_layers,
                num_selected_layers=num_multiscale_layers)
            adapter_in = hidden_size
        else:
            self.multiscale_fusion = None
            adapter_in = hidden_size

        self.vision_adapter = nn.Sequential(
            nn.Linear(adapter_in, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.topic_proj_img = nn.Linear(embed_dim, 64)

    def forward(self, images, return_topic=False):
        device = next(self.model.parameters()).device
        images = images.to(device)

        trainable = any(p.requires_grad for p in self.model.parameters())
        with torch.set_grad_enabled(trainable):
            if self.multiscale_fusion is not None:
                outputs = self._forward_model(
                    pixel_values=images, output_hidden_states=True)
                img_feats = self.multiscale_fusion(outputs.hidden_states)
            else:
                outputs = self._forward_model(pixel_values=images)
                img_feats = outputs.pooler_output

        img_feats = self.vision_adapter(img_feats)
        img_feats = F.normalize(img_feats, dim=-1)
        if return_topic:
            topic = F.normalize(self.topic_proj_img(img_feats.detach()), dim=-1)
            return img_feats, topic
        return img_feats


# ── Chinese-CLIP 文本编码器（冻结，蒸馏教师） ──
class CLIPTextEncoder(nn.Module):
    def __init__(self, model_name="OFA-Sys/chinese-clip-vit-base-patch16"):
        super().__init__()
        self.model = ChineseCLIPTextModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, texts):
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=52,
        ).to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return F.normalize(outputs.pooler_output, dim=-1)


# ── 完整检索模型 ──
class QwenCLIPRetriever(nn.Module):
    def __init__(self, qwen_name="Qwen/Qwen2.5-1.5B",
                 clip_name="OFA-Sys/chinese-clip-vit-base-patch16",
                 embed_dim=768, trainable_layers=0,
                 clip_trainable_layers=0, dropout=0.1,
                 use_lora=False, lora_r=4, lora_alpha=8,
                 clip_use_lora=False, clip_lora_r=4,
                 use_soft_prompt=False, num_prompt_tokens=8,
                 use_multiscale=False, num_multiscale_layers=4):
        super().__init__()
        self.text_encoder = QwenTextEncoder(
            qwen_name, embed_dim,
            trainable_layers=trainable_layers,
            dropout=dropout,
            use_lora=use_lora, lora_r=lora_r, lora_alpha=lora_alpha,
            use_soft_prompt=use_soft_prompt,
            num_prompt_tokens=num_prompt_tokens,
        )
        self.image_encoder = CLIPImageEncoder(
            clip_name,
            trainable_layers=clip_trainable_layers,
            embed_dim=embed_dim, dropout=dropout,
            use_lora=clip_use_lora, lora_r=clip_lora_r,
            use_multiscale=use_multiscale,
            num_multiscale_layers=num_multiscale_layers,
        )

    def forward(self, images=None, texts=None, return_topic=False):
        if images is not None:
            return self.image_encoder(images, return_topic=return_topic)
        if texts is not None:
            return self.text_encoder(texts, return_topic=return_topic)
        raise ValueError("Must provide either images or texts")
