# src/models/encoder.py

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer
from typing import List
from omegaconf import DictConfig

class LatentEncoder(nn.Module):
    """
    使用预训练的Transformer模型将文本状态编码为潜空间向量。
    """

    def __init__(self, config: DictConfig, device: torch.device):
        super().__init__()
        model_path = config.rl_agent.model_config.encoder_path
        print(f"Loading Encoder model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(device)
        self.device = device

        # 默认不训练Encoder，将其作为固定的特征提取器
        # 可以通过调用 train() 方法来使其可训练
        self.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def _mean_pooling(self, model_output, attention_mask: torch.Tensor) -> torch.Tensor:
        """对token embeddings进行平均池化以获得句子嵌入"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        """
        将一批文本编码为潜向量。

        Args:
            texts (List[str]): 需要编码的文本列表。

        Returns:
            torch.Tensor: 归一化的潜向量张量, shape (batch_size, latent_dim)。
        """
        # 将模型切换到评估模式
        self.model.eval()

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=1024,  # 编码器有自己的最大长度限制
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model(**inputs)
        embeddings = self._mean_pooling(outputs, inputs['attention_mask'])

        # 对嵌入进行L2归一化，使得距离计算更稳定
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)

        return normalized_embeddings