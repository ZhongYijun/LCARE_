# src/models/lge_encoder.py

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoTokenizer, PreTrainedModel
from typing import List


class LGE_Encoder(nn.Module):
    """
    [V-FINAL] Latent Go-Explore的编码器。
    它通过依赖注入接收一个预训练的Actor模型，并利用其词嵌入层来生成语义向量。
    这确保了潜空间与Actor的语义空间是100%对齐的。
    """

    def __init__(self, actor_model: PreTrainedModel, tokenizer: AutoTokenizer, device: torch.device):
        """
        初始化LGE编码器。

        Args:
            actor_model (PreTrainedModel): 已经实例化的、基础的Actor Transformer模型 (通常是PeftModel解包后的base_model)。
            tokenizer (AutoTokenizer): 与Actor共享的Tokenizer。
            device (torch.device): 模型应该在其上运行的设备。
        """
        super().__init__()

        # 直接引用Actor的embedding层，不复制权重，以节省内存并保证一致性
        self.embedding_layer = actor_model.get_input_embeddings()
        self.tokenizer = tokenizer
        self.device = device

        # 编码器自身不参与训练
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

        print("✅ LGE_Encoder initialized successfully, using the Actor's own embedding layer.")

    def _mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        对token embeddings进行平均池化以获得句子嵌入。
        这是从句子向量表征中获得高质量句子嵌入的标准方法。
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 256) -> torch.Tensor:
        """
        将一批文本编码为潜向量。

        Args:
            texts (List[str]): 需要编码的文本列表。
            batch_size (int): 处理时的内部批次大小。

        Returns:
            torch.Tensor: 归一化的潜向量张量, shape (batch_size, hidden_dim)。
        """
        self.eval()  # 确保在评估模式
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            inputs = self.tokenizer(
                batch_texts, padding=True, truncation=True,
                max_length=8192,  # 编码器的输入长度可以与Actor不同
                return_tensors="pt"
            ).to(self.device)

            # 直接使用embedding层进行查找，而不是完整的模型前向传播
            token_embeddings = self.embedding_layer(inputs.input_ids)

            # 平均池化得到句子/文本的表征
            sentence_embeddings = self._mean_pooling(token_embeddings, inputs.attention_mask)
            all_embeddings.append(sentence_embeddings)

        if not all_embeddings:
            # 从embedding层获取隐藏维度
            hidden_dim = self.embedding_layer.weight.shape[1]
            return torch.empty(0, hidden_dim, device=self.device)

        full_embeddings = torch.cat(all_embeddings, dim=0)
        # 对嵌入进行L2归一化，使得距离计算更稳定
        normalized_embeddings = F.normalize(full_embeddings, p=2, dim=1)

        return normalized_embeddings