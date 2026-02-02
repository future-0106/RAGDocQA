"""嵌入模型文件：Qwen3-Embedding GPU/CPU兼容版实现"""
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from typing import List
from config import DEVICE, MAX_EMBED_LENGTH
from utils import fix_cpu_compatibility, clean_text
import os

class Qwen3Embeddings:
    def __init__(self, model_path: str, device: torch.device = DEVICE):
        self.model_path = model_path
        self.device = device
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"❌ 模型路径不存在：{self.model_path}")

        # 修复CPU兼容性
        fix_cpu_compatibility()

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            padding_side='left',
            trust_remote_code=True,
            local_files_only=True
        )

        # 加载模型（兼容CPU/GPU）
        dtype = torch.float16 if (torch.cuda.is_available() and self.device.type == "cuda") else torch.float32
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            device_map="auto" if self.device.type == "cuda" else "cpu",
            local_files_only=True,
            torch_dtype=dtype
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"✅ Qwen3-Embedding加载成功（{self.device}）：{self.model_path}")

    @staticmethod
    def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """最后token池化（兼容左padding）"""
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文档文本"""
        if not texts:
            return []
        # 清洗文本
        texts = [clean_text(t) for t in texts]
        # 分词（兼容长文本截断）
        batch_dict = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_EMBED_LENGTH,
            return_tensors="pt"
        ).to(self.device)

        # 推理（禁用梯度）
        with torch.no_grad():
            outputs = self.model(** batch_dict)

        # 池化+归一化
        embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        # 转CPU列表（避免GPU内存占用）
        return embeddings.cpu().tolist()

    def embed_query(self, text: str) -> List[float]:
        """嵌入单条查询文本"""
        return self.embed_documents([text])[0]