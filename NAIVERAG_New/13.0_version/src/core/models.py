"""
模型模块：整合模型工厂、LLM、嵌入模型等
"""
import os
import re
import gc
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import requests
from pydantic import BaseModel, Field
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from langchain_huggingface import HuggingFaceEmbeddings
from config import ALL_RERANKER_MODELS, DEFAULT_RERANKER_MODEL
from config import ALL_MODELS, ALL_EMBEDDING_MODELS, MODEL_FACTORY_CONFIG, DEVICE, DEFAULT_MODEL, DEFAULT_EMBEDDING_MODEL


class ModelConfig(BaseModel):
    """模型配置基类"""
    name: str
    type: str  # local, api
    provider: str  # local, dashscope, openai等
    params: Dict[str, Any]
    description: str = ""


class DashScopeChatModel(LLM):
    """阿里云百炼API模型"""

    model_name: str = Field(default="qwen-turbo", description="模型名称")
    api_key: str = Field(default="", description="API密钥")
    temperature: float = Field(default=0.3, description="温度参数")
    max_tokens: int = Field(default=2000, description="最大token数")
    top_p: float = Field(default=0.9, description="top_p参数")
    api_base: str = Field(
        default=os.getenv("DASHSCOPE_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        description="API基础URL"
    )
    timeout: int = Field(default=30, description="请求超时时间")

    def __init__(self, **kwargs):
        model_name = kwargs.pop("model_name", "qwen-turbo")
        api_key = kwargs.pop("api_key", "")
        api_base = kwargs.pop("api_base", None)
        if api_base is None:
            api_base = os.getenv("DASHSCOPE_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        if not api_key:
            api_key = os.getenv("DASHSCOPE_API_KEY", "")
            if not api_key:
                raise ValueError("❌ 未配置阿里云百炼API Key，请在.env文件中设置DASHSCOPE_API_KEY")
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            api_base=api_base,
            **kwargs
        )

    @property
    def _llm_type(self) -> str:
        return f"dashscope_{self.model_name}"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "你是一个智能助手，请根据用户提供的信息回答问题。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stream": False
        }
        if stop:
            payload["stop"] = stop

        try:
            full_url = f"{self.api_base}/chat/completions"
            response = requests.post(
                full_url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"]
                else:
                    return f"❌ API返回格式错误: {result}"
            else:
                return f"❌ API调用失败 (HTTP {response.status_code}): {response.text}"
        except Exception as e:
            return f"❌ API调用异常: {e}"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "api_base": self.api_base
        }


class VLLMChatModel(LLM):
    """vLLM远程API模型"""

    api_base: str = Field(default="", description="API基础URL")
    model_name: str = Field(default="", description="模型名称")
    api_key: str = Field(default="dummy", description="API密钥")
    temperature: float = Field(default=0.3, description="温度参数")
    max_tokens: int = Field(default=1500, description="最大token数")
    timeout: int = Field(default=60, description="请求超时时间")

    def __init__(self, **kwargs):
        api_base = kwargs.pop("api_base", "")
        super().__init__(**kwargs)
        self.api_base = api_base

    @property
    def _llm_type(self) -> str:
        return f"vllm_{self.model_name}"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "你是一个智能助手，请根据用户提供的信息回答问题。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if stop:
            payload["stop"] = stop

        try:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"]
                else:
                    return f"❌ API返回格式错误: {result}"
            else:
                return f"❌ API调用失败 (HTTP {response.status_code}): {response.text}"
        except Exception as e:
            return f"❌ API调用异常: {e}"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "api_base": self.api_base,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }


class ModelScopeChatModel(LLM):
    """ModelScope API模型（GLM-5等）"""

    model_name: str = Field(default="ZhipuAI/GLM-5", description="模型名称")
    api_key: str = Field(default="", description="API密钥")
    temperature: float = Field(default=0.3, description="温度参数")
    max_tokens: int = Field(default=2000, description="最大token数")
    api_base: str = Field(
        default=os.getenv("MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1"),
        description="API基础URL"
    )
    timeout: int = Field(default=30, description="请求超时时间")

    def __init__(self, **kwargs):
        model_name = kwargs.pop("model_name", "ZhipuAI/GLM-5")
        api_key = kwargs.pop("api_key", "")
        api_base = kwargs.pop("api_base", None)
        if api_base is None:
            api_base = os.getenv("MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1")
        if not api_key:
            api_key = os.getenv("MODELSCOPE_API_KEY", "")
            if not api_key:
                raise ValueError("❌ 未配置ModelScope API Key，请在.env文件中设置MODELSCOPE_API_KEY")
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            api_base=api_base,
            **kwargs
        )

    @property
    def _llm_type(self) -> str:
        return f"modelscope_{self.model_name}"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "你是一个智能助手，请根据用户提供的信息回答问题。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False
        }
        if stop:
            payload["stop"] = stop

        try:
            full_url = f"{self.api_base}/chat/completions"
            response = requests.post(
                full_url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    message = result["choices"][0]["message"]
                    content = message.get("content", "")
                    # reasoning = message.get("reasoning_content", "")
                    # if reasoning:
                    #     return f"{reasoning}\n\n=== 最终答案 ===\n\n{content}"
                    return content
                else:
                    return f"❌ API返回格式错误: {result}"
            else:
                return f"❌ API调用失败 (HTTP {response.status_code}): {response.text}"
        except Exception as e:
            return f"❌ API调用异常: {e}"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "api_base": self.api_base
        }


class LocalChatModel(LLM):
    """本地模型 - 已适配聊天模板"""

    model_path: str
    max_new_tokens: int = 512          # 调高以支持更长回答
    temperature: float = 0.3
    top_p: float = 0.9
    repetition_penalty: float = 1.1    # 降低以避免抑制正常输出
    do_sample: bool = True
    device: Any = DEVICE

    _tokenizer: Any = None
    _model: Any = None
    _generation_config: Any = None

    def __init__(self, **data):
        super().__init__(**data)
        print(f"🔧 初始化本地模型: {self.model_path}")
        self._initialize_model()

    def _initialize_model(self):
        """初始化模型与tokenizer"""
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                padding_side="left"
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            torch_dtype = torch.float16 if str(self.device) == "cuda" else torch.float32
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True
            )
            self._model = self._model.to(self.device)

            self._generation_config = GenerationConfig(
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                do_sample=self.do_sample,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
                no_repeat_ngram_size=5,
                penalty_alpha=0.7,
            )
            print(f"✅ 本地模型加载成功！")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise

    @property
    def _llm_type(self) -> str:
        return f"local_{Path(self.model_path).name}"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        try:
            # ---------- 核心修改：使用聊天模板 ----------
            # 将原始 prompt 拆分为 system 和 user 两条消息
            # 注意：用户传入的 prompt 已经包含了上下文、问题和指令，我们直接将其作为 user 消息
            messages = [
                {"role": "system", "content": "你是一个严格基于文档内容回答问题的助手，只能使用提供的上下文，禁止编造。"},
                {"role": "user", "content": prompt}
            ]

            # 应用模型的聊天模板，自动添加生成前缀（如 <|im_start|>assistant）
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # 编码
            inputs = self._tokenizer(text, return_tensors="pt", padding=True)
            if str(self.device) == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # 生成
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    generation_config=self._generation_config,
                    **kwargs
                )

            # 提取新生成的 tokens（排除输入部分）
            generated_ids = outputs[0][len(inputs["input_ids"][0]):]
            response = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

            # 处理停止词
            if stop:
                for stop_word in stop:
                    if stop_word in response:
                        response = response.split(stop_word)[0]

            return response.strip()

        except Exception as e:
            print(f"❌ 生成失败: {e}")
            return "抱歉，生成回答时出现了错误。"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_path": self.model_path,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p
        }


class LocalEmbeddingModel:
    """本地嵌入模型"""
    def __init__(self, model_path: str, **params):
        self.model_path = model_path
        self.device = params.get("device", DEVICE)
        print(f"🔧 初始化本地嵌入模型: {model_path}")
        self._init_model()

    def _init_model(self):
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_path,
                model_kwargs={
                    'device': str(self.device),
                    'trust_remote_code': True,
                },
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': 4,
                }
            )
            test_embedding = self.embeddings.embed_query("测试文本")
            print(f"✅ 嵌入模型加载成功，向量维度: {len(test_embedding)}")
        except Exception as e:
            print(f"❌ 嵌入模型加载失败: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)


class ModelFactory:
    """模型工厂"""
    @staticmethod
    def create_llm(model_key: str) -> LLM:
        if model_key not in ALL_MODELS:
            raise ValueError(f"❌ 模型 '{model_key}' 不存在")
        model_config = ALL_MODELS[model_key]
        model_class = model_config["class"]
        params = model_config["params"].copy()
        if model_class == "DashScopeChatModel":
            api_key = params.get("api_key", "")
            if not api_key:
                api_key = os.getenv("DASHSCOPE_API_KEY", "")
                params["api_key"] = api_key
            api_base = os.getenv("DASHSCOPE_API_BASE")
            if api_base:
                params["api_base"] = api_base
            else:
                dashscope_config = MODEL_FACTORY_CONFIG.get("dashscope", {})
                params["api_base"] = dashscope_config.get("api_base",
                    "https://dashscope.aliyuncs.com/compatible-mode/v1")
            params["timeout"] = dashscope_config.get("timeout", 30)
            return DashScopeChatModel(**params)
        elif model_class == "ModelScopeChatModel":
            api_key = params.get("api_key", "")
            if not api_key:
                api_key = os.getenv("MODELSCOPE_API_KEY", "")
                params["api_key"] = api_key
            api_base = os.getenv("MODELSCOPE_BASE_URL")
            modelscope_config = MODEL_FACTORY_CONFIG.get("modelscope", {})
            if api_base:
                params["api_base"] = api_base
            else:
                params["api_base"] = modelscope_config.get("api_base",
                    "https://api-inference.modelscope.cn/v1")
            params["timeout"] = modelscope_config.get("timeout", 30)
            return ModelScopeChatModel(**params)
        elif model_class == "LocalChatModel":
            return LocalChatModel(**params)
        elif model_class == "VLLMChatModel":
            return VLLMChatModel(**params)
        else:
            raise ValueError(f"❌ 未知的模型类: {model_class}")

    @staticmethod
    def create_embedding_model(model_key: str):
        if model_key not in ALL_EMBEDDING_MODELS:
            raise ValueError(f"❌ 嵌入模型 '{model_key}' 不存在")
        model_config = ALL_EMBEDDING_MODELS[model_key]
        model_class = model_config["class"]
        params = model_config["params"].copy()
        if model_class == "LocalEmbeddingModel":
            return LocalEmbeddingModel(**params)
        else:
            raise ValueError(f"❌ 未知的嵌入模型类: {model_class}")

    @staticmethod
    def create_reranker_model(model_key: str):
        if model_key not in ALL_RERANKER_MODELS:
            raise ValueError(f"❌ 重排模型 '{model_key}' 不存在")
        model_config = ALL_RERANKER_MODELS[model_key]
        model_class = model_config["class"]
        params = model_config["params"].copy()
        if model_class == "LocalRerankerModel":
            from retrieval.retrieval import LocalRerankerModel
            return LocalRerankerModel(**params)
        else:
            raise ValueError(f"❌ 未知的重排模型类: {model_class}")

    @staticmethod
    def list_available_reranker_models() -> List[Dict[str, Any]]:
        models = []
        for key, config in ALL_RERANKER_MODELS.items():
            model_info = {
                "key": key,
                "type": config["type"],
                "description": config.get("description", ""),
                "params": config.get("params", {})
            }
            models.append(model_info)
        return models

    @staticmethod
    def list_available_models() -> List[Dict[str, Any]]:
        models = []
        for key, config in ALL_MODELS.items():
            model_info = {
                "key": key,
                "type": config["type"],
                "provider": config.get("provider", "unknown"),
                "description": config.get("description", ""),
                "params": config.get("params", {})
            }
            models.append(model_info)
        return models

    @staticmethod
    def list_available_embedding_models() -> List[Dict[str, Any]]:
        models = []
        for key, config in ALL_EMBEDDING_MODELS.items():
            model_info = {
                "key": key,
                "type": config["type"],
                "description": config.get("description", ""),
                "params": config.get("params", {})
            }
            models.append(model_info)
        return models


class MultiModelLLM(LLM):
    """统一的多模型LLM封装"""
    model_key: str = Field(default=DEFAULT_MODEL, description="模型配置键")
    _model_instance: Any = None

    def __init__(self, **data):
        if 'model_key' in data and data['model_key']:
            data.setdefault('model_key', data['model_key'])
        super().__init__(**data)
        self._init_model()

    def _init_model(self):
        if self.model_key not in ALL_MODELS:
            print(f"[WARNING] Model '{self.model_key}' not found, using default '{DEFAULT_MODEL}'")
            self.model_key = DEFAULT_MODEL
        self._model_instance = ModelFactory.create_llm(self.model_key)

    def switch_model(self, model_key: str):
        if model_key in ALL_MODELS:
            old_key = self.model_key
            # 释放旧模型显存
            if self._model_instance:
                del self._model_instance
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            self.model_key = model_key
            self._init_model()
            print(f"🔄 已切换模型: {old_key} -> {model_key}")
        else:
            print(f"❌ 模型 '{model_key}' 不存在")

    @property
    def _llm_type(self) -> str:
        config = ALL_MODELS.get(self.model_key, {})
        return f"multi_{config.get('provider', 'unknown')}_{self.model_key}"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        if not self._model_instance:
            self._init_model()
        response = self._model_instance._call(prompt, stop, run_manager, **kwargs)
        response = self._clean_response(response)
        return response

    def _clean_response(self, response: str) -> str:
        if not response or len(response) < 10:
            return "根据文档内容，我无法生成有效的回答。请尝试更具体的问题。"
        # 移除思考过程标记
        thought_patterns = [
            r'<think>.*?</think>',
            r'思考[:：].*?(?=答案[:：]|\n\n|$)',
            r'分析[:：].*?(?=答案[:：]|\n\n|$)',
            r'首先.*?然后.*?最后.*?(?=答案[:：]|\n\n|$)',
            r'根据.*?可以得出.*?(?=答案[:：]|\n\n|$)',
            r'(所以|因此|综上|由此可见|总的来说|总而言之).*?(?=答案[:：]|\n\n|$)',
        ]
        for pattern in thought_patterns:
            response = re.sub(pattern, '', response, flags=re.DOTALL)
        response = re.sub(r'[A-D]\.\s+.*?(?=\n[A-D]\.|\n\n|$)', '', response, flags=re.DOTALL)
        response = re.sub(r'选项[A-D]\s*[:：]?\s*', '', response)
        response = re.sub(r'正确答案?是?[：:]?\s*[A-D]', '', response)
        response = re.sub(r'^答案[：:]\s*', '', response)
        response = ' '.join(response.split())
        return response.strip()

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_key": self.model_key,
            "model_type": ALL_MODELS.get(self.model_key, {}).get("type", "unknown")
        }


class MultiEmbeddings:
    """统一的多嵌入模型封装"""
    def __init__(self, model_key: str = None):
        if model_key is None:
            model_key = DEFAULT_EMBEDDING_MODEL
        self.model_key = model_key
        self._model_instance = None
        self._init_model()

    def _init_model(self):
        if self.model_key not in ALL_EMBEDDING_MODELS:
            self.model_key = DEFAULT_EMBEDDING_MODEL
        self._model_instance = ModelFactory.create_embedding_model(self.model_key)

    def switch_model(self, model_key: str):
        if model_key == self.model_key:
            print(f"ℹ️ 嵌入模型已是 {model_key}，无需切换")
            return
        
        if model_key in ALL_EMBEDDING_MODELS:
            old_key = self.model_key
            if self._model_instance:
                del self._model_instance
                self._model_instance = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            self.model_key = model_key
            self._init_model()
            print(f"🔄 已切换嵌入模型: {old_key} -> {model_key}")
        else:
            print(f"❌ 嵌入模型 '{model_key}' 不存在")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self._model_instance:
            self._init_model()
        return self._model_instance.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        if not self._model_instance:
            self._init_model()
        return self._model_instance.embed_query(text)

    @property
    def embeddings(self):
        if not self._model_instance:
            self._init_model()
        return self._model_instance


class MultiReranker:
    """统一的多重排模型封装"""
    def __init__(self, model_key: str = None):
        if model_key is None:
            model_key = DEFAULT_RERANKER_MODEL
        self.model_key = model_key
        self._model_instance = None
        self._init_model()

    def _init_model(self):
        if self.model_key not in ALL_RERANKER_MODELS:
            self.model_key = DEFAULT_RERANKER_MODEL
        self._model_instance = ModelFactory.create_reranker_model(self.model_key)

    def switch_model(self, model_key: str):
        if model_key == self.model_key:
            print(f"ℹ️ 重排模型已是 {model_key}，无需切换")
            return
        
        if model_key in ALL_RERANKER_MODELS:
            old_key = self.model_key
            if self._model_instance:
                del self._model_instance
                self._model_instance = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            self.model_key = model_key
            self._init_model()
            print(f"🔄 已切换重排模型: {old_key} -> {model_key}")
        else:
            print(f"❌ 重排模型 '{model_key}' 不存在")

    def rerank_batch(self, query: str, documents: List[str], top_k: int = None):
        if not self._model_instance:
            self._init_model()
        return self._model_instance.rerank_batch(query, documents, top_k)

    def rerank(self, query: str, documents: List[str], top_k: int = None):
        if not self._model_instance:
            self._init_model()
        return self._model_instance.rerank(query, documents, top_k)


# 保持向后兼容性
QwenLLM = MultiModelLLM
QwenEmbeddings = MultiEmbeddings