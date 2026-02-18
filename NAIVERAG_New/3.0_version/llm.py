"""
大语言模型模块
"""
import os
import re
import torch
from typing import List, Dict, Any, Optional
from pydantic import Field
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from config import QWEN_LLM_PATH, LLM_CONFIG


class QwenLLM(LLM):
    """适配Qwen3-0.6B的大语言模型"""

    # 使用Pydantic Field声明字段
    model_path: str = Field(default=QWEN_LLM_PATH, description="模型路径")
    max_new_tokens: int = Field(default=LLM_CONFIG["max_new_tokens"], description="生成的最大token数")
    temperature: float = Field(default=LLM_CONFIG["temperature"], description="温度参数")
    top_p: float = Field(default=LLM_CONFIG["top_p"], description="top-p采样参数")
    repetition_penalty: float = Field(default=LLM_CONFIG["repetition_penalty"], description="重复惩罚")
    do_sample: bool = Field(default=LLM_CONFIG["do_sample"], description="是否采样")

    # 内部使用的字段
    _tokenizer: Any = None
    _model: Any = None
    _generation_config: Any = None
    _device: str = "cpu"

    def __init__(self, **data):
        super().__init__(**data)
        self._initialize_model()

    def _initialize_model(self):
        """初始化模型"""
        print(f"🔧 加载语言模型: {self.model_path}")

        # 检查本地模型路径
        if not os.path.exists(self.model_path):
            print("⚠️  模型路径不存在，尝试使用默认路径")
            self.model_path = QWEN_LLM_PATH

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"📱 使用设备: {self._device}")

        try:
            # 加载tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                padding_side="left"
            )

            # 设置pad_token
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            # 加载模型
            torch_dtype = torch.float16 if self._device == "cuda" else torch.float32
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map="auto" if self._device == "cuda" else None,
                low_cpu_mem_usage=True
            )

            if self._device == "cpu":
                self._model = self._model.to("cpu")

            # 设置生成配置
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

            print("✅ 语言模型加载成功！")

        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise

    @property
    def _llm_type(self) -> str:
        return "qwen_0.6b"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs
    ) -> str:
        try:
            # 编码输入
            inputs = self._tokenizer(prompt, return_tensors="pt", padding=True)

            if self._device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            else:
                inputs = {k: v for k, v in inputs.items()}

            # 生成
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    generation_config=self._generation_config,
                    **kwargs
                )

            # 解码输出
            generated_ids = outputs[0][len(inputs["input_ids"][0]):]
            response = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

            # 处理停止词
            if stop:
                for stop_word in stop:
                    if stop_word in response:
                        response = response.split(stop_word)[0]

            # 清理输出
            response = self._clean_response(response)

            return response.strip()

        except Exception as e:
            print(f"❌ 生成失败: {e}")
            return "抱歉，生成回答时出现了错误。"

    def _clean_response(self, response: str) -> str:
        """清理模型输出，移除所有思考过程和格式"""
        # 移除<think>标签及其内容
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)

        # 移除选择题格式
        response = re.sub(r'[A-D]\.\s+.*?(?=\n[A-D]\.|\n\n|$)', '', response, flags=re.DOTALL)
        response = re.sub(r'选项[A-D]\s*[:：]?\s*', '', response)
        response = re.sub(r'正确答案?是?[：:]?\s*[A-D]', '', response)

        # 移除思考过程标记
        thought_patterns = [
            r'思考[:：].*?(?=答案[:：]|\n\n|$)',
            r'分析[:：].*?(?=答案[:：]|\n\n|$)',
            r'首先.*?然后.*?最后.*?(?=答案[:：]|\n\n|$)',
            r'根据.*?可以得出.*?(?=答案[:：]|\n\n|$)',
        ]

        for pattern in thought_patterns:
            response = re.sub(pattern, '', response, flags=re.DOTALL)

        # 移除问题分析部分
        response = re.sub(r'问题.*?问的是.*?(?=答案[:：]|\n\n|$)', '', response, flags=re.DOTALL)

        # 移除连接词后面的思考
        response = re.sub(r'(所以|因此|综上|由此可见|总的来说|总而言之).*?(?=答案[:：]|\n\n|$)', '', response,
                          flags=re.DOTALL)

        # 移除答案标记，保留实际内容
        response = re.sub(r'答案[：:]?\s*', '', response)

        # 移除多余的空行和空白字符
        lines = response.split('\n')
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            if line and not self._is_thought_line(line):
                cleaned_lines.append(line)

        # 合并行
        cleaned_response = ' '.join(cleaned_lines)

        # 移除多余的空格
        cleaned_response = ' '.join(cleaned_response.split())

        # 如果清理后为空，返回默认信息
        if not cleaned_response or len(cleaned_response) < 10:
            return "根据文档内容，我无法生成有效的回答。请尝试更具体的问题。"

        return cleaned_response

    def _is_thought_line(self, line: str) -> bool:
        """判断一行是否是思考过程"""
        thought_keywords = [
            '思考：', '分析：', '首先', '然后', '最后',
            '所以', '因此', '综上', '由此可见', '总的来说',
            '选项', '正确答案', '应该选', '选择'
        ]

        line_lower = line.lower()
        for keyword in thought_keywords:
            if keyword in line_lower:
                return True
        return False

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_path": self.model_path,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p
        }