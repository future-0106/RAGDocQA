"""LLM文件：修复本地模型张量设备不匹配问题"""
import dashscope
from dashscope import Generation
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from typing import Optional

# ===== 模型工厂类 =====
class ModelFactory:
    @staticmethod
    def create_model(model_config):
        """根据配置创建模型实例"""
        model_type = model_config["type"]
        model_class = model_config["class"]
        model_params = model_config["params"]

        if model_type == "api":
            if model_class == "DashScopeChatModel":
                return DashScopeChatModel(**model_params)
            raise ValueError(f"不支持的API模型类: {model_class}")

        elif model_type == "local":
            if model_class == "LocalChatModel":
                return LocalChatModel(**model_params)
            raise ValueError(f"不支持的本地模型类: {model_class}")

        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

# ===== 阿里百炼API模型 =====
class DashScopeChatModel(BaseChatModel):
    model_name: str
    api_key: str
    temperature: float = 0.3
    max_tokens: int = 1500

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        dashscope_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                dashscope_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                dashscope_messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                dashscope_messages.append({"role": "system", "content": msg.content})

        if not self.api_key:
            raise ValueError("❌ DASHSCOPE_API_KEY未配置")
        dashscope.api_key = self.api_key

        try:
            response = Generation.call(
                model=self.model_name,
                messages=dashscope_messages,
                result_format='message',
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=stop or [],** kwargs
            )

            if response.status_code == 200:
                content = response.output.choices[0].message.content
            else:
                content = f"❌ API调用失败：{response.code} - {response.message}"
        except dashscope.APIError as e:
            content = f"❌ 阿里百炼API错误：{e.code} - {e.message}"
        except Exception as e:
            content = f"❌ LLM调用异常：{str(e)}"

        generation = ChatGeneration(message=AIMessage(content=content))
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self):
        return "dashscope-qwen"

    def invoke(self, prompt, **kwargs):
        if isinstance(prompt, str):
            messages = [HumanMessage(content=prompt)]
        elif isinstance(prompt, list) and all(isinstance(m, dict) for m in prompt):
            messages = []
            for m in prompt:
                if m["role"] == "user":
                    messages.append(HumanMessage(content=m["content"]))
                elif m["role"] == "assistant":
                    messages.append(AIMessage(content=m["content"]))
                elif m["role"] == "system":
                    messages.append(SystemMessage(content=m["content"]))
        else:
            messages = prompt

        result = self._generate(messages,** kwargs)
        return result.generations[0].message

# ===== 本地模型支持（修复张量设备不匹配）=====
class LocalChatModel(BaseChatModel):
    model_path: str
    temperature: float = 0.3
    max_tokens: int = 1500
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer: Optional[AutoTokenizer] = None
    model: Optional[AutoModelForCausalLM] = None

    def __init__(self, **kwargs):
        super().__init__(** kwargs)
        self._load_model()

    def _load_model(self):
        """加载本地模型（强制统一设备，关闭自动设备映射）"""
        # print(f"🔄 正在加载本地模型：{self.model_path} (目标设备：{self.device})")
        try:
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                padding_side="left"
            )

            # 关键修复1：关闭device_map="auto"，强制加载到指定device
            # 避免模型层分散在不同设备导致张量不匹配
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                # 移除 device_map="auto"，改为显式指定设备
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
            ).to(self.device)  # 强制将模型移到指定设备

            # 设置为评估模式
            self.model.eval()

            # 处理缺少pad_token的情况
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # print(f"✅ 本地模型加载完成（运行在{self.device}）")

        except Exception as e:
            raise ValueError(f"❌ 加载本地模型失败：{str(e)}")

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """生成响应（过滤无效参数 + 移除显式device）"""
        # 转换为模型输入格式
        conversation = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                conversation.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                conversation.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                conversation.append({"role": "assistant", "content": msg.content})

        # 构建输入
        input_ids = self.tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        # 强制将输入张量移到模型所在设备
        input_ids = input_ids.to(self.device)

        # ========== 核心修复：过滤无效的model_kwargs ==========
        # 只保留generate方法支持的参数，移除device等无效参数
        valid_generate_kwargs = {}
        # 定义generate支持的核心参数（根据Hugging Face官方文档）
        valid_params = [
            "max_new_tokens", "temperature", "do_sample", "pad_token_id",
            "eos_token_id", "top_p", "top_k", "repetition_penalty"
        ]
        for key, value in kwargs.items():
            if key in valid_params:
                valid_generate_kwargs[key] = value

        # 生成响应（使用过滤后的有效参数）
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=True if self.temperature > 0 else False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    # 仅传递过滤后的有效参数
                    **valid_generate_kwargs
                )

            # 解码输出（确保输出张量移回CPU再解码）
            response_ids = outputs[:, input_ids.shape[1]:].cpu()
            content = self.tokenizer.decode(response_ids[0], skip_special_tokens=True).strip()

        except Exception as e:
            content = f"❌ 本地模型生成失败：{str(e)}"
            # 打印详细设备信息，方便排查
            print(f"⚠️  设备排查：模型设备={next(self.model.parameters()).device}，输入设备={input_ids.device}")

        # 构造返回结果
        generation = ChatGeneration(message=AIMessage(content=content))
        return ChatResult(generations=[generation])
    @property
    def _llm_type(self):
        return "local-llm"

    def invoke(self, prompt, **kwargs):
        if isinstance(prompt, str):
            messages = [HumanMessage(content=prompt)]
        else:
            messages = prompt
        result = self._generate(messages,** kwargs)
        return result.generations[0].message