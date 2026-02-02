"""LLM文件：阿里百炼模型调用实现（增强错误处理）"""
import dashscope
from dashscope import Generation
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from config import MODEL_NAME, TEMPERATURE, MAX_TOKENS

class DashScopeChatModel(BaseChatModel):
    model_name: str = MODEL_NAME
    api_key: str = None
    temperature: float = TEMPERATURE
    max_tokens: int = MAX_TOKENS

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        # 转换消息格式
        dashscope_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                dashscope_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                dashscope_messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                dashscope_messages.append({"role": "system", "content": msg.content})

        # 校验API Key
        if not self.api_key:
            raise ValueError("❌ DASHSCOPE_API_KEY未配置，请在.env文件中添加")
        dashscope.api_key = self.api_key

        # 调用阿里百炼API
        try:
            response = Generation.call(
                model=self.model_name,
                messages=dashscope_messages,
                result_format='message',
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=stop or [],
                **kwargs
            )

            # 处理响应
            if response.status_code == 200:
                content = response.output.choices[0].message.content
            else:
                content = f"❌ API调用失败：{response.code} - {response.message}"
        except dashscope.APIError as e:
            content = f"❌ 阿里百炼API错误：{e.code} - {e.message}"
        except Exception as e:
            content = f"❌ LLM调用异常：{str(e)}"

        # 构造返回结果
        generation = ChatGeneration(message=AIMessage(content=content))
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self):
        return "dashscope-qwen"

    def invoke(self, prompt, **kwargs):
        """简化调用接口（兼容字符串/消息列表）"""
        if isinstance(prompt, str):
            messages = [HumanMessage(content=prompt)]
        elif isinstance(prompt, list) and all(isinstance(m, dict) for m in prompt):
            # 兼容原生dashscope消息格式
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

        result = self._generate(messages, **kwargs)
        return result.generations[0].message