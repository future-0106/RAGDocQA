"""LLM文件：阿里百炼模型调用实现"""
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
                stop=stop or [],
                **kwargs
            )

            if response.status_code == 200:
                content = response.output.choices[0].message.content
            else:
                content = f"API调用失败：{response.code} - {response.message}"
        except Exception as e:
            content = f"LLM调用异常：{str(e)}"

        generation = ChatGeneration(message=AIMessage(content=content))
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self):
        return "dashscope-qwen"