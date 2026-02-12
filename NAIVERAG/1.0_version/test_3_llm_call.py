"""测试3：验证阿里百炼LLM API调用（OpenAI兼容模式）- 流式输出+思考过程版"""
import os
import dotenv
from openai import OpenAI


# -------------------------- 核心配置说明 --------------------------
# 方式1：直接在代码中配置（测试用）
# os.environ["DASHSCOPE_API_KEY"] = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # 你的阿里百炼API Key
# os.environ["DASHSCOPE_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 阿里百炼OpenAI兼容地址

# 方式2：通过.env文件配置（推荐生产用）
# .env文件内容示例：
# DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

def test_dashscope_openai_compatible_stream():
    try:
        # 加载.env文件（优先读取环境变量）
        dotenv.load_dotenv()

        # 获取配置并验证
        api_key = os.getenv("DASHSCOPE_API_KEY")
        base_url = os.getenv("DASHSCOPE_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"

        if not api_key:
            raise ValueError("未配置DASHSCOPE_API_KEY环境变量，请检查.env文件或代码配置")

        # 初始化OpenAI客户端（指定阿里百炼配置）
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        # 构建请求消息
        messages = [
            {"role": "system", "content": "你是一个友好的助手，仅用于测试API调用"},
            {"role": "user", "content": "你是谁，介绍一下自己"}
        ]

        # 调用阿里百炼Qwen-turbo（流式+思考过程）
        print("🚀 开始调用阿里百炼LLM（流式输出+思考过程）...")
        completion = client.chat.completions.create(
            model="qwen-turbo",  # 支持qwen-turbo/qwen-plus/qwen-max等
            messages=messages,
            temperature=0,
            max_tokens=1024,
            extra_body={"enable_thinking": True},  # 开启思考过程
            stream=True  # 流式输出
        )

        # 解析流式响应
        is_answering = False  # 是否进入回复阶段
        print("\n" + "=" * 20 + "思考过程" + "=" * 20)

        for chunk in completion:
            delta = chunk.choices[0].delta
            # 处理思考过程内容
            if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                if not is_answering:
                    print(delta.reasoning_content, end="", flush=True)
            # 处理正式回复内容
            if hasattr(delta, "content") and delta.content:
                if not is_answering:
                    print("\n" + "=" * 20 + "完整回复" + "=" * 20)
                    is_answering = True
                print(delta.content, end="", flush=True)

        # 输出结束信息
        print("\n" + "=" * 45)
        print("✅ 阿里百炼LLM（流式+思考过程）调用成功！")

    except ValueError as e:
        print(f"❌ 配置错误：{str(e)}")
    except Exception as e:
        print(f"❌ LLM测试失败：{str(e)}")
        print("\n💡 常见解决方法：")
        print("   1. API Key错误：检查DASHSCOPE_API_KEY是否正确（以sk-开头）")
        print("   2. Base URL错误：确认是https://dashscope.aliyuncs.com/compatible-mode/v1")
        print("   3. 网络问题：确保能访问阿里百炼接口（国内网络无需翻墙）")
        print("   4. API额度不足：登录阿里百炼控制台（https://dashscope.console.aliyun.com/）检查额度")
        print("   5. 依赖缺失：执行pip install openai --upgrade安装最新版openai库")
        print("   6. 思考过程仅支持：qwen-turbo/qwen-plus/qwen-max等新版模型")


if __name__ == "__main__":
    # 检查依赖
    try:
        import openai

        openai_version = openai.__version__
        print(f"📌 OpenAI库版本：{openai_version} (要求≥1.0.0)")
        # 版本检查
        version_parts = openai_version.split('.')
        if int(version_parts[0]) < 1:
            print("⚠️ OpenAI库版本过低，执行：pip install openai --upgrade")
            exit(1)
    except ImportError:
        print("⚠️ 未安装openai库，执行：pip install openai --upgrade")
        exit(1)

    # 运行测试
    test_dashscope_openai_compatible_stream()