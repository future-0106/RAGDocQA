"""测试3：验证阿里百炼LLM API调用（OpenAI兼容模式）"""
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

def test_dashscope_openai_compatible():
    try:
        # 加载.env文件（优先读取环境变量）
        dotenv.load_dotenv()

        # 映射到OpenAI风格的环境变量（核心：按你的要求配置）
        os.environ["OPENAI_API_KEY"] = os.getenv("DASHSCOPE_API_KEY")
        os.environ["OPENAI_BASE_URL"] = os.getenv("DASHSCOPE_BASE_URL")

        # 验证环境变量是否配置
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("未配置DASHSCOPE_API_KEY环境变量，请检查.env文件或代码配置")
        if not os.environ.get("OPENAI_BASE_URL"):
            raise ValueError("未配置DASHSCOPE_BASE_URL环境变量，请检查.env文件或代码配置")

        # 初始化OpenAI客户端（自动读取OPENAI_API_KEY和OPENAI_BASE_URL）
        client = OpenAI()

        # 调用阿里百炼Qwen-turbo（OpenAI兼容格式）
        completion = client.chat.completions.create(
            model="qwen-turbo",  # 阿里百炼支持的模型名：qwen-turbo/qwen-plus/qwen-max等
            messages=[
                {"role": "system", "content": "你是一个友好的助手，仅用于测试API调用"},
                {"role": "user", "content": "你好，测试一下是否能正常回答"}
            ],
            temperature=0,  # 固定输出，便于测试
            max_tokens=1024
        )

        # 输出结果
        print("✅ 阿里百炼LLM（OpenAI兼容模式）调用成功！")
        print(f"📌 模型名称：{completion.model}")
        print(f"📌 回答内容：{completion.choices[0].message.content.strip()}")
        print(f"📌 消耗Token：{completion.usage.total_tokens}")

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

if __name__ == "__main__":
    # 先安装依赖提示（首次运行）
    try:
        import openai
        print(f"📌 OpenAI库版本：{openai.__version__} (要求≥1.0.0)")
    except ImportError:
        print("⚠️ 未安装openai库，执行：pip install openai --upgrade")
        exit(1)

    # 运行测试
    test_dashscope_openai_compatible()