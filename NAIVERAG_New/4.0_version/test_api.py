"""测试阿里云百炼API连接"""
import os
import requests
import json
from pathlib import Path


def load_env_from_config():
    """从config.py指定的路径加载.env文件"""
    # 使用与config.py相同的路径
    env_path = Path(r"D:\projects\fastapi_langchain_env\NAIVERAG_test\.env")

    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path, override=True)
            print(f"✅ 从 {env_path} 加载环境变量")
            return True
        except ImportError:
            print("⚠️  未安装python-dotenv")
            return False
    else:
        print(f"❌ 未找到 .env 文件: {env_path}")
        # 尝试在当前目录查找
        current_env = Path.cwd() / ".env"
        if current_env.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(current_env, override=True)
                print(f"✅ 从当前目录加载 .env 文件: {current_env}")
                return True
            except ImportError:
                return False
        return False


def test_dashscope_api():
    """测试DashScope API连接"""

    # 先加载环境变量
    load_env_from_config()

    # 获取API密钥
    api_key = os.getenv("DASHSCOPE_API_KEY", "")

    if not api_key:
        print("❌ 未设置DASHSCOPE_API_KEY环境变量")
        print("请检查.env文件是否存在并包含DASHSCOPE_API_KEY")
        return False

    print(f"🔑 API密钥已加载 (长度: {len(api_key)})")

    # 测试不同的API端点
    endpoints = [
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
    ]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 使用OpenAI兼容格式
    payload = {
        "model": "qwen-turbo",
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "temperature": 0.3,
        "max_tokens": 100
    }

    for endpoint in endpoints:
        print(f"\n🔗 测试端点: {endpoint}")
        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=10)
            print(f"📡 状态码: {response.status_code}")

            if response.status_code == 200:
                print("✅ 连接成功！")
                result = response.json()
                if "choices" in result:
                    content = result["choices"][0]["message"]["content"]
                    print(f"🤖 AI回复: {content}")
                else:
                    print(f"📄 响应: {json.dumps(result, ensure_ascii=False, indent=2)}")
                return True
            else:
                print(f"❌ 连接失败: {response.text}")

        except Exception as e:
            print(f"❌ 异常: {e}")

    return False


def check_environment():
    """检查环境配置"""
    print("🔍 检查环境配置:")
    print(f"当前目录: {os.getcwd()}")
    print(f"Python路径: {sys.executable}")

    # 检查.env文件
    env_paths = [
        r"D:\projects\fastapi_langchain_env\NAIVERAG_test\.env",
        ".env",
        Path.cwd() / ".env"
    ]

    for env_path in env_paths:
        if os.path.exists(env_path):
            print(f"\n✅ 找到.env文件: {env_path}")
            try:
                with open(env_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            if "DASHSCOPE_API_KEY" in line:
                                print(f"   {line[:20]}..." if len(line) > 20 else f"   {line}")
            except Exception as e:
                print(f"   读取错误: {e}")
            break
    else:
        print("❌ 未找到任何.env文件")

    # 检查环境变量
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if api_key:
        print(f"\n✅ 环境变量DASHSCOPE_API_KEY已设置 (长度: {len(api_key)})")
        if len(api_key) > 8:
            print(f"   密钥开头: {api_key[:8]}...")
    else:
        print("\n❌ 环境变量DASHSCOPE_API_KEY未设置")


if __name__ == "__main__":
    import sys

    print("🚀 阿里云百炼API测试")
    print("=" * 50)

    check_environment()
    print("\n" + "=" * 50)

    success = test_dashscope_api()

    print("\n" + "=" * 50)
    if success:
        print("🎉 API测试成功！")
    else:
        print("❌ API测试失败")
        print("\n💡 故障排除建议:")
        print("1. 检查API密钥是否正确")
        print("2. 确保已开通阿里云百炼服务")
        print("3. 检查网络连接")
        print("4. 确认API端点是否正确")

        print("\n📋 手动测试命令：")
        print('curl -X POST "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions" \\')
        print('  -H "Authorization: Bearer YOUR_API_KEY" \\')
        print('  -H "Content-Type: application/json" \\')
        print('  -d \'{"model": "qwen-turbo", "messages": [{"role": "user", "content": "Hello"}]}\'')