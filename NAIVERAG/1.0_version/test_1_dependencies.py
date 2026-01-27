"""测试1：验证所有核心依赖是否安装成功"""
try:
    # 核心依赖
    import os
    import dotenv
    from langchain_huggingface import HuggingFaceEmbeddings  # 替换弃用的导入
    import sentence_transformers
    import transformers
    import dashscope
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import Chroma

    print("✅ 所有依赖导入成功！")
    print(f"📌 transformers版本：{transformers.__version__} (要求≥4.51.0)")
    print(f"📌 sentence-transformers版本：{sentence_transformers.__version__} (要求≥2.7.0)")

except ImportError as e:
    print(f"❌ 依赖导入失败：{str(e)}")
    print("\n💡 解决方法：")
    if "sentence_transformers" in str(e):
        print("   执行：pip install --upgrade sentence-transformers")
    elif "langchain_huggingface" in str(e):
        print("   执行：pip install --upgrade langchain-huggingface")
    elif "dashscope" in str(e):
        print("   执行：pip install --upgrade dashscope")



