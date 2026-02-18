"""
主程序入口 - 使用数字选项的简洁界面
"""
import sys
import time
import shutil
import re
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

# 导入配置和工具
from config import *
from utils import setup_environment, check_imports, display_menu

# 设置环境
setup_environment()

if not check_imports():
    sys.exit(1)

# 导入各功能模块
from embeddings import QwenEmbeddings
from llm import QwenLLM
from document_processor import DocumentProcessor
from vector_store import ChromaDBManager
from rag_pipeline import QwenRAGPipeline
from file_manager import FileVectorizationManager


def handle_query(rag_pipeline):
    """处理查询"""
    question = input("\n📝 请输入您的问题: ").strip()
    if not question:
        print("❌ 问题不能为空")
        return

    # 清理用户输入
    question = question.rstrip('-').strip()

    # 执行查询
    start_time = time.time()
    result = rag_pipeline.query(question, k=SIMILARITY_TOP_K, score_threshold=SCORE_THRESHOLD)
    elapsed_time = time.time() - start_time

    # 显示结果
    print(f"\n📝 回答: {result['answer']}")
    print(f"⏱️  处理时间: {elapsed_time:.2f}秒")
    print(f"📊 检索到 {result['source_count']} 个相关文档块")

    # 显示详细来源
    if result['sources']:
        print(f"\n🔍 检索来源:")
        for i, source in enumerate(result['sources']):
            print(f"\n  [{i + 1}] {source['source_info']}")
            print(f"      相关度: {source['score']:.4f}")
            # 清理来源内容中的特殊字符
            clean_content = re.sub(r'[_\-\*\/\\]', ' ', source['content'])
            print(f"      内容: {clean_content[:200]}...")

    print("\n" + "-" * 50)


def handle_upload_single(file_manager, processor, vector_manager):
    """处理单个文件上传"""
    print("\n📤 上传文件并立即转换为向量")
    print("支持格式: PDF, TXT, MD")
    source_path = input("请输入文件路径 (或输入 'cancel' 取消): ").strip()

    if source_path.lower() == 'cancel':
        print("上传已取消")
        return

    if not source_path:
        print("❌ 请输入有效的文件路径")
        return

    # 上传并向量化
    start_time = time.time()
    success, message, documents = file_manager.upload_and_vectorize(
        source_path, processor, vector_manager
    )
    elapsed_time = time.time() - start_time

    if success:
        print(f"✅ {message}")
        print(f"⏱️  处理时间: {elapsed_time:.2f}秒")
        print(f"📄 生成 {len(documents)} 个文本块")
    else:
        print(f"❌ {message}")


def handle_upload_multiple(file_manager, processor, vector_manager):
    """处理批量文件上传"""
    print("\n📤 批量上传多个文件并立即转换为向量")
    print("支持格式: PDF, TXT, MD")
    print("请输入文件路径，用逗号或分号分隔多个文件")
    print("示例: /path/to/file1.pdf, /path/to/file2.txt")

    input_paths = input("请输入文件路径列表: ").strip()

    if not input_paths:
        print("❌ 请输入有效的文件路径")
        return

    # 分割文件路径
    paths = []
    for path in input_paths.replace(';', ',').split(','):
        path = path.strip()
        if path:
            paths.append(path)

    if not paths:
        print("❌ 没有找到有效的文件路径")
        return

    print(f"📋 发现 {len(paths)} 个文件:")
    for i, path in enumerate(paths, 1):
        print(f"  [{i}] {path}")

    confirm = input(f"确认上传这 {len(paths)} 个文件? (y/N): ").strip().lower()
    if confirm != 'y':
        print("批量上传已取消")
        return

    # 批量上传并向量化
    start_time = time.time()
    results = file_manager.upload_multiple_files(paths, processor, vector_manager)
    elapsed_time = time.time() - start_time

    print(f"\n📊 批量上传结果:")
    print(f"⏱️  总处理时间: {elapsed_time:.2f}秒")
    print(f"✅ 成功: {len(results['success'])} 个文件")
    print(f"❌ 失败: {len(results['failed'])} 个文件")
    print(f"📄 总文档块数: {results['total_documents']}")

    if results['success']:
        print("\n📋 成功文件:")
        for i, success_info in enumerate(results['success'], 1):
            print(f"  [{i}] {success_info['file']} - {success_info['document_count']} 个文档块")

    if results['failed']:
        print("\n❌ 失败文件:")
        for i, failed_info in enumerate(results['failed'], 1):
            print(f"  [{i}] {failed_info['file']} - {failed_info['message']}")


def handle_list_files(file_manager):
    """列出所有文件"""
    files = file_manager.list_data_files()
    print(f"\n📁 数据目录中的文件 ({len(files)} 个):")
    if files:
        for i, filename in enumerate(files, 1):
            file_path = DATA_DIR / filename
            if file_path.exists():
                size_kb = file_path.stat().st_size / 1024
                print(f"  [{i}] {filename} ({size_kb:.1f} KB)")
            else:
                print(f"  [{i}] {filename}")
    else:
        print("  (空)")


def handle_stats(vector_manager, file_manager):
    """显示系统状态"""
    print(f"\n📊 系统状态:")
    print(f"📁 数据目录: {DATA_DIR}")
    print(f"📂 向量存储目录: {CHROMA_DB_DIR}")

    # 向量存储统计
    stats = vector_manager.get_collection_stats()
    print(f"\n📈 向量存储统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 文件信息
    file_info = file_manager.get_file_info()
    if file_info:
        print(f"\n📁 数据目录文件 ({len(file_info)} 个):")
        for i, info in enumerate(file_info, 1):
            size_kb = info['size'] / 1024
            print(
                f"  [{i}] {info['name']} ({size_kb:.1f} KB, {info['type']}, 修改时间: {info['modified']})")
    else:
        print("\n📁 数据目录: (空)")


def handle_reload(file_manager, processor, vector_manager):
    """重新处理所有文件"""
    print("\n🔄 重新处理所有文件并重建向量存储")
    files = file_manager.list_data_files()

    if not files:
        print("❌ 数据目录中没有文件")
        return

    print(f"📋 数据目录中有 {len(files)} 个文件:")
    for i, filename in enumerate(files, 1):
        print(f"  [{i}] {filename}")

    confirm = input("确认重新处理所有文件并重建向量存储? (y/N): ").strip().lower()

    if confirm == 'y':
        # 收集所有文档
        all_documents = []
        for filename in files:
            file_path = DATA_DIR / filename
            documents = processor.process_file(str(file_path))
            all_documents.extend(documents)
            print(f"✅ 已处理: {filename} ({len(documents)} 个文档块)")

        if all_documents:
            # 删除旧的向量存储并创建新的
            if CHROMA_DB_DIR.exists():
                shutil.rmtree(CHROMA_DB_DIR, ignore_errors=True)
                time.sleep(1)

            # 创建新的向量存储
            print(f"\n🔄 正在创建新的向量存储，包含 {len(all_documents)} 个文档块...")
            vector_manager.create_from_documents(all_documents)

            # 显示统计信息
            stats = vector_manager.get_collection_stats()
            print(f"\n📊 向量存储统计:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        else:
            print("❌ 没有处理任何文档")


def main():
    """主函数"""
    try:
        # 1. 初始化模型
        print("\n1️⃣ 初始化模型...")
        embedding_model = QwenEmbeddings()
        llm = QwenLLM(**LLM_CONFIG)

        # 2. 初始化文档处理器
        processor = DocumentProcessor(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

        # 3. 初始化向量存储管理器
        print("\n2️⃣ 初始化向量存储管理器...")
        vector_manager = ChromaDBManager(embedding_model, persist_directory=str(CHROMA_DB_DIR))

        # 4. 初始化文件上传和向量化管理器
        print("\n3️⃣ 初始化文件上传和向量化管理器...")
        file_manager = FileVectorizationManager()

        # 5. 检查并加载向量存储
        existing_store = vector_manager.load()
        if existing_store:
            print("✅ 发现现有向量存储")
        else:
            print("📂 没有找到现有向量存储")

        # 6. 创建RAG流水线
        print("\n4️⃣ 创建RAG流水线...")
        rag_pipeline = QwenRAGPipeline(llm, vector_manager)

        print("\n✅ 系统初始化完成！")
        time.sleep(1)

        # 7. 主循环
        while True:
            display_menu()

            try:
                choice = input("\n请选择操作编号 [1-8]: ").strip()

                if choice == '1':
                    handle_query(rag_pipeline)
                elif choice == '2':
                    handle_upload_single(file_manager, processor, vector_manager)
                elif choice == '3':
                    handle_upload_multiple(file_manager, processor, vector_manager)
                elif choice == '4':
                    handle_list_files(file_manager)
                elif choice == '5':
                    handle_stats(vector_manager, file_manager)
                elif choice == '6':
                    handle_reload(file_manager, processor, vector_manager)
                elif choice == '7':
                    os.system('cls' if os.name == 'nt' else 'clear')
                elif choice == '8':
                    print("\n👋 再见！")
                    break
                else:
                    print("❌ 无效选择，请输入 1-8 之间的数字")

            except KeyboardInterrupt:
                print("\n\n👋 再见！")
                break
            except Exception as e:
                print(f"❌ 操作失败: {e}")

    except Exception as e:
        print(f"\n❌ 系统错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Qwen3-0.6B RAG系统 - 上传文件后立即转换为向量")
    parser.add_argument("--data", type=str, default="data", help="数据目录路径")
    parser.add_argument("--reset", action="store_true", help="重置向量存储（删除已有的）")

    args = parser.parse_args()

    if args.data != "data":
        DATA_DIR = Path(args.data)
        if not DATA_DIR.exists():
            print(f"❌ 指定的数据目录不存在: {args.data}")
            sys.exit(1)

    if args.reset:
        # 删除向量存储
        if CHROMA_DB_DIR.exists():
            shutil.rmtree(CHROMA_DB_DIR)
            print(f"🗑️  已删除向量存储目录: {CHROMA_DB_DIR}")
            time.sleep(0.5)

    sys.exit(main())