"""
主程序文件：统一模型管理RAG系统的主入口
"""
import os
import sys
from pathlib import Path

# 导入各模块
from config import *
from models import MultiModelLLM, MultiEmbeddings, ModelFactory
from documents import DocumentProcessor
from vector_store import ChromaDBManager, FileVectorizationManager
from rag_pipeline import QwenRAGPipeline, setup_environment, check_imports, display_menu, display_model_menu, \
    display_models


class RAGSystem:
    """统一模型管理RAG系统主类"""

    def __init__(self):
        """初始化系统"""
        # 设置环境
        setup_environment()

        # 检查必要的导入
        if not check_imports():
            print("❌ 系统初始化失败，缺少必要的依赖")
            sys.exit(1)

        print("=" * 60)
        print("🚀 统一模型管理RAG系统初始化中...")
        print("=" * 60)

        # 初始化组件
        self._init_components()

        print("✅ 系统初始化完成！")

    def _init_components(self):
        """初始化各个组件"""
        try:
            # 1. 初始化嵌入模型
            print("🔧 初始化嵌入模型...")
            self.embeddings = MultiEmbeddings()

            # 2. 初始化向量存储管理器
            print("🔧 初始化向量存储管理器...")
            self.vector_manager = ChromaDBManager(
                embedding_model=self.embeddings,
                persist_directory=str(CHROMA_DB_DIR)
            )

            # 3. 加载现有的向量存储
            print("🔧 加载向量存储...")
            self.vector_manager.load()

            # 4. 初始化LLM
            print("🔧 初始化大语言模型...")
            self.llm = MultiModelLLM()

            # 5. 初始化文档处理器
            print("🔧 初始化文档处理器...")
            self.document_processor = DocumentProcessor(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )

            # 6. 初始化文件管理器
            print("🔧 初始化文件管理器...")
            self.file_manager = FileVectorizationManager()

            # 7. 初始化RAG流水线
            print("🔧 初始化RAG流水线...")
            self.rag_pipeline = QwenRAGPipeline(self.llm, self.vector_manager)

        except Exception as e:
            print(f"❌ 组件初始化失败: {e}")
            raise

    def clear_screen(self):
        """清屏"""
        os.system('cls' if os.name == 'nt' else 'clear')

    def show_system_status(self):
        """显示系统状态"""
        print("\n📊 系统状态:")
        print("-" * 40)

        # 显示模型状态
        print(f"🤖 当前LLM模型: {self.llm.model_key}")
        print(f"📐 当前嵌入模型: {self.embeddings.model_key}")

        # 显示向量存储状态
        stats = self.vector_manager.get_collection_stats()
        if "error" not in stats:
            print(f"🗂️  向量存储文档数: {stats.get('document_count', '未知')}")
        else:
            print(f"🗂️  向量存储状态: 未加载")

        # 显示文件状态
        file_count = len(self.file_manager.list_data_files())
        print(f"📁 数据目录文件数: {file_count}")
        print(f"💾 使用设备: {DEVICE}")
        print("-" * 40)

    def handle_model_management(self):
        """处理模型管理"""
        while True:
            display_model_menu()
            choice = input("请选择操作 (1-5): ").strip()

            if choice == "1":
                # 查看当前模型状态
                self.show_system_status()

            elif choice == "2":
                # 切换LLM模型
                models_list = ModelFactory.list_available_models()
                display_models(models_list, "可用LLM模型")

                try:
                    model_keys = [m["key"] for m in models_list]
                    print(f"\n当前可用模型键值: {', '.join(model_keys)}")
                    new_model = input("请输入要切换的模型键值: ").strip()

                    if new_model in model_keys:
                        self.llm.switch_model(new_model)
                        # 重新初始化RAG流水线
                        self.rag_pipeline = QwenRAGPipeline(self.llm, self.vector_manager)
                    else:
                        print(f"❌ 模型 '{new_model}' 不存在")
                except Exception as e:
                    print(f"❌ 切换模型失败: {e}")

            elif choice == "3":
                # 切换嵌入模型
                models_list = ModelFactory.list_available_embedding_models()
                display_models(models_list, "可用嵌入模型")

                try:
                    model_keys = [m["key"] for m in models_list]
                    print(f"\n当前可用嵌入模型键值: {', '.join(model_keys)}")
                    new_model = input("请输入要切换的嵌入模型键值: ").strip()

                    if new_model in model_keys:
                        self.embeddings.switch_model(new_model)
                        # 需要重新初始化向量存储管理器
                        self.vector_manager = ChromaDBManager(
                            embedding_model=self.embeddings,
                            persist_directory=str(CHROMA_DB_DIR)
                        )
                        self.vector_manager.load()
                        # 重新初始化RAG流水线
                        self.rag_pipeline = QwenRAGPipeline(self.llm, self.vector_manager)
                        print("🔄 嵌入模型切换完成，可能需要重新处理文档")
                    else:
                        print(f"❌ 嵌入模型 '{new_model}' 不存在")
                except Exception as e:
                    print(f"❌ 切换嵌入模型失败: {e}")

            elif choice == "4":
                # 查看所有可用模型
                print("\n🤖 LLM模型列表:")
                llm_models = ModelFactory.list_available_models()
                display_models(llm_models)

                print("\n📐 嵌入模型列表:")
                embedding_models = ModelFactory.list_available_embedding_models()
                display_models(embedding_models)

            elif choice == "5":
                # 返回主菜单
                break
            else:
                print("❌ 无效的选择，请重新输入")

    def handle_rag_query(self):
        """处理RAG查询"""
        question = input("\n请输入您的问题: ").strip()

        if not question:
            print("⚠️  问题不能为空")
            return

        try:
            result = self.rag_pipeline.query(
                question,
                k=SIMILARITY_TOP_K,
                score_threshold=SCORE_THRESHOLD
            )

            print(f"\n💡 问题: {result['question']}")
            print(f"🤖 回答: {result['answer']}")
            print(f"\n📚 参考来源 ({result['source_count']} 个):")

            for i, source in enumerate(result['sources'], 1):
                print(f"\n  [{i}] {source['source_info']}")
                print(f"      相关度: {source['score']:.3f}")
                print(f"      内容: {source['content']}")

        except Exception as e:
            print(f"❌ 查询失败: {e}")

    def handle_single_file_upload(self):
        """处理单个文件上传"""
        file_path = input("\n请输入要上传的文件路径: ").strip()

        if not os.path.exists(file_path):
            print("❌ 文件不存在")
            return

        success, message, documents = self.file_manager.upload_and_vectorize(
            file_path,
            self.document_processor,
            self.vector_manager
        )

        if success:
            print(f"✅ {message}")
            print(f"📄 处理了 {len(documents)} 个文档块")
        else:
            print(f"❌ {message}")

    def handle_batch_file_upload(self):
        """处理批量文件上传"""
        files_input = input("\n请输入要上传的文件路径（多个文件用逗号分隔）: ").strip()

        if not files_input:
            print("❌ 请输入文件路径")
            return

        file_paths = [path.strip() for path in files_input.split(",") if path.strip()]

        if not file_paths:
            print("❌ 未找到有效的文件路径")
            return

        print(f"📁 准备处理 {len(file_paths)} 个文件...")

        results = self.file_manager.upload_multiple_files(
            file_paths,
            self.document_processor,
            self.vector_manager
        )

        print(f"\n📊 上传结果:")
        print(f"✅ 成功: {len(results['success'])} 个文件")
        print(f"❌ 失败: {len(results['failed'])} 个文件")
        print(f"📄 总文档块数: {results['total_documents']}")

        if results['failed']:
            print("\n失败的文件:")
            for fail in results['failed']:
                print(f"  - {fail['file']}: {fail['message']}")

    def handle_list_files(self):
        """处理文件列表显示"""
        files = self.file_manager.list_data_files()

        if not files:
            print("\n📁 数据目录为空")
            return

        print(f"\n📁 数据目录中的文件 ({len(files)} 个):")
        print("-" * 60)

        for i, filename in enumerate(files, 1):
            file_path = DATA_DIR / filename
            size = file_path.stat().st_size
            size_kb = size / 1024
            print(f"[{i}] {filename} ({size_kb:.1f} KB)")

        # 显示详细文件信息
        show_detail = input("\n是否显示详细文件信息? (y/n): ").lower()
        if show_detail == 'y':
            file_info = self.file_manager.get_file_info()
            for info in file_info:
                print(f"\n  文件名: {info['name']}")
                print(f"  大小: {info['size'] / 1024:.1f} KB")
                print(f"  类型: {info['type']}")
                print(f"  修改时间: {info['modified']}")

    def handle_reprocess_files(self):
        """重新处理所有文件并重建向量存储"""
        files = self.file_manager.list_data_files()

        if not files:
            print("❌ 数据目录为空，无需重新处理")
            return

        confirm = input(f"\n⚠️  确定要重新处理所有 {len(files)} 个文件吗? (y/n): ").lower()
        if confirm != 'y':
            print("❌ 取消操作")
            return

        print("🔄 开始重新处理所有文件...")

        # 清空现有向量存储
        self.vector_manager.vector_store = None

        total_documents = 0
        for i, filename in enumerate(files, 1):
            file_path = DATA_DIR / filename
            print(f"\n[{i}/{len(files)}] 处理文件: {filename}")

            documents = self.document_processor.process_file(str(file_path))

            if documents:
                if self.vector_manager.vector_store:
                    self.vector_manager.add_documents(documents)
                else:
                    self.vector_manager.create_from_documents(documents)

                total_documents += len(documents)
                print(f"✅ 处理完成，添加 {len(documents)} 个文档块")
            else:
                print(f"⚠️  文件处理失败或为空")

        print(f"\n🎉 所有文件重新处理完成!")
        print(f"📄 总共添加了 {total_documents} 个文档块")

    def run(self):
        """运行主循环"""
        self.clear_screen()
        self.show_system_status()

        while True:
            display_menu()
            choice = input("请选择操作 (1-9): ").strip()

            if choice == "1":
                # 基于文档内容回答问题
                self.handle_rag_query()

            elif choice == "2":
                # 上传单个文件并向量化
                self.handle_single_file_upload()

            elif choice == "3":
                # 批量上传多个文件
                self.handle_batch_file_upload()

            elif choice == "4":
                # 查看已上传文件列表
                self.handle_list_files()

            elif choice == "5":
                # 查看系统状态和统计
                self.show_system_status()

            elif choice == "6":
                # 重新处理所有文件并重建向量存储
                self.handle_reprocess_files()

            elif choice == "7":
                # 模型管理（切换模型）
                self.handle_model_management()

            elif choice == "8":
                # 清除屏幕
                self.clear_screen()
                self.show_system_status()

            elif choice == "9":
                # 退出系统
                print("\n👋 感谢使用，再见！")
                break

            else:
                print("❌ 无效的选择，请重新输入")

            # 每次操作后暂停一下
            if choice != "8":  # 清除屏幕不需要暂停
                input("\n按回车键继续...")


def main():
    """主函数"""
    try:
        # 创建系统实例并运行
        system = RAGSystem()
        system.run()
    except KeyboardInterrupt:
        print("\n\n👋 用户中断，退出系统")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 系统运行出错: {e}")
        print("请检查配置和依赖是否正确安装")
        sys.exit(1)


if __name__ == "__main__":
    main()