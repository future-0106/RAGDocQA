import json
import os
import glob
from pathlib import Path
from typing import List, Dict, Any


def convert_legal_corpus(file_path: str) -> List[str]:
    """转换法律领域语料库格式 (法律领域语料库1.json, 法律领域语料库2.json)
    格式: {"subTitle": "...", "dataTime": "...", "contentText": "..."}
    支持JSON数组和JSONL格式
    """
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        # 先尝试JSON数组格式
        f.seek(0)
        try:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    title = item.get('subTitle', '')
                    time = item.get('dataTime', '')
                    content = item.get('contentText', '')
                    
                    lines = []
                    if title:
                        lines.append(f"标题: {title}")
                    if time:
                        lines.append(f"时间: {time}")
                    if content:
                        lines.append(f"内容: {content}")
                    
                    if lines:
                        results.append('\n'.join(lines))
                return results
        except json.JSONDecodeError:
            pass
        
        # 尝试JSONL格式
        f.seek(0)
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    title = item.get('subTitle', '')
                    time = item.get('dataTime', '')
                    content = item.get('contentText', '')
                    
                    lines = []
                    if title:
                        lines.append(f"标题: {title}")
                    if time:
                        lines.append(f"时间: {time}")
                    if content:
                        lines.append(f"内容: {content}")
                    
                    if lines:
                        results.append('\n'.join(lines))
                except json.JSONDecodeError:
                    continue
    
    return results


def convert_qa_jsonl(file_path: str) -> List[str]:
    """转换QA问答格式 (DISC-Law-SFT-Pair-QA-released.jsonl, DISC-Law-SFT-Triplet-QA-released.jsonl)
    格式: {"id": "...", "input": "...", "output": "..."}
    """
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                question = item.get('input', '')
                answer = item.get('output', '')
                
                lines = []
                if question:
                    lines.append(f"问题: {question}")
                if answer:
                    lines.append(f"答案: {answer}")
                
                if lines:
                    results.append('\n'.join(lines))
    
    return results


def convert_law_item(file_path: str) -> List[str]:
    """转换法律条款格式 (law_item.jsonl)
    格式: {"title": "...", "classification": "...", "num": "...", "contents": "..."}
    """
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                title = item.get('title', '')
                classification = item.get('classification', '')
                num = item.get('num', '')
                contents = item.get('contents', '')
                
                lines = []
                if title:
                    lines.append(f"标题: {title}")
                if classification:
                    lines.append(f"分类: {classification}")
                if num:
                    lines.append(f"条款: {num}")
                if contents:
                    lines.append(f"内容: {contents}")
                
                if lines:
                    results.append('\n'.join(lines))
    
    return results


def convert_triplet_jsonl(file_path: str) -> List[str]:
    """转换Triplet格式 (DISC-Law-SFT-Pair.jsonl, DISC-Law-SFT-Triplet-released.jsonl)
    格式: {"id": "...", "input": "...", "output": "..."}
    """
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                input_text = item.get('input', '')
                output_text = item.get('output', '')
                
                lines = []
                if input_text:
                    lines.append(f"输入: {input_text}")
                if output_text:
                    lines.append(f"输出: {output_text}")
                
                if lines:
                    results.append('\n'.join(lines))
    
    return results


def convert_alpaca(file_path: str) -> List[str]:
    """转换Alpaca指令格式 (alpaca_dataset.json)
    格式: {"instruction": "...", "input": "...", "output": "..."}
    支持JSON数组和JSONL格式
    """
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        # 先尝试JSON数组格式
        f.seek(0)
        try:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    instruction = item.get('instruction', '')
                    input_text = item.get('input', '')
                    output_text = item.get('output', '')
                    
                    lines = []
                    if instruction:
                        lines.append(f"指令: {instruction}")
                    if input_text:
                        lines.append(f"输入: {input_text}")
                    if output_text:
                        lines.append(f"输出: {output_text}")
                    
                    if lines:
                        results.append('\n'.join(lines))
                return results
        except json.JSONDecodeError:
            pass
        
        # 尝试JSONL格式
        f.seek(0)
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    instruction = item.get('instruction', '')
                    input_text = item.get('input', '')
                    output_text = item.get('output', '')
                    
                    lines = []
                    if instruction:
                        lines.append(f"指令: {instruction}")
                    if input_text:
                        lines.append(f"输入: {input_text}")
                    if output_text:
                        lines.append(f"输出: {output_text}")
                    
                    if lines:
                        results.append('\n'.join(lines))
                except json.JSONDecodeError:
                    continue
    
    return results


def convert_file(file_path: str, output_dir: str) -> int:
    """转换单个文件"""
    filename = os.path.basename(file_path)
    name_without_ext = os.path.splitext(filename)[0]
    ext = os.path.splitext(filename)[1].lower()
    
    results = []
    
    # 根据文件名判断格式
    if '法律领域语料库' in filename:
        results = convert_legal_corpus(file_path)
    elif 'law_item' in filename:
        results = convert_law_item(file_path)
    elif 'QA' in filename:
        results = convert_qa_jsonl(file_path)
    elif 'Triplet' in filename or 'Pair' in filename:
        results = convert_triplet_jsonl(file_path)
    elif 'alpaca' in filename:
        results = convert_alpaca(file_path)
    else:
        print(f"未知格式: {filename}")
        return 0
    
    # 写入输出文件
    output_file = os.path.join(output_dir, f"{name_without_ext}.txt")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, item in enumerate(results):
            f.write(item)
            if i < len(results) - 1:
                f.write('\n---\n')
            else:
                f.write('\n')
    
    print(f"转换完成: {filename} -> {name_without_ext}.txt ({len(results)} 条记录)")
    return len(results)


def main():
    # 配置路径
    source_dir = r"C:\Users\32459\Desktop\可用劳动数据集"
    output_dir = r"C:\Users\32459\Desktop\可用劳动数据集\TXT"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有 JSON 和 JSONL 文件
    json_files = glob.glob(os.path.join(source_dir, "*.json"))
    jsonl_files = glob.glob(os.path.join(source_dir, "*.jsonl"))
    all_files = json_files + jsonl_files
    
    print(f"找到 {len(all_files)} 个文件需要转换\n")
    
    total_count = 0
    for file_path in sorted(all_files):
        count = convert_file(file_path, output_dir)
        total_count += count
    
    print(f"\n转换完成! 共转换 {len(all_files)} 个文件, {total_count} 条记录")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
