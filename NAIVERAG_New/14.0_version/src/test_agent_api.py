"""
法律Agent API 测试脚本 (Agentic RAG版本)
用法: python test_agent_api.py
"""
import requests
import json
import time

BASE_URL = "http://localhost:8000"

def print_response(name: str, response: requests.Response):
    """打印响应结果"""
    print(f"\n{'='*60}")
    print(f"📌 {name}")
    print(f"{'='*60}")
    print(f"状态码: {response.status_code}")
    try:
        data = response.json()
        print(f"响应内容:")
        print(json.dumps(data, indent=2, ensure_ascii=False))
    except:
        print(f"响应文本: {response.text[:500]}")

def check_declaration(response_data: dict, field: str = "answer") -> bool:
    """检查响应中是否包含数据来源声明"""
    content = response_data.get(field, "")
    return "⚠️ 声明" in content or "声明：" in content

def test_health():
    """测试健康检查"""
    response = requests.get(f"{BASE_URL}/api/health")
    print_response("健康检查", response)
    return response.status_code == 200

def test_consultation():
    """测试法律咨询"""
    data = {"question": "什么是劳动合同试用期？法律规定最长多久？"}
    response = requests.post(
        f"{BASE_URL}/api/agent/consultation",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    print_response("法律咨询", response)
    result = response.json()
    success = result.get("success", False)
    if success and check_declaration(result, "answer"):
        print("\n✅ 包含数据来源声明")
    return success

def test_contract_review():
    """测试合同审查"""
    data = {
        "contract_text": """
        劳动合同
        
        甲方: XX公司
        乙方: 张三
        
        1. 合同期限: 2024年1月1日至2027年12月31日
        2. 试用期: 6个月
        3. 月薪: 3000元 (试用期2500元)
        4. 工作时间: 每天9小时，每周6天
        5. 社保: 由乙方自行缴纳
        """
    }
    response = requests.post(
        f"{BASE_URL}/api/agent/contract-review",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    print_response("合同审查", response)
    result = response.json()
    success = result.get("success", False)
    if success and check_declaration(result, "review"):
        print("\n✅ 包含数据来源声明")
    return success

def test_document_generation():
    """测试文书生成 - 劳动仲裁申请书"""
    data = {
        "doctype": "劳动仲裁申请书",
        "facts": """申请人李四于2024年1月15日入职被申请人XX公司，担任销售经理。
        双方未签订书面劳动合同。
        被申请人自2024年3月起拖欠申请人工资，至今已欠发3个月工资共计15000元。
        被申请人未为申请人缴纳社会保险。
        申请人月工资标准为5000元。"""
    }
    response = requests.post(
        f"{BASE_URL}/api/agent/document",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    print_response("文书生成 - 劳动仲裁申请书", response)
    result = response.json()
    success = result.get("success", False)
    if success and check_declaration(result, "document"):
        print("\n✅ 包含数据来源声明")
    return success

def test_iou_generation():
    """测试借条生成 - 验证无外部数据时基于知识回答"""
    data = {
        "doctype": "借条",
        "facts": """借款人：李四
出借人：王五
借款金额：人民币10000元
借款日期：2024年1月1日
还款日期：2024年12月31日
利息：年利率10%"""
    }
    response = requests.post(
        f"{BASE_URL}/api/agent/document",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    print_response("借条生成（无外部数据时基于知识）", response)
    result = response.json()
    success = result.get("success", False)
    if success and check_declaration(result, "document"):
        print("\n✅ 包含数据来源声明")
    return success

def test_risk_assessment():
    """测试风险评估"""
    data = {
        "case_type": "劳动争议",
        "facts": """申请人主张与被申请人存在劳动关系，被申请人拖欠工资3个月，
        未签订劳动合同，未缴纳社保。申请人手头有工资条和考勤记录，
        但没有劳动合同。"""
    }
    response = requests.post(
        f"{BASE_URL}/api/agent/risk-assessment",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    print_response("风险评估", response)
    result = response.json()
    success = result.get("success", False)
    if success and check_declaration(result, "assessment"):
        print("\n✅ 包含数据来源声明")
    return success

def test_auto_route():
    """测试自动路由"""
    test_cases = [
        ("我想咨询一下劳动仲裁的流程", "procedure_guide"),
        ("帮我审查一下这份合同有没有风险", "contract_review"),
        ("我想写一个仲裁申请书", "document_generation"),
        ("这个案件能打赢吗", "consultation"),
        ("帮我写一个借条", "document_generation"),
    ]
    
    for question, expected_type in test_cases:
        data = {"question": question}
        response = requests.post(
            f"{BASE_URL}/api/agent/auto",
            json=data,
            headers={"Content-Type": "application/json"}
        )
        result = response.json()
        print_response(f"自动路由: {question}", response)
        
        if result.get("success"):
            if expected_type == "document_generation" and "need_input" in result.get("type", ""):
                print(f"✅ 正确识别为需要输入的文书生成")
            elif result.get("type") == expected_type or result.get("type") == "consultation":
                print(f"✅ 路由正确")
        time.sleep(1)

def test_multi_turn_conversation():
    """测试多轮对话"""
    session = requests.Session()
    
    print("\n" + "="*60)
    print("📌 多轮对话测试")
    print("="*60)
    
    # 第一轮
    data1 = {"question": "我想咨询劳动仲裁的问题"}
    response1 = session.post(
        f"{BASE_URL}/api/agent/consultation",
        json=data1,
        headers={"Content-Type": "application/json"}
    )
    print(f"\n第1轮: {data1['question']}")
    print(f"响应状态: {response1.status_code}")
    result1 = response1.json()
    if result1.get("success"):
        print(f"回复: {result1.get('answer', '')[:200]}...")
    
    # 第二轮（追问）
    data2 = {"question": "仲裁需要多长时间？"}
    response2 = session.post(
        f"{BASE_URL}/api/agent/consultation",
        json=data2,
        headers={"Content-Type": "application/json"}
    )
    print(f"\n第2轮: {data2['question']}")
    print(f"响应状态: {response2.status_code}")
    result2 = response2.json()
    if result2.get("success"):
        print(f"回复: {result2.get('answer', '')[:200]}...")
    
    return result1.get("success", False) and result2.get("success", False)

def main():
    print("\n🚀 法律Agent API 测试开始 (Agentic RAG版本)\n")
    
    print("=" * 60)
    print("测试1: 健康检查")
    print("=" * 60)
    if not test_health():
        print("❌ 服务未启动，请先启动服务")
        return
    
    print("\n" + "=" * 60)
    print("测试2: 法律咨询")
    print("=" * 60)
    test_consultation()
    
    print("\n" + "=" * 60)
    print("测试3: 合同审查")
    print("=" * 60)
    test_contract_review()
    
    print("\n" + "=" * 60)
    print("测试4: 文书生成 - 劳动仲裁申请书")
    print("=" * 60)
    test_document_generation()
    
    print("\n" + "=" * 60)
    print("测试5: 借条生成（验证无外部数据时基于知识回答）")
    print("=" * 60)
    test_iou_generation()
    
    print("\n" + "=" * 60)
    print("测试6: 风险评估")
    print("=" * 60)
    test_risk_assessment()
    
    print("\n" + "=" * 60)
    print("测试7: 自动路由")
    print("=" * 60)
    test_auto_route()
    
    print("\n" + "=" * 60)
    print("测试8: 多轮对话")
    print("=" * 60)
    test_multi_turn_conversation()
    
    print("\n" + "="*60)
    print("✅ 所有测试完成!")
    print("="*60)

if __name__ == "__main__":
    main()