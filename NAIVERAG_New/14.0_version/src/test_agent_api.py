"""
法律Agent API 测试脚本
用法: python test_agent_api.py
"""
import requests
import json
import time

BASE_URL = "http://localhost:9000"

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
    return response.json().get("success", False)

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
    return response.json().get("success", False)

def test_document_generation():
    """测试文书生成"""
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
    print_response("文书生成", response)
    return response.json().get("success", False)

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
    return response.json().get("success", False)

def test_auto_route():
    """测试自动路由"""
    test_cases = [
        "我想咨询一下劳动仲裁的流程",
        "帮我审查一下这份合同有没有风险",
        "我想写一个仲裁申请书",
        "这个案件能打赢吗",
    ]
    
    for question in test_cases:
        data = {"question": question}
        response = requests.post(
            f"{BASE_URL}/api/agent/auto",
            json=data,
            headers={"Content-Type": "application/json"}
        )
        print_response(f"自动路由: {question}", response)
        time.sleep(1)

def main():
    print("\n🚀 法律Agent API 测试开始\n")
    
    print("测试1: 健康检查")
    if not test_health():
        print("❌ 服务未启动，请先启动服务")
        return
    
    print("\n测试2: 法律咨询")
    test_consultation()
    
    print("\n测试3: 合同审查")
    test_contract_review()
    
    print("\n测试4: 文书生成")
    test_document_generation()
    
    print("\n测试5: 风险评估")
    test_risk_assessment()
    
    print("\n测试6: 自动路由")
    test_auto_route()
    
    print("\n" + "="*60)
    print("✅ 所有测试完成!")
    print("="*60)

if __name__ == "__main__":
    main()