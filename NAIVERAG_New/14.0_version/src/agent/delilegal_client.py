"""
得理开放平台API客户端
用于法规检索、类案检索等
"""
import requests
from typing import Dict, List, Optional


class DeliLegalClient:
    """得理开放平台API客户端"""
    
    BASE_URL = "https://openapi.delilegal.com"
    APPID = "QthdBErlyaYvyXul"
    SECRET = "EC5D455E6BD348CE8E18BE05926D2EBE"
    
    def __init__(self):
        self.headers = {
            "appid": self.APPID,
            "secret": self.SECRET,
            "Content-Type": "application/json"
        }
    
    def search_law(
        self, 
        keywords: str, 
        page: int = 1, 
        page_size: int = 5,
        sort_field: str = "correlation",
        sort_order: str = "desc",
        field_name: str = "semantic"
    ) -> Dict:
        """
        法规检索
        
        Args:
            keywords: 检索关键词
            page: 页码
            page_size: 每页数量
            sort_field: 排序字段 (correlation/time)
            sort_order: 排序方式 (asc/desc)
            field_name: 检索方式 (title:关键词检索, semantic:语义检索)
            
        Returns:
            API响应结果
        """
        url = f"{self.BASE_URL}/api/qa/v3/search/queryListLaw"
        
        payload = {
            "pageNo": page,
            "pageSize": page_size,
            "sortField": sort_field,
            "sortOrder": sort_order,
            "condition": {
                "keywords": [keywords],
                "fieldName": field_name
            }
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ 法规检索失败: {e}")
            return {"success": False, "error": str(e)}
    
    def get_law_detail(self, law_id: str, merge: bool = True) -> Dict:
        """
        获取法规详情
        
        Args:
            law_id: 法规ID（从检索结果中获取）
            merge: 是否合并内容
            
        Returns:
            法规详情
        """
        url = f"{self.BASE_URL}/api/qa/v3/search/lawInfo"
        
        params = {
            "lawId": law_id,
            "merge": str(merge).lower()
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ 获取法规详情失败: {e}")
            return {"success": False, "error": str(e)}
    
    def search_case(
        self,
        keywords: str,
        page: int = 1,
        page_size: int = 5,
        sort_field: str = "correlation",
        sort_order: str = "desc",
        case_year_start: Optional[int] = None,
        case_year_end: Optional[int] = None,
        court_level_arr: Optional[List[str]] = None,
        judgement_type_arr: Optional[List[str]] = None
    ) -> Dict:
        """
        类案检索
        
        Args:
            keywords: 检索关键词
            page: 页码
            page_size: 每页数量
            sort_field: 排序字段 (correlation/time)
            sort_order: 排序方式 (asc/desc)
            case_year_start: 案例年份起始
            case_year_end: 案例年份结束
            court_level_arr: 法院层级 ["0":最高院,"1":高院,"2":中院,"3":基层]
            judgement_type_arr: 文书类型 ["30":判决书,"31":裁决书,"32":调解书]
            
        Returns:
            API响应结果
        """
        url = f"{self.BASE_URL}/api/qa/v3/search/queryListCase"
        
        condition: Dict = {"keywordArr": [keywords]}
        
        if case_year_start is not None:
            condition["caseYearStart"] = str(case_year_start)
        if case_year_end is not None:
            condition["caseYearEnd"] = str(case_year_end)
        if court_level_arr:
            condition["courtLevelArr"] = court_level_arr
        if judgement_type_arr:
            condition["judgementTypeArr"] = judgement_type_arr
        
        payload = {
            "pageNo": page,
            "pageSize": page_size,
            "sortField": sort_field,
            "sortOrder": sort_order,
            "condition": condition
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ 类案检索失败: {e}")
            return {"success": False, "error": str(e)}
    
    def format_law_results(self, result: Dict) -> str:
        """格式化法规检索结果"""
        if not result.get("success"):
            return f"检索失败: {result.get('error', '未知错误')}"
        
        body = result.get("body", {})
        if not body:
            return "未找到相关法规"
        
        laws = body.get("list", [])
        if not laws:
            return "未找到相关法规"
        
        lines = [f"找到 {len(laws)} 条相关法规：\n"]
        
        for i, law in enumerate(laws, 1):
            title = law.get("title", "未知标题")
            level = law.get("levelName", "")
            publisher = law.get("publisherName", "")
            date = law.get("publishDate", "")
            law_id = law.get("lawsId", "")
            
            lines.append(f"\n【{i}】{title}")
            if level:
                lines.append(f"    类型: {level}")
            if publisher:
                lines.append(f"    发布: {publisher}")
            if date:
                lines.append(f"    日期: {date}")
            if law_id:
                lines.append(f"    ID: {law_id}")
        
        return "\n".join(lines)
    
    def format_case_results(self, result: Dict) -> str:
        """格式化类案检索结果"""
        if not result.get("success"):
            return f"检索失败: {result.get('error', '未知错误')}"
        
        body = result.get("body", {})
        if not body:
            return "未找到相关案例"
        
        cases = body.get("list", [])
        if not cases:
            return "未找到相关案例"
        
        lines = [f"找到 {len(cases)} 个相关案例：\n"]
        
        for i, case in enumerate(cases, 1):
            title = case.get("caseTitle", case.get("title", "未知标题"))
            court = case.get("courtName", "")
            date = case.get("judgementDate", "")
            case_type = case.get("caseType", "")
            
            lines.append(f"\n【案例 {i}】{title}")
            if court:
                lines.append(f"    法院: {court}")
            if date:
                lines.append(f"    日期: {date}")
            if case_type:
                lines.append(f"    类型: {case_type}")
        
        return "\n".join(lines)


def create_deli_client() -> DeliLegalClient:
    """创建得理客户端实例"""
    return DeliLegalClient()