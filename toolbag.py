"""
工具注册表（展示版）

生产版工具链：
  get_time              -> 本地时间
  get_weather           -> 真实天气 API
  get_history           -> SQLite 会话库关键词搜索（含敏感词过滤）
  search_knowledge_base -> RAG 全链路（切分 / embedding / 向量库 / rerank）

展示版只保留同名函数与签名，返回示例数据，
保证 services/ 下的服务代码可以直接 import 和跑通。
"""
from __future__ import annotations

import time


def get_time() -> dict:
    """查询当前时间。"""
    return {"time": time.strftime("%Y-%m-%d %H:%M:%S")}


def get_weather(city: str) -> dict:
    """查询指定城市天气（展示版返回示例数据）。"""
    return {"city": city, "weather": "示例：晴", "temp": "25摄氏度"}


def get_history(field: str) -> dict:
    """在历史会话里搜索关键词（展示版不接数据库）。"""
    return {"note": f"示例实现：生产版在 SQLite 会话库中搜索 {field!r}"}


def search_knowledge_base(query: str) -> dict:
    """检索知识库（展示版不接向量库）。"""
    return {"note": f"示例实现：生产版走 RAG 链路检索 {query!r}"}


TOOLBAG = {
    "get_time": get_time,
    "get_weather": get_weather,
    "get_history": get_history,
    "search_knowledge_base": search_knowledge_base,
}
