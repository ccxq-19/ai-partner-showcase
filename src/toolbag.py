"""
工具注册表（真实结构展示版）

和根目录 toolbag.py 的区别：
  - 根目录 toolbag.py：服务层演示用的示例注册表，返回示例数据，
    保证 services/ 下的服务可以直接 import 并启动
  - 本文件：真实工具结构，天气走 wttr.in、历史查 SQLite、知识库走 RAG 链路

生产版 77 行，依赖私有天气 SDK / 会话库 / RAG 模块；
展示版把这些依赖替换成公开实现，调用链与错误处理保持一致。
"""
from __future__ import annotations

import time

import requests

from src import db

# 历史搜索时过滤掉的泛词（搜这些词没有意义）
STOP_WORDS = ["记忆", "历史", "话题", "相关", "关于", " ", "的", "记录"]

# 涉及隐私的词：命中就不返回，避免把敏感信息喂给模型
SENSITIVE_WORDS = ["密码", "银行卡", "身份证号", "手机号", "邮箱", "地址", "姓名"]


def token_split(text: str, sep_tokens: list[str] | None = None) -> list[str]:
    """按分隔词把长句切成关键词；长词优先，避免'相关'先于'相关记录'命中。"""
    sep_tokens = sep_tokens or ["相关", "的", "关于", "和", " "]
    sep_tokens = sorted(sep_tokens, key=len, reverse=True)
    sep = "\u0001"
    temp = text
    for token in sep_tokens:
        temp = temp.replace(token, sep + token + sep)
    return [part for part in temp.split(sep) if part]


def get_time() -> dict:
    """查询当前时间。"""
    return {"time": time.strftime("%Y-%m-%d %H:%M:%S")}


def get_weather(city: str) -> dict:
    """查询城市天气（wttr.in 公开接口，无需 key）。"""
    try:
        resp = requests.get(
            f"https://wttr.in/{city}",
            params={"format": "j1", "lang": "zh-cn"},
            timeout=10,
        )
        resp.raise_for_status()
        current = resp.json()["current_condition"][0]
        return {
            "weather": current["weatherDesc"][0]["value"].strip(),
            "temp": f"{current['temp_C']}摄氏度",
            "feels_like": f"{current['FeelsLikeC']}摄氏度",
            "humidity": f"{current['humidity']}%",
        }
    except Exception as exc:  # 网络/解析失败都转成可读错误
        return {"error": f"天气查询失败：{exc}"}


def get_history(field: str) -> dict:
    """在历史会话里搜索关键词；跳过 tool 消息和敏感内容。"""
    keywords = token_split(field)
    keywords = [word for word in keywords if word not in STOP_WORDS]
    if not keywords:
        return {"error": "请提供更具体的关键词（人名、地名、物品等）"}

    if any(word in "".join(keywords) for word in SENSITIVE_WORDS):
        return {"error": "出于隐私保护，不能查询敏感信息"}

    results: dict[str, dict[int, str]] = {}
    for session_id in db.list_sessions():
        session = db.load_session(session_id)
        if not session:
            continue
        for index, message in enumerate(session["messages"]):
            if message.get("role") == "tool":
                continue
            content = message.get("content")
            if not content:
                continue
            if any(word in content for word in SENSITIVE_WORDS):
                continue
            if any(keyword in content for keyword in keywords):
                results.setdefault(session_id, {})[index] = content
    return results


def search_knowledge_base(query: str) -> dict:
    """走 RAG 链路检索知识库；懒加载，避免没配 key 时 import 就失败。"""
    try:
        from src.rag_pipeline import answer_with_rag

        return {"answer": answer_with_rag(query)}
    except Exception as exc:
        return {"error": f"知识库检索失败：{exc}"}


TOOLBAG = {
    "get_time": get_time,
    "get_weather": get_weather,
    "get_history": get_history,
    "search_knowledge_base": search_knowledge_base,
}


if __name__ == "__main__":
    print("time:", get_time())
    print("weather:", get_weather("北京"))
