"""
手写 Agent loop（展示版）

核心链路：
    模型决策 → 工具调用 → 结果回流 → 再决策 → 直到模型给出最终回答

生产版里这套循环跑在 Streamlit / FastAPI 中；这里抽成一个可独立阅读的最小实现。
重点看四件事：
  1. 流式 tool_calls 的分片拼接（按 index 归组）
  2. 最大轮数保险丝（防止模型无限调用工具）
  3. 先算后写（工具全部执行完，再一次性写入消息，避免“孤儿 tool_calls”）
  4. 工具容错（异常也转成 tool 消息回流，循环不炸）
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime

from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

SYSTEM_PROMPT = """你是一个可靠的 AI 助手。
需要实时信息或外部知识时，优先调用工具；拿到工具结果后继续判断，
直到能给出最终回答。不要编造工具没有返回的内容。"""

# ---------------------------------------------------------------------------
# 工具：说明书（给模型看） + 注册表（给代码用）
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "查询当前日期和时间",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名，如北京"}
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_history",
            "description": (
                "在历史会话中搜索包含指定关键词的聊天内容。"
                "field 必须填具体关键词（人名/地名/物品/话题），不要填“记忆”“历史”这类泛词。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "field": {"type": "string", "description": "要搜索的具体关键词"}
                },
                "required": ["field"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "在知识库中检索专业资料；当用户明确要求查知识库，或内置知识不足时使用",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "要检索的问题或关键词"}
                },
                "required": ["query"],
            },
        },
    },
]


def get_time() -> dict:
    return {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


def get_weather(city: str) -> dict:
    """示例实现：生产版接的是 fetch-my-weather 的真实天气数据。"""
    return {"city": city, "weather": "晴", "temp": "25°C", "source": "demo"}


def search_history(field: str) -> dict:
    """示例实现：生产版遍历 SQLite 中的全部会话，按关键词匹配并过滤敏感词。"""
    return {"keyword": field, "matches": [], "note": "demo：生产版查 SQLite 会话库"}


def search_knowledge_base(query: str) -> dict:
    """示例实现：生产版调用 rag_pipeline.answer_with_rag(query)。"""
    return {
        "query": query,
        "answer": "（demo）生产版会返回 RAG 检索 + Rerank 后的回答",
        "source": "demo",
    }


TOOLBAG = {
    "get_time": get_time,
    "get_weather": get_weather,
    "search_history": search_history,
    "search_knowledge_base": search_knowledge_base,
}


def execute_tool(name: str, args: dict) -> dict:
    """工具容错：工具不存在、参数错、内部异常，都转成可回流的 error 结果。"""
    try:
        func = TOOLBAG.get(name)
        if func is None:
            return {"error": f"工具不存在: {name}"}
        return func(**args)
    except Exception as exc:  # noqa: BLE001 - 工具边界必须兜底
        return {"error": f"工具执行失败: {exc}"}


# ---------------------------------------------------------------------------
# 核心循环
# ---------------------------------------------------------------------------
def run_agent(question: str, max_steps: int = 5) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for step in range(max_steps):
        stream = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            stream=True,
        )
        content = ""
        tool_call_buffer: dict[int, dict] = {}
        finish_reason = None

        for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            finish_reason = choice.finish_reason or finish_reason

            if delta.content:
                content += delta.content

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    buf = tool_call_buffer.setdefault(
                        tc.index,
                        {"id": "", "name": "", "arguments": ""},
                    )
                    if tc.id:
                        buf["id"] = tc.id
                    if tc.function and tc.function.name:
                        buf["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        buf["arguments"] += tc.function.arguments

        # 1) 模型直接给出最终回答
        if finish_reason == "stop":
            messages.append({"role": "assistant", "content": content})
            return content

        # 2) 模型要调用工具
        if finish_reason == "tool_calls":
            calls = []
            for index in sorted(tool_call_buffer):
                buf = tool_call_buffer[index]
                calls.append(
                    {
                        "id": buf["id"],
                        "type": "function",
                        "function": {
                            "name": buf["name"],
                            "arguments": buf["arguments"],
                        },
                    }
                )

            # 先算后写：所有工具先执行完、结果都拿到，再写消息。
            # 这样即使某个工具抛异常，也不会留下“有 tool_calls、没有 tool 回复”的孤儿消息。
            results = []
            for call in calls:
                try:
                    args = json.loads(call["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                results.append((call["id"], execute_tool(call["function"]["name"], args)))

            messages.append(
                {"role": "assistant", "content": None, "tool_calls": calls}
            )
            for call_id, result in results:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
            continue

        # 3) 其它情况：避免死循环，直接进入收尾
        break

    # 保险丝烧掉后：补一次不带 tools 的调用，强制模型基于已有信息总结
    final = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=False,
    )
    return final.choices[0].message.content


if __name__ == "__main__":
    print(run_agent("现在几点了？"))
