"""
LangGraph 版 Agent（展示版）

对照手写 loop：
    手写 for 循环            ↔  图的环（call_tools → call_model）
    手写 messages 上下文     ↔  State + add_messages reducer
    手写 session 隔离        ↔  checkpoint + thread_id
    手写“是否继续”判断       ↔  条件边 + should_continue
    手写 range(5) 保险丝     ↔  recursion_limit（默认 25）

结论：框架封装的是“编排”，不封装“决策”和“业务”。
"""
from __future__ import annotations

import json
import os
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage
from langchain_core.messages.utils import convert_to_openai_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from openai import OpenAI

from src.agent_loop import TOOLBAG, TOOLS

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

SYSTEM_PROMPT = """你是一个可靠的 AI 助手。
需要实时信息或外部知识时，优先调用工具；拿到工具结果后继续判断，
直到能给出最终回答。不要编造工具没有返回的内容。"""


class State(TypedDict):
    messages: Annotated[list, add_messages]


def call_model(state: State) -> dict:
    """节点 1：调用模型；流式拼接文本与 tool_calls，token 通过 writer 外传。"""
    messages = convert_to_openai_messages(state["messages"])
    writer = get_stream_writer()

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, *messages],
        tools=TOOLS,
        tool_choice="auto",
        stream=True,
    )

    text = ""
    tool_call_buffer: dict[int, dict] = {}
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta

        if delta.content:
            text += delta.content
            writer({"type": "token", "content": delta.content})

        if delta.tool_calls:
            for tc in delta.tool_calls:
                buf = tool_call_buffer.setdefault(
                    tc.index, {"id": "", "name": "", "arguments": ""}
                )
                if tc.id:
                    buf["id"] = tc.id
                if tc.function and tc.function.name:
                    buf["name"] = tc.function.name
                if tc.function and tc.function.arguments:
                    buf["arguments"] += tc.function.arguments

    # 拼成 LangChain AIMessage 的 tool_calls 结构
    tool_calls = []
    for index in sorted(tool_call_buffer):
        buf = tool_call_buffer[index]
        try:
            args = json.loads(buf["arguments"] or "{}")
        except json.JSONDecodeError:
            args = {}
        tool_calls.append({"id": buf["id"], "name": buf["name"], "args": args})

    return {"messages": [AIMessage(content=text, tool_calls=tool_calls)]}


def should_continue(state: State) -> str:
    """条件边：最后一条消息有 tool_calls 就继续执行工具，否则结束。"""
    last = state["messages"][-1]
    return "continue" if getattr(last, "tool_calls", None) else "end"


def call_tools(state: State) -> dict:
    """节点 2：执行工具；异常也转成 tool 消息，避免孤儿 tool_calls。"""
    last = state["messages"][-1]
    results = []

    for call in last.tool_calls:
        name = call["name"]
        args = call["args"]
        try:
            func = TOOLBAG.get(name)
            if func is None:
                result = {"error": f"工具不存在: {name}"}
            else:
                result = func(**args)
        except Exception as exc:  # noqa: BLE001 - 工具边界必须兜底
            result = {"error": f"工具执行失败: {exc}"}

        results.append(
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "content": json.dumps(result, ensure_ascii=False),
            }
        )

    return {"messages": results}


def build_graph():
    graph = StateGraph(State)
    graph.add_node("call_model", call_model)
    graph.add_node("call_tools", call_tools)

    graph.add_edge(START, "call_model")
    graph.add_conditional_edges(
        "call_model",
        should_continue,
        {"continue": "call_tools", "end": END},
    )
    graph.add_edge("call_tools", "call_model")

    # checkpoint 持久化：同一个 thread_id 自动续档
    return graph.compile(checkpointer=InMemorySaver())


app = build_graph()


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "demo-session"}}
    for mode, chunk in app.stream(
        {"messages": [{"role": "user", "content": "现在几点了？"}]},
        config,
        stream_mode=["custom", "updates"],
    ):
        print(mode, chunk)
