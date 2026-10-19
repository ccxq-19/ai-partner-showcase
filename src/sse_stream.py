"""
FastAPI SSE 流式对话（展示版）

协议：
    event: token   data: {"type":"token","content":"..."}
    event: node    data: {"node":"call_model"}
    event: done    data: [DONE]
    event: error   data: {"message":"..."}

工程要点：
  - 用异步生成器 + StreamingResponse，边生成边推送
  - 关闭代理缓冲（X-Accel-Buffering: no），否则 Nginx 会攒满才发
  - 异常也走 SSE 事件，前端能拿到可读错误
"""
from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv()

from src.langgraph_agent import app as agent_app  # noqa: E402

server = FastAPI(
    title="AI 助手 · SSE 流式对话",
    description="LangGraph Agent + FastAPI StreamingResponse（SSE）",
    version="0.1.0",
)


class ChatRequest(BaseModel):
    message: str
    session_id: str = "demo-session"


async def event_stream(req: ChatRequest):
    config = {"configurable": {"thread_id": req.session_id}}
    try:
        async for mode, chunk in agent_app.astream(
            {"messages": [{"role": "user", "content": req.message}]},
            config,
            stream_mode=["custom", "updates"],
        ):
            if mode == "custom":
                yield f"event: token\ndata: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            else:
                node = next(iter(chunk))
                yield (
                    "event: node\n"
                    f"data: {json.dumps({'node': node}, ensure_ascii=False)}\n\n"
                )
    except Exception as exc:  # noqa: BLE001 - 流式接口必须把错误变成事件
        yield (
            "event: error\n"
            f"data: {json.dumps({'message': str(exc)}, ensure_ascii=False)}\n\n"
        )
    finally:
        yield "event: done\ndata: [DONE]\n\n"


@server.post("/chat")
async def chat(req: ChatRequest):
    return StreamingResponse(
        event_stream(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 反代禁缓冲
        },
    )
