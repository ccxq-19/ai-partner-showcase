"""
FastAPI WebSocket 实时对话（展示版）

一个连接 = 一个会话；同一条连接内可多轮对话。

事件协议（6 类）：
    thinking     模型思考片段
    content      正文片段
    tool_call    模型决定调用工具
    tool_result  工具执行结果
    done         本轮结束
    error        错误（含不可恢复错误，随后 close 1008）

工程要点：
  - 同步工具用 asyncio.to_thread 丢线程池，避免阻塞事件循环
  - 工具异常转成 tool 消息回流，保证 tool_calls 永远有对应回复
  - 断连时清理连接；房间广播单独走 /ws/room/{room_id}
"""
from __future__ import annotations

import asyncio
import json
import os
import time

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from openai import AsyncOpenAI

from src.agent_loop import TOOLBAG, TOOLS

load_dotenv()

client = AsyncOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    timeout=httpx.Timeout(connect=15.0, read=120.0, write=60.0, pool=15.0),
)
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
MAX_STEPS = 5

SYSTEM_PROMPT = """你是一个可靠的 AI 助手。
需要实时信息或外部知识时，优先调用工具；拿到工具结果后继续判断，
直到能给出最终回答。不要编造工具没有返回的内容。"""

app = FastAPI(title="AI 助手 · WebSocket 实时对话")


class ConnectionManager:
    """聊天连接池 + 房间广播。"""

    def __init__(self) -> None:
        self.chat_connections: set[WebSocket] = set()
        self.rooms: dict[str, set[WebSocket]] = {}

    async def connect_chat(self, ws: WebSocket) -> None:
        await ws.accept()
        self.chat_connections.add(ws)

    def disconnect_chat(self, ws: WebSocket) -> None:
        self.chat_connections.discard(ws)

    async def join_room(self, room: str, ws: WebSocket) -> None:
        await ws.accept()
        self.rooms.setdefault(room, set()).add(ws)

    def leave_room(self, room: str, ws: WebSocket) -> None:
        members = self.rooms.get(room)
        if not members:
            return
        members.discard(ws)
        if not members:
            self.rooms.pop(room, None)

    async def broadcast(self, room: str, payload: dict) -> None:
        for member in list(self.rooms.get(room, set())):
            try:
                await member.send_json(payload)
            except Exception:  # noqa: BLE001 - 单个客户端坏掉不能影响其他人
                self.leave_room(room, member)


manager = ConnectionManager()


def _build_tool_calls(buffer: dict[int, dict]) -> list[dict]:
    calls = []
    for index in sorted(buffer):
        buf = buffer[index]
        calls.append(
            {
                "id": buf["id"],
                "type": "function",
                "function": {"name": buf["name"], "arguments": buf["arguments"]},
            }
        )
    return calls


@app.websocket("/ws/chat/{mode}")
async def chat_ws(ws: WebSocket, mode: str = "assistant") -> None:
    if mode not in {"assistant", "roleplay"}:
        # 先接受再拒绝，客户端能读到可读错误
        await ws.accept()
        await ws.send_json({"type": "error", "message": "mode 只能是 assistant 或 roleplay"})
        await ws.close(code=1008, reason="invalid mode")
        return

    await manager.connect_chat(ws)
    messages: list[dict] = []

    try:
        while True:
            data = await ws.receive_json()
            question = (data or {}).get("question", "").strip()
            if not question:
                await ws.send_json({"type": "error", "message": "question 不能为空"})
                continue

            messages.append({"role": "user", "content": question})
            finish_answer: str | None = None

            for _step in range(MAX_STEPS):
                stream = await client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}, *messages],
                    tools=TOOLS,
                    tool_choice="auto",
                    stream=True,
                )

                finish_reason = None
                full_text = ""
                full_thinking = ""
                tool_call_buffer: dict[int, dict] = {}

                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    delta = choice.delta
                    finish_reason = choice.finish_reason or finish_reason

                    if delta.content:
                        full_text += delta.content
                        await ws.send_json({"type": "content", "text": delta.content})

                    if getattr(delta, "reasoning_content", None):
                        full_thinking += delta.reasoning_content
                        await ws.send_json(
                            {"type": "thinking", "text": delta.reasoning_content}
                        )

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

                if finish_reason == "stop":
                    messages.append({"role": "assistant", "content": full_text})
                    finish_answer = full_text
                    break

                if finish_reason == "tool_calls":
                    calls = _build_tool_calls(tool_call_buffer)
                    for call in calls:
                        await ws.send_json({"type": "tool_call", "call": call})

                    # 先算后写：所有工具执行完，再写消息
                    results: list[tuple[str, dict]] = []
                    for call in calls:
                        name = call["function"]["name"]
                        raw_args = call["function"]["arguments"]
                        try:
                            args = json.loads(raw_args or "{}")
                            func = TOOLBAG.get(name)
                            if func is None:
                                result = {"error": f"工具不存在: {name}"}
                            else:
                                # 同步工具丢线程池，避免阻塞事件循环
                                result = await asyncio.to_thread(func, **args)
                        except Exception as exc:  # noqa: BLE001
                            result = {"error": f"工具执行失败: {exc}"}

                        await ws.send_json(
                            {"type": "tool_result", "id": call["id"], "result": result}
                        )
                        results.append((call["id"], result))

                    messages.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": calls,
                            "reasoning_content": full_thinking,
                        }
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

                break

            # 保险丝烧掉或异常结束：补一次不带 tools 的调用，强制收尾
            if finish_answer is None:
                final = await client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}, *messages],
                    stream=True,
                )
                full_text = ""
                async for chunk in final:
                    if not chunk.choices:
                        continue
                    content = chunk.choices[0].delta.content
                    if content:
                        full_text += content
                        await ws.send_json({"type": "content", "text": content})
                messages.append({"role": "assistant", "content": full_text})

            await ws.send_json({"type": "done", "reason": "stop"})

    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        try:
            await ws.send_json({"type": "error", "message": f"服务异常: {exc}"})
        except Exception:  # noqa: BLE001 - 客户端可能已经断开
            pass
    finally:
        manager.disconnect_chat(ws)


@app.websocket("/ws/room/{room_id}")
async def room_ws(ws: WebSocket, room_id: str) -> None:
    """房间广播：同一房间内任一客户端发言，广播给所有成员。"""
    await manager.join_room(room_id, ws)
    await ws.send_json({"type": "joined", "room": room_id})
    try:
        while True:
            text = await ws.receive_text()
            await manager.broadcast(
                room_id,
                {"type": "message", "room": room_id, "text": text, "ts": time.time()},
            )
    except WebSocketDisconnect:
        pass
    finally:
        manager.leave_room(room_id, ws)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "chat_connections": len(manager.chat_connections),
        "rooms": {room: len(members) for room, members in manager.rooms.items()},
    }
