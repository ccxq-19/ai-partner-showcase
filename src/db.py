"""
SQLite 会话存储（展示版）

生产版 db.py 229 行，额外包含：
  - auth_tokens 表：记住登录令牌只存 SHA256 哈希
  - 旧库迁移：补 role_mode / full_history 列
  - 令牌创建 / 校验 / 撤销

展示版保留核心表结构与 CRUD：
  sessions 一张表；messages / full_history 以 JSON 数组存储。
  每次操作独立连接，避免多线程共享连接的问题。
"""
from __future__ import annotations

import datetime
import json
import os
import sqlite3

DB_PATH = os.getenv(
    "SESSIONS_DB_PATH",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sessions.db"),
)


def get_connection() -> sqlite3.Connection:
    """每次操作独立连接；row_factory 让查询结果可以按字段名访问。"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """初始化表结构（首次运行自动创建）。"""
    conn = get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id   TEXT PRIMARY KEY,          -- 会话 ID（时间戳）
            nick_name    TEXT DEFAULT '',           -- 伴侣昵称
            nature       TEXT DEFAULT '',           -- 性格设定
            memory       TEXT DEFAULT '',           -- 长期记忆总结
            role_mode    TEXT DEFAULT 'assistant',  -- assistant / roleplay
            messages     TEXT DEFAULT '[]',         -- 最近消息（JSON 数组）
            full_history TEXT DEFAULT '[]',         -- 完整历史（JSON 数组）
            created_at   TEXT DEFAULT '',
            updated_at   TEXT DEFAULT ''
        )
        """
    )
    conn.commit()
    conn.close()


def _messages_to_json(messages: list[dict]) -> str:
    return json.dumps(messages, ensure_ascii=False)


def _json_to_messages(raw: str | None) -> list[dict]:
    if not raw:
        return []
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return []


def save_session(
    session_id: str,
    nick_name: str = "",
    nature: str = "",
    memory: str = "",
    role_mode: str = "assistant",
    messages: list[dict] | None = None,
    full_history: list[dict] | None = None,
) -> None:
    """保存会话：存在则更新，不存在则插入（UPSERT）。"""
    messages = messages or []
    full_history = full_history or []
    now = _now_iso()
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO sessions
            (session_id, nick_name, nature, memory, role_mode,
             messages, full_history, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            nick_name    = excluded.nick_name,
            nature       = excluded.nature,
            memory       = excluded.memory,
            role_mode    = excluded.role_mode,
            messages     = excluded.messages,
            full_history = excluded.full_history,
            updated_at   = excluded.updated_at
        """,
        (
            session_id,
            nick_name,
            nature,
            memory,
            role_mode,
            _messages_to_json(messages),
            _messages_to_json(full_history),
            now,
            now,
        ),
    )
    conn.commit()
    conn.close()


def load_session(session_id: str) -> dict | None:
    """加载单个会话；不存在返回 None。"""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return {
        "session_id": row["session_id"],
        "nick_name": row["nick_name"],
        "nature": row["nature"],
        "memory": row["memory"],
        "role_mode": row["role_mode"],
        "messages": _json_to_messages(row["messages"]),
        "full_history": _json_to_messages(row["full_history"]),
    }


def list_sessions() -> list[str]:
    """列出所有会话 ID，新的在前。"""
    conn = get_connection()
    rows = conn.execute(
        "SELECT session_id FROM sessions ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [row["session_id"] for row in rows]


def delete_session(session_id: str) -> None:
    conn = get_connection()
    conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()


def session_exists(session_id: str) -> bool:
    conn = get_connection()
    row = conn.execute(
        "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    conn.close()
    return row is not None


def count_sessions() -> int:
    conn = get_connection()
    row = conn.execute("SELECT COUNT(*) AS cnt FROM sessions").fetchone()
    conn.close()
    return row["cnt"] if row else 0


def _now_iso() -> str:
    return datetime.datetime.now().isoformat()


if __name__ == "__main__":
    # 最小演示：建 → 存 → 读 → 删
    init_db()
    sid = "demo-session"
    save_session(
        sid,
        nick_name="小助手",
        nature="专业、友好",
        messages=[{"role": "user", "content": "你好"}],
        full_history=[{"role": "user", "content": "你好"}],
    )
    print("saved:", load_session(sid))
    print("list:", list_sessions())
    delete_session(sid)
    print("after delete:", session_exists(sid))
