"""
长期记忆压缩（展示版）

当消息数超过阈值：
  1. 取最老的 N 条消息
  2. 调 LLM，把“已有记忆 + 老消息”压缩成新的长期记忆
  3. 只保留最近 KEEP_RECENT 条消息，旧的进记忆

工程要点：
  - 总结失败时宁可多带消息，也不丢历史
  - 截断后要保证最近消息以 user / 有内容的 assistant 开头
  - 记忆是“模型上下文”的一部分，不影响界面显示完整历史
"""
from __future__ import annotations

SUMMARY_THRESHOLD = 20
KEEP_RECENT = 10

SUMMARIZE_SYSTEM_PROMPT = """你是一个对话记忆整理助手。
把一段对话浓缩成精炼的“长期记忆”，供未来对话延续人设、回应用户喜好与过往。

输出要求：
1. 只输出总结内容本身，不要解释、前缀、标题
2. 用简洁的中文条目式表达
3. 保留：用户的喜好、习惯、情绪状态、重要经历、与助手的关系和互动模式
4. 结合已有记忆整合：保留重要信息，合并重复，删除过时
5. 总长度控制在 300 字以内

示例 1：用户：我喜欢跑步。助手：用户喜欢跑步，有运动习惯。
示例 2：用户：我最近在学习编程。助手：用户最近在学习编程，对技术感兴趣。
示例 3：用户：我最近有点焦虑。助手：用户近期情绪焦虑，可能面临现实压力。
示例 4：用户：我的项目上线了，心情很好。助手：用户因项目上线而心情很好。
"""


def summarize_memory(
    client,
    model: str,
    old_memory: str,
    nick_name: str,
    nature: str,
    old_messages: list[dict],
) -> str:
    """把旧记忆 + 老消息压缩成新记忆。"""
    chat_log = "\n".join(
        f"{'用户' if m['role'] == 'user' else nick_name}: {m['content']}"
        for m in old_messages
        if m["role"] != "tool" and m.get("content")
    )

    user_content = f"""当前助手：{nick_name}（性格：{nature}）

已有长期记忆：
{old_memory if old_memory else "（暂无）"}

需要整合进记忆的对话内容：
{chat_log}
"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": SUMMARIZE_SYSTEM_PROMPT.format(nick_name=nick_name),
            },
            {"role": "user", "content": user_content},
        ],
        max_tokens=2000,
    )
    return response.choices[0].message.content.strip()


def maybe_compress(
    client,
    model: str,
    nick_name: str,
    nature: str,
    memory: str,
    messages: list[dict],
) -> tuple[list[dict], str]:
    """超阈值就压缩；失败时原样返回，不丢历史。

    返回 (new_messages, new_memory)。
    """
    if len(messages) <= SUMMARY_THRESHOLD:
        return messages, memory

    old_messages = messages[:-KEEP_RECENT]
    recent_messages = messages[-KEEP_RECENT:].copy()

    # 截断后可能以 tool / 空 assistant 开头，往前挪到合法起点
    while recent_messages and not (
        recent_messages[0]["role"] == "user"
        or (
            recent_messages[0]["role"] == "assistant"
            and recent_messages[0].get("content")
        )
    ):
        recent_messages.pop(0)

    if not recent_messages:
        return messages, memory

    try:
        new_memory = summarize_memory(
            client, model, memory, nick_name, nature, old_messages
        )
        return recent_messages, new_memory
    except Exception:  # noqa: BLE001 - 总结失败不能丢消息
        return messages, memory
