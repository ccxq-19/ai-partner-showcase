"""
RAG 全链路（展示版）

    文档切分 → bge-m3 Embedding → 手写向量库 top-k 检索 → bge-reranker 精排 → 注入提示词

设计要点：
  - 切分用“重叠窗口”，避免语义在 chunk 边界被切断
  - 向量库存 SQLite（chunks + JSON 向量），不依赖 Chroma/Qdrant，便于理解底层
  - 先向量粗筛 top-k，再用交叉编码器 rerank 精排；rerank 失败时降级为粗筛结果
  - 提示词只允许基于参考资料回答；没有资料就明确说“没找到”
"""
from __future__ import annotations

import json
import os
import sqlite3
from math import sqrt
from pathlib import Path

import requests
from openai import OpenAI

# Embedding / Rerank 走硅基流动；对话走 DeepSeek
embedding_client = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1",
)
chat_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

EMBEDDING_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
CHAT_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

DB_PATH = Path(
    os.getenv(
        "VECTOR_DB_PATH",
        Path(__file__).resolve().parent.parent / "data" / "vector_store.db",
    )
)


def chunk_text(text: str, chunk_size: int = 300, chunk_overlap: int = 40) -> list[str]:
    """重叠窗口切分：步长 = chunk_size - chunk_overlap。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size")

    step = chunk_size - chunk_overlap
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        if start + chunk_size >= len(text):
            break
        start += step
    return chunks


def embed(texts: list[str]) -> list[list[float]]:
    """文本列表 → 向量列表。"""
    resp = embedding_client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in resp.data]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    len_a = sqrt(sum(x * x for x in a))
    len_b = sqrt(sum(y * y for y in b))
    if len_a == 0 or len_b == 0:
        return 0.0
    return dot / (len_a * len_b)


def rerank(query: str, documents: list[dict], top_k: int = 3) -> list[str]:
    """交叉编码器精排；失败时降级为粗筛顺序。"""
    if not documents:
        return []

    texts = [doc["text"] for doc in documents]
    try:
        resp = requests.post(
            "https://api.siliconflow.cn/v1/rerank",
            headers={"Authorization": f"Bearer {os.getenv('SILICONFLOW_API_KEY')}"},
            json={"model": RERANK_MODEL, "query": query, "documents": texts},
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
        if "error" in payload:
            raise RuntimeError(payload["error"])
        return [texts[item["index"]] for item in payload["results"][:top_k]]
    except Exception:  # noqa: BLE001 - 外部服务不可用时必须能降级
        return texts[:top_k]


class VectorStore:
    """最小向量库：SQLite 存文本 + JSON 向量，内存里做余弦相似度。"""

    def __init__(self, db_path: Path | str = DB_PATH):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS chunks ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "text TEXT NOT NULL, "
            "vector TEXT NOT NULL)"
        )
        self.conn.commit()
        rows = self.conn.execute("SELECT text, vector FROM chunks").fetchall()
        self.items = [{"text": text, "vector": json.loads(vector)} for text, vector in rows]

    def add_text(self, text: str) -> None:
        chunks = chunk_text(text)
        if not chunks:
            return
        vectors = embed(chunks)
        self.conn.executemany(
            "INSERT INTO chunks (text, vector) VALUES (?, ?)",
            [(chunk, json.dumps(vector)) for chunk, vector in zip(chunks, vectors)],
        )
        self.conn.commit()
        self.items.extend(
            {"text": chunk, "vector": vector} for chunk, vector in zip(chunks, vectors)
        )

    def search(self, query: str, k: int = 20) -> list[dict]:
        if not self.items:
            return []
        query_vector = embed([query])[0]
        scored = [
            {"text": item["text"], "score": cosine_similarity(query_vector, item["vector"])}
            for item in self.items
        ]
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:k]

    def clear(self) -> None:
        self.conn.execute("DELETE FROM chunks")
        self.conn.commit()
        self.items = []


def answer_with_rag(query: str, top_k: int = 20, rerank_k: int = 3) -> str:
    """检索 → 精排 → 只根据参考资料回答。"""
    store = VectorStore()
    hits = store.search(query, k=top_k)
    if not hits:
        return "资料里没找到相关内容——请先往知识库添加文档。"

    documents = rerank(query, hits, top_k=rerank_k)
    references = "\n\n".join(
        f"[资料 {index + 1}]\n{text}" for index, text in enumerate(documents)
    )

    system_prompt = f"""你是一个知识库问答助手。
请只根据下面的参考资料回答问题；如果资料里没有答案，就如实说“资料里没找到”。
不要补充参考资料之外的事实。

【参考资料】
{references}
"""
    response = chat_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        temperature=0.1,
        max_tokens=2000,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    # 最小演示：写入一句，检索一句
    store = VectorStore()
    store.clear()
    store.add_text("温度是控制大模型输出随机性的参数。值越高越发散，值越低越稳定。")
    print(answer_with_rag("温度是什么？"))
