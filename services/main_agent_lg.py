from toolbag import TOOLBAG
from typing import TypedDict, Annotated
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os,json
load_dotenv()
from services.agent_lg import app as agent_app
server = FastAPI(
    title="AI 伴侣 · LangGraph Agent 服务",
    description="基于 LangGraph 状态图的对话服务：checkpoint 多轮续档（thread_id）、工具调用（时间/天气/历史/知识库）。",
    version="0.1.0",
)

#请求模板
class ChatRequest(BaseModel):
    messages:str
    session_id:str

#响应模板
class ChatResponse(BaseModel):
    reply:str

@server.post("/chat",response_model=ChatResponse)
def chat(req:ChatRequest):
    config = {"configurable":{"thread_id":req.session_id}}
    result = agent_app.invoke(
        {"messages":[{"role":"user","content":req.messages}]},config
    )
    return {"reply":result["messages"][-1].content}
