from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()
from services.agent_lg import app as agent_app
import json

server = FastAPI()

class ChatRequest(BaseModel):
    message:str
    session_id:str

def generator(req):
    config  = {"configurable":{"thread_id":req.session_id}}
    for mode,chunk in  agent_app.stream(
            {"messages":[{"role":"user","content":req.message}]},
        config,
        stream_mode = ["custom","updates"]
    ):
        if mode == "custom":
            yield f"data:{json.dumps(chunk,ensure_ascii=False)}\n\n"

        else:
            mode_name = list(chunk.keys())[0]
            yield f"data:{json.dumps({'type':'node','node':mode_name},ensure_ascii=False)}\n\n"
    yield f"data:[DONE]\n\n"

@server.post("/chat")
def chat_stream(request: ChatRequest):
    return StreamingResponse(generator(request),media_type="text/event-stream")
