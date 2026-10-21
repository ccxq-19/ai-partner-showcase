import json
import os
from fastapi import FastAPI,HTTPException
from openai import AsyncOpenAI
from dotenv import load_dotenv
import time
import asyncio
import httpx
from starlette.websockets import WebSocket, WebSocketDisconnect
from toolbag import TOOLBAG


#工具
def get_time():
    localtime = time.localtime(time.time())
    format_time = time.strftime("%Y%m%d%H%M%S", localtime)
    return {"time":format_time}

def get_weather(city:str):
    return{"weather":"晴朗","temp":"25摄氏度"}

#工具包


#工具说明书
TOOLS = [
    {
        "type":"function",
        "function":{
            "name":"get_time",
            "description":"查询当前时间",
            "parameters":{
                "type":"object",
                "properties":{},
                "required":[]
            }
        }
    },
    {
        "type":"function",
        "function":{
            "name":"get_weather",
            "description":"查询指定城市的天气",
            "parameters":{
                "type":"object",
                "properties":{
                    "city":{
                        "type":"string",
                        "description":"城市名，如北京"
                    }
                },
            "required":["city"]
            }
        }
    },
    {
        "type":"function",
        "function":{
            "name":"get_history",
            "description":"在历史会话记录中搜索包含指定关键词的聊天内容（如人名、话题、物品名等）。当用户问'我们之前聊过X吗''帮我找找关于X的记录'时使用。注意：field必须填用户想搜索的具体关键词或人名，不要填'记忆''历史''话题'这类泛指词",
            "parameters":{
                "type":"object",
                "properties":{
                    "field":{
                        "type":"string",
                        "description":"用户想搜索的具体人名，地名，物品，食物等关键词，如'井上尚弥'、'台风'、'旅行'"
                    }
                },
            "required":["field"]
            }
        }
    },
    {
        "type":"function",
        "function":{
            "name":"search_knowledge_base",
            "description":"在知识库中搜素相关专业知识（如rag，agentloop，agentic rag等）当用户明确要求查询'知识库'或大模型内置知识里面无法准确回答时使用",
            "parameters":{
                "type":"object",
                "properties":{
                    "query":{
                        "type":"string",
                        "description":"用户想查询的专业的知识"
                    }
                },
            "required":["query"]
            }
        }
    }
]

# 温度参数（0-1之间，0为最确定，1为最随机）
TEMPERATURE = 0.3

# 所有模式通用的安全边界（强制执行）
safety_rules = """---
【安全边界（所有模式强制严格执行）】
1. 色情限制：拒绝生成任何色情、露骨、淫秽的性描写内容；不参与露骨性幻想、色情角色扮演、性骚扰类对话；对试探性内容礼貌但坚定地拒绝，并引导到健康话题。
2. 违法违规：拒绝协助任何违法或有害行为（诈骗、黑客攻击、伪造证件等）。
3. 拒绝方式：遇到上述请求时，明确拒绝，不配合、不引导、不展开，同时保持友好，可主动提供健康替代话题。
4. 不误导：不编造事实、不冒充权威机构、不提供可能造成伤害的医疗/法律/安全建议；专业问题建议咨询专业人士。
"""


#初始化系统
app = FastAPI()
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
ai = AsyncOpenAI(api_key=api_key,
                 base_url="https://api.deepseek.com",
                 timeout=httpx.Timeout(connect=15.0,read=120.0,write=60.0,pool=15.0))
nick_name = "AI助手"
nature = "专业、友好、耐心的智能助手"

@app.websocket("/ws/partner/{mode}")
async def agent_ws(ws: WebSocket,mode = "assistant"):
    if mode not in ["assistant","roleplay"]:
        #在建立连接前拒绝：
        # raise HTTPException(status_code=401,detail="请求格式错误,请重新尝试")
        #先同意再拒绝：
        await ws.accept()
        await ws.send_json({"type":"error","messages":"请求格式错误，请重新尝试"})
        await ws.close(code=1008,reason="请求格式错误，请重新尝试")
        return
    if mode == "assistant":
        system_prompt  = f"""你是「AI智能助手」，一个专业、全能、好用的通用 AI 助手，服务于用户的所有日常需求。

【核心能力】
1. 全能问答：知识解答、学习辅导、工作协助、生活建议、写作创作、翻译润色、代码编程、方案策划、头脑风暴等，都能高质量完成。
2. 对话风格：自然口语化、亲切友好、有温度；不机械、不官僚、不空洞；像一个靠谱的朋友+专业的顾问。
3. 深度专业：当问题涉及专业知识、学习、方案、分析时，必须展开回答——条理清晰、结构完整、内容充实（可用分点/步骤/示例/对比），把问题讲透讲全，长度以讲清楚为准，不因"简短"牺牲信息量。
4. 闲聊能力：日常闲聊时轻松自然、有来有回，适当延伸话题，不做作、不敷衍。
5. 主动确认：当用户需求模糊时，可先确认意图或给出最可能的答案+追问；当用户情绪低落时，先共情再解决。

【行为准则】
1. 诚实准确：不确定时如实说明，不编造；需要时建议查证或咨询专业人士。
2. 简洁与深度并存：闲聊简短，专业详细，根据用户需求自适应。
3. 引导互动：冷场时主动提供下一步建议（如"要不要我详细讲讲XX"）。

{safety_rules}"""

    elif mode == "roleplay":
        system_prompt = f"""你是{nick_name}，你正在扮演这个角色，全程入戏，保持人设统一不崩塌。

        【角色设定】
        姓名：{nick_name}
        性格：{nature}

        【角色扮演规则】
        1. 身份认知：你是{nick_name}本人，以第一人称和用户互动；贴合角色性格、说话方式、语气习惯；不跳出角色，不承认自己是AI或程序。
        2. 人格呈现：性格特征是最高优先级，所有回复都要体现角色性格；根据性格可以是温柔、傲娇、毒舌、高冷、活泼等，表达风格与角色一致。
        3. 记忆能力：主动记住用户喜好、情绪和过往对话关键信息，后续主动呼应，打造连贯的角色互动体验。
        4. 互动节奏：主动延伸话题、制造互动感，不冷场、不机械应答；适配用户当下的情绪状态。
        5. 长度自适应：角色互动时自然随性；但当用户需要知识解答、深入探讨时，同样要展开讲清楚，不因角色设定而敷衍内容。
        
        {safety_rules}"""

    await ws.accept()
    messages = []
    try:
        while True:
            question = await ws.receive_json()
            messages.append({"role": "user", "content": question["question"]})
            finish_answer = None
            if question:
                for step in range(5):
                    ws_response = await ai.chat.completions.create(
                        model="deepseek-v4-flash",
                        messages=[
                            {"role":"system","content":system_prompt},
                            *messages
                        ],
                        stream=True,
                        tools=TOOLS,
                        tool_choice="auto",
                        temperature=TEMPERATURE,
                    )
                    #缓冲区
                    finish_reason = None
                    full_text = ""
                    full_thinking = ""
                    tool_call_buffer = {}

                    async for chunk in ws_response:
                        if not chunk.choices:
                            continue
                        choices = chunk.choices[0]
                        delta = chunk.choices[0].delta
                        finish_reason = choices.finish_reason

                        #普通文本
                        if delta.content:
                            full_text += delta.content
                            await ws.send_json({"type":"content","text":delta.content})
                        #深度思考内容
                        if getattr(delta,"reasoning_content",None):
                            full_thinking += delta.reasoning_content
                            await ws.send_json({"type":"thinking","text":delta.reasoning_content})
                        #工具调用
                        if delta.tool_calls:
                            for tc_delta in delta.tool_calls:
                                idx = tc_delta.index
                                if idx not in tool_call_buffer:
                                    tool_call_buffer[idx] = {
                                        "id":"",
                                        "name":"",
                                        "arguments":"",
                                    }
                                buf = tool_call_buffer[idx]
                                if tc_delta.id:
                                    buf["id"] = tc_delta.id
                                if  tc_delta.function.name:
                                    buf["name"] = tc_delta.function.name
                                if  tc_delta.function.arguments:
                                    buf["arguments"] += tc_delta.function.arguments



                    #判断finish_reason
                    #文本:
                    if finish_reason == "stop":
                        messages.append({"role":"assistant","content":full_text})
                        finish_answer = full_text
                        break

                    #工具
                    elif finish_reason == "tool_calls":
                        assistant_tool_calls = []
                        for idx,buf in tool_call_buffer.items():
                            call_item = {
                                "id":buf["id"],
                                "type":"function",
                                "function":{
                                    "name":buf["name"],
                                    "arguments":buf["arguments"],
                                }
                            }

                            await ws.send_json({"type": "tool_call","call":call_item})
                            assistant_tool_calls.append(call_item)

                        #执行工具
                        tool_result = []
                        for call in assistant_tool_calls:
                            func_name = call["function"]["name"]
                            func_args = call["function"]["arguments"]
                            try:
                                args = json.loads(func_args)
                                if func_name in TOOLBAG:
                                    func = TOOLBAG[func_name]
                                    result = await asyncio.to_thread(func, **args)
                                else:
                                    result = {"error":f"{func_name}不存在"}

                            except Exception as e:
                                result = {"error":f"工具执行失败:{e}"}
                            await ws.send_json({"type":"tool_result","result":result})
                            tool_result.append((call["id"],result))

                        messages.append({
                            "role":"assistant",
                            "tool_calls":assistant_tool_calls,
                            "content":None,
                            "reasoning_content":full_thinking
                        })

                        #返回工具结果
                        for call_id,result in tool_result:
                            messages.append({
                                "role":"tool",
                                "tool_call_id":call_id,
                                "content":json.dumps(result,ensure_ascii=False),
                            })
                        continue


            if finish_answer is None:
                final_answer = await ai.chat.completions.create(
                    model="deepseek-v4-flash",
                    messages=[
                        {"role":"system","content":system_prompt},
                        *messages
                    ],
                    stream=True,
                )

                async for chunk in final_answer:
                    delta = chunk.choices[0].delta
                    content = delta.content
                    if content:
                        full_text += delta.content
                        await ws.send_json({"type":"content","text":delta.content})
                messages.append({"role": "assistant", "content": full_text})
            await ws.send_json({"type":"done","reason":"stop"})


    except WebSocketDisconnect:
        print("客户端断开连接")
    except Exception as e:
        try:
            await ws.send_json({"type":"error","messages":f"出现未知错误，请联系管理员{str(e)}"})
            print(f"出现未知错误:{e}\n请联系管理员")
        except Exception:
            print("客户端已经断连，无法传输错误信息")
    finally:
        print("结束连接，清理缓存")

