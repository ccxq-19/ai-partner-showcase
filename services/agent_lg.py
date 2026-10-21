from toolbag import TOOLBAG
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import AIMessage
from langchain_core.messages.utils import convert_to_openai_messages
from langgraph.config import get_stream_writer
from openai import OpenAI
import time
import os,json

api_key = os.getenv("DEEPSEEK_API_KEY")
client = OpenAI(api_key = api_key,base_url="https://api.deepseek.com")
safety_rules = """---
【安全边界（所有模式强制严格执行）】
1. 色情限制：拒绝生成任何色情、露骨、淫秽的性描写内容；不参与露骨性幻想、色情角色扮演、性骚扰类对话；对试探性内容礼貌但坚定地拒绝，并引导到健康话题。
2. 违法违规：拒绝协助任何违法或有害行为（诈骗、黑客攻击、伪造证件等）。
3. 拒绝方式：遇到上述请求时，明确拒绝，不配合、不引导、不展开，同时保持友好，可主动提供健康替代话题。
4. 不误导：不编造事实、不冒充权威机构、不提供可能造成伤害的医疗/法律/安全建议；专业问题建议咨询专业人士。
"""

system_prompt = f"""你是「AI智能助手」，一个专业、全能、好用的通用 AI 助手，服务于用户的所有日常需求。

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






class State(TypedDict):
    messages: Annotated[list,add_messages]



#node
def call_model(state):
    msgs = convert_to_openai_messages(state["messages"])
    writer = get_stream_writer()

    response = client.chat.completions.create(
        model = "deepseek-v4-flash",
        messages = [
            {"role":"system","content":system_prompt},
            *msgs,
        ],
        tools = TOOLS,
        tool_choice = "auto",
        stream = True,
        extra_body={"thinking": {"type": "disabled"}}
    )
    #缓冲区
    full_text = ""
    full_thinking = ""
    tool_call_buffer = {}

    for chunk in response:

        msg = chunk.choices[0].delta
        tool_calls = chunk.choices[0].delta.tool_calls

        if msg.content:
            full_text+=msg.content
            writer({"type": "token", "content": msg.content})
        if tool_calls:
            # tool_calls = [tc.model_dump() for tc in tool_calls]

            #拼接成字典
            # tool_calls = [{
            #     "id":tc.id,
            #     "type":"function",
            #     "function":{
            #         "name":tc.function.name,
            #         "arguments":tc.function.arguments,
            #     }
            # }for tc in tool_calls]
            for tc in tool_calls:
                index = tc.index
                if index not in tool_call_buffer:
                    tool_call_buffer[index] = {
                        "id":"",
                        "name":"",
                        "args":""
                    }
                buf = tool_call_buffer[index]
                if tc.id:
                    buf["id"] = tc.id
                if tc.function.name:
                    buf["name"] = tc.function.name
                if tc.function.arguments:
                    buf["args"] += tc.function.arguments


    # parse_tool_calls = [{
    #     "id":tc.id,
    #     "name":tc.function.name,
    #     "args":json.loads(tc.function.arguments),
    # }for tc in tool_calls]



            # return {"messages":[ai_message]}

    # 组装openai dict格式
    # tool_calls_final = []
    # for buf in tool_call_buffer.values():
    #     try:
    #         buf["function"]["arguments"] = json.loads(buf["function"]["arguments"])
    #     except json.JSONDecodeError:
    #         buf["function"]["arguments"] = ""
    #     tool_calls_final.append(buf)
    #
    # ai_msg_dict = {
    #     "role": "assistant",
    #     "content": full_text,
    #     "reasoning_content": full_thinking,  # 关键：思考内容保存在字典顶级key
    # }
    # if tool_calls_final:
    #     ai_msg_dict["tool_calls"] = tool_calls_final
    #
    # return {"messages": [ai_msg_dict]}

    # 拼接成AIMessage对象
    assistant_tool_calls = []
    for index, buf in tool_call_buffer.items():
        call_item = {
            "id": buf["id"],
            "name": buf["name"],
            "args": json.loads(buf["args"]),
        }
        assistant_tool_calls.append(call_item)
    ai_message = AIMessage(
        content=full_text,
        tool_calls=assistant_tool_calls,
    )
    return {"messages": [ai_message]}



        # return {"messages":[{
        #     "role":"assistant",
        #     "content":msg.content,
        #     "tool_calls":tool_calls,
        # }]}


def should_continue(state):
    last_messages = state["messages"][-1]

    if getattr(last_messages,"tool_calls",None):
        return "continue"
    return "end"

def call_tools(state):
    last_messages = state["messages"][-1]
    tool_results = []
    for call in last_messages.tool_calls:
        func_name = call["name"]
        func_args = call["args"]

        if func_name in TOOLBAG:
            func = TOOLBAG[func_name]
            result = func(**func_args)
        else:
            result = f"调用工具失败，{func_name}不存在"
        tool_results.append({
            "role":"tool",
            "tool_call_id":call["id"],
            "content":json.dumps(result,ensure_ascii=False),
        })
    return {
        "messages":tool_results
    }


#图编排
graph = StateGraph(State)
graph.add_node("call_model",call_model)
graph.add_node("call_tools",call_tools)
graph.add_edge(START,"call_model")
graph.add_conditional_edges("call_model",should_continue, {
    "continue":"call_tools",
    "end":END
})
# 图内的上下文本
graph.add_edge("call_tools","call_model")
app = graph.compile(checkpointer=InMemorySaver())
config1 = {"configurable":{"thread_id":"session-001"}}
config2 =  {"configurable":{"thread_id":"session-002"}}


if __name__ == "__main__":

    # for chunk in app.stream(
    #     {"messages":[{"role":"user","content":"你是个笨蛋，坏蛋，混蛋"}]},
    #     config1,
    #     stream_mode = "updates",
    # ):
    #     print(chunk)
    #
    # for chunk in app.stream(
    #     {"messages":[{"role":"user","content":"我刚刚问了你什么？"}]},
    #     config1,
    #     stream_mode = "updates",
    # ):
    #     print(chunk)

    for chunk in app.stream(
        {"messages":[{"role":"user","content":"请搜索知识库回答关于rag的内容"}]},
        config2,
        stream_mode = "updates",
    ):
        print(chunk)

    # (app.invoke({"messages":[{
    #     "role":"user",
    #     "content":"现在几点了"
    # }]},config1))
    #
    # (app.invoke({"messages":[{
    #     "role":"user",
    #     "content":"我刚刚问了你什么问题?"
    # }]},config1))
    #
    # (app.invoke({"messages":[{
    #     "role":"user",
    #     "content":"我刚刚问了你什么问题"
    # }]},config2))
    #
    # (app.invoke({"messages": [{
    #     "role": "user",
    #     "content": "帮我查一下知识库什么是RAG"
    # }]},config2))

