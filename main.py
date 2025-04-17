
from dotenv import load_dotenv
from langchain_experimental.utilities import PythonREPL
from langchain_deepseek import ChatDeepSeek
from langchain.agents import AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain_community.utilities.openweathermap import OpenWeatherMapAPIWrapper
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.agents import create_react_agent
from langchain import hub
from langchain.agents import AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain.agents import tool
from datetime import date
from langchain_core.tools import Tool
from langchain.memory import ConversationBufferMemory

from sqlalchemy import create_engine
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core import SQLDatabase
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from langchain_experimental.plan_and_execute.planners.chat_planner import load_chat_planner
from langchain_experimental.plan_and_execute.executors.agent_executor import load_agent_executor
from langchain_experimental.plan_and_execute.agent_executor import PlanAndExecute

import os
# path = os.getcwd()
# load_dotenv(f'{path}/api.env')

api_key = os.getenv("DEEPSEEK_API_KEY")
chatbot = ChatDeepSeek(
    api_key=api_key
    model="deepseek-reasoner",
    temperature=0,
    max_tokens=1024,
    timeout=60,
    max_retries=2)


USER=os.getenv("DB_USER")
PASSWORD=os.getenv("DB_PASSWORD")
HOST=os.getenv("DB_HOST")
NAME=os.getenv("DB_NAME")
engine = create_engine(f"mysql+pymysql://{USER}:{PASSWORD}@l{HOST}:3306/{NAME}")


# def db_engine(query):
#     return db_query_engine.query(query)

# from langchain.agents import Tool
# db_tool = Tool(
#     name="DatabaseQueryEngine",
#     func=db_engine,  # LlamaIndex问答引擎
#     description="用于查询mysql数据库得到与数据库问题相关的结果"
# )

# Settings.llm = llm
# Settings.embed_model=HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
# Settings.note_parser =SentenceSplitter(chunk_size=512, chunk_overlap=20)
# Settings.num_output = 512
# Settings.context_window = 3900
# Settings.transformations = [SentenceSplitter(chunk_size=1024,chunk_overlap=20))]

# doc_index = VectorStoreIndex.from_documents(documents,
#                                             embed_model=HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2"),
#                                             transformations=[SentenceSplitter(chunk_size=512, chunk_overlap=20)])
# rag_engine=doc_index.as_query_engine()
# def rag_query(query):
#     reponse = rag_engine.query(query)
#     return response
    
# from langchain.agents import Tool
# rag_tool = Tool(
#     name="DocumentQATool",
#     func=rag_query,  # LlamaIndex问答引擎
#     description="用于文档内容的问答"
# )


weather = OpenWeatherMapAPIWrapper()

@tool
def time(text:str):
    """
    returns today's date,used for any question related to today's date,
    the inputs should always be empty string, and this function output should always be today's date.
    """
    return str(date.today())

python_repl = PythonREPL()
repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run,
)

tools = load_tools(["wikipedia", "arxiv", "ddg-search","openweathermap-api"], llm=chatbot)

memory=ConversationBufferMemory(memory_key='chat_history')

planner = load_chat_planner(llm=chatbot)

# executor = load_agent_executor(llm=chatbot, tools=tools+[time]+[repl_tool]+[db_tool, rag_tool], verbose=True)
executor = load_agent_executor(llm=chatbot, tools=tools+[time]+[repl_tool], verbose=True)

from langchain_experimental.plan_and_execute.agent_executor import PlanAndExecute

multi_agent = PlanAndExecute(
    planner=planner,
    executor=executor,
    verbose=True,
    memory=memory
)

response = multi_agent.run('帮我查找一下最近购买次数最多的客户最近一次购买商品当天的所有订单中销售最多的商品是哪一件')


def chatbot_response(user_input):
    response = multi_agent.run(user_input)
    return response


import gradio as gr
chatbot_ui = gr.Interface(fn=chatbot_response,
                          inputs=gr.Textbox(label='you can ask me anything including the weather and the date...'),
                          outputs=gr.Textbox(label='output'))
if __name__=="__main__":
    chatbot_ui.launch()