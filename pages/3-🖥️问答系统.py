import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.llms import OpenAI
import openai
from llama_index.memory import ChatMemoryBuffer
from llama_index import ServiceContext, VectorStoreIndex
from llama_index.storage.storage_context import StorageContext
from llama_index import KnowledgeGraphIndex, SimpleDirectoryReader
from llama_index import StorageContext
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.llms.azure_openai import AzureOpenAI
import os
from llama_index.embeddings.openai import OpenAIEmbedding
import time
# 导入必要的库
from llama_index.llms.openai import OpenAI
from llama_index.evaluation import AnswerRelevancyEvaluator, ContextRelevancyEvaluator



st.set_page_config(page_title="华中农业大学问答系统",
                   page_icon="🦙",
                   layout="centered",
                   initial_sidebar_state="auto",
                   menu_items=None)

openai.api_key = st.secrets['openai_key']
st.title("🦙 华中农业大学问答系统")
st.info("查询基于知识图谱，由学校官网整理制作而成", icon="📃")

if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about HZAU!"}
    ]

@st.cache_resource(show_spinner=False)
def load_data_kg():
    with st.spinner(text="Loading and indexing the docs. This should take 1-2 minutes."):
        # load documents
        #llm = OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert on the LlamaIndex docs and your job is to answer user's questions. Assume that all questions are related to the LlamaIndex docs. Keep your answers being based on facts – do not hallucinate features.Your answer is output in Chinese")

        llm = AzureOpenAI(
            model="gpt-35-turbo-16k",
            deployment_name="gpt-35-turbo-16k",
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version="2024-02-15-preview",
            system_prompt="所有你输出的数据都必须是中文，处理数据时也用中文表示"
        )

        embedding_llm = OpenAIEmbedding(
            model="text-embedding-ada-002",
            deployment_name="text-embedding-ada-002",
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version="2024-02-15-preview",
        )

        username = "neo4j"
        password = "689100"
        url = "bolt://localhost:7687"
        database = "Neo4j"

        graph_store = Neo4jGraphStore(
            username=username,
            password=password,
            url=url,
            database=database,
        )
        storage_context = StorageContext.from_defaults(graph_store=graph_store)

        from llama_index.query_engine import KnowledgeGraphQueryEngine
        from llama_index.query_engine.knowledge_graph_query_engine import DEFAULT_NEO4J_NL2CYPHER_PROMPT
        query_engine = KnowledgeGraphQueryEngine(
            storage_context=storage_context,
            llm=llm,
            embedding_llm=embedding_llm,
            verbose=True,
            graph_query_synthesis_prompt=DEFAULT_NEO4J_NL2CYPHER_PROMPT
        )

    return query_engine


@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the docs. This should take 1-2 minutes."):
        # load documents
        #llm = OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert on the LlamaIndex docs and your job is to answer user's questions. Assume that all questions are related to the LlamaIndex docs. Keep your answers being based on facts – do not hallucinate features.Your answer is output in Chinese")
        llm = OpenAI(model="gpt-3.5-turbo", system_prompt="所有输出的数据都必须是中文，处理数据时也用中文表示，一切参数也使用中文进行表示;Keep your answers being based on facts – do not hallucinate features.Extrapolate as many relationships as you can from the prompt and generate tuples like (source, relation, target). Make sure there are always source, relation and target in the tuple.Example:prompt: John knows React, Golang, and Python. John is good at Software Engineering and Leadership.tuple: (John, knows, React); (John, knows, Golang); (John, knows, Python); (John, good_at, Software_Engineering); (John, good_at, Leadership)")
        embedding_llm = OpenAIEmbedding(
            model="text-embedding-3-large",
            deployment_name="text-embedding-3-large",
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version="2024-02-15-preview",
        )

        #embed_model="local:BAAI/bge-small-en-v1.5"

        service_context = ServiceContext.from_defaults(
            llm=llm,
            #embed_model=embed_model,
            embed_model=embedding_llm
        )
        documents = SimpleDirectoryReader(input_files=["./hzau.txt"]).load_data()

        username = "neo4j"
        password = "689100"
        url = "bolt://localhost:7687"
        database = "Neo4j"

        graph_store = Neo4jGraphStore(
            username=username,
            password=password,
            url=url,
            database=database,
        )
        storage_context = StorageContext.from_defaults(graph_store=graph_store)


        QueryEngine = KnowledgeGraphIndex.from_documents(
            documents,
            service_context=service_context,
            storage_context=storage_context,
            max_triplets_per_chunk=10,
        )

        return QueryEngine

index = load_data()
query_engine_kg = load_data_kg()

stream = st.checkbox("仅采用知识图谱进行查询", value=False)  # Add a checkbox for streaming

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        #st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_plus_context", verbose=True,memory=ChatMemoryBuffer.from_defaults(token_limit=10000))
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="openai", verbose=True, system_prompt="所有输出的数据都必须是中文，处理数据时也用中文表示")

if prompt := st.chat_input("Enter a prompt here"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    """Clears the chat history and resets the state."""
    st.session_state.messages = [
         {"role": "assistant", "content": "Ask me a question about HZAU documentation!"}
         ]
    del st.session_state["chat_engine"]
# Create the sidebar
st.sidebar.header("Options")

# Add the "Clear History" button to the sidebar
if st.sidebar.button("Clear History", key="clear"):
    clear_chat_history()


# 初始化评估器
answer_evaluator = AnswerRelevancyEvaluator(service_context = ServiceContext.from_defaults(
            llm=OpenAI(model="gpt-3.5-turbo"),
        ))
context_evaluator = ContextRelevancyEvaluator(service_context = ServiceContext.from_defaults(
            llm=OpenAI(model="gpt-3.5-turbo"),
        ))


# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if stream:
                response = query_engine_kg.query(prompt)
                st.write(response.response)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message)  # Add response to message history
            else:
                #
                start_time=time.time()
                response = st.session_state.chat_engine.chat(prompt)
                st.write(response.response)
                end_time=time.time()
                answer_time=end_time-start_time
                print("所用时间为: "+str(answer_time))
                #
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message) # Add response to message history

                # 在你的问答逻辑后添加评估代码
                answer_score = answer_evaluator.evaluate(prompt, str(response.response))
                # context_score = context_evaluator.evaluate(query, reference_context)

                # 输出评估得分
                print(f"Answer Relevancy Score: {answer_score.score}")
                # print(f"Context Relevancy Score: {context_score}")
