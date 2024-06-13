import streamlit as st

st.set_page_config(
    page_title="主页",
    page_icon="🏠",
)

st.write("# 欢迎使用本系统! 👋")


st.markdown(
    """
    **本系统是基于llamaindex+streamlit实现的知识图谱问答系统**

    ### 系统介绍
    - **知识图谱展示**：将neo4j中的数据可视化至前端页面
    - **知识图谱生成**：利用rebel模型+大模型进行构建
    - **问答系统**：llamaindex+openai构建知识图谱索引，进行问答
    
    
 
    ### 👈从侧边栏选择一个项目，看看能做什么吧👈
"""
)
