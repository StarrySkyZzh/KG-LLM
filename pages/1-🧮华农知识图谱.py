import streamlit as st
import streamlit.components.v1 as components
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import utils
import utils_t
from kb import KB
import pandas as pd
from pyvis.network import Network


st.title("🌳🌳:green[华中农业大学知识图谱]🌳🌳")
st.info("华中农业大学官网(www.hzau.edu.cn)", icon="📃")

# import streamlit as st
# from PIL import Image
#
# image = Image.open('imgs/hzau.jpeg')
#
# st.image(image, use_column_width=True)
st.sidebar.write("知识图谱")
st.sidebar.write("是结构化的语义知识库，用于以符号形式描述物理世界中的概念及其相互关系。其基本组成单位是“实体-关系-实体”三元组，以及实体及其相关属性-值对，实体间通过关系相互联结，构成网状的知识结构。")
st.sidebar.write("三元组是知识图谱的一种通用表示方式，即 G =（E, R, S)，其中 E 是知识库中的实体，R 是知识库中的关系，S 代表知识库中的三元组。")
@st.cache_resource(show_spinner=False)
def query_neo4j():
    with st.spinner('Connecting to Neo4j...'):
        net = Network(directed=True, width="700px", height="700px", cdn_resources="in_line")
        conn = utils.Neo4jConnection(uri=st.secrets['uri'], user=st.secrets['username'], pwd=st.secrets['password'])
        query="MATCH (n)-[r]->(b) RETURN (n),(b),(r)"
        result=conn.query(query)
        # print(type(result))
        color_entity = "pink"

        node_size="200px"
        for r in result:
            net.add_node(r['n']['id'], shape="dot", color=color_entity, node_size=node_size)
            net.add_node(r['b']['id'], shape="dot", color=color_entity, node_size=node_size)
            net.add_edge(r["n"]['id'], r['b']['id'],
                         title=r["r"].type, label=r["r"].type)

        # save network
        net.repulsion(
            node_distance=200,
            central_gravity=0.2,
            spring_length=200,
            spring_strength=0.05,
            damping=0.09
        )
        net.set_edge_smooth('dynamic')
        # net.show('./networks/' + filename, notebook=False)

        from IPython.display import display, HTML

        html = net.generate_html()
        with open("test.html", mode='w', encoding='utf-8') as fp:
            fp.write(html)

query_neo4j()

with st.container():
    #st.markdown("Creating success!")
    html_source_code = open("test.html", 'r', encoding='utf-8').read()
    components.html(html_source_code, width=700, height=700)