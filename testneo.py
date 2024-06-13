import streamlit as st
import streamlit.components.v1 as components
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import utils
import utils_t
from kb import KB
import pandas as pd
from pyvis.network import Network


@st.cache_resource(show_spinner=False)
def query_neo4j():
    with st.spinner('Connecting to Neo4j...'):
        net = Network(directed=True, width="700px", height="700px", cdn_resources="in_line")
        conn = utils.Neo4jConnection(uri=st.secrets['uri'], user=st.secrets['username'], pwd=st.secrets['password'])
        query="MATCH (n)-[r]->(b) RETURN (n),(b),(r)"
        result=conn.query(query)
        print(type(result))
        print(result)
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

