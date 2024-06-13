import streamlit as st
import streamlit.components.v1 as components
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import utils
import utils_t
from kb import KB
import pandas as pd

st.set_page_config(page_title="知识图谱生成",
                   layout="centered", 
                   initial_sidebar_state="auto", 
                   menu_items=None)

st.header("利用rebel模型生成知识图谱")


# Loading the model
st_model_load = st.text('Loading Rebel model. This should take 1-2 minutes.')

@st.cache_resource(show_spinner=False)
def load_model():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
    # tokenizer = AutoTokenizer.from_pretrained("Babelscape/mrebel-large",src_lang="zh_CN", tgt_lang="zh_CN")
    # model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/mrebel-large")
    print("Model loaded!")
    return tokenizer, model

tokenizer, model = load_model()
st.success('Model loaded!')
st_model_load.text("")

placeholder_str ="请输入文本"

if 'text' not in st.session_state:
    st.session_state.text = ""
text = st.session_state.text
text = st.text_area('Text:', value=text, height=300, disabled=False, max_chars=30000, placeholder=placeholder_str)

def ingest_to_neo4j(filename='relations.csv'):
    with st.spinner('Ingesting to Neo4j...'): 
        import time
        time.sleep(10)
        df = pd.read_csv('./data/' + filename)
        #df = df.drop(columns={'meta'})
        for col in df.columns:
            df[col] = df[col].apply(utils.sanitize).str.lower()

        conn = utils.Neo4jConnection(uri=st.secrets['uri'], user=st.secrets['username'], pwd=st.secrets['password'])

        # Loop through data and create Cypher query
        for i in range(len(df['head'])):

            query = f'''
                MERGE (head:Head {{name: "{df['head'][i]}"}})

                MERGE (tail:tail {{value: "{df['tail'][i]}"}})

                MERGE (head)-[:{df['type'][i]}]->(tail)
                '''
            result = conn.query(query, db=st.secrets['database'])
        st.success('Data ingested to Neo4j', icon="✅")

tab1, tab2 = st.tabs(["生成 & 展示", "保存至Neo4j"])



with tab1:
    button_text = "生成 & 展示"
    with st.spinner('正在构建 KG...'):
        # generate KB button
        if st.button(button_text):

            #kb = utils_t.from_text_to_kb_(text, model, tokenizer, verbose=True)
            kb = utils.from_text_to_kb(text, model, tokenizer, verbose=True)
            
            kb.save_csv(f"./data/{st.secrets['triplets_filename']}")

            # save chart
            #utils_t.save_network_html(kb, filename="network.html")
            utils.save_network_html(kb, filename="network.html")
            st.session_state.kb_chart = "./networks/network.html"
            # st.session_state.kb_text = kb.get_textual_representation()
            st.session_state.error_url = None


            # kb chart session state
            if 'kb_chart' not in st.session_state:
                st.session_state.kb_chart = None
            if 'kb_text' not in st.session_state:
                st.session_state.kb_text = None
            if 'error_url' not in st.session_state:
                st.session_state.error_url = None

            # show graph
            if st.session_state.error_url:
                st.markdown(st.session_state.error_url)
            elif st.session_state.kb_chart:
                with st.container():
                    st.subheader("KG构建成功")
                    st.markdown("*您可以与图形进行拖动缩放等交互。*")
                    html_source_code = open(st.session_state.kb_chart, 'r', encoding='utf-8').read()
                    components.html(html_source_code, width=700, height=700)
                    st.markdown(st.session_state.kb_text)
            st.success('KG构建成功!', icon="✅")
with tab2:
    st.button('注入Neo4j', on_click=ingest_to_neo4j)


