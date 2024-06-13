from pyvis.network import Network
import pandas as pd 
import math
import torch
from kb import KB

def extract_triplets_typed(text):
    triplets = []
    relation = ''
    text = text.strip()
    current = 'x'
    subject, relation, object_, object_type, subject_type = '','','','',''

    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").replace("tp_XX", "").replace("__en__", "").split():
        if token == "<triplet>" or token == "<relation>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(), 'head_type': subject_type, 'type': relation.strip(),'tail': object_.strip(), 'tail_type': object_type})
                relation = ''
            subject = ''
        elif token.startswith("<") and token.endswith(">"):
            if current == 't' or current == 'o':
                current = 's'
                if relation != '':
                    triplets.append({'head': subject.strip(), 'head_type': subject_type, 'type': relation.strip(),'tail': object_.strip(), 'tail_type': object_type})
                object_ = ''
                subject_type = token[1:-1]
            else:
                current = 'o'
                object_type = token[1:-1]
                relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '' and object_type != '' and subject_type != '':
        triplets.append({'head': subject.strip(), 'head_type': subject_type, 'type': relation.strip(),'tail': object_.strip(), 'tail_type': object_type})
    return triplets

def from_text_to_kb(text,
                    model,
                    tokenizer,
                    span_length=50,
                    verbose=False):
    # tokenize whole text
    inputs = tokenizer([text], return_tensors="pt")

    # compute span boundaries 计算文本段落的划分边界
    num_tokens = len(inputs["input_ids"][0])
    if verbose:
        print(f"Input has {num_tokens} tokens")
    num_spans = math.ceil(num_tokens / span_length)
    if verbose:
        print(f"Input has {num_spans} spans")
    overlap = math.ceil((num_spans * span_length - num_tokens) /
                        max(num_spans - 1, 1))
    spans_boundaries = []
    start = 0
    for i in range(num_spans):
        spans_boundaries.append([start + span_length * i,
                                 start + span_length * (i + 1)])
        start -= overlap
    if verbose:
        print(f"Span boundaries are {spans_boundaries}")

    # transform input with spans 这段代码将原始的输入文本按照计算得到的段落边界进行切分，并将每个段落转换为模型所需的输入格式
    tensor_ids = [inputs["input_ids"][0][boundary[0]:boundary[1]]
                  for boundary in spans_boundaries]
    tensor_masks = [inputs["attention_mask"][0][boundary[0]:boundary[1]]
                    for boundary in spans_boundaries]
    inputs = {
        "input_ids": torch.stack(tensor_ids),
        "attention_mask": torch.stack(tensor_masks)
    }

    # generate relations
    num_return_sequences = 3
    gen_kwargs = {
        "max_length": 256,
        "length_penalty": 0,
        "num_beams": 3,
        "num_return_sequences": num_return_sequences
    }
    generated_tokens = model.generate(
        **inputs,
        **gen_kwargs,
    )

    # model_inputs = tokenizer(text, max_length=256, padding=True, truncation=True, return_tensors='pt')
    # generated_tokens = model.generate(
    #     model_inputs["input_ids"].to(model.device),
    #     attention_mask=model_inputs["attention_mask"].to(model.device),
    #     decoder_start_token_id=tokenizer.convert_tokens_to_ids("tp_XX"),
    #     **gen_kwargs,
    # )

    # decode relations
    decoded_preds = tokenizer.batch_decode(generated_tokens,
                                           skip_special_tokens=False)

    # create kb
    kb = KB()
    i = 0
    for sentence_pred in decoded_preds:
        current_span_index = i // num_return_sequences
        relations = extract_triplets_typed(sentence_pred)
        for relation in relations:
            relation["meta"] = {
                "spans": [spans_boundaries[current_span_index]]
            }
            kb.add_relation(relation)
        i += 1

    return kb

def from_text_to_kb_(text,
                    model,
                    tokenizer,
                    span_length=50,
                    verbose=False):
    gen_kwargs = {
        "max_length": 1000,
        "length_penalty": 0,
        "num_beams": 3,
        "num_return_sequences": 3,
        "forced_bos_token_id": None,
    }

    model_inputs = tokenizer(text, max_length=1000, padding=True, truncation=True, return_tensors='pt')

    # Generate
    generated_tokens = model.generate(
        model_inputs["input_ids"].to(model.device),
        attention_mask=model_inputs["attention_mask"].to(model.device),
        decoder_start_token_id=tokenizer.convert_tokens_to_ids("tp_XX"),
        **gen_kwargs,
    )

    # Extract text
    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

    # create kb
    kb = KB()
    for sentence_pred in decoded_preds:
        relations = extract_triplets_typed(sentence_pred)
        for relation in relations:
            kb.add_relation_text(relation)

    return kb

def save_network_html(kb, filename="network.html"):
    # create network
    net = Network(directed=True, width="700px", height="700px", cdn_resources = "in_line")

    # nodes
    color_entity = "#00FF00"
    
    for e in kb.relations:
        net.add_node(e['head'], shape="circle", color=color_entity)
        net.add_node(e['tail'], shape="circle", color=color_entity)
    # edges
    for r in kb.relations:
        net.add_edge(r["head"], r["tail"],
                    title=r["type"], label=r["type"])

    # save network
    net.repulsion(
        node_distance=200,
        central_gravity=0.2,
        spring_length=200,
        spring_strength=0.05,
        damping=0.09
    )
    net.set_edge_smooth('dynamic')
    #net.show('./networks/' + filename, notebook=False)

    from IPython.display import display, HTML
    html = net.generate_html()
    with open("./networks/network.html", mode='w', encoding='utf-8') as fp:
        fp.write(html)



from neo4j import GraphDatabase


class Neo4jConnection:
    
    def __init__(self, uri, user, pwd):
        self.__uri = uri
        self.__user = user
        self.__pwd = pwd
        self.__driver = None
        try:
            self.__driver = GraphDatabase.driver(self.__uri, auth=(self.__user, self.__pwd))
        except Exception as e:
            print("Failed to create the driver:", e)
        
    def close(self):
        if self.__driver is not None:
            self.__driver.close()
        
    def query(self, query, db=None):
        assert self.__driver is not None, "Driver not initialized!"
        session = None
        response = None
        try: 
            session = self.__driver.session(database=db) if db is not None else self.__driver.session() 
            response = list(session.run(query))
        except Exception as e:
            print("Query failed:", e)
        finally: 
            if session is not None:
                session.close()
        return response

def sanitize(text):
    text = str(text).replace("'","").replace('"','').replace('{','').replace('}', '').replace(" ", "_").replace('-', '_')
    return text