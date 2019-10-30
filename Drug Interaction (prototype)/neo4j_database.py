from py2neo import Graph, Node, Relationship
from flask import Flask
import os
import pandas as pd
import tqdm



# set up authentication parameters

def init(bolt_uri = "bolt://54.154.137.197:7687",password="Limon88de00",db_path="/home/ubuntu/upload/thesaurus_data/thesaurus_cleaned.csv"):
    '''
    function which initializes the graph database
    :param bolt_uri:
    :param password:
    :param db_path:
    :return  graph1, result_subset, new_df:
    '''

    graph1 = Graph(bolt_uri, password=password)
    result_subset = pd.read_csv(db_path)
    new_df = result_subset.copy()
    return(graph1, result_subset, new_df)


def nodes_creation(df, graph1):
    '''
    function which creates all the nodes  (all the molecules interacting)
    :param df:
    :param graph1:
    :return None:
    '''

    # we want to remove any 2x entry
    princeps_unique = df["princeps"].unique().tolist()
    interaction_unique = df["interacted_with"].unique().tolist()

    all_drugs_OI = list(set(princeps_unique + interaction_unique))

    # appending nodes to the graoh database
    tx = graph1.begin()
    nodes = []
    for idx, _ in tqdm.tqdm(enumerate(all_drugs_OI)):
        a = Node("Drug", name=all_drugs_OI[idx])
        nodes.append(a)
    for el in nodes:
        graph1.create(el)
    tx.commit()
    print("all nodes created ({0}).".format(len(all_drugs_OI)))


def relationship_creation(df, graph1):
    '''
    function which update all the  created nodes (all the molecules interacting) with a relationship
    :param df:
    :param graph1:
    :return None:
    '''


    princeps_list = df["princeps"].values
    adverse_event = df["verdict"].values
    interaction_list = df["interacted_with"].values

    tx = graph1.begin()
    relationships = []

    for idx, _ in tqdm.tqdm(enumerate(interaction_list)):
        a = Node("Drug", name=princeps_list[idx])
        b = Node("Drug", name=interaction_list[idx])

        ab = Relationship(a, "INTERACTS_WITH", b, adverse_ev=adverse_event[idx])
        ba = Relationship(b, "INTERACTS_WITH", a, adverse_ev=adverse_event[idx])

        relationships.append(ab)
        relationships.append(ba)
    print(len(relationships))
    for el in relationships:
        try:
            graph1.merge(el, "Drug", "name")
        except:
            pass
    tx.commit()
    print("all relationships created ({0}).".format(len(interaction_list)))


def drug_query_interaction_where(queried_drug = "ANTI-INFLAMMATOIRES NON STÉROÏDIENS",node_property="name"):
    '''
    function which queries the graph based on a Where clause
    :param queried_drug:
    :return df_response:
    '''
    # query retrieving all the interactions & returns the molecules interacting
    query = "MATCH (n:Drug)-[r:INTERACTS_WITH]-(m) WHERE" + " n.name=\"{0}\"".format(
        str(queried_drug)) + " RETURN m."+"{0};".format(str(node_property))

    cursor = graph1.run(query)
    df_response = cursor.to_data_frame()
    return(df_response)



app = Flask(__name__)


@app.route("/")
def home():
    return "Hello, World!"

@app.route("/<query>", methods = ['GET'])
def predict_interaction(query):
    result = drug_query_interaction_where(queried_drug=query, node_property="name")
    return (",").join(result.values.flatten().tolist())


if __name__ == "__main__":
    graph1, result_subset, df = init()
    nodes_creation(df, graph1)
    relationship_creation(df, graph1)
    drug_query_interaction_where(queried_drug="ANTI-INFLAMMATOIRES NON STÉROÏDIENS", node_property="name")
    app.run(host='0.0.0.0', port =5000, debug=True)