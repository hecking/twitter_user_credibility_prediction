import networkx as nx
import pandas as pd
import numpy as np
from flask import Flask
from flask import jsonify
from collective_trust import *

app = Flask(__name__)

# Read evidence data
evidence_graph = nx.read_gml("user_graph_with_info_cleaned.gml", label='id').to_undirected()
node_df = pd.DataFrame.from_dict(dict(evidence_graph.nodes(data=True)), orient='index')
node_df['id'] = list(range(0, node_df.shape[0]))
node_df['value'] = (node_df.truetweets + 1) / (node_df.truetweets + node_df.falsetweets + 2)

# Keep only nodes with more than 2 tweets as evidence nodes
evidence_nodes = node_df[(node_df.truetweets + node_df.falsetweets) > 2]


# ego_net: Ego network of the node for which veracity should be predicted as GML string.
# ego: Node for which veracity should be predicted.
# type: Either truncated katz (katz) or collective regression (cr)
@app.route('/credibility/<ego_net>/<ego>/<strategy>')
def predict_veracity(ego_net, ego, strategy='katz'):

    query_graph = nx.parse_gml(ego_net).to_undirected()
    # Check if evidence is available
    common_nodes = np.intersect1d(np.array(query_graph.nodes), np.array(evidence_graph.nodes))
    if np.isin(ego, evidence_nodes.name):
        result = {ego : evidence_nodes.value[evidence_nodes.name == ego]}

    if common_nodes.size > 0:
        combined_graph = nx.compose(query_graph, evidence_graph).subgraph(common_nodes)

        if strategy == 'katz':
            veracity_predicted = predict_veracity_truncated_katz(combined_graph, evidence_nodes)
        else:
            veracity_predicted = predict_veracity_collective_regression(combined_graph, evidence_nodes, 0.9)

        result = veracity_predicted[ego]
    else:
        result = "No connection with evidence graph."

    return jsonify(result)
