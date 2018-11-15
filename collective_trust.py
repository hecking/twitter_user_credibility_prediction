import networkx as nx
import numpy as np
import pandas as pd


def get_fit(A1, A2, x, b, evidence, alpha):
    return np.sum(alpha**2 * np.square(A2.dot(A1.dot(x) + b) - evidence))


# Optimise veracity assignment by gradient descent.
def propagate_trust(A1, A2, b, alpha, evidence, init=None, learning_rate=0.1, verbose=True):

    if init is None:
        x = np.random.rand(np.size(A1, 0))
        x = x.reshape(x.shape[0], 1)
    else:
        x = init

    old_q = get_fit(A1, A2, x, b, evidence, alpha)

    if verbose:
        print("Q = " + str(old_q))

    gradient = alpha * (A2.dot(A1).T.dot(A2.dot(A1.dot(x) + b) - evidence))

    new_x = x - (learning_rate * gradient)

    new_x[new_x < 0] = 0
    new_x[new_x > 1] = 1

    new_q = get_fit(A1, A2, new_x, b, evidence, alpha)

    # Adjust learning rate in case of overshooting a local optima.
    while old_q < new_q:
        if verbose:
            print("adjust learning rate")
            learning_rate = learning_rate / 2
            new_x = x + learning_rate * gradient
            new_q = get_fit(A1, A2, new_x, b, evidence, alpha)

    if np.sum(np.square(x - new_x)) < 0.0001:
        return new_x
    else:
        return propagate_trust(A1, A2, b, alpha, evidence, new_x, learning_rate, verbose)


def predict_veracity_collective_regression(g, evidence, alpha, learning_rate=0.5, init=None, verbose=True):
    A = nx.to_numpy_matrix(g)
    A = A / np.sum(A, axis=1)
    A = np.nan_to_num(A)
    A1 = np.delete(np.delete(A, evidence.id, axis=0), evidence.id, axis=1)
    A2 = np.delete(np.take(A, evidence.id, axis=0), evidence.id, axis=1)
    ev_vals = np.array(evidence.value.values)
    ev_vals = ev_vals.reshape(ev_vals.shape[0], 1)

    b = np.take(np.delete(A, evidence.id, axis=0), evidence.id, axis=1).dot(ev_vals)

    veracity = propagate_trust(A1, A2, b, alpha, ev_vals, init, learning_rate, verbose)

    non_evidence = np.isin(range(0, len(g)), evidence.id, invert=True)
    node_names = np.array(list(nx.get_node_attributes(g, 'name').values()))[non_evidence]

    return dict(zip(node_names, veracity[:,0].reshape(-1,).tolist()[0]))


def truncated_katz(g, alpha=0.75):
    A = nx.to_numpy_matrix(g)

    return (alpha * A) + (alpha**2 * A.dot(A)) + (alpha**3 * A.dot(A).dot(A)) + (alpha**4 * A.dot(A).dot(A).dot(A).dot(A))


def predict_veracity_truncated_katz(g, evidence, alpha=0.75):
    K = truncated_katz(g, alpha)
    np.fill_diagonal(K, 0)
    ev_vals = np.array(evidence.value.values)
    ev_vals = ev_vals.reshape(ev_vals.shape[0], 1)

    veracity = np.nan_to_num(K[:, evidence.id].dot(ev_vals) / np.sum(K[:, evidence.id], axis=1))
    non_evidence = np.isin(range(0, len(g)), evidence.id, invert=True)

    node_names = np.array(list(nx.get_node_attributes(g, 'name').values()))[non_evidence]
    return dict(zip(node_names, veracity[non_evidence, 0].reshape(-1,).tolist()[0]))


def example():
    g = nx.read_gml("user_graph_with_info_cleaned.gml", label='id').to_undirected()

    node_df = pd.DataFrame.from_dict(dict(g.nodes(data=True)), orient='index')
    node_df['id'] = list(range(0, node_df.shape[0]))
    node_df['value'] = (node_df.truetweets + 1) / \
                                 (node_df.truetweets + node_df.falsetweets + 2)
    # Example run
    # Keep only nodes with more than 2 tweets as evidence nodes
    evidence_nodes = node_df[(node_df.truetweets + node_df.falsetweets) > 2]

    print("Truncated Katz")
    p_katz = predict_veracity_truncated_katz(g, evidence_nodes, 0.75)
    print("Collective regression")
    p_colreg = predict_veracity_collective_regression(g, evidence_nodes, 1)

    return p_colreg, p_katz

example()
