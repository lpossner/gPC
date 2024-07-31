import logging
from tqdm import tqdm
import numpy as np
import networkx as nx
import pygpc
from castle.algorithms import Notears, PC
from castle.datasets import DAG, IIDSimulation


# def simulate_equation(X, w, Z):
#     return X @ w + Z

# def simulate_dag(W, Z):
#     G =  nx.from_numpy_matrix(W, create_using=nx.DiGraph)
#     assert nx.is_directed_acyclic_graph(G), "W must represent a DAG"
#     X = np.zeros([Z.shape[1], Z.shape[0]])
#     ordered_vertices = list(nx.topological_sort(G))
#     for vertex in ordered_vertices:
#         parents = list(G.predecessors(vertex))
#         X[:, vertex] = simulate_equation(X[:, parents], W[parents, vertex], Z[vertex, :])
#     return X

# N_samples = 100
# N_nodes = 10
# N_edges = 20
# W_true = DAG.scale_free(n_nodes=N_nodes, n_edges=N_edges)
# Z = np.random.normal(loc=0, scale=1, size=(N_nodes, N_samples))

# X = simulate_dag(W_true, Z)


logging.disable(logging.CRITICAL)

N_nodes = 30
# N_edges = 120
N_learn_samples = 1000
N_res_samples = 1
method = "linear"
sem_type = "gauss"
noise_scale = 1

coords = np.array([120, 180, 220])

model = Notears()

W_result_lst = []
N_iterations = len(coords) * len(range(N_res_samples))

with tqdm(total=N_iterations) as pbar:
    for N_edges in coords:
        W_inner_result_lst = []
        for _ in range(N_res_samples):
            W_true = DAG.scale_free(n_nodes=N_nodes, n_edges=N_edges)
            data = IIDSimulation(
                W=W_true,
                n=N_learn_samples,
                method=method,
                sem_type=sem_type,
                noise_scale=noise_scale,
            )
            model.learn(data.X)
            W_inner_result_lst.append((W_true - model.causal_matrix).flatten())
            pbar.update(1)
        W_result_lst.append(W_inner_result_lst)

results = np.array(W_result_lst)
