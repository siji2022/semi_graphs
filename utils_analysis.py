# util files for dataset analysis
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.nn as geom_nn
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
from torch_geometric.utils import *
from torch_sparse import SparseTensor

def vis_N_nodes(N, data, edges):
    # generate a list from 0 to N
    # select_nodes_idx=list(range(N))
    select_nodes_idx=torch.arange(0, N-1, dtype=torch.long)
    # select_nodes=data[select_nodes_idx]
    # # selected edges
    # selected_edges = [] 
    # # loop edge_index
    # for start_nodes in (edges):
    #     if start_nodes[0] in select_nodes_idx and start_nodes[1] in select_nodes_idx:
    #         selected_edges.append(start_nodes)
    selected_edges,_=torch_geometric.utils.subgraph(subset=select_nodes_idx, edge_index=edges)
    # visualize the graph
    G = nx.Graph()
    G.add_nodes_from(select_nodes_idx.numpy())
    G.add_edges_from(selected_edges.t().tolist())
    plt.figure(figsize=(5,5))
    nx.draw(G, with_labels=True)
    # pyg utils to get subset of the graph
    # subset_data = data[select_nodes_idx]
    
    
def pca_analysis(data, y, labels_size):
    pca = PCA(n_components=2).fit_transform(data)
    # plot 2 figures
    plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    for i in range(labels_size):
        plt.scatter(pca[y == i, 0], pca[y== i, 1], s=4, label=f'Class {i}')
        
    # highlignt 1728 and its neighbors
    plt.subplot(1, 2, 2)
    for i in range(labels_size):
        plt.scatter(pca[y == i, 0], pca[y== i, 1], s=4, label=f'Class {i}')
    highlight_node = 1728   
    highlight_node_neighbors = [961, 1358, 2257, 2555, 2599]
    highlight_node_neighbors_based_similarity = [1299, 1728, 2577]
    plt.scatter(pca[highlight_node, 0], pca[highlight_node, 1], s=200, c='black', marker='x', label='Highlight Node')
    for neighbor in highlight_node_neighbors:
        plt.scatter(pca[neighbor, 0], pca[neighbor, 1], s=100, c='blue', marker='*')
    for neighbor in highlight_node_neighbors_based_similarity:
        plt.scatter(pca[neighbor, 0], pca[neighbor, 1], s=100, c='blue', marker='o')
    
    # legend on the right side of the plot
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(f'PCA')
    
def tsne_analysis(data, y, labels_size):
    tsne = TSNE(n_components=2).fit_transform(data)
    plt.figure(figsize=(5,5))
    for i in range(labels_size):
        plt.scatter(tsne[y == i, 0], tsne[y== i, 1], s=5, label=f'Class {i}')
    plt.legend()
    plt.title(f'TSNE')

def start_end_cluster_distribution(edges, y, labels_size):
    results=np.zeros((labels_size,labels_size))
    for edge in edges:
        start_node, end_node = edge
        start_class = y[start_node]
        end_class = y[end_node]
        
        results[start_class][end_class] += 1
    # 2 subplots

    fig, axs=plt.subplots(1,2,figsize=(12,5))
    axs=axs.flatten()
    
    sns.heatmap(results, annot=True, fmt='g',ax=axs[0])
    # heatmap in percentage
    sns.heatmap(results/np.sum(results,axis=1), annot=True, fmt='.1%',ax=axs[1])
    
    
def assortativity(edge_index) -> float:
    r"""The degree assortativity coefficient from the
    `"Mixing patterns in networks"
    <https://arxiv.org/abs/cond-mat/0209450>`_ paper.
    Assortativity in a network refers to the tendency of nodes to
    connect with other similar nodes over dissimilar nodes.
    It is computed from Pearson correlation coefficient of the node degrees.

    Args:
        edge_index (Tensor or SparseTensor): The graph connectivity.

    Returns:
        The value of the degree assortativity coefficient for the input
        graph :math:`\in [-1, 1]`

    Example:
        >>> edge_index = torch.tensor([[0, 1, 2, 3, 2],
        ...                            [1, 2, 0, 1, 3]])
        >>> assortativity(edge_index)
        -0.666667640209198
    """
    if isinstance(edge_index, SparseTensor):
        adj: SparseTensor = edge_index
        row, col, _ = adj.coo()
    else:
        assert isinstance(edge_index, Tensor)
        row, col = edge_index

    device = row.device
    out_deg = degree(row, dtype=torch.long)
    in_deg = degree(col, dtype=torch.long)
    degrees = torch.unique(torch.cat([out_deg, in_deg]))
    mapping = row.new_zeros(degrees.max().item() + 1)
    mapping[degrees] = torch.arange(degrees.size(0), device=device)

    # Compute degree mixing matrix (joint probability distribution) `M`
    num_degrees = degrees.size(0)
    src_deg = mapping[out_deg[row]]
    dst_deg = mapping[in_deg[col]]

    pairs = torch.stack([src_deg, dst_deg], dim=0)
    occurrence = torch.ones(pairs.size(1), device=device)
    pairs, occurrence = coalesce(pairs, occurrence)
    M = to_dense_adj(pairs, edge_attr=occurrence, max_num_nodes=num_degrees)[0]
    # normalization
    M /= M.sum()

    # numeric assortativity coefficient, computed by
    # Pearson correlation coefficient of the node degrees
    x = y = degrees.float()
    a, b = M.sum(0), M.sum(1)

    vara = (a * x**2).sum() - ((a * x).sum())**2
    varb = (b * x**2).sum() - ((b * x).sum())**2
    xy = torch.outer(x, y)
    ab = torch.outer(a, b)
    out = (xy * (M - ab)).sum() / (vara * varb).sqrt()
    return out.item()
    
    

def create_synthetic_data(FIGURES_DIR='graph_figures'):
    # Create a graph with 7 nodes
    N = 7
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 2, 2, 5, 0, 6, 1, 6, 3, 4, 4, 5, 3, 5],
                               [1, 0, 3, 2, 2, 4, 5, 2, 6, 0, 6, 1, 4, 3, 5, 4, 5, 3]], dtype=torch.long)
    # plot the graph
    plt.clf()
    G = nx.Graph()
    G.add_edges_from(edge_index.T.numpy())
    nx.draw(G, with_labels=True)
    path = osp.join(osp.dirname(osp.realpath(__file__)),
                    '.', FIGURES_DIR, 'sythetic_graph.png')
    plt.savefig(path)

    # create labels
    labels = torch.tensor([0, 0, 1, 1, 1, 1, 0], dtype=torch.long)
    train_mask = torch.tensor([True, False, False, True, False, True, False])
    test_mask = torch.tensor([False, True, True, False, True, False, False])
    return edge_index, labels, train_mask, test_mask, N, 2, None
    
def calc_p_dirichlet(x, L):
    energy=torch.sum(torch.diag(x.T@L@x)) +1e-3
    return torch.sqrt(energy)/x.shape[0]


def sim_graph(adj, th):
    # adj[adj>th]=1.0
    # adj[adj<th]=0
    # print(adj[0])
    # if th==1:
        # adj=torch.expm1(adj)
    D = torch.diag_embed(torch.sum(adj, dim=1))
    deg_inv_sqrt = torch.pow(D.diag(), -0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    deg_inv_sqrt = torch.diag_embed(deg_inv_sqrt)
    A_norm = deg_inv_sqrt@(adj)@deg_inv_sqrt   # CHECK !!!! 072624 if normalize L, acc is 0.1 equals[L_sys-I] Cora use this one has better performance
        # L_sys =  deg_inv_sqrt@(D-adj)@deg_inv_sqrt # this is the correct way of normalize L; iris use this one has better performance
    # elif th==2:
    #     adj[adj<0.95]=0.0
    #     adj[adj>0.95]=1.0
    #     D = torch.diag_embed(torch.sum(adj, dim=1))
    #     deg_inv_sqrt = torch.pow(D.diag(), -0.5)
    #     deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    #     deg_inv_sqrt = torch.diag_embed(deg_inv_sqrt)
    #     A_norm = deg_inv_sqrt@(adj)@deg_inv_sqrt   # CHECK !!!! 072624 if normalize L, acc is 0.1 equals[L_sys-I] Cora use this one has better performance
    #     # L_sys =  deg_inv_sqrt@(D-adj)@deg_inv_sqrt # this is the correct way of normalize L; iris use this one has better performance
    # elif th==3:
    #     adj=adj**2
    #     A_norm=adj
    # elif th==4:
    #     adj=adj
    #     A_norm=adj
    # elif th==5:
    #     adj=adj+torch.eye(adj.shape[0]).to(adj.device)*torch.e 
    #     A_norm=adj
    
    return A_norm 

def unsim_graph(adj, th):
    # adj[adj>th]=0.0
    # adj[adj<th]=1.0
    # adj.diagonal().fill_(1)
    D = torch.diag_embed(torch.sum(adj, dim=1))
    deg_inv_sqrt = torch.pow(D.diag(), -0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    deg_inv_sqrt = torch.diag_embed(deg_inv_sqrt)
    L_sys =  deg_inv_sqrt@(D-adj)@deg_inv_sqrt # this is the correct way of normalize L; iris use this one has better performance
    return L_sys
    
def create_infer_graph_from_x(x, th1, th2, x_size=0, plot_hist=False):
    adj = (x@x.T)
    norms=torch.norm(x, p=2, dim=1).unsqueeze(1)
    adj=adj/(norms@norms.T+1e-6)
    adj.fill_diagonal_(0)
    if x_size>0:
    # #     # make the non-diag elements to be 0
        adj[:x_size, :x_size].triu().fill_(0)
        adj[:x_size, :x_size].tril().fill_(0)
    # adj.diagonal().fill_(1)
    adj_sim=sim_graph(adj.clone(),th1)
    adj_unsim=unsim_graph(adj.clone(),th2)
    
    
    # # generate a random number from 1 to 100
    # rnd=np.random.randint(1, 1000)
    # if rnd>995 and plot_hist:
    #     plt.hist(adj.flatten().detach().cpu().numpy(), bins=100)
    #     # plt.hist(L_sys.flatten().detach().cpu().numpy(), bins=100)
    #     #     # use log on y axis
    #     plt.yscale('log')
    #     plt.xlabel('Similarity score')
    #     plt.title(f'sigmoid(X@X.T) histogram {plot_hist}')
    #     plt.savefig('./sigmoid_X_X_T_hist.png')
    #     plt.clf()
        
    return adj_sim, adj_unsim

def create_infer_graph_from_x_v5(x, th1, th2, x_size=0, plot_hist=False):
    adj = (x@x.T)
    # adj=torch.sigmoid(adj)
    norms=torch.norm(x, p=2, dim=1).unsqueeze(1)
    adj=adj/(norms@norms.T+1e-6)
    adj.fill_diagonal_(0)
    if x_size>0:
    # #     # make the non-diag elements to be 0
        adj[:x_size, :x_size].triu().fill_(0)
        adj[:x_size, :x_size].tril().fill_(0)
    # adj.diagonal().fill_(1)
    # lsys=sim_graph_v5(adj.clone(),th1)
    # adj_unsim=unsim_graph_v5(adj.clone(),th2)
    # # generate a random number from 1 to 100
    # rnd=np.random.randint(1, 1000)
    # if rnd>995 and plot_hist:
    #     plt.clf()
    #     plt.hist(adj.flatten().detach().cpu().numpy(), bins=100)
    #     # plt.hist(L_sys.flatten().detach().cpu().numpy(), bins=100)
    #     #     # use log on y axis
    #     plt.yscale('log')
    #     plt.xlabel('Similarity score')
    #     plt.title(f'sigmoid(X@X.T) histogram {plot_hist}')
    #     plt.savefig('./sigmoid_X_X_T_hist.png')
    #     plt.clf()
        
    return  adj

def sim_graph_v5(adj, th):
    # adj[adj<th]=0.0
    # adj[adj>th]=1
    # adj=torch.expm1(adj)
    D = torch.diag_embed(torch.sum(adj, dim=1))
    deg_inv_sqrt = torch.pow(D.diag(), -0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    deg_inv_sqrt = torch.diag_embed(deg_inv_sqrt)
    # A_norm = deg_inv_sqrt@(adj)@deg_inv_sqrt   # CHECK !!!! 072624 if normalize L, acc is 0.1 equals[L_sys-I] Cora use this one has better performance
    L_sys =  deg_inv_sqrt@(D-adj)@deg_inv_sqrt # this is the correct way of normalize L; iris use this one has better performance
    
    # return L_sys.detach() 
    return adj
# def unsim_graph_v5(adj, th):
    # # adj[adj>th]=0.0
    # # adj[adj<th]=1.0
    # # adj.diagonal().fill_(1)
    # D = torch.diag_embed(torch.sum(adj, dim=1))
    # deg_inv_sqrt = torch.pow(D.diag(), -0.5)
    # deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    # deg_inv_sqrt = torch.diag_embed(deg_inv_sqrt)
    # L_sys =  deg_inv_sqrt@(D-adj)@deg_inv_sqrt # this is the correct way of normalize L; iris use this one has better performance
    # return L_sys
    
