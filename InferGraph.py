# implement the topology learning algorithm based on Node2Vec
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.nn as geom_nn
import numpy as np
import matplotlib.pyplot as plt
import os
import gc
from torch.nn import Embedding
from torch.utils.data import DataLoader, TensorDataset
from torch_sparse import SparseTensor
import networkx as nx
import os.path as osp
import sys
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import to_dense_adj
from sklearn.linear_model import LogisticRegression
import pandas as pd
from utils_analysis import *
from utils_model import *

# suppress warnings
import warnings
warnings.filterwarnings("ignore")


# fix seed
# torch.manual_seed(0)
# np.random.seed(0)
# DATA_SET = 'Pubmed'
DATA_SET = 'Cora'
# DATA_SET = 'CiteSeer'
# SPLIT='public'
# SPLIT = 'full'
SPLIT = 'random'
LOAD_MODEL = False
EPS = 1e-15
FIGURES_DIR = 'graph_figures'


device = torch.device(
    "cuda:1") if torch.cuda.is_available() else torch.device("cpu")
Training_EPS = 200

DEBUG = False

# def calc_p_dirichlet(x, L):
#     energy=torch.sum(torch.diag(x.T@L@x))
#     return torch.sqrt(energy)/x.shape[0]

# def create_infer_graph_from_x(x, th1, th2):
#     # print('create graph in forward')
#     # adj = torch.sigmoid(decode+self.bias)
#     decode = x.to(device)
#     decode = (decode@decode.T).detach()
#     # adj = decode
#     adj = torch.sigmoid(decode)

#     adj_sim=adj.clone()
#     adj_sim[adj_sim>th1]=1.0
#     adj_sim[adj_sim<th2]=0.0
#     I=torch.eye(adj_sim.shape[0]).to(device)

    
    

#     D = torch.diag_embed(torch.sum(adj_sim, dim=1))
    
#     deg_inv_sqrt = torch.pow(D.diag(), -0.5)
#     deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
#     deg_inv_sqrt = torch.diag_embed(deg_inv_sqrt)
#     A_infered = deg_inv_sqrt@adj_sim@deg_inv_sqrt # CHECK !!!! 072624 if normalize L, acc is 0.1 equals[L_sys-I]
#     L_sys =  deg_inv_sqrt@(D-adj_sim)@deg_inv_sqrt # this is the correct way of normalize L
    
#     # D = torch.diag_embed(torch.sum(adj, dim=1))
#     # deg_inv_sqrt = torch.pow(D.diag(), -0.5)
#     # deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
#     # deg_inv_sqrt = torch.diag_embed(deg_inv_sqrt)
#     # adj = deg_inv_sqrt@adj@deg_inv_sqrt
#     # L_sys = torch.div(L, D.diag().reshape(-1, 1))
#     # L_sys = torch.div(torch.div(L, D.diag().reshape(1, -1)), D.diag().reshape(-1, 1))
#     return A_infered, L_sys
    
# class GNN(nn.Module):
#     def __init__(self, feature_dim, hidden_dim, create_graph, drop, th1, th2, mode=0, saved_graph=None):
#         '''mode=0: use gcn, mode=1: use MLP, mode=2: use InferGraph'''
#         super(GNN, self).__init__()
#         self.nn1 = nn.Linear(feature_dim, hidden_dim)
#         self.nn2 = nn.Linear(hidden_dim, hidden_dim)
#         self.gcn1 = geom_nn.GCNConv(feature_dim, hidden_dim)
#         self.gcn2 = geom_nn.GCNConv(hidden_dim, hidden_dim)
#         self.clf = nn.Linear(hidden_dim, NUM_CLASSES)

#         self.l_sys = None
#         self.drop = drop
#         self.hidden_dim = hidden_dim
#         self.feature_dim = feature_dim
#         self.th1 = th1
#         self.th2 = th2
#         self.create_graph = create_graph
#         self.saved_graph=saved_graph
#         self.mode=mode

#     def forward(self, data, x=None, Debug=False):
#         data_x, edge, attr = data.x, data.edge_index, data.edge_attr
 

#         if self.l_sys is None and self.create_graph and self.saved_graph is None:
#             self.l_sys, self.l_for_energy = create_infer_graph_from_x(data_x, self.th1, self.th2)
#             self.adj = self.l_sys
#             # change adj matrix to adjancy list
#             # edge = torch_geometric.utils.dense_to_sparse(adj)[0]
#             # self.edge = edge  # used for gcn
#         elif self.l_sys is None:
#             self.l_sys = self.saved_graph
        

#         if Debug and self.create_graph:
#             #     # print(f'D max: {D.max()}, D min: {D.min()}')
#             # print(f'adj max: {adj.max()}, adj min: {adj. min()}')
#             # plot adj distribution
#             path = osp.join(osp.dirname(osp.realpath(__file__)), '.',
#                             FIGURES_DIR, f'adj_hist.png')
#             plt.clf()
#             plt.hist(self.adj.flatten().detach().cpu().numpy(), bins=100)
#             # use log on y axis
#             plt.yscale('log')
#             plt.savefig(path)
#         #     # print(f'edge max: {edge.max()}, edge min: {edge.min()}')
#         #     # print(L_sys[0][0], L_sys[1][1])
#         #     print(
#         #         f'emb max: {emb.max():3f}, decode min: {decode.min():3f}, bias: {self.bias.min():3f}, L_sys min: {L_sys.min():3f}')
#         if self.mode==0:
#             out1=F.relu(F.dropout(self.gcn1(data_x, edge),self.drop, training=self.training))
#             out2=F.relu(F.dropout(self.gcn2(out1, edge),self.drop, training=self.training))
        
#         if self.mode==1:
#             out1 = F.relu(F.dropout((self.nn1(data_x)), self.drop,
#                                     training=self.training))
#             out2 = F.relu(F.dropout((self.nn2(out1)), self.drop,
#                                     training=self.training))
#         if self.mode==2:
#             out1 = F.relu(F.dropout(self.l_sys@(self.nn1(data_x)), self.drop,
#                                training=self.training))
#             out2 = F.relu(F.dropout((self.l_sys@self.nn2(out1)), self.drop,
#                                training=self.training))
        
       
#         if self.create_graph or self.saved_graph is not None :
#             p_norm = calc_p_dirichlet(out2, self.l_sys)
#         else:
#             p_norm = torch.tensor(0)
#         # out = F.dropout(F.relu(self.nn(data_x)),0.2, training=self.training)
#         # out1=F.relu(self.nn1(L@out))
#         # out2=F.dropout(F.relu(self.gcn1(data_x, edge)), self.drop,
#         #                        training=self.training) #acc 0.7580
#         # concat the two outputs
#         # out=torch.concat((out1, out2), dim=1)
#         out = self.clf(out2)

#         return None, out, p_norm

#     def __repr__(self):
#         return '{}(hs={}, drop={})'.format(self.__class__.__name__, self.hidden_dim, self.drop)



def load_dataset(dataset,split):
    path = osp.join(osp.dirname(osp.realpath(__file__)),
                    '..', 'data', 'Planetoid')
    from torch_geometric.datasets import Planetoid
    dataset = Planetoid(path, name=dataset, split=split)
    data = dataset[0]
    edge_index = data.edge_index
    labels = data.y
    train_mask = data.train_mask
    test_mask = data.test_mask
    N = data.num_nodes
    # print(N)

    return edge_index, labels, train_mask, test_mask, N, 7, data


def train(model, data, optimizer, crit,  Debug=False):
    '''
    train 1 epoch
    '''
    model.train()
    total_loss = 0
    i = 0

    optimizer.zero_grad()

    adj, pred, p_norm = model(data, Debug=Debug)

    loss = crit(pred[train_mask], labels[train_mask].to(
        device))

    loss.backward()
    optimizer.step()
    total_loss += loss.item()
    i += 1
    return total_loss/train_mask.sum()


@torch.no_grad()
def test(model, data, labels,  mask, crit):
    model.eval()
    # return acc
    _, pred, p_norm = model(data)
    pred = pred.cpu()
    p_norm = p_norm.cpu()
    loss = crit(pred[mask], labels[mask])
    if DEBUG:
        print(pred[mask].argmax(dim=-1)[:10], labels[mask][:10])
    # calculate accuracy
    acc = ((pred[mask].argmax(dim=-1) == labels[mask]
            ).sum().float() / mask.sum()).cpu()
    # save pred into pt file
    # torch.save(pred, f'./results/{DATA_SET}_pred_Infer.pt')
    
    return loss, acc, p_norm





def run_experiment(model, data):
    # edge_index, labels, train_mask, test_mask, N, NUM_CLASSES, data = load_dataset()
    data = data.to(device)
    # print(model.__repr__())
    # load model
    model_path = osp.join(osp.dirname(osp.realpath(__file__)),
                          '.', 'models', 'node2vec_my.pth')
    if os.path.exists(model_path) and LOAD_MODEL:
        try:
            model.load_state_dict(torch.load(model_path))
            print('model loaded')
        except:
            pass
    optimizer = torch.optim.Adam(list(model.parameters()), lr=0.01)
    train_losss = []
    train_accs = []
    test_accs = []

    for epoch in range(1, Training_EPS+1):

        crit = nn.CrossEntropyLoss()
        if epoch % 100 == 0:
            DEBUG = True
        else:
            DEBUG = False
        train_loss = train(model, data,  optimizer,
                           crit, Debug=DEBUG)
        loss, train_acc, p_norm = test(model, data, labels, train_mask, crit)
        loss, test_acc, p_norm = test(model, data, labels, test_mask, crit)
        # if DEBUG:
        #     print(
        #         f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, p_norm: {p_norm:.4f}')

        train_losss.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

    # plot
    plt.figure(figsize=(10, 5))

    # plt.plot(train_losss, label='train loss')
    plt.plot(train_accs, label='train acc')
    plt.plot(test_accs, label='test acc')
    plt.legend(loc='best')
    plt.grid()
    plt.title(f'train loss and acc')
    path = osp.join(osp.dirname(osp.realpath(__file__)),
                    '.', FIGURES_DIR, 'train_loss_acc_my.png')
    plt.savefig(path)
    plt.clf()
    return np.max(train_accs), np.max(test_accs), p_norm
    # plot loss and acc on two y axis
    # save model

    # torch.save(model.state_dict(), model_path)


# for walk_length in [ 5,7,11,19, 23, 29]:


hidden_dim = 128
drop = 0.5
th1 = 0.99
th2 = 1
for DATA_SET in ['Cora']:
# for DATA_SET in ['Cora','CiteSeer', 'Pubmed']:
    # for SPLIT in ['public','full', 'random']:
    for SPLIT in ['random']:
        
        for mode in [2]:
            for drop in [0.2]:
            # for drop in [0, 0.2, 0.5, 0.7]:
                if mode==2:
                    th1_list=[0.5,0.9]
                    # th1_list=[0.5,0.6,0.7, 0.8, 0.9, 0.95]
                else:
                    th1_list=[0]
                for th1 in th1_list:        
                    train_accs = []
                    test_accs = []
                    p_norms = []
                    RUN_TIMES = 10
                    
                    for i in range(RUN_TIMES):
                        edge_index, labels, train_mask, test_mask, N, NUM_CLASSES, data = load_dataset(DATA_SET, SPLIT)
                        model = GNN(feature_dim=data.x.shape[1],out_dim=NUM_CLASSES, hidden_dim=hidden_dim, create_graph=True,
                                    drop=drop, th1=th1, th2=th2, mode=mode, saved_graph=None).to(device)
                        train_acc, test_acc, p_norm = run_experiment(model, data)
                        train_accs.append(train_acc.item())
                        test_accs.append(test_acc.item())
                        p_norms.append(p_norm.item())
                        # clear nvidia memory
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                    print(f'completed {RUN_TIMES},mode={mode},hs={hidden_dim}, drop={drop}, th1={model.th1}, th2={model.th2} train acc: {np.mean(train_accs):.4f},{np.std(train_accs):.4f}, test acc: {np.mean(test_accs):.4f}, {np.std(test_accs):.4f}, p_norm: {np.mean(p_norms):.4f}, {np.std(p_norms):.4f}')
                    # save into csv file
                    results = pd.DataFrame(
                        {'train_acc': train_accs, 'test_acc': test_accs, 'p_norm': p_norms})
                    # if mode==0:
                    #     results.to_csv(
                    #         f'./results/{DATA_SET}_{SPLIT}_hs_{hidden_dim}_drop_{drop}_GCN.csv')
                    # if mode==1:
                    #     results.to_csv(f'./results/{DATA_SET}_{SPLIT}_hs_{hidden_dim}_drop_{drop}_MLP.csv')
                    # if mode==2:
                    #     results.to_csv(
                    #     f'./results/{DATA_SET}_{SPLIT}_hs_{hidden_dim}_drop_{drop}_th1_{th1}_th2_{th2}.csv')
                        

            print('-----------------')
