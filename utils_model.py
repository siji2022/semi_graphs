# util files for dataset analysis
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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


def train(model, data, optimizer, crit,  Debug=False):
    '''
    train 1 epoch
    '''
    model.train()
    total_loss = 0
    i = 0

    optimizer.zero_grad()

    adj, pred, p_norm = model(data, Debug=Debug)
    device=data.x.device
    loss = crit(pred[data.train_mask], data.y[data.train_mask].to(
        device))

    loss.backward()
    optimizer.step()
    total_loss += loss.item()
    i += 1
    return total_loss/data.train_mask.sum()

def train_iris(model, data_loader, optimizer, crit):
    '''
    train 1 epoch
    '''
    model.train()
    total_loss = 0
    i = 0
    optimizer.zero_grad()
    for data in data_loader:
        x,y=data
        _, pred, p_norm = model(x)
        loss=crit(pred, y)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        i += 1

    return total_loss/len(data_loader)

def train_iris_emb(model, data_loader, optimizer, crit, crit_mse):
    '''
    train 1 epoch
    '''
    model.train()
    total_loss = 0
    i = 0
    optimizer.zero_grad()
    for data in data_loader:
        x,y=data
        (cloest_centers, distance), pred, p_norm = model(x)
        loss=crit(pred, y)

        loss+=crit_mse(cloest_centers, x)
        # loss+=crit_mse(distance, torch.zeros_like(distance))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        i += 1

    return total_loss/len(data_loader)

@torch.no_grad()
def test(model, data, labels,  mask, crit):
    model.eval()
    # return acc
    _, pred, p_norm = model(data)
    pred = pred.cpu()
    p_norm = p_norm.cpu()
    loss = crit(pred[mask], labels[mask])
    # if DEBUG:
    #     print(pred[mask].argmax(dim=-1)[:10], labels[mask][:10])
    # calculate accuracy
    acc = ((pred[mask].argmax(dim=-1) == labels[mask]
            ).sum().float() / mask.sum()).cpu()
    # save pred into pt file
    # torch.save(pred, f'./results/{DATA_SET}_pred_Infer.pt')
    
    return loss, acc, p_norm

@torch.no_grad()
def test_iris(model, test_data_loader, train_data_loader, crit):
    model.eval()
    # check implimentation 072924
    test_x=[]
    test_y=[]
    ref_x=[]
    
    for test_data in test_data_loader:
        t_x, t_y = test_data
        test_x.append(t_x)
        test_y.append(t_y)
    if train_data_loader is not None:
        ref_data = next(iter(train_data_loader))
        for train_data in train_data_loader:
            t_x, _ = train_data
            ref_x.append(t_x)
    test_x=torch.cat(test_x)
    test_y=torch.cat(test_y)
    if train_data_loader is not None:
        ref_x=torch.cat(ref_x)


    # return acc
    if train_data_loader is not None:
        _, pred, p_norm = model(test_x, ref_x)
    else:
        _, pred, p_norm = model(test_x, None)
    loss = crit(pred, test_y)
    # pred = pred.cpu()
    # p_norm = p_norm.cpu()
    acc=(pred.argmax(dim=-1) == test_y).sum().float()/len(test_y)
    # save pred into pt file
    # torch.save(pred, f'./results/{DATA_SET}_pred_Infer.pt')
    
    return loss, acc.cpu(), p_norm.cpu()


@torch.no_grad()
def test_iris_emb(model, test_data_loader, crit):
    model.eval()
    
    y=[]
    pred_y=[]
    total_loss=0
    for test_data in test_data_loader:
        test_x, test_y = test_data
        _, pred, p_norm = model(test_x)
        loss = crit(pred, test_y)
        total_loss+=loss
        y.append(test_y)
        pred_y.append(pred)
    y=torch.cat(y)
    pred_y=torch.cat(pred_y)
    acc=(pred_y.argmax(dim=-1) == y).sum().float()/len(y)
    # save pred into pt file
    # torch.save(pred, f'./results/{DATA_SET}_pred_Infer.pt')
    
    return loss, acc.cpu(), p_norm.cpu()
    
class GNN(nn.Module):
    def __init__(self, feature_dim, hidden_dim, out_dim, create_graph, drop, th1, th2, mode=0, saved_graph=None):
        '''mode=0: use gcn, mode=1: use MLP, mode=2: use InferGraph'''
        super(GNN, self).__init__()
        self.nn1 = nn.Linear(feature_dim, hidden_dim)
        self.nn2 = nn.Linear(hidden_dim, hidden_dim)
        self.gcn1 = geom_nn.GCNConv(feature_dim, hidden_dim)
        self.gcn2 = geom_nn.GCNConv(hidden_dim, hidden_dim)
        self.clf = nn.Linear(hidden_dim, out_dim)

        self.l_sys = None
        self.drop = drop
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.th1 = th1
        self.th2 = th2
        self.create_graph = create_graph
        self.saved_graph=saved_graph
        self.mode=mode

    def forward(self, data, x=None, Debug=False):
        data_x, edge, attr = data.x, data.edge_index, data.edge_attr
 

        if self.l_sys is None and self.create_graph and self.saved_graph is None:
            self.l_sys, self.l_for_energy = create_infer_graph_from_x(data_x, self.th1, self.th2)
            self.adj = self.l_sys
            # change adj matrix to adjancy list
            # edge = torch_geometric.utils.dense_to_sparse(adj)[0]
            # self.edge = edge  # used for gcn
        elif self.l_sys is None:
            self.l_sys = self.saved_graph
        

        # if Debug and self.create_graph:
        #     #     # print(f'D max: {D.max()}, D min: {D.min()}')
        #     # print(f'adj max: {adj.max()}, adj min: {adj. min()}')
        #     # plot adj distribution
        #     path = osp.join(osp.dirname(osp.realpath(__file__)), '.',
        #                     FIGURES_DIR, f'adj_hist.png')
        #     plt.clf()
        #     plt.hist(self.adj.flatten().detach().cpu().numpy(), bins=100)
        #     # use log on y axis
        #     plt.yscale('log')
        #     plt.savefig(path)
        #     # print(f'edge max: {edge.max()}, edge min: {edge.min()}')
        #     # print(L_sys[0][0], L_sys[1][1])
        #     print(
        #         f'emb max: {emb.max():3f}, decode min: {decode.min():3f}, bias: {self.bias.min():3f}, L_sys min: {L_sys.min():3f}')
        if self.mode==0:
            out1=F.relu(F.dropout(self.gcn1(data_x, edge),self.drop, training=self.training))
            out2=F.relu(F.dropout(self.gcn2(out1, edge),self.drop, training=self.training))
        
        if self.mode==1:
            out1 = F.relu(F.dropout((self.nn1(data_x)), self.drop,
                                    training=self.training))
            out2 = F.relu(F.dropout((self.nn2(out1)), self.drop,
                                    training=self.training))
        if self.mode==2:
            out1 = F.relu(F.dropout(self.l_sys@(self.nn1(data_x)), self.drop,
                               training=self.training))
            out2 = F.relu(F.dropout((self.l_sys@self.nn2(out1)), self.drop,
                               training=self.training))
        
       
        if self.create_graph or self.saved_graph is not None :
            p_norm = calc_p_dirichlet(out2, self.l_for_energy)
        else:
            p_norm = torch.tensor(0)
        # out = F.dropout(F.relu(self.nn(data_x)),0.2, training=self.training)
        # out1=F.relu(self.nn1(L@out))
        # out2=F.dropout(F.relu(self.gcn1(data_x, edge)), self.drop,
        #                        training=self.training) #acc 0.7580
        # concat the two outputs
        # out=torch.concat((out1, out2), dim=1)
        out = self.clf(out2)

        return None, out, p_norm

    def __repr__(self):
        return '{}(hs={}, drop={})'.format(self.__class__.__name__, self.hidden_dim, self.drop)
    
    
# create GNN2 model extends GNN
class GNN2(GNN):
    def __init__(self, feature_dim, hidden_dim, out_dim, create_graph, drop, th1, th2, mode=0, saved_graph=None):
        super(GNN2, self).__init__(feature_dim, hidden_dim, out_dim, create_graph, drop, th1, th2, mode, saved_graph)
        

    def forward(self, data, ref_data=None):
        # ref_data is used to create the graph
        used_data_size=data.shape[0]
        
        
        if self.mode==0:
            out1 = F.relu(F.dropout((self.nn1(data)), self.drop,
                                    training=self.training))
            out2 = F.relu(F.dropout((self.nn2(out1)), self.drop,
                                    training=self.training))
            p_norm = torch.tensor(0)
            
        if self.mode==1:
            if ref_data is not None:
                # concat data and ref_data
                data=torch.concat((data, ref_data), dim=0)
            _, self.l_sys = create_infer_graph_from_x(data, self.th1, self.th2)

                
            out1 = F.relu(F.dropout(self.l_sys@(self.nn1(data)), self.drop,
                               training=self.training))
            out2 = F.relu(F.dropout((self.l_sys@self.nn2(out1)), self.drop,
                               training=self.training))
            
            p_norm = calc_p_dirichlet(out2, self.l_sys)
            out2=out2[:used_data_size]
        
            

        out = self.clf(out2)

        return None, out, p_norm
    
    
    
   
# create GNN3 model extends GNN; create a emb layer to represent the cluster centroids
class GNN3(GNN):
    def __init__(self, feature_dim, hidden_dim, out_dim, create_graph, drop, th1, th2, mode=0, saved_graph=None, num_centroids=3):
        super(GNN3, self).__init__(feature_dim, hidden_dim, out_dim, create_graph, drop, th1, th2, mode, saved_graph)
        self.num_centroids= num_centroids # inital to be the same as out_dim; but it can be different, thinking elbow methods in k-means
        self.centroids=Embedding(self.num_centroids, feature_dim, sparse=False)
        
        self.reset_parameters()

    def reset_parameters(self):
        self.centroids.reset_parameters()
        
    def forward(self, data):
        # ref_data is used to create the graph
        used_data_size=data.shape[0]
        
        
        if self.mode==0:
            out1 = F.relu(F.dropout((self.nn1(data)), self.drop,
                                    training=self.training))
            out2 = F.relu(F.dropout((self.nn2(out1)), self.drop,
                                    training=self.training))
            p_norm = torch.tensor(0)
            
            
        if self.mode==1:
            # find the closest centroids index
            distance=torch.cdist(data, self.centroids.weight, p=2)
            _, idx=torch.min(distance, dim=1)
            closest_centers=self.centroids.weight[idx]
            

            
            # concat data and ref_data
            data=torch.concat((data, self.centroids.weight), dim=0)
            self.a_norm, self.l_sys = create_infer_graph_from_x(data, self.th1, self.th2, used_data_size)

                
            out2 = F.relu(F.dropout(self.l_sys@(self.nn1(data)), self.drop,
                               training=self.training))
            # out2 = F.relu(F.dropout((self.l_sys@self.nn2(out1)), self.drop,
            #                    training=self.training))
            
            p_norm = calc_p_dirichlet(out2, self.l_sys)
            out2=out2[:used_data_size]
        
            

        out = self.clf(out2)
        if self.mode==0:
            closest_centers=torch.zeros_like(data)
            distance=torch.zeros_like(data)
    

        return (closest_centers,distance), out, p_norm
    
    def plot_centroinds(self, train_loader):
        # get all data from train_loader
        # x, y = next(iter(train_loader))
        # x=x.cpu().detach()
        # y=y.cpu().detach()
     
        # plot the centroids
        x_centers=self.centroids.weight.cpu().detach()
        center_size=self.num_centroids
        
        # all_data=torch.concat(( x_centers,x), dim=0)
        pca_model = PCA(n_components=2).fit(x_centers.numpy() )
        pca_data=pca_model.transform(x_centers.numpy())
        plt.clf()
        # highlight the centers
        for i in range(center_size):
            plt.scatter(pca_data[i, 0], pca_data[i, 1], s=100, c='red', marker='o')
            
        for x,y in train_loader:
            x=x.cpu().detach()
            y=y.cpu().detach()
            
            pca_data=pca_model.transform(x.numpy())
            # color the data points using y
            colors = ['red', 'green', 'blue']
            for i in range(3):
                plt.scatter(pca_data[:, 0][y == i], pca_data[:, 1][y == i], s=20, label=f'Class {i}', color=colors[i])
        # plt.legend()
        path = osp.join(osp.dirname(osp.realpath(__file__)), '.',
                        'graph_figures', f'centroids.png')
        plt.savefig(path)
        
        return path