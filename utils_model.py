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
from sklearn import datasets
from sklearn.model_selection import train_test_split
device = torch.device(
    "cuda:1") if torch.cuda.is_available() else torch.device("cpu")

def save_model(model, path):
    torch.save(model.state_dict(), path)
    
def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model

def load_dataset_1(train_size=0.2, dataset='Cora'):
    path = osp.join(osp.dirname(osp.realpath(__file__)),
                    '..', 'data', 'Planetoid')
    from torch_geometric.datasets import Planetoid
    dataset = Planetoid(path, name=dataset, split='random')
    data = dataset[0]
    x=data.x
    y=data.y
     # normalize x
    # x=(x-x.mean(axis=0))/(x.std())
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size,stratify=y)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.2)
    x_train = torch.tensor(x_train, dtype=torch.float).to(device)
    x_val= torch.tensor(x_val, dtype=torch.float).to(device)
    x_test = torch.tensor(x_test, dtype=torch.float).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    y_val= torch.tensor(y_val, dtype=torch.long).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)
    #  create dataloader
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    test_dataset = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    print('train size:', x_train.shape)
    print('test_size:', x_test.shape)
    return train_loader, val_loader, test_loader, x.to(device) 

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

def fit_iris_emb(model, data, optimizer, crit_mse):
    '''
    train 1 epoch
    '''
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    cloest_centers = model.fit(data)
    
    loss=crit_mse(cloest_centers, data)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()


    return total_loss

def train_iris_emb(model, data_loader, optimizer, crit):
    '''
    train 1 epoch
    '''
    model.train()
    total_loss = 0
    i = 0
    optimizer.zero_grad()
    for data in data_loader:
        x,y=data
        pred, p_norm = model(x)
        loss=crit(pred, y)
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
        pred, p_norm = model(test_x)
        loss = crit(pred, test_y)
        total_loss+=loss
        y.append(test_y)
        pred_y.append(pred)
    y=torch.cat(y)
    pred_y=torch.cat(pred_y)
    acc=(pred_y.argmax(dim=-1) == y).sum().float()/len(y)
    f1= f1_score(y, pred_y.argmax(dim=-1),7)
    # save pred into pt file
    # torch.save(pred, f'./results/{DATA_SET}_pred_Infer.pt')
    
    return loss, acc.cpu(), p_norm.cpu(), f1
    
class GNN(nn.Module):
    def __init__(self, feature_dim, hidden_dim, out_dim, create_graph, drop, th1, th2, mode=0, saved_graph=None):
        '''mode=0: use gcn, mode=1: use MLP, mode=2: use InferGraph'''
        super(GNN, self).__init__()
        self.nn1 = nn.Linear(feature_dim, hidden_dim)
        self.nn2 = nn.Linear(hidden_dim, hidden_dim)
        self.nn3 = nn.Linear(hidden_dim, hidden_dim)
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
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

    def forward(self, data, x=None, Debug=False):
        data_x, edge, attr = data.x, data.edge_index, data.edge_attr
 

        if self.l_sys is None and self.create_graph and self.saved_graph is None:
            self.a_norm, self.l_sys = create_infer_graph_from_x(data_x, self.th1, self.th2)
            self.a_norm.requires_grad=False
            self.l_sys.requires_grad=False
    

        elif self.l_sys is None:
            self.l_sys = self.saved_graph
        
        if self.mode==0:
            out1=F.relu(F.dropout(self.gcn1(data_x, edge),self.drop, training=self.training))
            # out1=self.bn1(out1)
            out2=F.relu(F.dropout(self.gcn2(out1, edge),self.drop, training=self.training))
            # out2=self.bn2(out2)
        
        if self.mode==1:
            out1 = F.relu(F.dropout((self.nn1(data_x)), self.drop,
                                    training=self.training))
            # out1=self.bn1(out1)
            out2 = F.relu(F.dropout((self.nn2(out1)), self.drop,
                                    training=self.training))
            # out2=self.bn2(out2)
        if self.mode==2:
            I=torch.eye(data_x.shape[0], device=device)
            out1 = F.relu(F.dropout((self.a_norm)@(self.nn1(data_x)), self.drop,
                               training=self.training))
            # out1=self.bn1(out1)
            out2 = F.relu(F.dropout(self.l_sys@self.nn2(out1), self.drop,
                               training=self.training))
            # out2=self.bn2(out2)
            
            # out3 = F.relu(F.dropout((self.a_norm@self.nn3(out2)), self.drop,
            #                    training=self.training))
            # out3=self.bn3(out3)
        
       
        if self.create_graph or self.saved_graph is not None :
            p_norm = calc_p_dirichlet(data_x, self.l_sys)
        else:
            p_norm = torch.tensor(0)
        # out = F.dropout(F.relu(self.nn(data_x)),0.2, training=self.training)
        # out1=F.relu(self.nn1(L@out))
        # out2=F.dropout(F.relu(self.gcn1(data_x, edge)), self.drop,
        #                        training=self.training) #acc 0.7580
        # concat the two outputs
        # out=torch.concat((out1, out2), dim=1)
        out = self.clf(out2+out1)

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
        # initialize the centroids weight
        self.centroids.weight.data=self.centroids.weight*0.001
        
    def fit(self, data):
        # train the embedding layer
         # find the closest centroids index
        distance=torch.cdist(data, self.centroids.weight, p=2)
        _, idx=torch.min(distance, dim=1)
        closest_centers=self.centroids.weight[idx]
        return closest_centers
            
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
            # concat data and ref_data
            data=torch.concat((data, self.centroids.weight.detach()), dim=0)
            
            # if self.l_sys is None and self.create_graph and self.saved_graph is None:
            #     self.a_norm, self.l_sys = create_infer_graph_from_x(data, self.th1, self.th2, used_data_size)
                
            # elif self.l_sys is None:
            #     self.l_sys = self.saved_graph
            self.a_norm, self.l_sys = create_infer_graph_from_x(data, self.th1, self.th2, used_data_size, not self.training)

                
            out1 = F.relu(F.dropout(self.l_sys@self.a_norm@(self.nn1(data)), self.drop,
                               training=self.training))
            out2 = F.relu(F.dropout((self.nn2(out1)), self.drop,
                               training=self.training))
            
            p_norm = calc_p_dirichlet(out2, self.l_sys)
            out2=out2[:used_data_size]
        
            

        out = self.clf(out2)
    

        return out, p_norm
    
    def plot_centroinds(self, train_loader, number_of_clusters=3):     
        # plot the centroids
        x_centers=self.centroids.weight.cpu().detach()
        center_size=self.num_centroids
        
        # all_data=torch.concat(( x_centers,x), dim=0)
        pca_model = PCA(n_components=2).fit(x_centers.numpy() )
        pca_data=pca_model.transform(x_centers.numpy())
        plt.clf()
        # highlight the centers
        for i in range(center_size):
            plt.scatter(pca_data[i, 0], pca_data[i, 1], s=100, c='red', marker='o', alpha=0.5)
            
        for x,y in train_loader:
            x=x.cpu().detach()
            y=y.cpu().detach()
            
            pca_data=pca_model.transform(x.numpy())
            # color the data points using y
            colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'black']
            for i in range(number_of_clusters): # this is the cluster size
                plt.scatter(pca_data[:, 0][y == i], pca_data[:, 1][y == i], s=20, label=f'Class {i}', color=colors[i])
        # plt.legend()
        path = osp.join(osp.dirname(osp.realpath(__file__)), '.',
                        'graph_figures', f'centroids.png')
        plt.savefig(path)
        plt.clf()
        return path
    
    
# create GNN4 model extends GNN; create a emb layer to represent the cluster centroids
# based on GNN3, but use  m masks to generate more centers (mXc)
class GNN4(GNN3):
    def __init__(self, feature_dim, hidden_dim, out_dim, create_graph, drop, th1, th2, mode=0, 
                 saved_graph=None, num_centroids=3, m=3):
        super(GNN3, self).__init__(feature_dim, hidden_dim, out_dim, create_graph, drop, th1, th2, mode, saved_graph)
        
        self.m = m # number of k-means kernals
        self.num_centroids= num_centroids # number of centers for each k-means
        # self.centroids=[ Embedding(self.num_centroids, self.kernal_features, sparse=False) for _ in range(self.m)]
        # initialize the centroids weight
        self.centroids=nn.Parameter(torch.Tensor(self.m, self.num_centroids, feature_dim))
        self.nn1_v4=[]
        self.nn2_v4=[]
        self.gcn1_v4=[]
        self.gcn2_v4=[]
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        # self.centroids_nn=[]
        for _ in range(self.m):
            # self.centroids_nn.append(nn.Linear(self.kernal_features, 1024).to(device))
            self.nn1_v4.append(nn.Linear(feature_dim, hidden_dim).to(device))
            # self.nn1_v4.append(nn.Linear(hidden_dim, hidden_dim).to(device))
            self.nn2_v4.append(nn.Linear(hidden_dim, hidden_dim).to(device))
            # self.nn2_v4.append(nn.Linear(hidden_dim, hidden_dim).to(device))
            self.gcn1_v4.append(geom_nn.GCNConv(feature_dim, hidden_dim).to(device))
            self.gcn2_v4.append(geom_nn.GCNConv(hidden_dim, hidden_dim).to(device))
        self.nn1_v4=nn.ModuleList(self.nn1_v4) # Need this otherwise the parameters are not reconized by the optimizer
        self.nn2_v4=nn.ModuleList(self.nn2_v4)
        self.gcn1_v4=nn.ModuleList(self.gcn1_v4)
        self.gcn2_v4=nn.ModuleList(self.gcn2_v4)
        # self.centroids_nn=nn.ModuleList(self.centroids_nn)
        self.clf = nn.Linear(hidden_dim*self.m, out_dim)
        
        # self.reset_parameters()

    # def reset_parameters(self):
        # set the centroids weight
        # torch.nn.init.xavier_uniform_(self.centroids)
        # self.kernal_idxs=[torch.randperm(self.feature_dim,device=device)[:self.kernal_features] for _ in range(self.m)]

        # check initialization
        # print('kernal sample index shape', self.kernal_idxs[0].shape)
        # print('centroids [0] shape', self.centroids[0].weight.shape)
        
        
        
    def fit(self, data, optimizer, crit_mse, epochs=100):
        self.train()
        total_loss=0
        loss_history=[]
        for _ in range(epochs):
            loss=None
            total_loss=0
               
            for i in range(self.m):
                optimizer.zero_grad()
                mask=self.kernal_idxs[i]
                distance=torch.cdist(self.centroids_nn[i](data[:, mask]),self.centroids_nn[i](self.centroids[i]), p=2)
                _, idx=torch.min(distance, dim=1)

                closest_centers=self.centroids[i][idx]
                # if loss is None:
                loss=crit_mse(closest_centers, data[:, mask])
                # print(distance.max(),loss.item())
                loss.backward()
                # clip the grad
                # torch.nn.utils.clip_grad_norm_(self.centroids[i], 10)

                optimizer.step()
                total_loss += loss.item()
            loss_history.append(total_loss)
        # print(' centroids weight sum', torch.sum(self.centroids))
        # calcualte dynamic th1
        # self.th1=[]
        # for i in range(self.m):
        #     centers=self.centroids[i].detach()
        #     # centers_norm=torch.norm(centers, p=2, dim=1).unsqueeze(1)
        #     # cos_centers=centers@(centers.T)/(centers_norm@centers_norm.T+1e-6)
        #     cos_centers=centers@(centers.T)
        #     # set the diagonal to 0
        #     cos_centers.diagonal().fill_(0)
        #     th1=cos_centers.max()
        #     self.th1.append(th1)

        return loss_history
    
    def fit1(self, train_loader, optimizer, crit_mse, epochs=100):
        all_x=[]
        for data in train_loader:
            x, y =data
            all_x.append(x)
        all_x=torch.concat(all_x)
        all_x=all_x # only use first 140 data points to fit the centroids
        all_x=all_x.unsqueeze(0)
        all_x=all_x[:,:self.num_centroids]
        params=[]
        for i in range(self.m):
            params.append(all_x)
        params=torch.concat(params)
        # use all_x as self.centroids
        self.centroids=nn.Parameter(params,requires_grad=False)
        # print('centroids shape', self.centroids.shape)
        return 0
            
    def forward(self, data):
        # ref_data is used to create the graph
        used_data_size=data.shape[0]
        
        if self.mode==0:
            outs=[]
            for i in range(self.m):
                # mask=self.kernal_idxs[i]
                out1 = F.relu(F.dropout((self.nn1_v4[i](data[:])), self.drop,
                                        training=self.training))
                # out1=self.bn1(out1)
                out2 = F.relu(F.dropout((self.nn2_v4[i](out1)), self.drop,
                                        training=self.training))
                # batch norm out2
                # out2=self.bn2(out2)
                outs.append(out2)
        
            outs=torch.concat(outs,dim=1)
            out = self.clf(outs)
            p_norm = torch.tensor(0)
            
        if self.mode==1:
            # concat data and ref_data
            outs=[]
            p_norms=[]
            for i in range(self.m):
                # mask=self.kernal_idxs[i]
                # calculate th1
                centers=self.centroids[i].detach()
                temp=torch.concat((data[:], centers), dim=0)
                # th1=self.th1[i]
                
                self.a_norm, self.l_sys = create_infer_graph_from_x(temp, self.th1, self.th2, used_data_size, not self.training)
         
                # pick the best th based on p_norm
                # th_best=0
                # p_norm_min=100000
                # for th in [0.1, 0.2, 0.3,0.4,0.5]:
                #     a_norm, l_norm= create_infer_graph_from_x(temp, th, th, used_data_size, not self.training)
                #     p_norm = calc_p_dirichlet(temp, l_norm)
                #     if p_norm<p_norm_min:
                #         p_norm_min=p_norm
                #         th_best=th
                        
                #         self.a_norm, self.l_sys = a_norm, l_norm
                # print('best th',th, p_norm)
                # t=self.a_norm+self.l_sys
                # self.a_norm=t
                # self.l_sys=t
                
                out1 = F.relu(F.dropout((self.a_norm@self.nn1_v4[i](temp)), self.drop,
                                training=self.training))
                # out1=self.bn1(out1) # bn not improve performance when adam optimizer is used.
                # out1 = F.relu(F.dropout((self.l_sys@self.nn1_v4[i+1](out1)), self.drop,
                #                 training=self.training))
                out2 = F.relu(F.dropout((self.l_sys@self.nn2_v4[i](out1)), self.drop,
                                training=self.training))
                # out2=self.bn2(out2)
                # out2 = F.relu(F.dropout((self.a_norm@self.nn2_v4[i+1](out2)), self.drop,
                #                 training=self.training))
    
   
                p_norm = calc_p_dirichlet(temp, self.l_sys)

                out2=out2[:used_data_size]

                outs.append(out2)
                p_norms.append(p_norm)
                
    
            outs=torch.concat(outs,dim=1)

            out = self.clf(outs)
            
            p_norms=torch.stack(p_norms)
            p_norm=torch.mean(p_norms)
            
        return out, p_norm
    
    def plot_centroinds(self, train_loader, number_of_clusters=3):     
        # plot the centroids
        x_centers=self.centroids_nn[0](self.centroids[0]).detach().cpu().numpy()
        # fill nan with 0
        x_centers[x_centers!=x_centers]=0
        center_size=self.num_centroids
        
        # all_data=torch.concat(( x_centers,x), dim=0)
        pca_model = PCA(n_components=2).fit(x_centers )
        pca_data=pca_model.transform(x_centers)
        plt.clf()
        # highlight the centers
        for i in range(center_size):
            plt.scatter(pca_data[i, 0], pca_data[i, 1], s=100, c='red', marker='o', alpha=0.5)
            
        for x,y in train_loader:
            mask=self.kernal_idxs[0]
            x=self.centroids_nn[0](x[:,mask]).cpu().detach()
            y=y.cpu().detach()
            
            pca_data=pca_model.transform(x)
            # color the data points using y
            colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'black']
            for i in range(number_of_clusters): # this is the cluster size
                plt.scatter(pca_data[:, 0][y == i], pca_data[:, 1][y == i], s=20, label=f'Class {i}', color=colors[i])
        # plt.legend()
        path = osp.join(osp.dirname(osp.realpath(__file__)), '.',
                        'graph_figures', f'centroids.png')
        plt.savefig(path)
        plt.clf()
        return path
    
    
        
# create GNN4 model extends GNN; create emb layer use encoder/deconder style
class GNN5(GNN3):
    def __init__(self, feature_dim, hidden_dim, out_dim, create_graph, drop, th1, th2, mode=0, 
                 saved_graph=None, num_centroids=3):
        super(GNN3, self).__init__(feature_dim, hidden_dim, out_dim, create_graph, drop, th1, th2, mode, saved_graph)
        
        self.num_centroids= num_centroids # number of centers for each k-means
        self.nn1_v4=[]
        self.nn2_v4=[]
        self.gcn1_v4=[]
        self.gcn2_v4=[]
        self.encoder=[]
        self.decoder=[]
        self.bn = nn.BatchNorm1d(hidden_dim)
     
        self.encoder.append(nn.Linear(feature_dim, hidden_dim).to(device))
        self.encoder.append(nn.Linear(hidden_dim, hidden_dim).to(device))
        self.decoder.append(nn.Linear(hidden_dim, hidden_dim).to(device))
        self.decoder.append(nn.Linear(hidden_dim, feature_dim).to(device))
        
        
        self.nn1_v4.append(nn.Linear(feature_dim, hidden_dim).to(device))
        # self.nn1_v4.append(nn.Linear(hidden_dim, hidden_dim).to(device))
        self.nn2_v4.append(nn.Linear(hidden_dim, hidden_dim).to(device))
        # self.nn2_v4.append(nn.Linear(hidden_dim, hidden_dim).to(device))
        self.nn1_v4=nn.ModuleList(self.nn1_v4) # Need this otherwise the parameters are not reconized by the optimizer
        self.nn2_v4=nn.ModuleList(self.nn2_v4)
       
        self.encoder=nn.ModuleList(self.encoder)
        self.decoder=nn.ModuleList(self.decoder)
        self.clf = nn.Linear(hidden_dim, out_dim)
        self.clf_encoder=nn.Linear(feature_dim, out_dim)
        
        # self.gcn1_v4.append(geom_nn.GCNConv(feature_dim, hidden_dim,add_self_loops=True, improved=True).to(device))
        # self.gcn2_v4.append(geom_nn.GCNConv(hidden_dim, hidden_dim, add_self_loops=True, improved=True).to(device))
        # self.gcn1_v4.append(geom_nn.SAGEConv(feature_dim, hidden_dim).to(device))
        # self.gcn2_v4.append(geom_nn.SAGEConv(hidden_dim, hidden_dim).to(device))
        # self.gcn1_v4.append(geom_nn.GraphConv(feature_dim, hidden_dim).to(device)) # not working
        # self.gcn2_v4.append(geom_nn.GraphConv(hidden_dim, hidden_dim).to(device)) # not working
        self.gcn1_v4.append(geom_nn.ChebConv(feature_dim, hidden_dim, K=3).to(device))
        self.gcn2_v4.append(geom_nn.ChebConv(hidden_dim, hidden_dim, K=3).to(device))
        # self.gcn1_v4.append(geom_nn.GATConv(feature_dim, hidden_dim).to(device)) #not working
        # self.gcn2_v4.append(geom_nn.GATConv(hidden_dim, hidden_dim).to(device))  #not working
        self.gcn1_v4=nn.ModuleList(self.gcn1_v4)
        self.gcn2_v4=nn.ModuleList(self.gcn2_v4)
        
        
        
    def fit(self, train_loader, optimizer, crit_mse,crit_c, epochs=100):
        self.train()
        total_loss=0
        loss_history=[]
        all_x=[]
        for data in train_loader:
            x, y =data
            all_x.append(x)
        all_x=torch.concat(all_x)
        all_x=all_x[:self.num_centroids]
        self.centroids=nn.Parameter(all_x,requires_grad=False)
        for _ in range(epochs):
            loss=None
            total_loss=0
               
            optimizer.zero_grad()
            for data in train_loader:
                x, y =data
                x=F.dropout(F.relu(self.encoder[0](x)),self.drop, training=self.training)
                x=F.dropout(F.relu(self.encoder[1](x)),self.drop, training=self.training)
                
                out=F.dropout(F.relu(self.decoder[0](x)),self.drop, training=self.training)
                out=self.decoder[1](out)
                
                pred_c=self.clf_encoder(out)
                loss=crit_mse(out, data[0]) # reconstruction x vs x_hat
                # loss_c=crit_c(pred_c, y)    # reconstruction prediction on classification y vs y_hat
                # loss=loss+loss_c # remove, it only has 0.5 acc
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            loss_history.append(total_loss)
        # don't modify encoder/decoder's weight
        # self.encoder[0].requires_grad=False
        # self.encoder[1].requires_grad=False
        # self.decoder[0].requires_grad=False
        # self.decoder[1].requires_grad=False
        return loss_history

            
    def forward(self, data):
        # ref_data is used to create the graph
        used_data_size=data.shape[0]
        
        if self.mode==0:
            outs=[]
            out1 = F.relu(F.dropout((self.nn1_v4[0](data[:])), self.drop,
                                    training=self.training))
            out1=self.bn(out1)
            out2 = F.relu(F.dropout((self.nn2_v4[0](out1)), self.drop,
                                    training=self.training))
            outs.append(out2)
        
            outs=torch.concat(outs,dim=1)
            out = self.clf(outs)
            p_norm = torch.tensor(0)
            
        if self.mode==1:
            # concat data and ref_data
            centers=self.centroids.detach()
            temp=torch.concat((data, centers), dim=0)

            encoded=F.relu(self.encoder[0](temp))
            encoded=F.relu(self.encoder[1](encoded))
            reconstructed=F.relu(self.decoder[0](encoded))
            reconstructed=self.decoder[1](reconstructed)
            
            # temp=torch.concat((data, reconstructed), dim=0)
            self.l_sys = create_infer_graph_from_x_v5(reconstructed, self.th1, self.th2, used_data_size, not self.training)
            
            # out1 = F.relu(F.dropout((self.l_sys@self.nn1_v4[0](temp)), self.drop,
            #                 training=self.training))
            # convert dense to edge_index
            edge_index, weight=torch_geometric.utils.dense_to_sparse(self.l_sys)
            out1 = F.relu(F.dropout((self.gcn1_v4[0](temp,edge_index, weight)), self.drop,
                            training=self.training))
            
            # out1=self.bn(out1)
      
            out2 = F.relu(F.dropout((self.gcn2_v4[0](out1,edge_index, weight)), self.drop,
                            training=self.training))


            # p_norm = calc_p_dirichlet(reconstructed, self.l_sys)
            p_norm = torch.tensor(0)
            out2=out2[:used_data_size]
            out = self.clf(out2)

            
        return out, p_norm
    
