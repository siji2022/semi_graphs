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
DATA_SET = 'Iris'

LOAD_MODEL = False
EPS = 1e-15
FIGURES_DIR = 'graph_figures'
from sklearn import datasets
from sklearn.model_selection import train_test_split

device = torch.device(
    "cuda:1") if torch.cuda.is_available() else torch.device("cpu")
Training_EPS = 100
DEBUG = False






def run_experiment(model, train_loader, val_loader, test_loader, x):
    optimizer = torch.optim.Adam(list(model.parameters()), lr=0.01)
    train_losss = []
    train_accs = []
    test_accs = []

    crit_mse=nn.MSELoss()
    fit_losss=[]
    if model.mode==1:
        fit_losss = model.fit1(train_loader, optimizer, crit_mse, 100)
        # path=osp.join(osp.dirname(osp.realpath(__file__)),'.',FIGURES_DIR,'iris_emb_fit_loss.png')
        # plt.plot(fit_losss)
        # plt.savefig(path)

        
    for epoch in range(1, Training_EPS+1):
        crit_c = nn.CrossEntropyLoss()
        
        if epoch % 100 == 0:
            DEBUG = True
        else:
            DEBUG = False
        
        train_loss = train_iris_emb(model, train_loader, optimizer, crit_c)
        loss, train_acc, p_norm = test_iris_emb(model, train_loader, crit_c)
        loss, test_acc, p_norm = test_iris_emb(model, test_loader, crit_c)
        # if DEBUG:
        #     print(
        #         f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, p_norm: {p_norm:.4f}')

        train_losss.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        # if epoch%10==0:
        #     model.plot_centroinds(train_loader)

    # plot 2 y axis
    plt.figure(figsize=(10, 5))
    fig, ax1 = plt.subplots()
    # ax2=ax1.twinx()
    # ax2.plot(train_losss, label='train loss')
    ax1.plot(train_accs, label='train acc')
    ax1.plot(test_accs, label='test acc')
    ax1.legend(loc='best')
    # ax2.legend(loc='best')
    plt.grid()
    plt.title(f'train loss and acc')
    path = osp.join(osp.dirname(osp.realpath(__file__)),
                    '.', FIGURES_DIR, 'iris_emb_train_loss_acc_my.png')
    plt.savefig(path)
    plt.clf()
    return np.max(train_accs), np.max(test_accs), p_norm
    # plot loss and acc on two y axis
    # save model

    # torch.save(model.state_dict(), model_path)


# for walk_length in [ 5,7,11,19, 23, 29]:


hidden_dim = 64
th2 = 0.2
th1=0.1
# DATA_SET='Cora'
DATA_SET='Citeseer'
train_size=140

m=1
for train_size in [0.1,]:
# for train_size in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
# for train_size in [140,210,280,350,630,840,1050,1260 ]:
# for train_size in [840 ]:
    train_loader, val_loader, test_loader, x  = load_dataset_1(train_size, DATA_SET)
    num_centroids=int(train_size*2708)
    num_centroids=140
    for mode in [0,1]:
        for drop in [0.2]:
        # for drop in [0, 0.2, 0.5]:
            if mode==1:
                # th1_list=[0.5]
                th1_list=[1]
                th2_list=[0.2]
            else:
                th1_list=[0]
                th2_list=[0]
            for th1 in th1_list:  
                for th2 in th2_list:
                    train_accs = []
                    test_accs = []
                    p_norms = []
                    RUN_TIMES = 1
                    
                    for i in range(RUN_TIMES):
                        
                        model = GNN4(feature_dim=x.shape[1],out_dim=7, hidden_dim=hidden_dim, create_graph=True,
                                    drop=drop, th1=th1, th2=th2, mode=mode, saved_graph=None,num_centroids=num_centroids,
                                    m=m).to(device)
                        train_acc, test_acc, p_norm = run_experiment(model, train_loader,val_loader, test_loader, x)
                        
                        # model.plot_centroinds(train_loader, 7)
                        
                        train_accs.append(train_acc.item())
                        test_accs.append(test_acc.item())
                        p_norms.append(p_norm.item())
                        # clear nvidia memory
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                        
                        # save model
                        path=osp.join(osp.dirname(osp.realpath(__file__)),
                                    '.', 'models',f'v4_{DATA_SET}_mode_{mode}_{train_size}_hs_{hidden_dim}_drop_{drop}_th1_{th1}_th2_{th2}_iter_{i}.pt')
                        # save_model(model, path)
                    print(f'completed {RUN_TIMES},train_size={train_size}, mode={mode}, hs={hidden_dim}, drop={drop}, th1={th1}, th2={th2}, m={model.m}, num_centroids={model.num_centroids}, train acc: {np.mean(train_accs):.4f},{np.std(train_accs):.4f}, test acc: {np.mean(test_accs):.4f}, {np.std(test_accs):.4f}, p_norm: {np.mean(p_norms):.4f}, {np.std(p_norms):.4f}')
                    # # # save into csv file
                    results = pd.DataFrame(
                        {'train_acc': train_accs, 'test_acc': test_accs, 'p_norm': p_norms})
                    # if mode==0:
                    #     results.to_csv(
                    #         f'./results/{DATA_SET}_{train_size}_hs_{hidden_dim}_drop_{drop}_MLP.csv')
                    # if mode==1:
                    #     results.to_csv(
                    #     f'./results/{DATA_SET}_{train_size}_hs_{hidden_dim}_drop_{drop}_th1_{th1}_th2_{th2}_centroids_{num_centroids}.csv')
                    
                   
        print('-----------------')
