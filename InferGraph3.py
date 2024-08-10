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





def load_dataset(train_size=0.2):
    
    iris=datasets.load_iris()
    x=iris.data
    y=iris.target
    # normalize x
    x=(x-x.mean(axis=0))/x.std(axis=0)
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
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    print('train size:', x_train.shape)
    print('test_size:', x_test.shape)
    return train_loader, val_loader, test_loader, x_train 
    

    






def run_experiment(model, train_loader, val_loader, test_loader, x):
    
    optimizer = torch.optim.Adam(list(model.parameters()), lr=0.01)
    train_losss = []
    train_accs = []
    test_accs = []

    crit_mse=nn.MSELoss()
    fit_losss=[]
    for epoch in range(1, Training_EPS*2+1):
        fit_loss=fit_iris_emb(model, x, optimizer, crit_mse)
        fit_losss.append(fit_loss)
    plt.figure()
    plt.plot(fit_losss)
    plt.grid()
    plt.title('fit loss')
    path = osp.join(osp.dirname(osp.realpath(__file__)),
                    '.', FIGURES_DIR, 'iris_emb_fit_loss.png')
    plt.savefig(path)
    plt.clf()
        
    for epoch in range(1, Training_EPS+1):
        crit_c = nn.CrossEntropyLoss()
        
        if epoch % 100 == 0:
            DEBUG = True
        else:
            DEBUG = False
        
        train_loss = train_iris_emb(model, train_loader, optimizer, crit_c)
        loss, train_acc, p_norm, f1 = test_iris_emb(model, train_loader, crit_c)
        loss, test_acc, p_norm, f1 = test_iris_emb(model, test_loader, crit_c)
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


hidden_dim = 128
th2 = 1
DATA_SET='Cora'
# for train_size in [  0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
for train_size in [ 140]:
    train_loader, val_loader, test_loader, x  = load_dataset_1(train_size)
    for mode in [ 0,1]:
        for drop in [0.2]:
        # for drop in [0, 0.2, 0.5]:
            if mode==1:
                # th1_list=[0.5]
                th1_list=[0.15]
                center_list=[7]
            else:
                th1_list=[0]
                center_list=[3]
                
            for th1 in th1_list:  
                for num_centroids in center_list:      
                    train_accs = []
                    test_accs = []
                    p_norms = []
                    RUN_TIMES = 1
                    
                    for i in range(RUN_TIMES):
                        
                        model = GNN3(feature_dim=x.shape[1],out_dim=7, hidden_dim=hidden_dim, create_graph=True,
                                    drop=drop, th1=th1, th2=th2, mode=mode, saved_graph=None,num_centroids=num_centroids).to(device)
                        train_acc, test_acc, p_norm = run_experiment(model, train_loader,val_loader, test_loader, x)
                        
                        model.plot_centroinds(train_loader, 7)
                        
                        train_accs.append(train_acc.item())
                        test_accs.append(test_acc.item())
                        p_norms.append(p_norm.item())
                        # clear nvidia memory
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                        
                        # save model
                        path=osp.join(osp.dirname(osp.realpath(__file__)),
                                    '.', 'models',f'v3_{DATA_SET}_mode_{mode}_{train_size}_hs_{hidden_dim}_drop_{drop}_th1_{th1}_centroids_{num_centroids}.pt')
                        save_model(model, path)
                    print(f'completed {RUN_TIMES},train_size={train_size}, mode={mode}, hs={hidden_dim}, drop={drop}, th1={model.th1}, th2={model.th2} train acc: {np.mean(train_accs):.4f},{np.std(train_accs):.4f}, test acc: {np.mean(test_accs):.4f}, {np.std(test_accs):.4f}, p_norm: {np.mean(p_norms):.4f}, {np.std(p_norms):.4f}')
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
