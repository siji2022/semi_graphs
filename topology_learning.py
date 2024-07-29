# implement the topology learning algorithm based on Node2Vec
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geom_nn
import numpy as np
import matplotlib.pyplot as plt
import os

from torch.nn import Embedding
from torch.utils.data import DataLoader, TensorDataset
from torch_sparse import SparseTensor
import networkx as nx
import os.path as osp
import sys
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import to_dense_adj
from sklearn.linear_model import LogisticRegression

# suppress warnings
import warnings
warnings.filterwarnings("ignore")

try:
    import torch_cluster  # noqa
    random_walk = torch.ops.torch_cluster.random_walk
except ImportError:
    random_walk = None

# fix seed
# torch.manual_seed(0)
# np.random.seed(0)

LOAD_MODEL = False
EPS = 1e-15
FIGURES_DIR = 'graph_figures'
ADJ_SPASE_WEIGHT = 0
FEATURE_LOSS_WEIGHT = 1


Training_EPS = 200

DEBUG = False



class Node2Vec(torch.nn.Module):
    def __init__(self, N, embedding_dim, walk_length, context_size,
                 walks_per_node=1, p=1, q=1, num_negative_samples=1,
                 num_nodes=None, sparse=False):
        super(Node2Vec, self).__init__()

        if random_walk is None:
            raise ImportError('`Node2Vec` requires `torch-cluster`.')

        self.N = N

        assert walk_length >= context_size

        self.embedding_dim = embedding_dim
        self.walk_length = walk_length - 1
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.p = p
        self.q = q
        self.num_negative_samples = num_negative_samples

        self.embedding = Embedding(N, embedding_dim, sparse=False)
        # self.embedding = nn.Linear(N, embedding_dim, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()

    def forward(self, batch=None):
        """Returns the embeddings for the nodes in :obj:`batch`."""
        emb = self.embedding.weight

        return emb if batch is None else emb[batch]

    def __repr__(self):
        return '{}({}, {}, walk={}, context={}, walks_per_node={}, p={}, q={})'.format(self.__class__.__name__,
                                                                                       self.N, self.embedding_dim, self.walk_length+1, self.context_size, self.walks_per_node, self.p, self.q)


class TopDecoder(nn.Module):
    def __init__(self, N, embedding_dim, walk_length, context_size,
                 walks_per_node=1, p=1, q=1, num_negative_samples=1,
                 num_nodes=None, sparse=False):
        super(TopDecoder, self).__init__()
        self.encoder = Node2Vec(N, embedding_dim, walk_length, context_size,
                                walks_per_node, p, q, num_negative_samples,
                                num_nodes, sparse)
        dim = 128
        self.decoder = nn.Linear(embedding_dim, 64)
        # self.decoder=nn.Linear(1433,256)
        self.nn = nn.Linear(1433, dim)
        # self.nn1 = nn.Linear(128, 128)
        # self.gcn=geom_nn.GCNConv(1433, dim)
        self.clf = nn.Linear(dim, NUM_CLASSES)
        self.bias = nn.Parameter(torch.zeros(1))
        self.l_sys = None

    def forward(self, data, x=None, create_graph=True, Debug=False):
        data_x, edge, attr = data.x, data.edge_index, data.edge_attr
        emb = self.encoder()
        # decode = self.decoder(emb)
        # decode=emb
        decode=data_x
        decode = (decode@decode.T)

        # sigmoid

        if create_graph or self.l_sys is None:
            # adj = torch.sigmoid(decode+self.bias)
            adj = torch.sigmoid(decode).detach()
            # find the 50th percentile of adj as th1; 95th percentile as th2
            # th1=adj.flatten().kthvalue(int(adj.numel()*0.5)).values #0.73
            # th2=adj.flatten().kthvalue(int(adj.numel()*0.95)).values #0.95
            # print(th1, th2)
            th1=0.8
            th2=1.0
            
            adj[adj < th1] = 1.0
            adj[adj < th2] = 0.0
            I = torch.eye(adj.size(0)).to(device)
            # adj_self = torch.diag_embed(adj.diag()) # remove selfloop
            # adj = adj-adj_self
            # add self loop
            adj= adj + I
            D = torch.diag_embed(torch.sum(adj, dim=1))
            # normalize L, divide by D
            L = D-adj
            L_sys = torch.diag_embed(1.0/D.diag())@L
            # L_sys = torch.div(L, D.diag().reshape(-1, 1))
            # L_sys = torch.div(torch.div(L, D.diag().reshape(1, -1)), D.diag().reshape(-1, 1))
            self.l_sys = L_sys
            # no need to do back prop for l_sys
            # self.l_sys.requires_grad=False

        # D_inv=torch.inverse(D)
        # L_sym=D_inv@L@D_inv
        # edge=to_dense_adj(edge).squeeze(0).to(device)
        if Debug:
            #     # print(f'D max: {D.max()}, D min: {D.min()}')
            # print(f'adj max: {adj.max()}, adj min: {adj. min()}')
            # plot adj distribution
            path = osp.join(osp.dirname(osp.realpath(__file__)), '.',
                            FIGURES_DIR, f'adj_hist.png')
            plt.clf()
            plt.hist(L_sys.flatten().detach().cpu().numpy(), bins=100)
            # use log on y axis
            plt.yscale('log')
            plt.savefig(path)
        #     # print(f'edge max: {edge.max()}, edge min: {edge.min()}')
        #     # print(L_sys[0][0], L_sys[1][1])
        #     print(
        #         f'emb max: {emb.max():3f}, decode min: {decode.min():3f}, bias: {self.bias.min():3f}, L_sys min: {L_sys.min():3f}')

        out = F.relu(self.l_sys@F.dropout((self.nn(data_x)), 0.8,
                        training=self.training))  # 0.5300 X only
       
        p_norm = torch.sum((self.l_sys@data_x))
        # out = F.dropout(F.relu(self.nn(data_x)),0.2, training=self.training)
        # out1=F.relu(self.nn1(L@out))
        # out=F.relu(self.gcn(data_x, edge)) #acc 0.7580
        out = self.clf(out)

        return emb, adj, out, p_norm

    def __repr__(self):
        return self.encoder.__repr__()


# start data preparation


def create_synthetic_data():
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


def load_dataset():
    path = osp.join(osp.dirname(osp.realpath(__file__)),
                    '..', 'data', 'Planetoid')
    from torch_geometric.datasets import Planetoid
    dataset = Planetoid(path, name='Cora')
    data = dataset[0]
    edge_index = data.edge_index
    labels = data.y
    train_mask = data.train_mask
    test_mask = data.test_mask
    N = data.num_nodes
    print(N)

    return edge_index, labels, train_mask, test_mask, N, 7, data


device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def pos_sample(batch, adj, walks_per_node, walk_length, context_size, p, q):

    batch = batch.repeat(walks_per_node)

    rowptr, col, _ = adj.csr()

    rw = random_walk(rowptr, col, batch, walk_length, p, q)
    if not isinstance(rw, torch.Tensor):
        rw = rw[0]

    walks = []
    num_walks_per_rw = 1 + walk_length + 1 - context_size
    for j in range(num_walks_per_rw):
        walks.append(rw[:, j:j + context_size])
    return torch.cat(walks, dim=0)


def neg_sample(batch, adj, walks_per_node, num_negative_samples, walk_length, context_size):

    batch = batch.repeat(walks_per_node * num_negative_samples)

    rw = torch.randint(adj.sparse_size(0),
                       (batch.size(0), walk_length))
    rw = torch.cat([batch.view(-1, 1), rw], dim=-1)

    walks = []
    num_walks_per_rw = 1 + walk_length + 1 - context_size
    for j in range(num_walks_per_rw):
        walks.append(rw[:, j:j + context_size])
    return torch.cat(walks, dim=0)


def train_dataloader(N, edge_index, model, batch_size=4):
    adj = SparseTensor(row=edge_index[0],
                       col=edge_index[1], sparse_sizes=(N, N))
    adj = adj.to('cpu')
    # if not isinstance(batch_size, torch.Tensor):
    #     batch_size = torch.tensor(batch_size)
    # sample batch_size get segmentation fault(core dumped) error,
    sample_batch_size = torch.tensor(1)
    pos_rw = pos_sample(sample_batch_size, adj, model.encoder.walks_per_node,
                        model.encoder.walk_length, model.encoder.context_size, model.encoder.p, model.encoder.q)

    neg_rw = neg_sample(sample_batch_size, adj, model.encoder.walks_per_node,
                        model.encoder.num_negative_samples, model.encoder.walk_length, model.encoder.context_size)
    dataset = TensorDataset(pos_rw, neg_rw)
    return DataLoader(dataset, shuffle=True, batch_size=batch_size)


def train(model, data, train_loader, optimizer, crit, ENABLE_FEATURE_LOSS=False, Debug=False):
    '''
    train 1 epoch
    '''
    model.train()
    total_loss = 0
    i = 0
    for pos_rw, neg_rw in train_loader:
        # print(pos_rw.shape)
        pos_rw = pos_rw.to(device)
        neg_rw = neg_rw.to(device)
        optimizer.zero_grad()
        # Postive loss and neg loss copied from source code; I don't understand this implementation.
        # Positive loss.
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()
        emb, adj, pred, p_norm = model(data, Debug=Debug)

        h_start = model.encoder.embedding(start).view(pos_rw.size(0), 1,
                                                      model.encoder.embedding_dim)
        h_rest = model.encoder.embedding(rest.view(-1)).view(pos_rw.size(0), -1,
                                                             model.encoder.embedding_dim)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        pos_loss = -torch.log(torch.sigmoid(out) + EPS).mean()

        # Negative loss.
        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()

        h_start = model.encoder.embedding(start).view(neg_rw.size(0), 1,
                                                      model.encoder.embedding_dim)
        h_rest = model.encoder.embedding(rest.view(-1)).view(neg_rw.size(0), -1,
                                                             model.encoder.embedding_dim)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_loss = -torch.log(1 - torch.sigmoid(out) + EPS).mean()

        # want adj sparse, just add it all together

        loss = pos_loss + neg_loss + p_norm*ADJ_SPASE_WEIGHT

        # if i == len(train_loader)-1 and ENABLE_FEATURE_LOSS:
        if ENABLE_FEATURE_LOSS:
            loss2 = crit(pred[train_mask], labels[train_mask].to(
                device))*FEATURE_LOSS_WEIGHT
            # print(loss2.item())
            loss = loss+loss2

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        i += 1
    return total_loss/len(train_loader)


@torch.no_grad()
def test(model, data, labels,  mask, crit):
    model.eval()
    # return acc
    pred = model(data)[2].cpu()
    loss = crit(pred[mask], labels[mask])
    if DEBUG:
        print(pred[mask].argmax(dim=-1)[:10], labels[mask][:10])
    # calculate accuracy
    acc = ((pred[mask].argmax(dim=-1) == labels[mask]
            ).sum().float() / mask.sum()).cpu()
    return loss, acc


def adj_from_z(z):
    ''' create adj matrix from z using Z@Z.T '''
    ''' and choose the top values '''
    adj = np.matmul(z, z.T)
    # adj=adj-np.min(adj)
    # set adj diagonal to zero
    np.fill_diagonal(adj, 0)
    # normalize the adj matrix

    # print('ZTZ', adj)
    # flatten adj
    adj_flat = adj.flatten()
    # top K values
    k = 7*2
    topk_ind = np.argpartition(adj_flat, -k)[-k:]
    topk = adj_flat[topk_ind]
    min_topk = np.min(topk)
    # print(min_topk)

    adj[adj < min_topk] = 0
    adj[adj >= min_topk] = 1
    np.fill_diagonal(adj, 0)
    return adj


@torch.no_grad()
def plot_points(model, data, labels, colors, fn=''):
    model.eval()
    z = model.encoder().cpu().numpy()

    # adj = adj_from_z(z)
    adj = model(data)[1].cpu().numpy()
    # print(adj)
    adj[adj < 0.5] = 0
    adj[adj >= 0.5] = 1
    np.fill_diagonal(adj, 0)
    # print(np.sum(adj))
    # create a graph using adj
    if N < 10:
        # clear the graph
        plt.clf()

        G = nx.from_numpy_array(adj)
        nx.draw(G, with_labels=True)
        path = osp.join(osp.dirname(osp.realpath(__file__)), '.',
                        FIGURES_DIR, f'sythetic_graph_node2vec_{fn}_my.png')
        plt.savefig(path)
        plt.clf()
    # z = TSNE(n_components=NUM_CLASSES).fit_transform(z)
    # TSNE error:"ValueError: perplexity must be less than n_samples"; using PCA
    from sklearn.decomposition import PCA
    z = PCA(n_components=2).fit_transform(z)
    y = labels.numpy()

    plt.figure(figsize=(8, 8))
    for i in range(NUM_CLASSES):
        plt.scatter(z[y == i, 0], z[y == i, 1], s=10, color=colors[i])
    plt.axis('off')
    # save the plot
    path = osp.join(osp.dirname(osp.realpath(__file__)),
                    '.', FIGURES_DIR, 'node2vec_my.png')
    plt.savefig(path)


colors = [
    '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535', '#ffd700'
]

edge_index, labels, train_mask, test_mask, N, NUM_CLASSES, data = load_dataset()


def run_experiment(model):
    edge_index, labels, train_mask, test_mask, N, NUM_CLASSES, data = load_dataset()
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
    ENABLE_FEATURE_LOSS = False
    for epoch in range(1, Training_EPS+1):
        loader = train_dataloader(N, edge_index, model, batch_size=128)
        crit = nn.CrossEntropyLoss()
        if epoch % 200 == 0:
            DEBUG = True
        else:
            DEBUG = False
        train_loss = train(model, data, loader, optimizer,
                           crit, ENABLE_FEATURE_LOSS, Debug=DEBUG)
        loss, train_acc = test(model, data, labels, train_mask, crit)
        loss, test_acc = test(model, data, labels, test_mask, crit)
        if DEBUG:
            print(
                f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
            plot_points(model, data, labels, colors, epoch)
        # train_losss.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        if train_loss < 100.:  # learn the adj first
            ENABLE_FEATURE_LOSS = True

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
    # plot loss and acc on two y axis
    # save model

    # torch.save(model.state_dict(), model_path)

# for walk_length in [ 5,7,11,19, 23, 29]:
for walk_length in [ 5]:
    for i in range(10):
        model = TopDecoder(N, embedding_dim=128, walk_length=walk_length, context_size=walk_length, walks_per_node=1, p=1, q=0.1,
                        num_negative_samples=1, num_nodes=None, sparse=False).to(device)
        run_experiment(model)
    print('-----------------')
    print('-----------------')
    
# for walk_length in [ 5,7,11,19, 23, 29]:
#     for i in range(5):
#         model = TopDecoder(N, embedding_dim=256, walk_length=walk_length, context_size=walk_length, walks_per_node=10, p=1, q=0.1,
#                         num_negative_samples=1, num_nodes=None, sparse=False).to(device)
#         run_experiment(model)
#     print('-----------------')
#     print('-----------------')
    

