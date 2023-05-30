import pandas as pd
import numpy as np
import gzip
import networkx as nx
import random
random.seed(10)

import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F

import torch_geometric
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, Linear, to_hetero

from sklearn.metrics import precision_score, recall_score, f1_score
    
class GCN(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(-1, hidden_dim)
        self.conv2 = GCNConv(-1, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
    
class GraphSAGE(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_dim)
        self.conv2 = SAGEConv((-1, -1), output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
    
class GAT(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_dim, add_self_loops=False)
        self.lin1 = Linear(-1, hidden_dim)
        self.conv2 = GATConv((-1, -1), output_dim, add_self_loops=False)
        self.lin2 = Linear(-1, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = x.relu()
        x = self.conv2(x, edge_index) + self.lin2(x)
        return x
    
class GNN():
    def __init__(self):
        self.model = None
        self.epochs = 200
        self.node_embed_size = 100
        self.hidden_dim = 128
        self.output_dim = 128
        self.neg_sampling_ratio = 0.5
        self.seed = 10
        torch.manual_seed(self.seed)
    
    def _train(self, g_train, GNN_variant):
        adj = nx.to_scipy_sparse_array(g_train)
        pos_edge_index = torch_geometric.utils.from_scipy_sparse_matrix(adj)[0]
        neg_edge_index = sample_negative_edges(adj, pos_edge_index, self.neg_sampling_ratio)
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        num_nodes = g_train.number_of_nodes()
        node_embed = torch.rand(num_nodes, self.node_embed_size)
        
        if GNN_variant == 'GCN':
            self.model = GCN(self.hidden_dim, self.output_dim)
        elif GNN_variant == 'GraphSAGE':
            self.model = GraphSAGE(self.hidden_dim, self.output_dim)
        elif GNN_variant == 'GAT':
            self.model = GAT(self.hidden_dim, self.output_dim)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005, weight_decay=5e-4)    
        targets = torch.cat([torch.ones(pos_edge_index.shape[1]), torch.zeros(neg_edge_index.shape[1])])
        edge_index, targets = shuffle_predictions_targets(edge_index, targets)
        
        print(f'{GNN_variant}:')
        for i in range(self.epochs+1):
            self.model.train()
            optimizer.zero_grad()
            
            output = self.model(node_embed, edge_index)
            u = torch.index_select(output, 0, edge_index[0, :])
            v = torch.index_select(output, 0, edge_index[1, :])
            pred = torch.sum(u * v, dim=-1)
            
            loss = mse_loss(pred, targets)
            loss.backward()
            optimizer.step()
            
            if i % 50 == 0:
                print(f'Epoch: {i}, Loss: {loss:.4f}')
     
    def _eval(self, g_test):
         with torch.no_grad():
                self.model.eval()
                
                adj = nx.to_scipy_sparse_array(g_test)
                pos_edge_index = torch_geometric.utils.from_scipy_sparse_matrix(adj)[0]
                neg_edge_index = sample_negative_edges(adj, pos_edge_index, self.neg_sampling_ratio)
                edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
                num_nodes = g_test.number_of_nodes()
                node_embed = torch.rand(num_nodes, self.node_embed_size)
                
                output = self.model(node_embed, edge_index)
                u = torch.index_select(output, 0, edge_index[0, :])
                v = torch.index_select(output, 0, edge_index[1, :])
                pred = torch.sum(u * v, dim=-1)
                
                norm_pred = normalize_arr(pred, pred.min(), pred.max())
                norm_pred = np.where(norm_pred >= 0.5, 1, 0)
                targets = targets = torch.cat([torch.ones(pos_edge_index.shape[1]), torch.zeros(neg_edge_index.shape[1])]).detach().numpy()
                
                precision = precision_score(targets, norm_pred)
                recall = recall_score(targets, norm_pred)
                f1 = f1_score(targets, norm_pred)
                
                print(f'Precision: {precision:.4f}')
                print(f'Recall: {recall:.4f}')
                print(f'F1-Score: {f1:.4f}')
            
###HELPER FUNCIONS### 

def mse_loss(pred, target):
    return (pred - target.to(pred.dtype)).pow(2).mean()

def normalize_arr(arr, x_min, x_max):
    new_arr = []
    for i in arr:
        new_arr.append((i-x_min)/(x_max-x_min))
    return np.array(new_arr)

def sample_negative_edges(adj, edge_index, neg_sampling_ratio):
    num_neg_edges = int(edge_index.shape[1] * neg_sampling_ratio)
    possible_edges = adj.shape[0] * adj.shape[1]
    index_to_edge = lambda i: (i // adj.shape[1], i % adj.shape[1])
    negative_edges = set()
    
    while len(negative_edges) < num_neg_edges:
        edge = np.random.randint(0, possible_edges, dtype=np.int64)
        i, j = index_to_edge(edge)
        if adj[i, j] == 0 and adj[j, i] == 0 and edge not in negative_edges and i != j:
            negative_edges.add(edge)
    negative_edges = [index_to_edge(i) for i in negative_edges]
    negative_edges = torch.tensor([[e[0] for e in negative_edges], [e[1] for e in negative_edges]])
    
    return negative_edges

def shuffle_predictions_targets(edge_index, targets):
    shuffle = list(zip(edge_index[0], edge_index[1], targets))
    random.shuffle(shuffle)
    u, v, targets = zip(*shuffle)
    return torch.tensor([list(u), list(v)]), torch.tensor(list(targets))