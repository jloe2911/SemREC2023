import pandas as pd
import numpy as np
import operator
import gzip
import networkx as nx
import random
random.seed(10)

from node2vec import Node2Vec

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
        self.node_embed_size = 200
        self.hidden_dim = 200
        self.output_dim = 200
        self.neg_sampling_ratio = 0.5
        self.seed = 10
        torch.manual_seed(self.seed)
    
    def _train(self, g_train, GNN_variant, bool_node2vec):
        adj = nx.to_scipy_sparse_array(g_train)
        pos_edge_index = torch_geometric.utils.from_scipy_sparse_matrix(adj)[0]
        neg_edge_index = sample_negative_edges(adj, pos_edge_index, self.neg_sampling_ratio)
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        num_nodes = g_train.number_of_nodes()
        
        ###Node2Vec###
        if bool_node2vec == 'Node2Vec':
            node2vec = Node2Vec(g_train, dimensions=200, walk_length=30, num_walks=200)
            node2vec_model = node2vec.fit(window=10, min_count=1, batch_words=4)
            self.node_embeds = torch.tensor(np.array([node2vec_model.wv.get_vector(node) for node in g_train.nodes()]))
        ######
        else:
            self.node_embeds = torch.rand(num_nodes, self.node_embed_size)
        
        if GNN_variant == 'GCN':
            self.model = GCN(self.hidden_dim, self.output_dim)
        elif GNN_variant == 'GraphSAGE':
            self.model = GraphSAGE(self.hidden_dim, self.output_dim)
        elif GNN_variant == 'GAT':
            self.model = GAT(self.hidden_dim, self.output_dim)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005, weight_decay=5e-4)    
        targets = torch.cat([torch.ones(pos_edge_index.shape[1]), torch.zeros(neg_edge_index.shape[1])])
        edge_index, targets = shuffle_predictions_targets(edge_index, targets)
        
        print(f'{GNN_variant} + {bool_node2vec}:')
        for i in range(self.epochs+1):
            self.model.train()
            optimizer.zero_grad()
            
            embeds = self.model(self.node_embeds, edge_index)
            u = torch.index_select(embeds, 0, edge_index[0, :])
            v = torch.index_select(embeds, 0, edge_index[1, :])
            pred = torch.sum(u * v, dim=-1)
            pred = (pred - pred.min()) / (pred.max() - pred.min())
            
            loss = mse_loss(pred, targets)
            loss.backward()
            optimizer.step()
            
            if i % 50 == 0:
                print(f'Epoch: {i}, Loss: {loss:.4f}')
     
    def _eval(self, g_test, max_num):
         with torch.no_grad():
                self.model.eval()
                
                adj = nx.to_scipy_sparse_array(g_test)
                pos_edge_index = torch_geometric.utils.from_scipy_sparse_matrix(adj)[0]
                neg_edge_index = sample_negative_edges(adj, pos_edge_index, self.neg_sampling_ratio)
                edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
                
                output = self.model(self.node_embeds, edge_index)
                u = torch.index_select(output, 0, edge_index[0, :])
                v = torch.index_select(output, 0, edge_index[1, :])
                pred = torch.sum(u * v, dim=-1)
                pred = (pred - pred.min()) / (pred.max() - pred.min())
                
                ###Model as Binary Classification Problem###
                pred = pred.detach().numpy()
                pred = np.where(pred >= 0.5, 1, 0)
                targets = torch.cat([torch.ones(pos_edge_index.shape[1]), torch.zeros(neg_edge_index.shape[1])]).detach().numpy()
                
                precision = precision_score(targets, pred)
                recall = recall_score(targets, pred)
                f1 = f1_score(targets, pred)
                
                print(f'Precision: {precision:.3f}, Recall: {recall:.3f}, F1-Score: {f1:.3f}')
                print()
                ######
                
                print(f'head, relation -> tail?')
                hits1, hits10 = eval_hits(tail_pred=1, g_test=g_test, pos_edge_index=pos_edge_index, output=output, max_num=max_num)
                print(f'hits@1: {hits1:.3f}, hits@10: {hits10:.3f}')
                print('-------------------------------------------')
                
                #print(f'tail, relation -> head?')
                #hits1, hits10 = eval_hits(tail_pred=0, g_test=g_test, pos_edge_index=pos_edge_index, output=output, max_num=100)
                #print(f'hits@1: {hits1:.3f}, hits@10: {hits10:.3f}')
                
###HELPER FUNCIONS### 

def mse_loss(pred, target):
    return (pred - target.to(pred.dtype)).pow(2).mean()

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

def eval_hits(tail_pred, g_test, pos_edge_index, output, max_num):    
    top1 = 0
    top10 = 0
    n = pos_edge_index.size()[1]

    for idx in range(n):

        if tail_pred == 1:
            x = torch.index_select(output, 0, pos_edge_index[0, idx])
        else:
            x = torch.index_select(output, 0, pos_edge_index[1, idx])
        
        candidates_embeds = sample_negative_edges_idx(idx=idx,tail_pred=tail_pred,g_test=g_test,pos_edge_index=pos_edge_index,output=output,max_num=max_num)

        dist = np.linalg.norm(candidates_embeds.detach().numpy() - x.detach().numpy(), axis=1) # Euclidean distance
        dist_dict = {i: dist[i] for i in range(0, len(dist))} 

        sorted_dict = dict(sorted(dist_dict.items(), key=operator.itemgetter(1), reverse=True))
        sorted_keys = list(sorted_dict.keys())

        ranks_dict = {sorted_keys[i]: i for i in range(0, len(sorted_keys))}
        try:
            rank = ranks_dict[pos_edge_index[1, idx].item()]
            if rank <= 1:
                top1 += 1
            if rank <= 10:
                top10 += 1
        except:
            rank=999
    return top1/n, top10/n

def sample_negative_edges_idx(idx, tail_pred, g_test, pos_edge_index, output, max_num):
    num_neg_samples = 0
    candidates = []
    nodes = list(range(g_test.number_of_nodes()))
    random.shuffle(nodes)
    
    while num_neg_samples < max_num:    
        if tail_pred == 1:
            t = nodes[num_neg_samples]
            h = torch.index_select(output, 0, pos_edge_index[0, idx])
            if (h,t) not in g_test.edges():
                candidates.append(t)
        else: 
            t = torch.index_select(output, 0, pos_edge_index[1, idx])
            h = nodes[num_neg_samples]
            if (h,t) not in g_test.edges():
                candidates.append(h)
        num_neg_samples += 1
    candidates_embeds = torch.index_select(output, 0, torch.tensor(candidates))
    return candidates_embeds