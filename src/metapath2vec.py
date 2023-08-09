import pandas as pd
import networkx as nx
import gzip

import torch
from torch_geometric.data import HeteroData

def insert_entry(entry, ntype, dic):
    if ntype not in dic:
        dic[ntype] = {}
    node_id = len(dic[ntype])
    if entry not in dic[ntype]:
         dic[ntype][entry] = node_id
    return dic

def get_node_dict(df):
    '''Create a dict of node-types -> each dictionary further consists of a dictionary mapping a node to an ID'''
    node_dict = {}
    for triple in df.values.tolist():
        src = triple[0]
        src_type = 'Class'
        dest = triple[2]
        dest_type = 'Class'
        insert_entry(src, src_type, node_dict)
        insert_entry(dest, dest_type, node_dict)
    return node_dict

def get_edge_dict(df, node_dict, etype, rev):
    '''Create a dict of edge-types -> the key is the edge-type and the value is a list of (src ID, dest ID) tuples'''
    df = df[df['p'] == etype]
    src = [node_dict['Class'][index] for index in df['s']]
    dst = [node_dict['Class'][index] for index in df['o']]
    if rev == 1:
        return torch.tensor([dst, src])
    else: 
        return torch.tensor([src, dst])

def get_heterodata(df_train, df_test):
    node_dict_train = get_node_dict(df_train)
    g_train = HeteroData()
    g_train['Class'].num_nodes = len(node_dict_train['Class'])
    for p in df_train['p'].unique():
        g_train[('Class', p, 'Class')].edge_index = get_edge_dict(df_train, node_dict_train, p, 0)
    for p in df_train['p'].unique():
        g_train[('Class', 'rev_ '+p, 'Class')].edge_index = get_edge_dict(df_train, node_dict_train, p, 1)

    node_dict_test = get_node_dict(df_test)
    g_test = HeteroData()
    g_test['Class'].num_nodes = len(node_dict_test['Class'])
    for p in df_test['p'].unique():
        g_test[('Class', p, 'Class')].edge_index = get_edge_dict(df_test, node_dict_test, p, 0)
    for p in df_test['p'].unique():
        g_test[('Class', 'rev_'+p, 'Class')].edge_index = get_edge_dict(df_test, node_dict_test, p, 1)

    return g_train, g_test