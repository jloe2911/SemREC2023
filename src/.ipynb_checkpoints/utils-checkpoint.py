import pandas as pd
import networkx as nx
import gzip

import torch
from torch_geometric.data import HeteroData

import re

def create_graph(df):
    graph = nx.MultiDiGraph()
    node_num = 0
    edge_num = 0
    nodes = dict()
    edges = dict()

    for i, row in df.iterrows():
        if row['s'] not in nodes:
            nodes[row['s']] = node_num
            node_num += 1
        if row['o'] not in nodes:
            nodes[row['o']] = node_num
            node_num += 1
        if row['p'] not in edges:
            edges[row['p']] = edge_num
            edge_num += 1
        graph.add_edge(nodes[row['s']], nodes[row['o']], type = edges[row['p']])
        
    return graph

def load_ore_files(pathfilename):
    
    nt_file = open(pathfilename,'r',newline='\n')
    lines = nt_file.read().split('\r\n')

    entity_dict = dict({'s':[],'p':[],'o':[]})

    for line in lines[:-1]:
        split = line.split(' ')
        entity_dict['s'].append(re.findall(r'(?<=\().+',split[0])[0])
        entity_dict['p'].append(re.findall(r'.+?(?=\()',split[0])[0])
        entity_dict['o'].append(split[1].replace(')',''))

    df = pd.DataFrame(entity_dict)
    
    return df

def load_ore_graphs(path, train_file, test_file):
    print('Running...', train_file, test_file)

    df_train = load_ore_files(path+train_file)
    g_train = create_graph(df_train)
    
    df_train_filter_subclass = df_train[df_train['p'] == 'SubClassOf']
    g_train_filter_subclass = create_graph(df_train_filter_subclass)
    
    df_train_filter_assertion = df_train[df_train['p'] == 'ClassAssertion']
    g_train_filter_assertion = create_graph(df_train_filter_assertion)
    
    print(f'# Train - Triplets: {len(df_train)}, # Nodes: {g_train.number_of_nodes()}, # Edges: {g_train.number_of_edges()}')

    df_test = load_ore_files(path+test_file)
    g_test = create_graph(df_test)
    
    df_test_filter_subclass = df_test[df_test['p'] == 'SubClassOf']
    g_test_filter_subclass = create_graph(df_test_filter_subclass)
    
    df_test_filter_assertion = df_test[df_test['p'] == 'ClassAssertion']
    g_test_filter_assertion = create_graph(df_test_filter_assertion)
    
    print(f'# Test - Triplets: {len(df_test)}, # Nodes: {g_test.number_of_nodes()}, # Edges: {g_test.number_of_edges()}')
    
    print()
    
    return g_train, g_train_filter_subclass, g_train_filter_assertion, \
           g_test, g_test_filter_subclass, g_test_filter_assertion

def load_clg_files(pathfilename):
    
    nt_file = open(pathfilename,'r')
    lines = nt_file.readlines()
    lines = lines[0].split(' .')
    
    entity_dict = dict({'s':[],'p':[],'o':[]})
    
    for line in lines[:-1]:
        entities = line.split(' ')
        entity_dict['s'].append(entities[0])
        entity_dict['p'].append(entities[1])
        entity_dict['o'].append(entities[2])
        
    df = pd.DataFrame(entity_dict)

    return df

def load_clg_graphs(path, train_file, test_file):
    print('Running...', train_file, test_file)

    df_train = load_clg_files(path+train_file)
    g_train = create_graph(df_train)
    
    df_train_filter_subclass = df_train[df_train['p'] == '<http://www.w3.org/2000/01/rdf-schema#subClassOf>']
    g_train_filter_subclass = create_graph(df_train_filter_subclass)
    
    df_train_filter_assertion = df_train[df_train['p'] == '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>']
    g_train_filter_assertion = create_graph(df_train_filter_assertion)
    
    print(f'# Train - Triplets: {len(df_train)}, # Nodes: {g_train.number_of_nodes()}, # Edges: {g_train.number_of_edges()}')

    df_test = load_clg_files(path+test_file)
    g_test = create_graph(df_test)
    
    df_test_filter_subclass = df_test[df_test['p'] == '<http://www.w3.org/2000/01/rdf-schema#subClassOf>']
    g_test_filter_subclass = create_graph(df_test_filter_subclass)
    
    df_test_filter_assertion = df_test[df_test['p'] == '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>']
    g_test_filter_assertion = create_graph(df_test_filter_assertion)
    
    print(f'# Test - Triplets: {len(df_test)}, # Nodes: {g_test.number_of_nodes()}, # Edges: {g_test.number_of_edges()}')
    
    print()
    
    return g_train, g_train_filter_subclass, g_train_filter_assertion, \
           g_test, g_test_filter_subclass, g_test_filter_assertion

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

def get_edge_dict(df, node_dict, etype):
    '''Create a dict of edge-types -> the key is the edge-type and the value is a list of (src ID, dest ID) tuples'''
    df = df[df['p'] == etype]
    src = [node_dict['Class'][index] for index in df['s']]
    dst = [node_dict['Class'][index] for index in df['o']]
    edge_dict = torch.tensor([src, dst])
    return edge_dict

def get_heterodata(df_train, df_test):
    node_dict_train = get_node_dict(df_train)
    g_train = HeteroData()
    g_train['Class'].num_nodes = len(node_dict_train['Class'])
    for p in df_train['p'].unique():
      g_train[('Class', p, 'Class')].edge_index = get_edge_dict(df_train, node_dict_train, p)

    node_dict_test = get_node_dict(df_test)
    g_test = HeteroData()
    g_test['Class'].num_nodes = len(node_dict_test['Class'])
    for p in df_test['p'].unique():
      g_test[('Class', p, 'Class')].edge_index = get_edge_dict(df_test, node_dict_test, p)

    return g_train, g_test