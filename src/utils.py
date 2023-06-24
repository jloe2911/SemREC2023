import pandas as pd
import networkx as nx
import gzip

import torch

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
        
    return graph, nodes, edges

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