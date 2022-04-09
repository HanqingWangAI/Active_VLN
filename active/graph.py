# from param import args
from math import pi
import numpy as np
import networkx as nx


class GraphBatch:
    def __init__(self, batch_size):
        self.graphs = []
        self.batch_size = batch_size
        for i in range(batch_size):
            self.graphs.append(nx.Graph())

    def reset(self):
        self.graphs = []
        for i in range(self.batch_size):
            self.graphs.append(nx.Graph())
    
    def add_edge(self, obs):
        for i, ob in enumerate(obs):
            n = ob['viewpoint']
            # if ended[i]:  # if ended, will not add ghost nodes
            #     continue
            for c in ob['candidate']: # add ghost nodes
                ghost_vid = c['viewpointId']
                from_vid = n
            
                self.graphs[i].add_edge(from_vid, ghost_vid, weight=1)
                self.graphs[i].add_edge(ghost_vid, from_vid, weight=1)
    
    def get_paths(self):
        return [dict(nx.all_pairs_dijkstra_path(g)) for g in self.graphs]