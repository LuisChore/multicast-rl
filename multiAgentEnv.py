
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from Graph import Graph


'''
Multi agent environment
Parameters: DAG, target, list of agent indexes
'''
class MultiCast(object):
    def __init__(self,graph,agents,target):
        self.fig = plt.figure()
        self.fig.add_subplot(111)
        self.graph = graph
        self.target = target
        self.agents = agents.copy()
        self.number_agents = len(agents)
        self.initial_state = agents.copy()
        self.agents_done = self.count_agents_done()
        self.done = self.is_done()
        self.edges_used = self.initialize_edges()
        self.graphic_graph = self.create_graph()

    def create_graph(self):
        G = nx.DiGraph()
        for i in range(self.graph.nodes):
            G.add_node(i)
        for u in range(self.graph.nodes):
            for v,w in self.graph.adj[u]:
                G.add_edge(u,v,weight = w)
        return G
    def initialize_edges(self):
        Edges = {}
        for u in range(self.graph.nodes):
            for v,w in self.graph.adj[u]:
                i = min(u,v)
                j = max(u,v)
                Edges[(i,j)] = (False,w)
        return Edges

    def render(self,sleep = 2):
        color_map = []
        for node in self.graphic_graph:
            if node in self.agents:
                color_map.append('red')
            else:
                color_map.append('blue')
        pos = nx.planar_layout(self.graphic_graph)
        nx.draw(self.graphic_graph,pos,node_color=color_map,with_labels = True)
        labels = nx.get_edge_attributes(self.graphic_graph,'weight')
        nx.draw_networkx_edge_labels(self.graphic_graph ,pos,edge_labels = labels)
        plt.ion()
        plt.show()
        plt.pause(sleep)

    def is_done(self):
        return self.number_agents == self.agents_done

    #just for initialization & reset, it's updated over time
    def count_agents_done(self):
        agents_done = self.number_agents
        for agent in self.agents:
            if agent != self.target:
                agents_done -= 1
        return agents_done

    def get_graph(self):
        return self.graph

    def reset(self):
        self.agents = self.initial_state.copy()
        self.agents_done = self.count_agents_done()
        self.edges_used = self.initialize_edges()
        return self.agents

    def step(self,action):
        # add action valitadion
        cost = 0
        for i in range(len(action)):
            if self.agents[i] != action[i]:
                u = min(self.agents[i],action[i])
                v = max(self.agents[i],action[i])
                used,w = self.edges_used[(u,v)]
                if used == False:
                    self.edges_used[(u,v)] = (True,w)
                    cost += w
                self.agents[i] = action[i]
                if self.target == self.agents[i]:
                    self.agents_done += 1
        return self.agents,cost,self.is_done()
