import random
import numpy as np
import collections
import networkx as nx
from collections import defaultdict 
import math

# This contains the logic of the model described in Törnberg, P. (2022) "How digital media drive affective polarization through partisan sorting" PNAS.
# Please see the paper for detailed description of the model.
class sorting_model:
    
    def __init__(self,k=2,m=10,n=4,nragents=100,gamma=0.,h=8,c=1, custom_network = None):        
        self.m = m
        self.k = k
        self.n = n
        self.nragents = nragents 
        self.c = c
        self.gamma = gamma
        self.reset()
        self.h = h
        self.custom_network = (custom_network is not None)
        self.network = self.custom_network if self.custom_network else self.grid_2d_moore_graph(nragents, periodic=True)        
        self.nodeid = {nid:nr for nr,nid in enumerate(self.network.nodes)}
        
    #Generating a lattice torus with moore neighbors. 
    def grid_2d_moore_graph(self,nragents,periodic=False,create_using=None):
        
        m = nragents**0.5
        if int(m) != m:
            raise TypeError("Check parameter. With a square grid torus, so the square root of the number of agents must be an integer.")
            
        m = int(m)
        n = m
        
        G=nx.empty_graph(0,None)
        G.name="grid_2d_moore_graph"
        rows=range(m)
        columns=range(n)
        G.add_nodes_from( (i,j) for i in rows for j in columns )
        G.add_edges_from( ((i,j),(i-1,j)) for i in rows for j in columns if i>0 )
        G.add_edges_from( ((i,j),(i,j-1)) for i in rows for j in columns if j>0 )

        G.add_edges_from( ((i,j),(i-1,j-1)) for i in rows for j in columns if j>0 and i>0 )
        G.add_edges_from( ((i,j),(i-1,j+1)) for i in rows for j in columns if i>0 and j<n-1 )

        if periodic:
            if n>2:
                G.add_edges_from( ((i,0),(i,n-1)) for i in rows )
                G.add_edges_from( ((i,0),(i-1,n-1)) for i in rows if i>0)
                G.add_edges_from( ((i,0),(i+1,n-1)) for i in rows if i<n-1)

            if m>2:
                G.add_edges_from( ((0,j),(m-1,j)) for j in columns )
                G.add_edges_from( ((0,j),(m-1,j-1)) for j in columns if j>0)
                G.add_edges_from( ((0,j),(m-1,j+1)) for j in columns if j<m-1)

            #Diagonal to diagonal
            G.add_edge( (0,0),(m-1,n-1) )
            G.add_edge( (m-1,0),(0,n-1) )

        G.name="periodic_grid_2d_graph(%d,%d)"%(m,n)
        return G
        
        
    def reset(self):
        self.agent_fixed = np.random.randint(self.k, size=(self.nragents))
        self.agent_flex = np.random.randint(self.m, size=(self.nragents,self.n))
        
        #List all unique combinations between nodes of same and different groups, preprocessing for faster run
        self.within = [(i1,i2) for i1,n1 in enumerate(self.agent_fixed) for i2,n2 in enumerate(self.agent_fixed) if n1==n2 and i1<i2]
        self.between = [(i1,i2) for i1,n1 in enumerate(self.agent_fixed) for i2,n2 in enumerate(self.agent_fixed) if n1!=n2 and i1<i2]
    
    
    #Run the model
    def run(self,steps,breakat=10):
        stepssincechange = 0
        for step in range(steps):                
            if self.step():
                stepssincechange = 0 
            else:
                stepssincechange += 1
            if breakat is not None and stepssincechange > breakat*self.nragents:
                break
                
            
    def step(self):
        #Select a node at random to update
        node = random.choice(list(self.network.nodes))
        nodeid = self.nodeid[node]

        #Select interlocutors: neighbors + randomly selected
        neigh = list(self.network.neighbors(node))
        nrrandom = int(self.gamma*len(neigh))
        neighbors = random.sample(neigh,len(neigh)-nrrandom) + random.sample(list(self.network.nodes),nrrandom)

        #Pick the other node based on the similarity
        weights = np.array([self.similarity(nodeid,self.nodeid[neighbor])**self.h for neighbor in neighbors])
        if sum(weights)==0: #No similarities with any neighbor: no change
            return False
        
        weights = weights/sum(weights)
        
        #Randomly pick another node on the basis of weights, urn model
        othernode = np.random.choice(len(neighbors),p=weights)
        othernodeid = self.nodeid[neighbors[othernode]]
                
        #Pick dimension for which the other node that is different than ours
        diff = np.array([1 if self.agent_flex[nodeid][dimension]!=self.agent_flex[othernodeid][dimension] else 0 for dimension in range(self.n)])
        
        #They're already the same
        if sum(diff)==0:
            return False
        
        diff = diff/sum(diff)
        dimension = np.random.choice(self.n,p=diff)
        
        #Set node dimension to other node's value
        self.agent_flex[nodeid][dimension] = self.agent_flex[othernodeid][dimension]        
        return True
     
    def similarity(self,a1,a2):        
        return ((self.c if self.agent_fixed[a1] == self.agent_fixed[a2] else 0) + sum([1 if self.agent_flex[a1][i] == self.agent_flex[a2][i] else 0 for i in range(self.n)]))/(self.c+self.n)     
       
    def fraction_shared_flex(self,a1,a2):
        return sum([1 if self.agent_flex[a1][i] == self.agent_flex[a2][i] else 0 for i in range(self.n)])/self.n
    
    def calculate_sorting(self):
        within = np.mean([self.fraction_shared_flex(a,b) for a,b in self.within])
        between = np.mean([self.fraction_shared_flex(a,b) for a,b in self.between])
        return within - between

