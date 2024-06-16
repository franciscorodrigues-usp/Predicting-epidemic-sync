
# coding: utf-8

# Francisco A. Rodrigues, University of São Paulo. http://conteudo.icmc.usp.br/pessoas/francisco

# # Epidemic spreading on networks

# Initially, we have to import the libraries necessary for our simulations.

# In[1]:


from numpy  import *
import numpy as np
#import matplotlib.pyplot as plt
import networkx as nx
import random as random
random.seed(100) #set the random seed. Important to reproduce our results.


# Let us generate an artificial random network. The network is the medium in which the infectious agent or information spread.

# In[2]:


# Network parameters
N = 100 #number of nodes
av_degree = 8 # average degree
p = float(av_degree)/float(N) #probability of connection in the ER model
m = int(av_degree/2) # number of nodes included at each time step in the BA model
kappa = av_degree # number of neighbors in the WS model

#G = nx.barabasi_albert_graph(N,m) # generate a BA network
#str_net = 'BA'

#G=nx.read_edgelist("nets/celegans.txt", nodetype=int)# if the data file has only two columns.
#str_net = 'celegans'

#G=nx.read_edgelist("nets/hamsterster.txt", nodetype=int)# if the data file has only two columns.
#str_net = 'hamsterster'

#G=nx.read_edgelist("nets/googleplus.txt", nodetype=int)# if the data file has only two columns.
#str_net = 'googleplus'

#G= nx.read_edgelist("nets/advogato.txt", nodetype=int, data=(('weight',float),)) # Read the network
#str_net = 'advogato'

G= nx.read_edgelist("nets/USairports.txt", nodetype=int, data=(('weight',float),)) # Read the network
str_net = 'USairports'

G = G.to_undirected()
Gcc=sorted(nx.connected_component_subgraphs(G), key = len, reverse=True)
G=Gcc[0]
G = nx.convert_node_labels_to_integers(G, first_label=0)


# The basic network properties.

# In[3]:


N = len(G) # Number of nodes
vk = dict(G.degree())
vk = list(vk.values())
av_degree = np.mean(vk)
print("Number of nodes:", N)
print("Average degree: ", av_degree)


# In[4]:


# function to simulate the SIR dynamics starting from a set of nodes stored in the variable "seed"
def SIR(G, seeds, beta=0.3, mu=1):    
    def find(v, i): # function to find the positions of element i in vector v
        l = []
        pos = 0
        for x in v:
            if(x == i):
                l.append(pos)
            pos = pos + 1
        return l

    #Reactive process: SIR dynamics
    vector_states = np.zeros(N) # vector that stores the states of the vertices
    vector_states[seeds] = 1 # vector that stores the states
    ninfected = len(seeds)
    t = 0 # start in t= 0 
    vt = list() # this list stores the time step
    vI = list() # this list stores the fraction of infected nodes
    vR = list() # this list stores the fraction of recovered nodes
    vS = list() # this list stores the fraction of susceptible nodes
    # Reactive model simulation
    while ninfected > 0: # Simulate while we can find infected nodes
        infected = find(vector_states,1) # list of infected nodes
        for i in infected: # try to infect the neighbors
            neigs = G.neighbors(i)
            for j in neigs:
                if np.random.rand() < beta:
                    if(vector_states[j] != 2): # verify if the node is not recovered
                        vector_states[j] = 1
        for k in infected: # try to recover the infected nodes
            if np.random.rand() < mu:
                vector_states[k] = 2
        ninfected = len(find(vector_states,1))
        vI.append(ninfected/N)
        vR.append(len(find(vector_states,2))/N)
        vS.append(len(find(vector_states,0))/N)
        t = t + 1
        vt.append(t)
    return vI, vS, vR, vt


# In[5]:


vrho_Rf = np.zeros(N)
Ns = 20
beta = 0.2
mu = 1.0
vrho_matrix = np.zeros((N,Ns), dtype=float)
for seed_node in G.nodes(): #starting from each node
    seeds = [seed_node] 
    rec = []
    for ns in np.arange(0, Ns):
        vI, vS, vR, vt = SIR(G, seeds, beta, mu)
        rec.append(vR[-1])
        vrho_matrix[seed_node][ns] = vR[-1]
    vrho_Rf[seed_node] = np.mean(rec)


# In[6]:


file_name = 'out/vrho_matrix_' + str_net + '_' + str(N) + '_' + str(np.round(mean(vk)*10)/10) + '_beta_' + str(beta) + '_mu' + str(mu) + '.txt'
np.savetxt(file_name,vrho_matrix, delimiter=',', fmt='%1.6f')




# In[9]:


#network Measures
vk = dict(G.degree())
vk = list(vk.values())

vcc = []
for i in G.nodes():
    vcc.append(nx.clustering(G, i))
vcc= np.array(vcc)

CLC = dict(nx.closeness_centrality(G))
CLC = list(CLC.values())
B = dict(nx.betweenness_centrality(G))
B = list(B.values())
EC = dict(nx.eigenvector_centrality(G, max_iter = 1000))
EC = list(EC.values())
PR = dict(nx.pagerank(G, alpha=0.85))
PR = list(PR.values())
KC= dict(nx.core_number(G))
KC = list(KC.values())

M = np.column_stack((vk,vcc,CLC, B, EC, PR, KC, vrho_Rf))



M = np.column_stack((vk,vcc,CLC, B, EC, PR, KC, vrho_Rf))
file_name = 'out/' + str_net + '_' + str(N) + '_' + str(np.round(mean(vk)*10)/10) + '_beta_' + str(beta) + '_mu' + str(mu) + '.txt'
np.savetxt(file_name, M, delimiter=',', fmt='%1.6f')



# write only the measures
M = np.column_stack((vk,vcc,CLC, B, EC, PR, KC))
file_name = 'out/Measures_' + str_net + '_' + str(N) + '_' + str(np.round(mean(vk)*10)/10) + '_beta_' + str(beta) + '_mu' + str(mu) + '.txt'
np.savetxt(file_name, M, delimiter=',', fmt='%1.6f')

