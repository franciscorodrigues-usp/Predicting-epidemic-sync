#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#"""
#Created on Thu Sep 12 21:54:10 2019

#@author: franciscoaparecidorodrigues
#"""

#from IPython import get_ipython
#get_ipython().magic('reset -sf')

import networkx as nx
import numpy as np
import EoN
import matplotlib.pyplot as plt
# Pandas is used for data manipulation
import pandas as pd
from scipy.linalg import expm
import math as math


#N = 200
#av_degree = 8
#p = av_degree/(N-1)
#G = nx.gnp_random_graph(N, p, seed=None, directed=False)

#G=nx.read_edgelist("nets/euroroad.txt", nodetype=int)
#G= nx.read_edgelist("nets/USairports.txt", nodetype=int, data=(('weight',float),)) # Read the network
G= nx.read_edgelist("nets/lesmis_Net.txt", nodetype=int, data=(('weight',float),)) # Read the network
#GG= nx.read_edgelist("nets/celegans.txt", nodetype=int, data=(('weight',float),)) # Read the network

#Q=0.4
#G = GG.copy()
#nx.double_edge_swap(G, nswap=Q * len(G.edges()), max_tries=1e75)

G = G.to_undirected()
Gcc=sorted(nx.connected_component_subgraphs(G), key = len, reverse=True)
G=Gcc[0]
G = nx.convert_node_labels_to_integers(G, first_label=0)
N = len(G)
M = G.number_of_edges()
print('Number of nodes:', N)
print('Number of edges:', M)

L = nx.normalized_laplacian_matrix(G)
e = np.linalg.eigvals(L.A)
print("Largest eigenvalue:", max(e))
print("Smallest eigenvalue:", min(e))
main_eigenvalue = max(e)

#Epidemic spreading with time
mu = 1.
#beta = (mu/main_eigenvalue)
beta = 0.3
rho = np.zeros(N)
plt.figure()
Ns = 50

r_all= []
for i in G.nodes():
    r = []
    for s in np.arange(0,Ns):        
        t, S, I, R = EoN.fast_SIR(G, beta, mu, initial_infecteds = i)
        r.append(R[-1])
    #rho[i] = max(R)/len(G.nodes())
    r_all.append(r)
    rho[i] = np.mean(r)/len(G.nodes())
    plt.plot(t, I, color='red', linestyle='dashed',linewidth=0.5, marker='o',markersize=0.5)
    plt.xlabel("t", fontsize=15)
plt.ylabel("Fraction of infected nodes", fontsize=15)
plt.legend()
plt.grid(True)
plt.show(True)  
rho=np.asarray(rho)
plt.hist(rho, bins='auto')  # arguments are passed to np.histogram
plt.show(True)
print(rho)


#Network measures
N = len(G) # Number of nodes
vk = dict(G.degree())
vk = list(vk.values())
#print("Degree:", vk)

cc = dict(nx.clustering(G))
cc = list(cc.values())
#print("Clustering coefficient:", cc)

bt = dict(nx.betweenness_centrality(G))
bt = list(bt.values())
#print("Betweenness centrality:", bt)

eg = list(nx.eigenvector_centrality(G, max_iter=1000).values())
#print("Eigenvector centrality: ", eg)

cl = dict(nx.closeness_centrality(G))
cl = list(cl.values())
#print("Closeness centrality:", bt)

kc= dict(nx.core_number(G))
kc = list(kc.values())
#print("K-core:", kc)
print("Measures calculated...")


def accessibility(G):
    vk = dict(G.degree())
    vk = list(vk.values())
    A = nx.adjacency_matrix(G)
    P = np.zeros((N,N), dtype = 'float')
    for i in np.arange(0, N):
        for j in np.arange(0, N):
            if(vk[i] > 0):
                P[i,j] = A[i,j]/vk[i]
    P2 = expm(P)/np.exp(1)
    vacc = np.zeros(N, dtype = float)
    for i in np.arange(0, N):
        acc = 0
        for j in np.arange(0,N):
            if(P2[i,j] > 0):
                acc = acc + P2[i,j]*math.log(P2[i,j])
        acc = np.exp(-acc)
        vacc[i] = acc
    return vacc

acc= list(accessibility(G))

import pandas as pd
M = np.column_stack((vk,cc, bt, cl, kc, eg, acc, rho))
#df = pd.DataFrame(M, columns = ['vk', 'cc', 'bt', 'cl', 'kc', 'eg', 'acc', 'com', 'rho'])
df = pd.DataFrame({'K':vk,'CC':cc, 'Bt':bt,'CL':cl, 'KC':kc, 'EG':eg, 'ACC':acc, 'rho':rho})

df.to_csv('out.csv', index=False)

r_all = np.array(r_all)
R = pd.DataFrame(r_all)
R.to_csv('fraction_of_recovered.csv', index=False)