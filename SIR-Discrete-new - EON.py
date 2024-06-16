
# coding: utf-8

# Francisco A. Rodrigues, University of São Paulo. http://conteudo.icmc.usp.br/pessoas/francisco

# # Epidemic spreading on networks

# Initially, we have to import the libraries necessary for our simulations.



import numpy as np
#import matplotlib.pyplot as plt
import networkx as nx
import random as random
random.seed(100) #set the random seed. Important to reproduce our results.
from scipy.linalg import expm
import math as math
import os
import EoN



# Let us generate an artificial random network. The network is the medium in which the infectious agent or information spread.


## Network parameters
#N = 100 #number of nodes
#av_degree = 8 # average degree
#p = float(av_degree)/float(N) #probability of connection in the ER model
#m = int(av_degree/2) # number of nodes included at each time step in the BA model
#kappa = av_degree # number of neighbors in the WS model
#
#G = nx.barabasi_albert_graph(N,m) # generate a BA network
#str_net = 'BA'
##############    
    
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

Ns = 30
mu = 1.0


file = 'nets-small/'
#arr = os.listdir(file)
files = []
for item in os.listdir(file):
    if not item.startswith('.') and os.path.isfile(os.path.join(file, item)):
        files.append(item)

print(files)
nnet = 0
for net in files:
    print(net)
    
    x = net
    x = x.split('.')
    str_net = x[0]
    print(str_net)
    
    # read the network
    G = nx.read_edgelist(file + net)
    G.remove_edges_from(nx.selfloop_edges(G))
    G = G.to_undirected()
    Gcc = sorted(nx.connected_components(G), key = len, reverse=True)
    G = G.subgraph(Gcc[0])
    G = nx.convert_node_labels_to_integers(G, first_label=0)

    
    N = len(G) # Number of nodes
    vk = dict(G.degree())
    vk = list(vk.values())
    av_degree = np.mean(vk)
    print("Number of nodes:", N)
    print("Average degree: ", av_degree)
    
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
    acc= list(accessibility(G))

    mu = 1.0
    A = nx.to_numpy_matrix(G)
    e = np.max(np.linalg.eigvals(A))
    lambda_c = mu/e
    #vbeta = np.arange(0.5*lambda_c,20*lambda_c, lambda_c)
    vbeta = np.arange(0.05,1,0.1)

    for beta in vbeta:
        vrho_Rf = np.zeros(N)
        vrho_matrix = np.zeros((N,Ns), dtype=float)
        for seed_node in G.nodes(): #starting from each node
            seeds = [seed_node] 
            rec = []
            for ns in np.arange(0, Ns):
                t, vS, vI, vR = EoN.Gillespie_SIR(G, beta, mu, initial_infecteds = seeds)
                rec.append(vR[-1]/N)
                vrho_matrix[seed_node][ns] = vR[-1]/N
            vrho_Rf[seed_node] = np.mean(rec)
        
        file_name = 'out-new/EON/' + str_net + '_matrix_infections_' + str(N) + '_' + str(np.round(np.mean(vk)*10)/10) + '_beta_' + str(int(beta*100)/100) + '_mu' + str(mu) + '.txt'
        np.savetxt(file_name,vrho_matrix, delimiter=',', fmt='%1.6f')
    
        M = np.column_stack((vk,vcc,CLC, B, EC, PR, KC, acc, vrho_Rf))
        file_name = 'out-new/EON/' + str_net +'_measures_rho_N' + str(N) + '_k' + str(np.round(np.mean(vk)*10)/10) + '_beta_' + str(int(beta*100)/100) + '_mu' + str(mu) + '.txt'
        np.savetxt(file_name, M, delimiter=',', fmt='%1.6f')



# write only the measures
#M = np.column_stack((vk,vcc,CLC, B, EC, PR, KC))
#file_name = 'out-new/Measures_' + str_net + '_' + str(N) + '_' + str(np.round(mean(vk)*10)/10) + '_beta_' + str(beta) + '_mu' + str(mu) + '.txt'
#np.savetxt(file_name, M, delimiter=',', fmt='%1.6f')

