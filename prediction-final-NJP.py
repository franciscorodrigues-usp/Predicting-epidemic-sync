#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 13:33:03 2020

@author: franciscoaparecidorodrigues
"""
########## Critical treshold #########
import networkx as nx


def momment_of_degree_distribution(G,m):
    M = 0
    N = len(G)
    for i in G.nodes:
        M = M + G.degree(i)**m
    M = M/N
    return M


# vnet_name = ['bitcoinalpha',
#               'cit-DBLP', 
#               'cong-votes' ,
#               'dolphins', 
#               'email-Eu-core',
#               'Gnutella08',
#               'hamsterster',
#               'inf-USAir97',
#               'jazz',
#               'moreno_blogs_blogs',
#               'netscience',
#               'polblogs',
#               'USairport_2010',
#               'USairport500']

# vnet_name2 = ['_measures_rho_N3775_k7.5_beta_',
#               '_measures_rho_N12495_k7.9_beta_',
#               '_measures_rho_N219_k4.8_beta_',
#               '_measures_rho_N62_k5.1_beta_',
#               '_measures_rho_N986_k32.6_beta_',
#               '_measures_rho_N6299_k6.6_beta_', 
#               '_measures_rho_N1788_k14.0_beta_',
#               '_measures_rho_N332_k12.8_beta_',
#               '_measures_rho_N198_k27.7_beta_',
#               '_measures_rho_N1222_k27.4_beta_',
#               '_measures_rho_N379_k4.8_beta_',
#               '_measures_rho_N1222_k27.4_beta_',
#               '_measures_rho_N1572_k21.9_beta_',
#               '_measures_rho_N500_k11.9_beta_']


vnet_name = ['bitcoinalpha',
              'email-Eu-core',
              'Gnutella08',
              'hamsterster',
              'polblogs',
              'USairport500']
             
vnet_name2 = ['_measures_rho_N3775_k7.5_beta_',
              '_measures_rho_N986_k32.6_beta_',
              '_measures_rho_N6299_k6.6_beta_', 
              '_measures_rho_N1788_k14.0_beta_',
              '_measures_rho_N1222_k27.4_beta_',
              '_measures_rho_N500_k11.9_beta_']


#vnet_name = ['dolphins']
#vnet_name2 = ['_measures_rho_N62_k5.1_beta_']

# test files
# for i in range(0,len(vnet_name)):
#     print('(**************')
#     net_name = vnet_name[i]
#     net_name2 =vnet_name2[i]
#     print('Net:',net_name)
#     G= nx.read_edgelist('nets/' + net_name + '.txt', nodetype=int, data=(('weight',float),)) # Read the network
#     mu = 1
#     lambdac = mu*momment_of_degree_distribution(G,1)/momment_of_degree_distribution(G,2)
#     print('lambdac =', lambdac)

###############################

# Pandas is used for data manipulation
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


#Nested k-fold cross validation
def regression_nested_crossvalidation_RF(X, y, n_folds,k_folds, model, hparameters):
    # define the external cross validation
    cv_outer = KFold(n_folds, shuffle=True)
    verror = list()
    vR2 = list()
    # CV external lood
    y_out = []
    vimportant = []
    for train_i, test_i in cv_outer.split(X):
        X_train, y_train = X[train_i], y[train_i]
        X_test, y_test = X[test_i], y[test_i]
        # internal CV (training and validation sets)
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)        
        X_test = scaler.transform(X_test)     
        
        grid_search_cv = GridSearchCV(model, hparameters, cv=k_folds, 
                                      scoring='neg_mean_squared_error', refit=True)
        # find the best hyperparameters
        result = grid_search_cv.fit(X_train, y_train)
        best_model = result.best_estimator_
        # prediction in the test fold
        y_pred = best_model.predict(X_test) 
        # evaluate and store the results
        error = metrics.mean_squared_error(y_test,y_pred)
        R2 = metrics.r2_score(y_test,y_pred)
        print("Mean Square Error:", error)
        verror.append(error)
        vR2.append(R2)
        #print('Best hyperparameters (extern): ', result.best_params_)
        print('.',end="")
        y_out.append((y_pred, y_test))
        vimportant.append(best_model.feature_importances_)
        
    print('\n')
    av_error = np.mean(verror)
    std_error = np.std(verror)
    av_R2 = np.mean(vR2)
    std_R2 = np.std(vR2)
    vimportances = np.array(vimportant)
    imp = np.mean(vimportant,axis = 0)
    imp = imp/np.sum(imp)
    return av_error,std_error,av_R2, std_R2, best_model, y_out, imp


#Nested k-fold cross validation
def regression_nested_crossvalidation_LR(X, y, n_folds,k_folds, model, hparameters):
    # define the external cross validation
    cv_outer = KFold(n_folds, shuffle=True)
    verror = list()
    vR2 = list()
    # CV external lood
    y_out = []
    for train_i, test_i in cv_outer.split(X):
        X_train, y_train = X[train_i], y[train_i]
        X_test, y_test = X[test_i], y[test_i]
        # internal CV (training and validation sets)
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)        
        X_test = scaler.transform(X_test)     
        
        grid_search_cv = GridSearchCV(model, hparameters, cv=k_folds, 
                                      scoring='neg_mean_squared_error', refit=True)
        # find the best hyperparameters
        result = grid_search_cv.fit(X_train, y_train)
        best_model = result.best_estimator_
        # prediction in the test fold
        y_pred = best_model.predict(X_test) 
        # evaluate and store the results
        error = metrics.mean_squared_error(y_test,y_pred)
        R2 = metrics.r2_score(y_test,y_pred)
        print("Mean Square Error:", error)
        verror.append(error)
        vR2.append(R2)
        #print('Best hyperparameters (extern): ', result.best_params_)
        print('.',end="")
        y_out.append((y_pred, y_test))
        
    print('\n')
    av_error = np.mean(verror)
    std_error = np.std(verror)
    av_R2 = np.mean(vR2)
    std_R2 = np.std(vR2)
    return av_error,std_error,av_R2, std_R2, best_model, y_out



for i in range(0,len(vnet_name)):
    print('(**************')
    net_name = vnet_name[i]
    net_name2 =vnet_name2[i]
    print('Net:',net_name)
    
    G= nx.read_edgelist('nets/' + net_name + '.txt', nodetype=int, data=(('weight',float),)) # Read the network
    mu = 1
    lambdac = mu*momment_of_degree_distribution(G,1)/momment_of_degree_distribution(G,2)
    
    verror_RF = list()
    verror_LR = list()
    verror_NN = list()
    vR2_RF = list()
    vR2_LR = list()
    vR2_NN = list()
    
    #betas = np.arange(0.05,1,0.1)
    betas = np.arange(0.05,1,0.1)
    betas = np.round(betas*100)/100
    vimport = []
    for beta in betas:
    
        # Read the network
        str_data = 'out-new-2020/' + net_name + net_name2 + str(beta) +'_mu1.0.txt'
        
        data = pd.read_table(str_data, sep = ',', header = None)
        data.columns = ['vk','vcc','CLC', 'B', 'EC', 'PR', 'KC','ACC', 'rho']
        list_labels = list(data.columns)
        rho = data['rho']
        list_labels = list(data.columns)
        
    #    corr = data.corr()
    #    #Plot Correlation Matrix using Matplotlib
    #    plt.figure(figsize=(7, 7))
    #    plt.imshow(corr, cmap='Blues', interpolation='none', aspect='auto')
    #    plt.colorbar()
    #    plt.xticks(range(len(corr)), corr.columns, rotation='vertical')
    #    plt.yticks(range(len(corr)), corr.columns);
    #    plt.suptitle('Correlation between variables', fontsize=15, fontweight='bold')
    #    plt.grid(False)
    #    plt.show()
        
        ##############################
        data2 = data.to_numpy()
        nrow,ncol = data2.shape
        y = data2[:,-1]
        X = data2[:,0:ncol-1]
        features = data.columns[0:-1]
        ##############################
        
        n_folds = 10 # cv external folds
        k_folds = 10 # cv internal folds (training set)
        model = RandomForestRegressor()
        #model = LinearRegression()
        parameters = {}
        av_error, std_error, av_R2,std_R2, model, y_out, imp = regression_nested_crossvalidation_RF(X, y, n_folds,k_folds, model, parameters)
        print('RF important:', imp)
        for w in y_out:
            #m = np.max(w[0])
            plt.scatter(w[0],w[1], s=80, edgecolors='white', color="blue", alpha=0.5)
        
        #plot the straight line
        plt.plot([0,np.max(y_out[0][:])],[0,np.max(y_out[0][:])],'r--')
        plt.title(str(model)+'_' + str(beta) + '_R2:' + str(av_R2))
        plt.ylabel("data", fontsize=20)
        plt.xlabel("prediction", fontsize=20)
        #plt.savefig('plots/' + net_name + '_RF_' + str(model)+'_beta_' + str(beta) + '.svg')
        #plt.savefig('plots/' + net_name + '_RF_' + str(model)+'_beta_' + str(beta) + '.pdf')
        plt.savefig('plots/' + net_name + '_RF_' + str(model)+'_beta_' + str(beta) + '.svg')

    
        plt.show(True)
        
        verror_RF.append(av_error)
        vR2_RF.append(av_R2)
        
        # Importances
        importances = imp
        indices = np.argsort(importances)
        lmeas_order = []
        for i in indices:
            lmeas_order.append(list_labels[i])
        #fig = plt.figure()
        #plt.title('Feature Importances')
        plt.figure(figsize=(6,6))
        plt.tight_layout()
        
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), lmeas_order, fontsize=25)
        plt.xlabel('Relative Importance',fontsize=25)
        plt.xticks(color='k', size=20)
        plt.yticks(color='k', size=20)
        #plt.xlim([0.0, 0.25])
        #plt.savefig('importance' + net_name + '.svg')
        plt.show(True)
        vimport.append(importances)
            
        
        
        # Linear Regression
        model = LinearRegression()
        parameters = {}
        av_error, std_error,av_R2,std_R2, model, y_out = regression_nested_crossvalidation_LR(X, y, n_folds,k_folds, model, parameters)
        
        for w in y_out:
            #m = np.max(w[0])
            plt.scatter(w[0],w[1], s=80, edgecolors='white', color="blue", alpha=0.5)
        
        plt.plot([0,np.max(y_out[0][:])],[0,np.max(y_out[0][:])],'r')
        plt.title(str(model)+'_beta:' + str(beta) + '_R2:' + str(av_R2))
        #plt.savefig('plots/' + net_name + '_Linear_' + str(model)+ '_beta_' + str(beta) + '.svg')
        #plt.savefig('plots/' + net_name + '_Linear_' + str(model)+ '_beta_' + str(beta) + '.pdf')
        plt.savefig('plots/' + net_name + '_Linear_' + str(model)+ '_beta_' + str(beta) + '.svg')
    
        plt.show(True)
    
        verror_LR.append(av_error)
        vR2_LR.append(av_R2)
        
        
    #    parameters = {"hidden_layer_sizes": [(5,),(5,5),(5,5,5)], 
    #                 "activation": ["identity", "logistic", "tanh", "relu"], 
    #                 "solver": ["lbfgs", "sgd", "adam"], 
    #                 #"alpha": [0.00005,0.0005],
    #                 #"learning_rate": ['constant', 'invscaling', 'adaptive']
    #                 }
    #    model = MLPRegressor(max_iter=7000)
    #    av_error, std_error, model, y_predicted = regression_nested_crossvalidation(X, y, n_folds,k_folds, model, parameters)
    #    
    #    for w in y_predicted:
    #        m = np.max(w[0])
    #        plt.plot(w[0],w[1],'bo')
    #    plt.plot([0,np.max(y_predicted[0][:])],[0,np.max(y_predicted[0][:])],'r')
    #    plt.title(str(model))
    #    plt.show(True)
    #    
    #    verror_NN.append(av_error)
    #    
    
    plt.plot(betas,vR2_RF,'o-r', label= 'Random Forest')
    plt.plot(betas,vR2_LR,'o-b', label = 'Linear Regression')
    #plt.plot(betas,verror_NN,'o-g', label ='Neural Network')
    plt.legend(fontsize = 15)
    plt.xlabel("beta", fontsize=20)
    plt.ylabel("R2", fontsize=20)
    
    plt.axvline(x=lambdac, color = 'gray', linestyle = '--')
    #plt.savefig('plots/' + net_name + 'R2_' + '.svg')
    #plt.savefig('plots/' + net_name + 'R2' + '.pdf')
    plt.savefig('plots/' + net_name + 'R2' + '.svg')
    
    plt.show(True)
    
    
    #importances
    alpha = ['K', 'C', 'CC','B', 'EC', 'PR', 'KC','ACC']
    xlabels = list(betas)
    data = np.array(vimport)
    data = data.transpose()
  
      
         
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    #color map: https://chrisalbon.com/python/basics/set_the_color_of_a_matplotlib/
    cax = ax.matshow(data, interpolation='nearest', cmap=plt.cm.get_cmap('Blues', 10),
                     vmin=0,vmax = 1) #plt.cm.binary, plt.cm.Blues
    ax.xaxis.set_ticks_position('bottom')
    cb = fig.colorbar(cax, label='importance', shrink=1)
    cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=15)
    
    
    ax.set_xticklabels(['']+xlabels)
    ax.set_yticklabels(['']+alpha)
    plt.xlabel("beta", fontsize=25)
    plt.xticks(color='k', size=20)
    plt.yticks(color='k', size=20)
    
    #plt.savefig('plots/' + net_name + '_importance.pdf')
    plt.savefig('plots/' + net_name + '_importance.svg')
    plt.savefig('plots/' + net_name + '_importance.pdf')
    plt.show()


    
