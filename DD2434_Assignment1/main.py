# -*- coding: utf-8 -*-
"""
Main file for assignment 1 of DD2434

#author Henrik Hellström
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import assignment1_functions as funcs

#Number of data points in dataset
n = 101
#Number of dimensions
d = 16
algorithm = "PCA"
#algorithm = "MDS"
#algorithm = "isomap"

#Load dataset from file
(Y, types, names) = funcs.loadDataset("dataset/zoo.data")

#Preprocessing on Y
Y = funcs.preprocessing(Y)
print("Y.shape =", Y.shape)

if algorithm == "PCA":
    #Take SVD of Y
    U_full, S, VH = np.linalg.svd(Y)
    
    #Get the 2 columns of U corresponding to the highest singular values
    U = U_full[:, 0:2]
    
    #Map Y to 2-dimensional space
    X = np.matmul(U.T, Y)
    
    print("U=", U)
    print("U[:,0]", np.sort(U[:,0]))
    print("U[:,1]", np.sort(U[:,1]))
    
elif algorithm == "MDS":
    #Form distance matrix
    D = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                D[i,j] = 0
            else:
                y_i = Y[:,i]
                y_j = Y[:,j]
                dist = np.linalg.norm(y_i-y_j)
                D[i,j] = dist*dist
    #Distance between seal and sealion
    print("D[74,75] =", D[74,75]) 
    #Distance between dolphin and honeybee
    print("D[19,39] =", D[19,39])
    
    #Form S matrix with double centering
    S = -0.5*(D - 1/n*np.matmul(D, np.ones((n,n))) -
              1/n*np.matmul(np.ones((n,n)), D) +
              1/(n*n)*np.matmul(np.matmul(np.ones((n,n)), D), np.ones((n,n))))
    
    #Take the eigenvalue decomposition
    l, U = np.linalg.eig(S)
    l = np.real(l)
    for i in range(n):
        if l[i] < 0.001:
            l[i] = 0
    l = np.sqrt(l)
    U = np.real(U)
    Lambda = np.diag(l)
    print("Lambda.shape =", Lambda.shape)
    print("U.shape =", U.shape)
    
    #Form the low-dimensional X with MDS
    X = np.matmul(np.matmul(np.eye(2 ,n), Lambda), U.T)

elif algorithm == "isomap":
    #Form distance matrix
    D = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                D[i,j] = 0
            else:
                y_i = Y[:,i]
                y_j = Y[:,j]
                dist = np.linalg.norm(y_i-y_j)
                D[i,j] = dist*dist
    
    #Construct graph
    #p = 17
    p = 30
    for i in range(n):
        col = D[:,i]
        infinity = 1000
        col[i] = infinity #The distance to your own position should be infinite
        minpos_list = []
        dist_list = []
        for j in range(p):
            minpos = np.argmin(col)
            dist_list.append(col[minpos])
            col[minpos] = infinity
            minpos_list.append(minpos)
        col = 1000*np.ones((n, 1))
        for j in range(len(minpos_list)):
            minpos = minpos_list[j]
            col[minpos] = dist_list[j]
        D[:,i] = col.ravel()
    G = D
    
    #Floyd-Warshall algorithm
    D_fw = infinity*np.ones((n,n))
    #Initialize 1-step distance
    D_fw = G
    #More than 1 step
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if D_fw[i,j] > D_fw[i,k] + D_fw[k,j]:
                    D_fw[i,j] = D_fw[i,k] + D_fw[k,j]
    
    #Check if graph is connected
    print("np.max(D_fw) =", np.max(D_fw))
    if np.max(D_fw) < 1000:
        print("Graph is connected!")
    
    #Run MDS on the distance matrix
    #Form S matrix with double centering
    S = -0.5*(D - 1/n*np.matmul(D, np.ones((n,n))) -
              1/n*np.matmul(np.ones((n,n)), D) +
              1/(n*n)*np.matmul(np.matmul(np.ones((n,n)), D), np.ones((n,n))))
    
    #Take the eigenvalue decomposition
    l, U = np.linalg.eig(S)
    l = np.real(l)
    for i in range(n):
        if l[i] < 0.001:
            l[i] = 0
    l = np.sqrt(l)
    U = np.real(U)
    Lambda = np.diag(l)
    print("Lambda.shape =", Lambda.shape)
    print("U.shape =", U.shape)
    
    #Form the low-dimensional X with MDS
    X = np.matmul(np.matmul(np.eye(2 ,n), Lambda), U.T)
else:
	print("Unknown algorithm, cannot run program. Please see README.txt file")
	sys.exit(0)

annotated_coords = [] 
#Plot results
for i in range(n):
    style = 'bo'
    if types[i] == 1:
        style = 'bo'
    elif types[i] == 2:
        style = 'go'
    elif types[i] == 3:
        style = 'ro'
    elif types[i] == 4:
        style = 'co'
    elif types[i] == 5:
        style = 'mo'
    elif types[i] == 6:
        style = 'yo'
    elif types[i] == 7:
        style = 'ko'
    else:
        print("unknown type")
    plt.plot(X[0,i], X[1,i], style)
    used = False
    for j in range(len(annotated_coords)):
        if annotated_coords[j] == (X[0,i], X[1,i]):
            used = True
    if used == False:
        plt.annotate(names[i], (X[0,i], X[1,i]))
        annotated_coords.append((X[0,i], X[1,i]))

patch1 = mpatches.Patch(color='blue', label='Mammals')
patch2 = mpatches.Patch(color='green', label='Birds')
patch3 = mpatches.Patch(color='red', label='Reptiles')
patch4 = mpatches.Patch(color='cyan', label='Fish')
patch5 = mpatches.Patch(color='magenta', label='Amphibians')
patch6 = mpatches.Patch(color='yellow', label='Insects')
patch7 = mpatches.Patch(color='black', label='Invertebrate')
plt.legend(handles=[patch1, patch2, patch3, patch4, patch5, patch6, patch7])
plt.show()    
    
