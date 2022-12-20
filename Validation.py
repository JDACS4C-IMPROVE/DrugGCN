import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def Validation(n_fold,X,Y,size=0.25,seed=20):
    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y)
    list_train_fold = []
    list_val_fold = []
    list_train = []
    Number = X.shape[0]//n_fold
    for i in range(X.shape[0]):
        list_train.append(i)

    # if condition returns False, AssertionError is raised:
    assert n_fold > 0, "n_fold should be at least 1"
    
    # when n_fold is 1
    if n_fold-1 == 0: 
        list_val = []
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=size, shuffle=True, random_state=seed)
        for j in list(Y_val.index):
            list_val.append(j)
        list_train_fold.append(np.setdiff1d(list_train,list_val))
        list_val_fold.append(list_val)

    # when n_fold is greater than 1      
    if n_fold-1 > 0:
        for i in range(n_fold)[::-1]:
            list_val = []
            if i==n_fold-1:
                for j in range(Number*i,X.shape[0]):
                    list_val.append(j)
                list_train_fold.append(np.setdiff1d(list_train,list_val))
                list_val_fold.append(list_val)
                
            
            if i != n_fold-1: 
                for j in range(Number*i,Number*(i+1)):
                    list_val.append(j)
                list_train_fold.append(np.setdiff1d(list_train,list_val))
                list_val_fold.append(list_val)

    return list_train_fold, list_val_fold


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
 

