#import yaml
import sys, os
import tensorflow.compat.v1 as tf
import numpy as np
import time
import pandas as pd
import tensorflow as tf
from Validation import Validation, createFolder
from scipy.sparse import coo_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy import stats
from lib import models, graph, coarsening, utils
from scipy.sparse import coo_matrix
import druggcn
import candle
from pathlib import Path

file_path = os.path.dirname(os.path.realpath(__file__))

def initialize_parameters(default_model="druggcn_default_model.txt"):

    # Build benchmark object
    common = druggcn.DrugGCN(
        file_path,
        default_model,
        "tensorflow",
        prog="Graph Convolutional Network for Drug Response Prediction Using Gene Expression Data (DrugGCN)",
        desc="DrugGCN drug response prediction model",
    )

    # Initialize parameters
    gParameters = candle.finalize_parameters(common)

    return gParameters

def run(gParameters): 

    args = candle.ArgumentStruct(**gParameters)

    if tf.test.gpu_device_name():
        print("Default GPU Device:{}".format(tf.test.gpu_device_name()))
    else:
        print("GPU not available")

    fdir = Path(__file__).resolve().parent
    if args.output_dir is not None:
        outdir = args.output_dir
    else:
        outdir = fdir/f"/results"
    os.makedirs(outdir, exist_ok=True)
    
    PPI_data = args.ppi_data
    Response_data = args.response_data
    Gene_data = args.gene_data
    n_fold = args.n_fold
    test_size = args.test_size
    num_epochs = args.epochs
    batch_size = args.batch_size
    brelu = args.brelu
    pool = args.pool_type
    regularization = args.regularization
    dropout = args.dropout
    learning_rate = args.learning_rate
    decay_rate = args.decay_rate
    momentum = args.momentum
    Name = args.name
    F = args.f
    K = args.k
    p = args.pool
    M = args.dense
    
    data_PPI = pd.read_csv(str(args.data_dir) + "/" + PPI_data)
    data_PPI.drop(['Unnamed: 0'], axis='columns', inplace=True)
    data_IC50 = pd.read_csv(str(args.data_dir) + "/" + Response_data)
    data_IC50.drop(['Unnamed: 0'], axis='columns', inplace=True)
    data_Gene = pd.read_csv(str(args.data_dir) + "/" + Gene_data)
    data_Gene.drop(['Unnamed: 0'], axis='columns', inplace=True)
    data_Gene = np.array(data_Gene)

    df = np.array(data_PPI)
    A = coo_matrix(df,dtype=np.float32)
    print(A.nnz)
    graphs, perm = coarsening.coarsen(A, levels=6, self_connections=False)
    L = [graph.laplacian(A, normalized=True) for A in graphs]
    graph.plot_spectrum(L)

    n_fold = n_fold
    PCC = []
    SPC = []
    RMSE = []

    X_train, X_test, Y_train, Y_test = train_test_split(data_Gene, data_IC50, 
                                                                  test_size=test_size, shuffle=True, random_state=args.rng_seed)


    for cv in range(n_fold):   
        Y_pred = np.zeros([Y_test.shape[0], Y_test.shape[1]])
        Y_test = np.zeros([Y_test.shape[0], Y_test.shape[1]])
        j = 0
        #for i in range(Y.test.shape[1]):
        for i in range(Y_test.shape[1]):
            data1 = data_IC50.iloc[:,i]
            data1 = np.array(data1)
            data_minmax = data1[~np.isnan(data1)]
            min = data_minmax.min()
            max = data_minmax.max()
            data1 = (data1 - min) / (max - min)

            train_data_split, test_data_split, train_labels_split, test_labels_split = train_test_split(data_Gene, data1, 
                                                                  test_size=test_size, shuffle=True, random_state=args.rng_seed)
            train_data = np.array(train_data_split[~np.isnan(train_labels_split)]).astype(np.float32)


            list_train, list_val = Validation(n_fold,train_data,train_labels_split)

            train_data_V = train_data[list_train[cv]]
            val_data = train_data[list_val[cv]]
            test_data = np.array(test_data_split[:]).astype(np.float32)
            train_labels = np.array(train_labels_split[~np.isnan(train_labels_split)]).astype(np.float32)
            train_labels_V = train_labels[list_train[cv]]
            val_labels = train_labels[list_val[cv]]
            test_labels = np.array(test_labels_split[:]).astype(np.float32)
            train_data_V = coarsening.perm_data(train_data_V, perm)
            val_data = coarsening.perm_data(val_data, perm)
            test_data = coarsening.perm_data(test_data, perm)

            common = {}
            common['num_epochs']     = num_epochs
            common['batch_size']     = batch_size
            common['decay_steps']    = train_data.shape[0] / common['batch_size']
            common['eval_frequency'] = 10 * common['num_epochs']
            common['brelu']          = brelu
            common['pool']           = pool

            common['regularization'] = regularization
            common['dropout']        = dropout
            common['learning_rate']  = learning_rate
            common['decay_rate']     = decay_rate
            common['momentum']       = momentum
            common['F']              = F
            common['K']              = K
            common['p']              = p
            common['M']              = M
            common['dir_name']       = outdir

            if True:
                name = Name
                params = common.copy()

            model = models.cgcnn(L, **params)
            loss, t_step = model.fit(train_data_V, train_labels_V, val_data, val_labels)

            Y_pred[:, j] = model.predict(test_data)
            Y_test[:, j] = test_labels
            j = j+1

        np.savez((str(outdir) + "/" + "GraphCNN_CV_{}".format(cv)), Y_true=Y_test, Y_pred=Y_pred)


def main():
    gParameters = initialize_parameters()
    run(gParameters)

if __name__ == "__main__":
    main()