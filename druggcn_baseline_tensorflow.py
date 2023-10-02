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
from candle.file_utils import directory_tree_from_parameters
#from candle.file_utils import get_file
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import json

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
    F = args.f
    K = args.k
    p = args.pool
    M = args.dense

    # get data from server or candle
    data_file_path = candle.get_file(args.processed_data, args.data_url + args.processed_data, datadir = args.data_dir, cache_subdir = None)
    #print(data_file_path)
    
    data_PPI = pd.read_csv(str(args.data_dir) + "/data_processed/" + PPI_data)
    data_PPI.drop(['Unnamed: 0'], axis='columns', inplace=True)
    data_IC50 = pd.read_csv(str(args.data_dir) + "/data_processed/" + Response_data)
    cell_id = pd.DataFrame(data_IC50['Unnamed: 0']).rename(columns={"Unnamed: 0": "CancID"}) # dataframe of cell ids only
    data_IC50.drop(['Unnamed: 0'], axis='columns', inplace=True)
    drug_list = list(data_IC50.columns) # list of drug names
    data_Gene = pd.read_csv(str(args.data_dir) + "/data_processed/" + Gene_data)
    data_Gene.drop(['Unnamed: 0'], axis='columns', inplace=True)
    data_Gene = np.array(data_Gene)

    df = np.array(data_PPI)
    A = coo_matrix(df, dtype=np.float32)
    #print(A.nnz)
    graphs, perm = coarsening.coarsen(A, levels=args.levels, self_connections=False)
    L = [graph.laplacian(A, normalized=True) for A in graphs]
    #graph.plot_spectrum(L)

    n_fold = n_fold
    PCC = []
    SCC = []
    RMSE = []
    MSE = []

    X_train, X_test, Y_train, Y_test = train_test_split(data_Gene, data_IC50, 
                                                                  test_size=test_size, shuffle=True, random_state=args.rng_seed)


    # initialize scaler object
    scaler = MinMaxScaler()
    # fit and transform response data
    data_IC50_scaled = pd.DataFrame(scaler.fit_transform(data_IC50.copy()))
    # get cell ids for test dataset
    test_cell_ids = cell_id[cell_id.index.isin(list(Y_test.index))]
        
    # loop through each n fold and drug to get train dataset size
    batch_sizes = []
    for cv in range(n_fold): 
        for i in range(data_IC50_scaled.shape[1]):
            data1 = data_IC50_scaled.iloc[:,i]
            train_data_split, test_data_split, train_labels_split, test_labels_split = train_test_split(data_Gene, data1, 
                                                                    test_size=test_size, shuffle=True, random_state=args.rng_seed)
        
            train_data = np.array(train_data_split[~np.isnan(train_labels_split)]).astype(np.float32)
            train_labels = np.array(train_labels_split[~np.isnan(train_labels_split)]).astype(np.float32)
            list_train, list_val = Validation(n_fold,train_data,train_labels,args.val_size,args.rng_seed)
            batch_sizes.append(train_data[list_train[cv]].shape[0])
    # get minimum batch size
    min_batch_size = min(batch_sizes)
    if batch_size > min_batch_size:
        print(f"Please use a batch size equal to or less than {min_batch_size}.")
        print("Exiting...")
        sys.exit(1)
    
    # train model
    for cv in range(n_fold):   
        Y_pred = np.zeros([Y_test.shape[0], Y_test.shape[1]])
        Y_test = np.zeros([Y_test.shape[0], Y_test.shape[1]])
        
        # initialize dataframes to hold predictions and true values for validation dataset per drug
        all_val_pred = pd.DataFrame()
        all_val_test = pd.DataFrame()

        j = 0
        
        # loop through each drug
        for i in range(Y_test.shape[1]):
            data1 = data_IC50_scaled.iloc[:,i]

            train_data_split, test_data_split, train_labels_split, test_labels_split = train_test_split(data_Gene, data1, 
                                                                  test_size=test_size, shuffle=True, random_state=args.rng_seed)
    
            train_data = np.array(train_data_split[~np.isnan(train_labels_split)]).astype(np.float32)
            train_labels = np.array(train_labels_split[~np.isnan(train_labels_split)]).astype(np.float32)

            list_train, list_val = Validation(n_fold,train_data,train_labels,args.val_size,args.rng_seed)

            train_data_V = train_data[list_train[cv]]
            val_data = train_data[list_val[cv]]
            test_data = np.array(test_data_split[:]).astype(np.float32)
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
            common['dir_name']       = args.output_dir

            params = common.copy()

            model = models.cgcnn(L, **params)
            loss, t_step = model.fit(train_data_V, train_labels_V, val_data, val_labels)

            # make predictions with test dataset
            Y_pred[:, j] = model.predict(test_data)
            Y_test[:, j] = test_labels

            # make predictions with validation dataset
            val_pred = pd.DataFrame({drug_list[j]: model.predict(val_data)})
            val_test = pd.DataFrame({drug_list[j]: val_labels})
            all_val_pred = pd.concat([all_val_pred, val_pred], axis=1)
            all_val_test = pd.concat([all_val_test, val_test], axis=1)
            
            j = j+1

        # inverse scaling of values and reformat data for test dataset
        Y_test = pd.concat([pd.Series(test_labels_split.index), pd.DataFrame(scaler.inverse_transform(Y_test), columns = drug_list)], axis=1).set_index(0)
        Y_pred = pd.concat([pd.Series(test_labels_split.index), pd.DataFrame(scaler.inverse_transform(Y_pred), columns = drug_list)], axis=1).set_index(0)
        Y_test_t = pd.merge(test_cell_ids, Y_test, how="left", left_index=True, right_index=True)
        Y_pred_t = pd.merge(test_cell_ids, Y_pred, how="left", left_index=True, right_index=True)
        new_Y_test = pd.melt(Y_test_t, id_vars = "CancID", value_vars = Y_test_t.columns)
        new_Y_test = new_Y_test.rename(columns = {"variable": "DrugID", "value": "True"})
        new_Y_pred = pd.melt(Y_pred_t, id_vars = "CancID", value_vars = Y_pred_t.columns)
        new_Y_pred = new_Y_pred.rename(columns = {"variable": "DrugID", "value": "Pred"})
        true_pred_df = new_Y_test.merge(new_Y_pred, how="inner", on=["CancID","DrugID"])

        # save predictions - long format
        true_pred_df.to_csv(str(args.output_dir) + "/" + "raw_predictions_CV_{}.csv".format(cv),index=False)

        # save predictions - wide format
        np.savez((str(args.output_dir) + "/" + "GraphCNN_CV_{}".format(cv)), Y_true=Y_test, Y_pred=Y_pred)

        # inverse scaling of values and reformat data for validation dataset
        CV_test = pd.DataFrame(scaler.inverse_transform(all_val_test), columns = all_val_test.columns)
        CV_pred = pd.DataFrame(scaler.inverse_transform(all_val_pred), columns = all_val_pred.columns)
        new_CV_test = pd.melt(CV_test, value_vars = CV_test.columns)
        new_CV_test = new_CV_test.rename(columns = {"variable": "DrugID", "value": "True"})
        new_CV_pred = pd.melt(CV_pred, value_vars = CV_pred.columns)
        new_CV_pred = new_CV_pred.rename(columns = {"variable": "DrugID", "value": "Pred"})
        new_CV_all = pd.concat([new_CV_test, new_CV_pred], axis = 1)
        # filter out any null values
        new_CV_all = new_CV_all[new_CV_all["True"].notnull()]

        # get evaluation metrics for validation dataset
        RMSE.append(mean_squared_error(new_CV_all["True"], new_CV_all["Pred"], squared = False))
        MSE.append(mean_squared_error(new_CV_all["True"], new_CV_all["Pred"], squared = True))
        PCC.append(stats.pearsonr(new_CV_all["True"], new_CV_all["Pred"])[0])
        SCC.append(stats.spearmanr(new_CV_all["True"], new_CV_all["Pred"])[0])
    
    # get average of evaluation metrics for validation dataset
    rmse_avg = np.mean(RMSE)
    mse_avg = np.mean(MSE)
    pcc_avg = np.mean(PCC)
    scc_avg = np.mean(SCC)

    val_scores = {"val_loss": float(mse_avg), "pcc": float(pcc_avg), "scc": float(scc_avg), "rmse": float(rmse_avg)}

    # Supervisor HPO
    print("\nIMPROVE_RESULT val_loss:\t{}\n".format(val_scores["val_loss"]))
    with open(Path(args.output_dir) / "scores.json", "w", encoding="utf-8") as f:
        json.dump(val_scores, f, ensure_ascii=False, indent=4)


def main():
    gParameters = initialize_parameters()
    run(gParameters)
    print("Done.")

if __name__ == "__main__":
    main()
