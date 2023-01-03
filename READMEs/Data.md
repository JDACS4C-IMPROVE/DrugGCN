Deep learning (DL) models built using popular DL frameworks can take various types of data from simple CSV to more complex structures such as `.pt` with PyTorch and `TFRecords` with TensorFlow.
Constructing datasets for drug response prediction (DRP) models generally requires combining heterogeneous data such as cancer and drug information and treatment response values.
We distinguish between two types of data:
- __ML data__. Data that can be directly consumed by prediction models for training and testing (e.g., `TFRecords`).
- __Raw data__. Data that are used to generate ML data (e.g., treatment response values, cancer and drug info). These usually include data files from drug sensitivity studies such as CCLE, CTRP, gCSI, GDSC, etc.

As part of model curation, the original data that is provided with public DRP models is copied to an FTP site. The full path is https://ftp.mcs.anl.gov/pub/candle/public/improve/model_curation_data/ . For each model, a subdirectory is created for storing the model's data.

The raw data and ML data are located, respectively, in `data` and `data_processed` folders. The data for DrugGCN can be found in:
- `data`: https://ftp.mcs.anl.gov/pub/candle/public/improve/model_curation_data/DrugGCN/druggcn_data.tar.gz  
- `data_processed`: https://ftp.mcs.anl.gov/pub/candle/public/improve/model_curation_data/DrugGCN/druggcn_data_processed.tar.gz  

Preprocessing scripts are often required to generate ML data from raw data. However, not all public repositories provide the necessary scripts.


# Raw data
The raw data includes the following for the L1000 and Var1000 datasets:
1) Gene expression data. `EXP.csv`: Gene expression data for 734 cell lines obtained from the authors' GitHub page, [GDSC_DATASET_S1-S12.zip](https://github.com/Jinyu2019/Suppl-data-BBpaper/blob/master/GDSC_DATASET_S1-S12.zip), `Table_S1_GDSC_Gene_expression.csv`.
2) Protein info. `PPI_INFO.txt`: List of proteins including their display names and descriptions from the STRING database. This data can be downloaded from the STRING database [here](https://stringdb-static.org/download/protein.info.v11.5/9606.protein.info.v11.5.txt.gz).
3) Protein links. `PPI_LINK.txt`: Protein network data (full network, scored links between proteins) from the STRING database. This data can be downloaded from the STRING database [here](https://stringdb-static.org/download/protein.links.v11.5/9606.protein.links.v11.5.txt.gz).
4) List of genes. `LIST.txt`: 663 or 1000 genes for the L1000 and Var1000 datasets respectively.

The raw data is available in: https://ftp.mcs.anl.gov/pub/candle/public/improve/model_curation_data/DrugGCN/data/


# ML data
The script `dataProcess.py` uses raw data to generate ML data that can be used to train and test with DrugGCN. The necessary raw data are automatically downloaded from the FTP server using the `candle_lib` utility function `get_file()` and processed. 

The ML data files are available in FTP: https://ftp.mcs.anl.gov/pub/candle/public/improve/model_curation_data/DrugGCN/data_processed/. These files can be automatically downloaded from the FTP server using the `candle_lib` utility function `get_file()`.

The ML data files include the following:

- __Cancer features__. L1000 and Var1000 folders with gene expression and protein-protein interaction files: 
  - groupEXP.csv. Gene expression data of selected genes.
  - groupEXP_foldChange.csv. Fold change of gene expression data of selected genes.
  - groupPPI.csv. Protein-protein interaction data of selected genes where values are weights of the interactions that reflect the amount of available evidence of the interaction between two genes.
- __Response data__. IC50 or AUC values of 201 drugs and 734 cell lines.
  - Table_S6_GDSC_Drug_response_IC50.csv
  - Table_S7_GDSC_Drug_response_AUC.csv

The user can specify which dataset, L1000 or Var1000, and which response data, IC50 or AUC, to use when training and testing the model.

# Using your own data
Ultimately, we want to be able to train models with other datasets (not only the ones provided with the model repo). This requires the preprocessing scripts to be available and reproducible.
