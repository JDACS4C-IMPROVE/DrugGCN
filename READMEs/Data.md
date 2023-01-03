Deep learning (DL) models built using popular DL frameworks can take various types of data from simple CSV to more complex structures such as `.pt` with PyTorch and `TFRecords` with TensorFlow.
Constructing datasets for drug response prediction (DRP) models generally requires combining heterogeneous data such as cancer and drug information and treatment response values.
We distinguish between two types of data:
- __ML data__. Data that can be directly consumed by prediction models for training and testing (e.g., `TFRecords`).
- __Raw data__. Data that are used to generate ML data (e.g., treatment response values, cancer and drug info). These usually include data files from drug sensitivity studies such as CCLE, CTRP, gCSI, GDSC, etc.

As part of model curation, the original data that is provided with public DRP models is copied to an FTP site. The full path is https://ftp.mcs.anl.gov/pub/candle/public/improve/model_curation_data/ . For each model, a subdirectory is created for storing the model's data.

The raw data and ML data are located, respectively, in `data` and `data_processed` folders. E.g., the data for DrugGCN can be found in:
- https://ftp.mcs.anl.gov/pub/candle/public/improve/model_curation_data/DrugGCN/data/
- https://ftp.mcs.anl.gov/pub/candle/public/improve/model_curation_data/DrugGCN/data_processed/

Preprocessing scripts are often required to generate ML data from raw data. However, not all public repositories provide the necessary scripts.


# Raw data
The raw data is downloaded from GDSC website (version 6.0) and refers here to three types of data:
1) Dose-independent drug response values.
`PANCANCER_IC.csv`: drug and cell ids, IC50 values and other metadata (201 drugs and 734 cell lines).
2) Cancer sample information. `PANCANCER_Genetic_feature.csv`: 735 binary features that include coding variants and copy number alterations.
3) Drug information. `drug_smiles.csv`: SMILES strings of drug molecules. The SMILES were retrieved from PubChem using CIDs (Druglist.csv). The script `preprocess.py` provides functions to generate this file.

The raw data is available in: https://ftp.mcs.anl.gov/pub/candle/public/improve/model_curation_data/DrugGCN/data/


# ML data
The script `dataProcess.py` uses raw data to generate ML data that can be used to train and test with DrugGCN. The necessary raw data are automatically downloaded from the FTP server using the `candle_lib` utility function `get_file()` and processed when running this script. 

The ML data files are available in FTP: https://ftp.mcs.anl.gov/pub/candle/public/improve/model_curation_data/GraphDRP/data_processed/. These files can be automatically downloaded from the FTP server using the `candle_lib` utility function `get_file()`.

The ML data files include the following:

- __Cancer features__. L1000 and Var1000 folders with gene expression and protein-protein interaction files: 
  - __groupEXP.csv__. Gene expression data of selected genes.
  - __groupEXP_foldChange.csv__. Fold change of gene expression data of selected genes.
  - __groupPPI.csv__. Protein-protein interaction data of selected genes where values are weights of the interactions that reflect the amount of available evidence of the interaction between two genes.
- __Response data__. IC50 or AUC values.
  - Table_S6_GDSC_Drug_response_IC50.csv
  - Table_S7_GDSC_Drug_response_AUC.csv

# Using your own data
Ultimately, we want to be able to train models with other datasets (not only the ones provided with the model repo). This requires the preprocessing scripts to be available and reproducible.
