# DrugGCN
DrugGCN is a framework for the prediction of **Drug** response using a Graph Convolutional Network (**GCN**).


## Structure
DrugGCN uses a graph convolutional network model to learn genomic features of cell lines with a graph structure for drug response prediction. This framework integrates protein-protein interaction (PPI) network data and gene expression data utilizing genes with high predictive power to construct a model for each drug. The input of the GCN model is an undirected input graph G=(V,E,W) where V is a set of vertices, E is a set of edges, and W is a weighted adjacency matrix. Vertices and edges of the graph indicate genes and interactions between genes respectively. The use of localized filters in DrugGCN aids in the detection of local features in a biological network such as subnetworks of genes that can contribute together to drug response.


## Data sources
The primary data sources that have been used to construct datasets for model training and testing (i.e., ML data) include:
- GDSC - cell line and drug ids, treatment response, cell line omics data
- Library of Integrated Network-Based Cellular Signatures (LINCS) L1000 project - list of genes
- STRING - PPI network data


## Data and preprocessing
CCL omics data and treatment response data (IC50 and AUC) were downloaded from the authors' GitHub page, [GDSC_DATASET_S1-S12.zip](https://github.com/Jinyu2019/Suppl-data-BBpaper/blob/master/GDSC_DATASET_S1-S12.zip). The [protein links](https://stringdb-static.org/download/protein.links.v11.5/9606.protein.links.v11.5.txt.gz) and [protein info](https://stringdb-static.org/download/protein.info.v11.5/9606.protein.info.v11.5.txt.gz) files (version 11.5) for *Homo sapiens* can be obtained from the STRING website. Refer to [Data](Data.md) for more info regarding the raw data provided with the original DrugGCN model repo and preprocessing scripts allowing to generate ML data for model training and testing.


## Evaluation
Four evaluation schemes were used for the analysis of prediction performance.

- L1000-IC50: Gene set of 663 genes using the list of landmark genes derived from LINCS L1000 project and IC50 drug response values.
- L1000-AUC: Gene set of 663 genes using the list of landmark genes derived from LINCS L1000 project and AUC drug response values.
- Var1000-IC50: Gene set of the top 1000 variable genes and IC50 drug response values.
- Var1000-AUC: Gene set of the top 1000 variable genes and AUC drug response values.


## URLs
- Original GitHub: [https://github.com/BML-cbnu/DrugGCN](https://github.com/BML-cbnu/DrugGCN)
- IMPROVE GitHub: [https://github.com/JDACS4C-IMPROVE/DrugGCN/tree/develop](https://github.com/JDACS4C-IMPROVE/DrugGCN/tree/develop)
- Data: https://ftp.mcs.anl.gov/pub/candle/public/improve/model_curation_data/DrugGCN/


## Reference
Kim S, Bae S, Piao Y, Jo K. Graph Convolutional Network for Drug Response Prediction Using Gene Expression Data. *Mathematics*. 2021; 9(7):772. https://doi.org/10.3390/math9070772
