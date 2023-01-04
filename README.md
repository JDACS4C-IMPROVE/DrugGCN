# DrugGCN
A framework for the prediction of **Drug** response using a Graph Convolutional Network (**GCN**).

## Model

See [Model](READMEs/Model.md) for more details.

## Data

See [Data](READMEs/Data.md) for more details.

## Preprocessing input data

Prepare the following files and save them to a folder: `groupEXP.csv`, `groupEXP_foldChange.csv`, `groupPPi.csv`,  `LIST.txt`.

```
python dataProcess.py -h
python dataProcess.py inputPath outputPath
```

Example

```
python dataProcess.py data/L1000/ data_processed/L1000/
```

## Training the model

```
python druggcn_baseline_tensorflow.py
```

Hyperparameters of the model can be adjusted in this file `druggcn_default_model.txt`.
