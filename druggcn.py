import os
import candle

file_path = os.path.dirname(os.path.realpath(__file__))

# additional definitions
additional_definitions = [
    {
        "name": "k",
        "type": int,
        "nargs": "+",
        "help": "polynomial orders (i.e. filter sizes)",
    },
    {   
        "name": "f",
        "type": int,
        "nargs": "+", 
        "help": "number of features",
    },
    {
        "name": "pool_type",
        "type": str,
        "help": "pooling type for max pooling (mpool1) or average pooling (apool1)",
    },
    {   
        "name": "brelu",
        "type": str,
        "help": "bias and ReLu type. Options are one bias per filter (b1relu) or one bias per vertex per filter (b2relu)",
    },
    {   
        "name": "decay_rate",
        "type": float, 
        "help": "Base of exponential decay. No decay with 1.",
    },
    {   
        "name": "test_size",
        "type": float, 
        "help": "test data size",
    },
    {  
        "name": "n_fold",
        "type": int, 
        "help": "number of folds in cross-validation",
    },
    {  
        "name": "regularization",
        "type": int,
        "help": "L2 regularizations of weights and biases",
    },
    {  
        "name": "ppi_data",
        "type": str,
        "help": "path to protein-protein interaction (PPI) network data",
    },
    {   
        "name": "response_data",
        "type": str,
        "help": "path to drug response data",
    },
    {   
        "name": "gene_data",
        "type": str, 
        "help": "path to gene expression data",
    },  
    {   
        "name": "summary_dir",
        "type": str, 
        "help": "path to folder for tensorflow summary data",
    },    
]

# required definitions
required = [
    "data_url",
    "model_name",
    "dense",
    "epochs",
    "batch_size",
    "dropout",
    "learning_rate",
    "pool",
    "momentum",
    "rng_seed",
    "output_dir",
    "ckpt_directory"
]

# initialize class
class DrugGCN(candle.Benchmark):
    def set_locals(self):
        """
        Functionality to set variables specific for the benchmark
        - required: set of required parameters for the benchmark.
        - additional_definitions: list of dictionaries describing the additional parameters for the benchmark.
        """
        if required is not None: 
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definisions = additional_definitions
