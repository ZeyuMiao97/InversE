# InversE
InversE: Improving Bilinear Knowledge Graph Embeddings with Canonical Tensor Decomposition
# InversE
This software can be used to reproduce the results in our "InversE: Improving Bilinear Knowledge Graph Embeddings with Canonical Tensor Decomposition" paper. It can be also used to learn `InversE` models for other datasets. The software can be also used as a framework to implement new tensor factorization models
# Dependencies
* `Python` version 2.7
* `Numpy` version 1.13.1
* `pytorch`
# Usage
To run InversE on a dataset `D`, do the following steps:
* `cd` to the directory where `main.py` is  
* Run `python learn.py --dataset D --model InversE --rank 4000 --optimizer Adagrad --learning_rate 1e-2 --batch_size 100 --regularizer N3 --reg 1e-2 --max_epochs 200 --valid 5`  
