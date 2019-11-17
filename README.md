GRGNN -- Gene Regulatory Graph Neural Network
===============================================================================

About
-----

Gene regulatory graph neural network (GRGNN): an end-to-end approach to reconstruct GRNs from scratch utilizing the gene expression data, in both a supervised and a semi-supervised framework. 

Preprocessing script is provided, readers can use the data directly or generate the data by downloading the DREAM5 challenge data from https://www.synapse.org/#!Synapse:syn3130840

Requirements
------------

Tested with Python 3.7.3, Pytorch 1.12.0 on Ubuntu 16.04

Required python libraries: gensim and scipy; all python libraries required by pytorch_DGCNN are networkx, tqdm, sklearn etc.

If you want to enable embeddings for link prediction, please install the network embedding software 'node2vec' in "software" (if the included one does not work).

Installation
------------
Type

    bash ./install.sh

to install the required software and libraries. [Node2vec](https://github.com/aditya-grover/node2vec) and [DGCNN](https://github.com/muhanzhang/pytorch_DGCNN) are included in software folder. 


Usages
------

(Optional): Preprocessing DREAM5 data

    python Preprocessing_DREAM5.py 3
    python Preprocessing_DREAM5.py 4

In this program, for simple, dataa3 means E.coli dataset, data4 means S. cerevisae dataset
Train E.coli and test on S. cerevisae with default parameters, Type:

    python Main_inductive_ensembl.py  --traindata-name data3 --testdata-name data4

Train E.coli and test on S. cerevisae with hop 1 and embedding, Type:

    python Main_inductive_ensembl.py  --traindata-name data3 --testdata-name data4 --hop 1 --use-embedding

Train S. cerevisae and test on E.coli with hop 1 and embedding, Type:

    python Main_inductive_ensembl.py  --traindata-name data3 --testdata-name data4 --hop 1 --use-embedding


References:
------------
1. SEAL code: https://github.com/muhanzhang/SEAL
2. Dream data: http://dreamchallenges.org/project/dream-5-network-inference-challenge/ 

