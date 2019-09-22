GRGNN -- Gene Regulatory Graph Neural Network
===============================================================================

About
-----

Gene regulatory graph neural network (GRGNN): an end-to-end approach to reconstruct GRNs from scratch utilizing the gene expression data, in both a supervised and a semi-supervised framework. 

The codes are modified for blind review. Sorry for not provide the DREAM challenge data for the size limitation of the submitting system is 10MB. 

Preprocessing script is provided, readers can generate the data by downloading the DREAM5 challenge data from https://www.synapse.org/#!Synapse:syn3130840


Installation
------------

Install [PyTorch](https://pytorch.org/).

Type

    bash ./install.sh

to install the required software and libraries. Node2vec and DGCNN are included in software/ folder. 


Usages
------

In folder data/, extract dream datasets by typing:
    
    tar xzvf dream.tar.gz
    cd ..

Or download DREAM5 data from DREAM official websites at https://www.synapse.org/#!Synapse:syn3130840, type:

    python Preprocessing_DREAM5.py 3
    python Preprocessing_DREAM5.py 4

In this program, for simple, dream3 means E.coli dataset, dream4 means S. cerevisae dataset
Train E.coli and test on S. cerevisae with default parameters, Type:

    python Main_inductive_ensembl.py  --traindata-name dream3 --testdata-name dream4

Train E.coli and test on S. cerevisae with hop 1 and embedding, Type:

    python Main_inductive_ensembl.py  --traindata-name dream3 --testdata-name dream4 --hop 1 --use-embedding

Train S. cerevisae and test on E.coli with hop 1 and embedding, Type:

    python Main_inductive_ensembl.py  --traindata-name dream3 --testdata-name dream4 --hop 1 --use-embedding


Requirements
------------

Tested with Python 3.7.3, Pytorch 1.2.0 on Ubuntu 16.04.

Required python libraries: gensim and scipy; all python libraries required by pytorch_DGCNN such as networkx, tqdm, sklearn etc.

If you want to enable embeddings for link prediction, please install the network embedding software 'node2vec' in "software/" (if the included one does not work).

References:
------------

1. SEAL code: https://github.com/muhanzhang/SEAL
2. Dream data: http://dreamchallenges.org/project/dream-5-network-inference-challenge/ 

