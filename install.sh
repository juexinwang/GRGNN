#!/bin/bash

cd software/pytorch_DGCNN
cd lib
make -j4
cd "$(dirname "$0")"
pip install --user numpy
pip install --user scipy
pip install --user networkx
pip install --user tqdm
pip install --user sklearn
pip install --user gensim
