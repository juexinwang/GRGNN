import numpy as np
import networkx as nx
import pickle

pos_graphs = pickle.load(open("pos_graphs.pickle", "rb")) 
neg_graphs = pickle.load(open("neg_graphs.pickle", "rb"))