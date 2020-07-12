from __future__ import print_function
import nltk
import pdb
import zinc_grammar
import numpy as np
import h5py
import molecule_vae



# f = open('rockyou-processed.txt','r')
L = []
with open('data/rockyou-processed.txt', 'r') as f:
    L = f.readlines()
    L = [w.strip() for w in L]
L = list(filter(lambda a: a.find("'")==-1, L))
L = list(filter(lambda a: a!='', L))
# f.close()

MAX_LEN=277
NCHARS = len(zinc_grammar.GCFG.productions())

def to_one_hot(smiles):
    """ Encode a list of smiles strings to one-hot vectors """
    assert type(smiles) == list
    prod_map = {}
    for ix, prod in enumerate(zinc_grammar.GCFG.productions()):
        prod_map[prod] = ix
    tokenize = molecule_vae.get_zinc_tokenizer(zinc_grammar.GCFG)
    tokens = map(tokenize, smiles)
    parser = nltk.ChartParser(zinc_grammar.GCFG)
    parse_trees = [parser.parse(t).__next__() for t in tokens]
    productions_seq = [tree.productions() for tree in parse_trees]
    indices = [np.array([prod_map[prod] for prod in entry], dtype=int) for entry in productions_seq]
    one_hot = np.zeros((len(indices), MAX_LEN, NCHARS), dtype=np.float32)
    mask = np.zeros((len(indices), MAX_LEN))
    for i in range(len(indices)):
        num_productions = len(indices[i])
        one_hot[i][np.arange(num_productions),indices[i]] = 1.
        one_hot[i][np.arange(num_productions, MAX_LEN),-1] = 1.
        mask[i][np.arange(num_productions+1)] = 1
    return one_hot, mask

# print(to_one_hot[line])

MAX_STEPS1 = 100000

OH = np.zeros((MAX_STEPS1,MAX_LEN,NCHARS))
maskv = np.zeros((MAX_STEPS1, MAX_LEN))
for i in range(0, MAX_STEPS1, 100):
    print('Processing: i=[' + str(i) + ':' + str(i+100) + ']')
    onehot, masks = to_one_hot(L[i:i+100])
    OH[i:i+100,:,:] = onehot
    maskv[i:i+100,:] = masks

h5f = h5py.File('data/zinc_grammar_dataset_100000.h5','w')
h5f.create_dataset('data', data=OH)
h5f.create_dataset('mask', data=maskv)
h5f.close()
