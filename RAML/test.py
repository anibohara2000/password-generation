import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input, Lambda, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import LambdaCallback
import math
from tensorflow.keras import backend as K
from tensorflow.keras.losses import categorical_crossentropy
import argparse
from train import *

def sample(preds, verbose=0, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    if verbose:
        paired = sorted([(chr(i), preds[i]) for i in range(len(preds))], key=lambda x: x[1], reverse=True)
        print(paired[:5])
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
def generate(temp=1.0, verbose=0):
    p = ''
    j = 0
    while True:
        x = model.predict(one_hot_encoding([p], temp=TEMP)[0])
        y = x[0, j, :]
        char = chr(sample(y, temp, verbose))
        print(char)
        if char == chr(VOCAB_SIZE + 1):
            break
        p += char
        j += 1
        if j == MAX_WORD_LEN:
            break
    return p

def reconstruct_probs(word):
    p = ''
    j = 0
    while p != word:
        x = model.predict(one_hot_encoding([p], temp=TEMP)[0])
        y = x[0, j, :]
        paired = sorted([(chr(i), y[i]) for i in range(len(y))], key=lambda x: x[1], reverse=True)
        print(paired[:5])        

        print(word[j], y[ord(word[j])])
        
        p += word[j]
        j += 1
    return

if __name__ == '__main__':
    np.random.seed(42)
    tf.compat.v1.set_random_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', '-w', required=True, help='Weights checkpoint file')
    args = parser.parse_args()
    
    data = []
    with open('../rockyou-processed.txt', 'r') as f: 
        data = f.readlines()
        data = [w.strip() for w in data]

    model = create_model()
    model.load_weights(args.weights)
    print("Weights loaded")
