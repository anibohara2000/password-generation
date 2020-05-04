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

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    print(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
def generate(model, temperature=1.0):
    p = ''
    j = 0
    while True:
        x = model.predict(one_hot_encoding([p])[0])
        y = x[0, j, :]
        char = chr(sample(y, temperature))
        if char == chr(VOCAB_SIZE + 1):
            break
        p += char
        j += 1
        if j == MAX_WORD_LEN:
            break
    return p

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
