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

def reconstruct(word):
    test_x, test_y = one_hot_encoding([word] * BATCH_SIZE)
    preds = vae.predict([test_x, test_y], batch_size=BATCH_SIZE)
    print([chr(x) for x in np.argmax(preds[0, :, :], axis=-1)])

def encoder_outputs(word):
    test_x, test_y = one_hot_encoding([word] * BATCH_SIZE)
    means, stds = encoder.predict(test_x, batch_size=BATCH_SIZE)
    print("Means: ", means[0, :10])
    prints("Stds: ", stds[0, :10])

def decode(words=None, p=None, verbose=True):
    n = None
    if words is None:
        if p is None:
            n = 2
        else:
            n = len(p)
        words = []
        for i in range(n):
            words.append(data[np.random.randint(0, TRAINING_DATA_SIZE)])
    else:
        n = len(words)

    if p is None:
        p = [1.0 / n for _ in range(n)]

    print(words)
    states_value = np.zeros((BATCH_SIZE, LATENT_DIM))
    for i in range(n):
        m, std = encoder.predict(one_hot_encoding([words[i]] * BATCH_SIZE))
        states_value += p[i] * (m + std * np.random.normal(size=(LATENT_DIM, )))

    # generate empty target sequence of length 1
    decoded_sentence = ""
    target_seq = one_hot_encoding([decoded_sentence] * BATCH_SIZE)[1][:, [0], :]

    first_time = True
    h, c = None, None
    idx = 0

    while True:
        if first_time:
            # feeding in states sampled with the mean and std provided by encoder
            # and getting current LSTM states to feed in to the decoder at the next step
            output_tokens, h, c = generator.predict([target_seq, states_value], batch_size=BATCH_SIZE)
            first_time = False
        else:
            # reading output token
            output_tokens, h, c = stepper.predict([target_seq, h, c], batch_size=BATCH_SIZE)

        # sample a token
        sampled_token_index = np.argmax(output_tokens[0, 0, :])
        sampled_token = chr(sampled_token_index)
        if verbose:
            print(sorted(list([(chr(x), output_tokens[0, 0, x]) for x in range(VOCAB_SIZE + 2)]), key=lambda x: x[1], reverse=True)[:5])
        idx += 1

        # exit condition: either hit max length
        # or find stop character.
        if sampled_token == chr(VOCAB_SIZE + 1) or idx == MAX_WORD_LEN:
            break

        decoded_sentence += sampled_token

        # Update the target sequence (of length 1).
        target_seq = one_hot_encoding([decoded_sentence] * BATCH_SIZE)[1][:, [idx], :]
    return decoded_sentence


if __name__ == '__main__':
    np.random.seed(42)
    tf.set_random_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', '-w', required=True, help='Weights checkpoint file')
    args = parser.parse_args()
    
    data = []
    with open('../rockyou-processed.txt', 'r') as f: 
        data = f.readlines()
        data = [w.strip() for w in data]

    vae, encoder, generator, stepper = create_model()
    vae.load_weights(args.weights)
    print("Weights loaded")
