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

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
    def __len__(self):
        return math.floor(len(self.data) / self.batch_size)
    def __getitem__(self, idx):
        batch_data = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x, batch_y = one_hot_encoding(batch_data)
        return ([batch_x, batch_y], batch_x)

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
        sampled_token_index = np.argmax(output_tokens[0, idx, :])
        sampled_token = chr(sampled_token_index)
        if verbose:
            print(sorted(list([(chr(x), output_tokens[0, idx, x]) for x in range(VOCAB_SIZE + 2)]), key=lambda x: x[1], reverse=True)[:5])
        idx += 1

        # exit condition: either hit max length
        # or find stop character.
        if sampled_token == chr(VOCAB_SIZE + 1) or idx == MAX_WORD_LEN:
            break

        decoded_sentence += sampled_token

        # Update the target sequence (of length 1).
        target_seq = one_hot_encoding([decoded_sentence] * BATCH_SIZE)[1][:, [idx], :]
    return decoded_sentence

def on_epoch_end(epoch, _):
    epoch += 1
    if epoch % 10 == 0:
        model.save_weights('e{:0>3d}_weights.h5'.format(epoch))

def one_hot_encoding(batch_input):
    batch_size = len(batch_input)
    x = np.zeros((batch_size, MAX_WORD_LEN, VOCAB_SIZE + 2), dtype=np.bool)
    y = np.zeros((batch_size, MAX_WORD_LEN, VOCAB_SIZE + 2), dtype=np.bool)
    for j in range(batch_size):
        x[j][0][VOCAB_SIZE]=1
        input = batch_input[j]
        for i in range(1, MAX_WORD_LEN):
            if (i-1<len(input)):
                index=ord(input[i-1])
            else:
                index=VOCAB_SIZE+1
            x[j][i][index]=1
            y[j][i-1][index]=1
        y[j][MAX_WORD_LEN-1][VOCAB_SIZE+1]=1
    return y,x # Encoder input, Decoder input. Encoder input does not contain Start token at start, Decoder input does

def create_model():
    # VAE model = encoder + decoder
    # build encoder model
    x = Input(shape=(None, INPUT_DIM))

    # LSTM encoding
    h = LSTM(units=INTERMEDIATE_DIM)(x)

    # VAE Z layer
    z_mean = Dense(units=LATENT_DIM)(h)
    z_log_sigma = Dense(units=LATENT_DIM)(h)

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(BATCH_SIZE, LATENT_DIM), mean=0., stddev=1.0)
        return z_mean + z_log_sigma * epsilon

    # note that "output_shape" isn't necessary with the TensorFlow backend
    # so you could write `Lambda(sampling)([z_mean, z_log_sigma])`
    z = Lambda(sampling, output_shape=(LATENT_DIM,))([z_mean, z_log_sigma])

    z_reweighting = Dense(units=INTERMEDIATE_DIM, activation="linear")
    z_reweighted = z_reweighting(z)

    # "next-word" data for prediction
    decoder_words_input = Input(shape=(None, INPUT_DIM,))

    # decoded LSTM layer
    decoder_h = LSTM(INTERMEDIATE_DIM, return_sequences=True, return_state=True, input_shape=(None, INPUT_DIM))

    # todo: not sure if this initialization is correct
    h_decoded, _, _ = decoder_h(decoder_words_input, initial_state=[z_reweighted, z_reweighted])
    decoder_dense = TimeDistributed(Dense(INPUT_DIM, activation="softmax"))
    decoded_onehot = decoder_dense(h_decoded)

    # end-to-end autoencoder
    vae = Model([x, decoder_words_input], decoded_onehot)

    # encoder, from inputs to latent space
    encoder = Model(x, [z_mean, z_log_sigma])

    # generator, from latent space to reconstructed inputs -- for inference's first step
    decoder_state_input = Input(shape=(LATENT_DIM,))
    _z_rewighted = z_reweighting(decoder_state_input)
    _h_decoded, _decoded_h, _decoded_c = decoder_h(decoder_words_input, initial_state=[_z_rewighted, _z_rewighted])
    _decoded_onehot = decoder_dense(_h_decoded)
    generator = Model([decoder_words_input, decoder_state_input], [_decoded_onehot, _decoded_h, _decoded_c])

    # RNN for inference
    input_h = Input(shape=(INTERMEDIATE_DIM,))
    input_c = Input(shape=(INTERMEDIATE_DIM,))
    __h_decoded, __decoded_h, __decoded_c = decoder_h(decoder_words_input, initial_state=[input_h, input_c])
    __decoded_onehot = decoder_dense(__h_decoded)
    stepper = Model([decoder_words_input, input_h, input_c], [__decoded_onehot, __decoded_h, __decoded_c])

    def vae_loss(x, x_decoded_onehot):
        xent_loss = K.categorical_crossentropy(x, x_decoded_onehot)
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
        loss = xent_loss + kl_loss
        return loss

    optimizer = RMSprop(learning_rate=0.001)
    vae.compile(optimizer=optimizer, loss=vae_loss)
    vae.summary()
    return vae, encoder, generator, stepper

if __name__ == '__main__':
    np.random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', '-w', required=True, help='Weights checkpoint file')
    args = parser.parse_args()
    
    data = []
    with open('rockyou-processed.txt', 'r') as f:
        data = f.readlines()
        data = [w.strip() for w in data]

    VOCAB_SIZE = 128
    MAX_WORD_LEN = 50
    BATCH_SIZE = 128
    INPUT_DIM = VOCAB_SIZE + 2
    INTERMEDIATE_DIM = 512
    LATENT_DIM = 256
    TRAINING_DATA_SIZE = 10000

    vae, encoder, generator, stepper = create_model()
    vae.load_weights(args.weights)
