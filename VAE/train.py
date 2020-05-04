import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input, Lambda, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import RMSprop, Adam, Nadam
from tensorflow.keras.callbacks import LambdaCallback, CSVLogger, LearningRateScheduler
import math
from tensorflow.keras import backend as K
from tensorflow.keras.losses import categorical_crossentropy
import argparse

VOCAB_SIZE = 128
MAX_WORD_LEN = 50
BATCH_SIZE = 128
INPUT_DIM = VOCAB_SIZE + 2
INTERMEDIATE_DIM = 512
LATENT_DIM = 256
TRAINING_DATA_SIZE = 10000

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

def scheduler(epoch):
    if epoch == 0:
        return 0.01
    last_loss = 0.0
    with open('training.log', 'r') as f:
        last_loss = float(f.readlines()[-1].strip().split(',')[-1])
    if last_loss < 1.0:
        return 0.01
    else:
        return 0.001

def on_epoch_end(epoch, _):
    # if epoch % 10 == 0:
    vae.save_weights('e{:0>3d}_weights.h5'.format(epoch))

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

    z_reweighting_h = Dense(units=INTERMEDIATE_DIM, activation="linear")
    z_reweighting_c = Dense(units=INTERMEDIATE_DIM, activation="linear")
    z_reweighted_h = z_reweighting_h(z)
    z_reweighted_c = z_reweighting_c(z)

    # "next-word" data for prediction
    decoder_words_input = Input(shape=(None, INPUT_DIM,))

    # decoded LSTM layer
    decoder_h = LSTM(INTERMEDIATE_DIM, return_sequences=True, return_state=True, input_shape=(None, INPUT_DIM))

    # todo: not sure if this initialization is correct
    h_decoded, _, _ = decoder_h(decoder_words_input, initial_state=[z_reweighted_h, z_reweighted_c])
    decoder_dense = TimeDistributed(Dense(INPUT_DIM, activation="softmax"))
    decoded_onehot = decoder_dense(h_decoded)

    # end-to-end autoencoder
    vae = Model([x, decoder_words_input], decoded_onehot)

    # encoder, from inputs to latent space
    encoder = Model(x, [z_mean, z_log_sigma])

    # generator, from latent space to reconstructed inputs -- for inference's first step
    decoder_state_input = Input(shape=(LATENT_DIM,))
    _z_rewighted_h = z_reweighting_h(decoder_state_input)
    _z_rewighted_c = z_reweighting_c(decoder_state_input)
    _h_decoded, _decoded_h, _decoded_c = decoder_h(decoder_words_input, initial_state=[_z_rewighted_h, _z_rewighted_c])
    _decoded_onehot = decoder_dense(_h_decoded)
    generator = Model([decoder_words_input, decoder_state_input], [_decoded_onehot, _decoded_h, _decoded_c])

    # RNN for inference
    input_h = Input(shape=(INTERMEDIATE_DIM,))
    input_c = Input(shape=(INTERMEDIATE_DIM,))
    __h_decoded, __decoded_h, __decoded_c = decoder_h(decoder_words_input, initial_state=[input_h, input_c])
    __decoded_onehot = decoder_dense(__h_decoded)
    stepper = Model([decoder_words_input, input_h, input_c], [__decoded_onehot, __decoded_h, __decoded_c])

    def vae_loss(x, x_decoded_onehot):
        xent_loss = categorical_crossentropy(x, x_decoded_onehot)
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
        loss = xent_loss + kl_loss
        return loss

    optimizer = Adam(learning_rate=0.001)
    vae.compile(optimizer=optimizer, loss=vae_loss)
    vae.summary()
    return vae, encoder, generator, stepper

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', '-w', help='Weights file if to start training from this')
    parser.add_argument('--initial_epoch', '-i', help='Number of epochs already done if weights file provided', type=int)
    args = parser.parse_args()

    np.random.seed(42)
    tf.set_random_seed(42)

    
    data = []
    with open('../rockyou-processed.txt', 'r') as f:
        data = f.readlines()
        data = [w.strip() for w in data]

    vae, encoder, generator, stepper = create_model()

    training_data_generator = DataGenerator(data[:TRAINING_DATA_SIZE], BATCH_SIZE)
    callback = LambdaCallback(on_epoch_end=on_epoch_end)
    csv_logger = CSVLogger('training.log')
    lr_scheduler = LearningRateScheduler(scheduler, verbose=1)
    if args.weights is not None:
        history = vae.load_weights(args.weights)
        assert args.initial_epoch is not None
        history = vae.fit(training_data_generator, epochs=500, callbacks=[callback, csv_logger], verbose=1, initial_epoch=args.initial_epoch)
    else:
        history = vae.fit(training_data_generator, epochs=500, callbacks=[callback, csv_logger], verbose=1)
    
