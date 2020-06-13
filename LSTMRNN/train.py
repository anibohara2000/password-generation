import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Input, Lambda, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import RMSprop, Adam, Nadam
from tensorflow.keras.callbacks import LambdaCallback, CSVLogger, LearningRateScheduler
import math
from tensorflow.keras import backend as K
from tensorflow.keras.losses import categorical_crossentropy
import argparse

VOCAB_SIZE = 128
MAX_WORD_LEN = 50
BATCH_SIZE = 128
INTERMEDIATE_DIM = 512
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
        return (batch_x, batch_y)

def on_epoch_end(epoch, _):
    # if epoch % 10 == 0:
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
    return x,y

def create_model():
    print('Building model...')
    inputs = Input(shape=(None, VOCAB_SIZE + 2))
    
    rnn_layer = LSTM(units=INTERMEDIATE_DIM, return_sequences=True)
    dense_layer = TimeDistributed(Dense(units=VOCAB_SIZE + 2, activation='softmax'))
    
    rnn_output = rnn_layer(inputs)
    outputs = dense_layer(rnn_output)

    model = Model(inputs=inputs, outputs=outputs)

    optimizer = RMSprop(learning_rate=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    model.summary()
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', '-w', help='Weights file if to start training from this')
    parser.add_argument('--initial_epoch', '-i', help='Number of epochs already done if weights file provided', type=int)
    args = parser.parse_args()

    np.random.seed(42)
    tf.compat.v1.set_random_seed(42)

    
    data = []
    with open('../rockyou-processed.txt', 'r') as f:
        data = f.readlines()
        data = [w.strip() for w in data]

    model = create_model()

    training_data_generator = DataGenerator(data[:TRAINING_DATA_SIZE], BATCH_SIZE)
    callback = LambdaCallback(on_epoch_end=on_epoch_end)
    csv_logger = CSVLogger('training.log')
    kwargs = {}
    if args.weights is not None:
        assert args.initial_epoch is not None
        model.load_weights(args.weights)
        kwargs['initial_epoch'] = args.initial_epoch
    history = model.fit(training_data_generator, epochs=500, callbacks=[callback, csv_logger], verbose=1, **kwargs)
    
