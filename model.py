# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import pickle
import random
import time

from keras.callbacks import Callback
from keras.layers import Dense, Embedding, LSTM, TimeDistributed
from keras.models import load_model, Sequential
import numpy as np

from vectorizer import Vectorizer
from utils import print_cyan, print_green, print_red
from utils import sample_preds, reshape_for_stateful_rnn, find_random_seeds


# Live samples the model after each epoch, which can be very useful when
# tweaking parameters and/or dataset
class LiveSamplerCallback(Callback):
    def __init__(self, meta_model):
        super(LiveSamplerCallback, self).__init__()
        self.meta_model = meta_model

    def on_epoch_end(self, epoch, logs=None):
        print()
        print_green('Sampling model...')
        self.meta_model.update_sample_model_weights()
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print('Using diversity:', diversity)
            self.meta_model.sample(diversity=diversity)
            print('-' * 50)


# We wrap the keras model in our own metaclass that handles text loading,
# provides convient train and sample functions.
class MetaModel:
    def __init__(self):
        self.train_model = None
        self.sample_model = None
        self.seeds = None
        self.vectorizer = None

    # Builds the underlying keras model
    def _build_models(self, batch_size, embedding_size, rnn_size, num_layers):
        model = Sequential()
        model.add(Embedding(self.vectorizer.vocab_size,
                            embedding_size,
                            batch_input_shape=(batch_size, None)))
        for layer in range(num_layers):
            model.add(LSTM(rnn_size,
                           stateful=True,
                           return_sequences=True))
        model.add(TimeDistributed(Dense(self.vectorizer.vocab_size,
                                        activation='softmax')))
        # With sparse_categorical_crossentropy we can leave as labels as
        # integers instead of one-hot vectors
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='rmsprop')
        model.summary()

        # Keep a separate model with batch_size 1 for training
        self.train_model = model
        config = model.get_config()
        config[0]['config']['batch_input_shape'] = (1, None)
        self.sample_model = Sequential.from_config(config)
        self.sample_model.trainable = False

    def update_sample_model_weights(self):
        self.sample_model.set_weights(self.train_model.get_weights())

    def train(self, data_dir, word_tokens, pristine_input, pristine_output,
              batch_size, seq_length, seq_step, embedding_size, rnn_size,
              num_layers, num_epochs, skip_sampling):
        print_green('Loading data...')
        load_start = time.time()

        text = open(os.path.join(data_dir, 'input.txt')).read()
        self.seeds = find_random_seeds(text)
        self.vectorizer = Vectorizer(text, word_tokens,
                                     pristine_input, pristine_output)

        data = self.vectorizer.vectorize(text)
        x = reshape_for_stateful_rnn(data[:-1], batch_size,
                                     seq_length, seq_step)
        y = reshape_for_stateful_rnn(data[1:], batch_size,
                                     seq_length, seq_step)
        # Y data needs an extra axis to work with the sparse categorical
        # crossentropy loss function
        y = y[:, :, np.newaxis]

        print('x.shape:', x.shape)
        print('y.shape:', y.shape)
        load_end = time.time()
        print_red('Data load time', load_end - load_start)

        print_green('Building model...')
        model_start = time.time()
        self._build_models(batch_size, embedding_size, rnn_size, num_layers)
        model_end = time.time()
        print_red('Model build time', model_end - model_start)

        print_green('Training...')
        train_start = time.time()
        # Train the model
        callbacks = []
        if not skip_sampling:
            callbacks.append(LiveSamplerCallback(self))
        self.train_model.fit(x, y,
                             batch_size=batch_size,
                             shuffle=False,
                             epochs=num_epochs,
                             verbose=1,
                             callbacks=callbacks)
        self.update_sample_model_weights()
        train_end = time.time()
        print_red('Training time', train_end - train_start)

    def sample(self, seed=None, length=None, diversity=1.0):
        self.sample_model.reset_states()

        if length is None:
            length = 100 if self.vectorizer.word_tokens else 500

        if seed is None:
            seed = random.choice(self.seeds)
            print('Using seed: ', end='')
            print_cyan(seed)
            print('-' * 50)

        preds = None

        # Feed in seed string
        print_cyan(seed, end=' ' if self.vectorizer.word_tokens else '')
        seed_vector = self.vectorizer.vectorize(seed)
        for char_index in np.nditer(seed_vector):
            preds = self.sample_model.predict(np.array([[char_index]]),
                                              verbose=0)

        sampled_indices = np.array([], dtype=np.int32)
        # Sample the model one token at a time
        for i in range(length):
            char_index = 0
            if preds is not None:
                char_index = sample_preds(preds[0][0], diversity)
            sampled_indices = np.append(sampled_indices, char_index)
            preds = self.sample_model.predict(np.array([[char_index]]),
                                              verbose=0)
        sample = self.vectorizer.unvectorize(sampled_indices)
        print(sample)
        return sample

    # Don't pickle the keras models, better to save directly
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['train_model']
        del state['sample_model']
        return state


# Save the keras model directly and pickle our meta model class
def save(model, data_dir):
    keras_file_path = os.path.join(data_dir, 'model.h5')
    pickle_file_path = os.path.join(data_dir, 'model.pkl')
    model.sample_model.save(filepath=keras_file_path)
    pickle.dump(model, open(pickle_file_path, 'wb'))
    print_green('Model saved to', pickle_file_path, keras_file_path)


# Load the meta model and restore its internal keras model
def load(data_dir):
    keras_file_path = os.path.join(data_dir, 'model.h5')
    pickle_file_path = os.path.join(data_dir, 'model.pkl')
    model = pickle.load(open(pickle_file_path, 'rb'))
    model.sample_model = load_model(keras_file_path)
    return model
