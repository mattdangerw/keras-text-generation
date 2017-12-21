# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import pickle
import random
import sys
import time

from keras.callbacks import Callback
from keras.layers import Dense, Dropout, Embedding, LSTM, TimeDistributed
from keras.models import load_model, Sequential
import numpy as np

from vectorizer import Vectorizer
from utils import print_cyan, print_green, print_red
from utils import sample_preds, shape_for_stateful_rnn, find_random_seeds


class LiveSamplerCallback(Callback):
    """
    Live samples the model after each epoch, which can be very useful when
    tweaking parameters and/or the dataset.
    """
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


class MetaModel:
    """
    We wrap the keras model in our own metaclass that handles text loading,
    and provides convenient train and sample functions.
    """
    def __init__(self):
        self.train_model = None
        self.sample_model = None
        self.seeds = None
        self.vectorizer = None

    # Read in our data and validation texts
    def _load_data(self, data_dir, word_tokens, pristine_input, pristine_output,
                   batch_size, seq_length, seq_step):
        try:
            with open(os.path.join(data_dir, 'input.txt')) as input_file:
                text = input_file.read()
        except FileNotFoundError:
            print_red("No input.txt in data_dir")
            sys.exit(1)

        skip_validate = True
        try:
            with open(os.path.join(data_dir, 'validate.txt')) as validate_file:
                text_val = validate_file.read()
                skip_validate = False
        except FileNotFoundError:
            pass # Validation text optional

        # Find some good default seed string in our source text.
        self.seeds = find_random_seeds(text)
        # Include our validation texts with our vectorizer
        all_text = text if skip_validate else '\n'.join([text, text_val])
        self.vectorizer = Vectorizer(all_text, word_tokens,
                                     pristine_input, pristine_output)

        data = self.vectorizer.vectorize(text)
        x, y = shape_for_stateful_rnn(data, batch_size, seq_length, seq_step)
        print('x.shape:', x.shape)
        print('y.shape:', y.shape)

        if skip_validate:
            return x, y, None, None

        data_val = self.vectorizer.vectorize(text_val)
        x_val, y_val = shape_for_stateful_rnn(data_val, batch_size,
                                              seq_length, seq_step)
        print('x_val.shape:', x_val.shape)
        print('y_val.shape:', y_val.shape)
        return x, y, x_val, y_val

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
            model.add(Dropout(0.2))
        model.add(TimeDistributed(Dense(self.vectorizer.vocab_size,
                                        activation='softmax')))
        # With sparse_categorical_crossentropy we can leave as labels as
        # integers instead of one-hot vectors
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
        model.summary()

        # Keep a separate model with batch_size 1 for training
        self.train_model = model
        config = model.get_config()
        config[0]['config']['batch_input_shape'] = (1, None)
        self.sample_model = Sequential.from_config(config)
        self.sample_model.trainable = False

    def update_sample_model_weights(self):
        """Sync training and sampling model weights"""
        self.sample_model.set_weights(self.train_model.get_weights())

    def train(self, data_dir, word_tokens, pristine_input, pristine_output,
              batch_size, seq_length, seq_step, embedding_size, rnn_size,
              num_layers, num_epochs, live_sample):
        """Train the model"""
        print_green('Loading data...')
        load_start = time.time()
        x, y, x_val, y_val = self._load_data(data_dir, word_tokens,
                                             pristine_input, pristine_output,
                                             batch_size, seq_length, seq_step)
        load_end = time.time()
        print_red('Data load time', load_end - load_start)

        print_green('Building model...')
        model_start = time.time()
        self._build_models(batch_size, embedding_size, rnn_size, num_layers)
        model_end = time.time()
        print_red('Model build time', model_end - model_start)

        print_green('Training...')
        train_start = time.time()
        validation_data = (x_val, y_val) if (x_val is not None) else None
        callbacks = [LiveSamplerCallback(self)] if live_sample else None
        self.train_model.fit(x, y,
                             validation_data=validation_data,
                             batch_size=batch_size,
                             shuffle=False,
                             epochs=num_epochs,
                             verbose=1,
                             callbacks=callbacks)
        self.update_sample_model_weights()
        train_end = time.time()
        print_red('Training time', train_end - train_start)

    def sample(self, seed=None, length=None, diversity=1.0):
        """Sample the model"""
        self.sample_model.reset_states()

        if length is None:
            length = 100 if self.vectorizer.word_tokens else 500

        if seed is None:
            seed = random.choice(self.seeds)
            print('Using seed: ', end='')
            print_cyan(seed)
            print('-' * 50)

        preds = None
        seed_vector = self.vectorizer.vectorize(seed)
        # Feed in seed string
        print_cyan(seed, end=' ' if self.vectorizer.word_tokens else '')
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


def save(model, data_dir):
    """Save the keras model directly and pickle our meta model class"""
    keras_file_path = os.path.join(data_dir, 'model.h5')
    pickle_file_path = os.path.join(data_dir, 'model.pkl')
    model.sample_model.save(filepath=keras_file_path)
    pickle.dump(model, open(pickle_file_path, 'wb'))
    print_green('Model saved to', pickle_file_path, keras_file_path)


def load(data_dir):
    """Load the meta model and restore its internal keras model"""
    keras_file_path = os.path.join(data_dir, 'model.h5')
    pickle_file_path = os.path.join(data_dir, 'model.pkl')
    model = pickle.load(open(pickle_file_path, 'rb'))
    model.sample_model = load_model(keras_file_path)
    return model
