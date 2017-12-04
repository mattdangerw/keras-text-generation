# -*- coding: utf-8 -*-

from __future__ import print_function
from collections import Counter
from keras.callbacks import Callback
from keras.layers import Dense, Embedding, LSTM, TimeDistributed
from keras.models import Sequential, load_model
import numpy as np
import os
import pickle
import random
import time

from utils import print_cyan, print_green, print_red
from utils import sample_preds, word_tokenize, word_detokenize


# Live samples the model after each epoch, which can be very useful when
# tweaking parameters and/or dataset
class LiveSamplerCallback(Callback):
    def __init__(self, meta_model):
        self.meta_model = meta_model

    def on_epoch_end(self, epoch, logs={}):
        print()
        print_green('Sampling model...')
        length = 100 if self.meta_model.word_tokens else 500
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print('Using diversity:', diversity)
            self.meta_model.sample(length=length, diversity=diversity)
            print('-' * 50)


# We wrap the keras model in our own metaclass that handles text loading,
# provides convient train and sample functions.
class MetaModel:
    def __init__(self):
        pass

    def tokenize(self, text):
        if not self.pristine_input:
            text = text.lower()
        if self.word_tokens:
            if self.pristine_input:
                return text.split()
            return word_tokenize(text)
        return text

    def detokenize(self, tokens):
        if self.word_tokens:
            if self.pristine_output:
                return ' '.join(tokens)
            return word_detokenize(tokens)
        return ''.join(tokens)

    def vectorize(self, text):
        tokens = self.tokenize(text)
        indices = []
        for token in tokens:
            if token in self.token_indices:
                indices.append(self.token_indices[token])
            else:
                print_red('Ignoring unrecognized token:', token)
        return np.array(indices, dtype=np.int32)

    def unvectorize(self, vector):
        tokens = [self.indices_token[index] for index in vector.tolist()]
        return self.detokenize(tokens)

    # Heuristic attempt to find some good seed strings in the input text
    def _find_random_seeds(self, text, num_seeds=50, max_seed_length=50):
        lines = text.split('\n')
        # Take a random sampling of lines
        if (len(lines) > num_seeds * 4):
            lines = random.sample(lines, num_seeds * 4)
        # Take the top quartile based on length so we get decent seed strings
        lines = sorted(lines, key=lambda line: len(line), reverse=True)
        lines = lines[:num_seeds]
        # Split on the first whitespace before max_seed_length
        lines = [line[:max_seed_length].rsplit(None, 1)[0] for line in lines]
        return lines

    # Reads in the input text and converts to a vector
    def _load_text(self, data_dir):
        text = open(os.path.join(data_dir, 'input.txt')).read()
        self.seeds = self._find_random_seeds(text)

        tokens = self.tokenize(text)
        print('corpus length:', len(tokens))
        token_counts = Counter(tokens)
        # Sort so most common tokens come first in our vocabulary
        tokens = [x[0] for x in token_counts.most_common()]
        self.token_indices = {x: i for i, x in enumerate(tokens)}
        self.indices_token = {i: x for i, x in enumerate(tokens)}
        self.vocab_size = len(tokens)
        print('vocab size:', self.vocab_size)

        return self.vectorize(text)

    # Reformat our data vector to feed into our model. Tricky with stateful rnn
    def _reshape_for_stateful_rnn(self, sequence):
        passes = []
        # Take strips of our data at every step size up to our seq_length and
        # cut those strips into seq_length sequences
        for offset in range(0, self.seq_length, self.seq_step):
            pass_samples = sequence[offset:]
            samples = pass_samples.size // self.seq_length
            pass_samples = np.resize(pass_samples, (samples, self.seq_length))
            passes.append(pass_samples)
        # Stack our samples together and make sure they fit evenly into batches
        all_samples = np.concatenate(passes)
        num_batches = all_samples.shape[0] // self.batch_size
        num_samples = num_batches * self.batch_size
        # Now the tricky part, we need to reformat our data so the first
        # sequence in the nth batch picks up exactly where the first sequence
        # in the (n - 1)th batch left off, as the lstm cell state will not be
        # reset between batches in the stateful model.
        reshuffled = np.zeros((num_samples, self.seq_length), dtype=np.int32)
        for batch_index in range(self.batch_size):
            # Take a slice of num_batches consecutive samples
            slice_start = batch_index * num_batches
            slice_end = slice_start + num_batches
            index_slice = all_samples[slice_start:slice_end, :]
            # Spread it across each of our batches in the same index position
            reshuffled[batch_index::self.batch_size, :] = index_slice
        return reshuffled

    # Builds the underlying keras model
    def _build_models(self, embedding_size, rnn_size, num_layers):
        model = Sequential()
        model.add(Embedding(self.vocab_size,
                            embedding_size,
                            batch_input_shape=(self.batch_size, None)))
        for layer in range(num_layers):
            model.add(LSTM(rnn_size,
                           stateful=True,
                           return_sequences=True))
        model.add(TimeDistributed(Dense(self.vocab_size,
                                        activation='softmax')))
        # With sparse_categorical_crossentropy we can leave as labels as
        # integers instead of one-hot vectors
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='rmsprop')
        model.summary()
        self.keras_model = model

    def train(self, data_dir, word_tokens, pristine_input, pristine_output,
              batch_size, seq_length, seq_step, embedding_size, rnn_size,
              num_layers, num_epochs, skip_sampling):
        # Store metadata we also need for sampling...
        self.word_tokens = word_tokens
        self.pristine_input = pristine_input
        self.pristine_output = pristine_output
        if self.pristine_input:
            self.pristine_output = True
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.seq_step = seq_step

        print_green('Loading data...')
        load_start = time.time()
        data = self._load_text(data_dir)
        x = self._reshape_for_stateful_rnn(data[:-1])
        y = self._reshape_for_stateful_rnn(data[1:])
        # Y data needs an extra axis to work with the sparse categorical
        # crossentropy loss function
        y = y[:, :, np.newaxis]

        print('x.shape:', x.shape)
        print('y.shape:', y.shape)
        load_end = time.time()
        print_red('Data load time', load_end - load_start)

        print_green('Building model...')
        model_start = time.time()
        self._build_models(embedding_size, rnn_size, num_layers)
        model_end = time.time()
        print_red('Model build time', model_end - model_start)

        print_green('Training...')
        train_start = time.time()
        # Train the model
        callbacks = []
        if not skip_sampling:
            callbacks.append(LiveSamplerCallback(self))
        self.keras_model.fit(x, y,
                             batch_size=self.batch_size,
                             shuffle=False,
                             epochs=num_epochs,
                             verbose=1,
                             callbacks=callbacks)

        self.keras_model.reset_states()
        train_end = time.time()
        print_red('Training time', train_end - train_start)

    def sample(self, seed=None, length=500, diversity=1.0):
        self.keras_model.reset_states()

        if seed is None:
            seed = random.choice(self.seeds)
            print('Using seed: ', end='')
            print_cyan(seed)
            print('-' * 50)

        # FIXME: Is there a way to make the current sample smaller not a batch
        # size vector?
        current_sample = np.zeros((self.batch_size, 1))
        preds = None

        # Feed in seed string
        print_cyan(seed, end=' ' if self.word_tokens else '')
        seed_vector = self.vectorize(seed)
        for char_index in np.nditer(seed_vector):
            current_sample.fill(char_index)
            preds = self.keras_model.predict(current_sample,
                                             batch_size=self.batch_size,
                                             verbose=0)

        sampled_indices = np.array([], dtype=np.int32)
        # Sample the model one token at a time
        for i in range(length):
            char_index = 0
            if preds is not None:
                char_index = sample_preds(preds[0][0], diversity)
            current_sample.fill(char_index)
            sampled_indices = np.append(sampled_indices, char_index)
            preds = self.keras_model.predict(current_sample,
                                             batch_size=self.batch_size,
                                             verbose=0)
        sample = self.unvectorize(sampled_indices)
        print(sample)
        return sample

    # Don't pickle the keras model, better to save it directly
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['keras_model']
        return state


# Save the keras model directly and pickle our meta model class
def save(model, data_dir):
    keras_file_path = os.path.join(data_dir, 'model.h5')
    pickle_file_path = os.path.join(data_dir, 'model.pkl')
    model.keras_model.save(filepath=keras_file_path)
    pickle.dump(model, open(pickle_file_path, 'wb'))
    print_green('Model saved to', pickle_file_path, keras_file_path)


# Load the meta model and restore its internal keras model
def load(data_dir):
    keras_file_path = os.path.join(data_dir, 'model.h5')
    pickle_file_path = os.path.join(data_dir, 'model.pkl')
    model = pickle.load(open(pickle_file_path, 'rb'))
    model.keras_model = load_model(keras_file_path)
    return model
