from __future__ import print_function
import colorama
from keras.callbacks import Callback
from keras.layers import Dense, Embedding, LSTM
from keras.models import Sequential, load_model
import numpy as np
import os
import pickle
import sys
import time

colorama.init()

def print_green(*args, **kwargs):
    sys.stdout.write(colorama.Fore.GREEN)
    return print(*args, colorama.Style.RESET_ALL, **kwargs)

def print_red(*args, **kwargs):
    sys.stdout.write(colorama.Fore.RED)
    return print(*args, colorama.Style.RESET_ALL, **kwargs)

class Struct:
    def __init__(self, **args):
        self.__dict__.update(args)

class LiveSamplerCallback(Callback):
    def __init__(self, model):
        self.our_model = model

    def on_epoch_end(self, epoch, logs={}):
        print()
        print_green('Sampling model...')
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print('Sampling with diversity', diversity)
            print('-' * 50)
            print(self.our_model.sample(Struct(length=400, seed=' ', temperature=1.0)))
            print('-' * 50)

class Model:
    def __init__(self):
        super().__init__()

    def load_text(self, data_dir):
        text = open(os.path.join(data_dir, 'input.txt')).read().lower()
        print('corpus length:', len(text))

        all_chars = sorted(list(set(text)))
        self.vocab_size = len(all_chars)
        print('total chars:', self.vocab_size)
        self.char_indices = dict((c, i) for i, c in enumerate(all_chars))
        self.indices_char = dict((i, c) for i, c in enumerate(all_chars))

        return self.vectorize(text)

    def vectorize(self, text):
        indices = list(map(lambda char: self.char_indices[char], text))
        return np.array(indices, dtype=np.int32)

    def unvectorize(self, vector):
        chars = map(lambda index: self.indices_char[index], vector.tolist())
        return ''.join(chars)

    # Reformat our data vector to feed into our model. Tricky with stateful rnns
    def reshape_for_stateful_rnn(self, sequence):
        passes = []
        # Take strips of our data at every step size up to our seq_length and
        # cut those strips into seq_length sequences
        for offset in range(0, self.seq_length, self.seq_step):
            pass_samples = sequence[offset:]
            samples = pass_samples.size // self.seq_length
            pass_samples = np.resize(pass_samples, (samples, self.seq_length))
            passes.append(pass_samples)
        # Stack our samples together and make sure they fit evenly into batches.
        all_samples = np.concatenate(passes)
        num_batches = all_samples.shape[0] // self.batch_size
        # Now the tricky part, we need to reformat our data so the first sequence in
        # the nth batch picks up exactly where the first sequence in the (n - 1)th
        # batch left off, as the lstm cell state will not be reset between batches
        # in the stateful model.
        reshuffled = np.zeros((num_batches * self.batch_size, self.seq_length), dtype=np.int32)
        for batch_index in range(self.batch_size):
            reshuffled[batch_index::self.batch_size, :] = all_samples[batch_index * num_batches:(batch_index + 1) * num_batches, :]
        return reshuffled

    def build_model(self, args):
        keras_model = Sequential()
        keras_model.add(Embedding(self.vocab_size, args.embedding_size, batch_size=args.batch_size))
        for layer in range(args.num_layers):
            keras_model.add(LSTM(args.rnn_size, stateful=True, return_sequences=True))
        keras_model.add(Dense(self.vocab_size, activation='softmax'))
        # With sparse_categorical_crossentropy we can leave as labels as integers
        # instead of one-hot vectors
        keras_model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')
        return keras_model

    def train(self, args):
        print_green('Loading data...')
        load_start = time.time()
        data = self.load_text(args.data_dir)

        self.batch_size = args.batch_size
        self.seq_length = args.seq_length
        self.seq_step = args.seq_step
        x = self.reshape_for_stateful_rnn(data[:-1])
        y = self.reshape_for_stateful_rnn(data[1:])
        # Y data needs an extra axis to work with the sparse categorical crossentropy
        # loss function
        y = y[:,:,np.newaxis]

        print('x.shape:', x.shape)
        print('y.shape:', y.shape)
        load_end = time.time()
        print_red('Data load time', load_end - load_start)

        print_green('Building model...')
        model_start = time.time()
        self.keras_model = self.build_model(args)
        self.keras_model.summary()
        model_end = time.time()
        print_red('Model build time', model_end - model_start)

        print_green('Training...')
        train_start = time.time()
        # Train the model
        callbacks = []
        if not args.skip_sampling:
            callbacks.append(LiveSamplerCallback(self))
        history = self.keras_model.fit(x, y,
            batch_size=self.batch_size,
            shuffle=False,
            epochs=args.num_epochs,
            verbose=1,
            callbacks=callbacks)

        self.keras_model.reset_states()
        train_end = time.time()
        print_red('Training time', train_end - train_start)

    def sample_preds(self, preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def sample(self, args):
        self.keras_model.reset_states()

        full_sample = np.array([], dtype=np.int32)
        # FIXME: Is there a way to make the current sample smaller not a batch_size vector?
        current_sample=np.zeros((self.batch_size, 1))
        preds = None

        # Feed in seed string
        if args.seed:
            seed_vector = self.vectorize(args.seed)
            for char_index in np.nditer(seed_vector):
                current_sample.fill(char_index)
                full_sample = np.append(full_sample, char_index)
                preds = self.keras_model.predict(current_sample, batch_size=self.batch_size, verbose=0)

        # Sample the model one character/word at a time
        for i in range(args.length):
            char_index = 0
            if preds is not None:
                char_index = self.sample_preds(preds[0][0], args.temperature)
            current_sample.fill(char_index)
            full_sample = np.append(full_sample, char_index)
            preds = self.keras_model.predict(current_sample, batch_size=self.batch_size, verbose=0)
        return self.unvectorize(full_sample)

    # Don't pickle the keras model
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['keras_model']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

def save(model, data_dir):
    keras_file_path = os.path.join(data_dir, 'model.h5')
    pickle_file_path = os.path.join(data_dir, 'model.pkl')
    model.keras_model.save(filepath=keras_file_path)
    pickle.dump(model, open(pickle_file_path, 'wb'))

def load(data_dir):
    keras_file_path = os.path.join(data_dir, 'model.h5')
    pickle_file_path = os.path.join(data_dir, 'model.pkl')
    model = pickle.load(open(pickle_file_path, 'rb'))
    model.keras_model = load_model(keras_file_path)
    return model
