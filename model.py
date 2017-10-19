from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
import numpy as np
import sys

class Model(Sequential):
    def __init__(self, args, loader):
        super().__init__()

        self.loader = loader
        self.batch_size = args.batch_size
        self.seq_length = args.seq_length
        self.seq_step = args.seq_step

        self.add(Embedding(loader.vocab_size, args.embedding_size, batch_size=args.batch_size))
        for layer in range(args.num_layers):
            self.add(LSTM(args.rnn_size, stateful=True, return_sequences=True))
        self.add(Dense(loader.vocab_size, activation='softmax'))
        # With sparse_categorical_crossentropy we can leave as labels as integers
        # instead of one-hot vectors
        self.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')
        self.summary()

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

    def sample_preds(self, preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def sample(self, seed_string=' ', length=400, diversity=1.0):
        self.reset_states()

        full_sample = np.array([], dtype=np.int32)
        # FIXME: Is there a way to make the current sample smaller not a batch_size vector?
        current_sample=np.zeros((self.batch_size, 1))
        preds = None

        # Feed in seed string
        seed_vector = self.loader.vectorize(seed_string)
        if seed_string:
            for char_index in np.nditer(seed_vector):
                current_sample.fill(char_index)
                full_sample = np.append(full_sample, char_index)
                preds = self.predict(current_sample, batch_size=self.batch_size, verbose=0)

        # Sample the model character by character
        for i in range(length):
            char_index = 0
            if preds is not None:
                char_index = self.sample_preds(preds[0][0], diversity)
            current_sample.fill(char_index)
            full_sample = np.append(full_sample, char_index)
            preds = self.predict(current_sample, batch_size=self.batch_size, verbose=0)
        return self.loader.unvectorize(full_sample)

    def train(self, data, num_epochs, live_sample=True):
        x = self.reshape_for_stateful_rnn(data[:-1])
        y = self.reshape_for_stateful_rnn(data[1:])
        # Y data needs an extra axis to work with the sparse categorical crossentropy
        # loss function
        y = y[:,:,np.newaxis]
        print('x.shape:', x.shape)
        print('y.shape:', y.shape)

        # Train the model
        for iteration in range(num_epochs):
            print('Epoch', iteration + 1)
            history = self.fit(x, y,
                batch_size=self.batch_size,
                epochs=1,
                verbose=1,
                shuffle=False)

            if live_sample:
                # Output generated text after each epoch
                for diversity in [0.2, 0.5, 1.0, 1.2]:
                    print('Sampling with diversity', diversity)
                    print('-' * 50)
                    print(self.sample(diversity=diversity))
                    print('-' * 50)
            self.reset_states()
