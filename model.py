from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.utils.data_utils import get_file
import numpy as np
import sys

class Model(Sequential):
    def __init__(args, vocab_size):
        super.__init()
        self.batch_size = args.batch_size

        self.add(Embedding(vocab_size, args.embedding_size, batch_size=args.batch_size))
        for layer in range(args.num_layers)
            self.add(LSTM(args.rnn_size, stateful=True, return_sequences=True))
        self.add(Dense(vocab_size, activation='softmax'))
        # With sparse_categorical_crossentropy we can leave as labels as integers
        # instead of one-hot vectors
        self.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')
        self.summary()

    # Reformat our data vector to feed into our model. Tricky with stateful rnns
    def reshape_for_stateful_rnn(self, sequence):
        passes = []
        # Take strips of our data at every step size up to our sequence_length and
        # cut those strips into sequence_length sequences
        for offset in range(0, sequence_length, step):
            pass_samples = sequence[offset:]
            samples = pass_samples.size // sequence_length
            pass_samples = np.resize(pass_samples, (samples, sequence_length))
            passes.append(pass_samples)
        # Stack our samples together and make sure they fit evenly into batches.
        all_samples = np.concatenate(passes)
        num_batches = all_samples.shape[0] // self.batch_size
        # Now the tricky part, we need to reformat our data so the first sequence in
        # the nth batch picks up exactly where the first sequence in the (n - 1)th
        # batch left off, as the lstm cell state will not be reset between batches
        # in the stateful model.
        reshuffled = np.zeros((num_batches * self.batch_size, sequence_length), dtype=np.int32)
        for batch_index in range(self.batch_size):
            reshuffled[batch_index::self.batch_size, :] = all_samples[batch_index * num_batches:(batch_index + 1) * num_batches, :]
        return reshuffled

    def train(self, data, epochs, sample):
        x = self.reshape_for_stateful_rnn(data[:-1])
        x = self.reshape_for_stateful_rnn(x)
        y = self.reshape_for_stateful_rnn(y)

# Y data needs an extra axis to work with the sparse categorical crossentropy
# loss function
y = y[:,:,np.newaxis]

print('x.shape:', x.shape)
print('y.shape:', y.shape)

print('Building model...')
model = Sequential()
# We won't specify a complete input_shape here, which allows this model to work
# with any length input.
model.add(Embedding(vocab_size, 32, batch_size=batch_size))
model.add(LSTM(128, stateful=True, return_sequences=True))
model.add(LSTM(128, stateful=True, return_sequences=True))
model.add(Dense(vocab_size, activation='softmax'))
# With sparse_categorical_crossentropy we can leave as labels as integers
# instead of one-hot vectors
model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')
model.summary()

# Helper function to sample an index from a probability array
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Train the model
for iteration in range(1, 80):
    print('Epoch', iteration)
    history=model.fit(x, y,
        batch_size=batch_size,
        epochs=1,
        verbose=1,
        shuffle=False)

    # Output generated text after each epoch
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('Sampling with diversity', diversity)
        print('-' * 50)

        model.reset_states()
        seed_string='the '
        x_sample=np.zeros((batch_size, 1))
        for t, char in enumerate(seed_string[:-1]):
            x_sample.fill(char_indices[char])
            model.predict(x_sample, batch_size=batch_size, verbose=0)

        sys.stdout.write(seed_string),
        last_char = seed_string[-1]
        last_index = char_indices[last_char]
        for i in range(400):
            x_sample.fill(last_index)
            preds = model.predict(x_sample, batch_size=batch_size, verbose=0)
            last_index = sample(preds[0][0], diversity)
            last_char = indices_char[last_index]
            sys.stdout.write(last_char)
        sys.stdout.flush()
        print()
        print('-' * 50)

    model.reset_states()
