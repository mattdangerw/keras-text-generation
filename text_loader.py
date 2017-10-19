import numpy as np
import os
import pickle

# Helper function to sample an index from a probability array
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

class TextLoader:
    def __init__(self, datadir):
        print('Reading input...')
        text = open(os.path.join(datadir, 'input.txt')).read().lower()
        print('corpus length:', len(text))

        all_chars = sorted(list(set(text)))
        self.vocab_size = len(all_chars)
        print('total chars:', self.vocab_size)
        self.char_indices = dict((c, i) for i, c in enumerate(all_chars))
        self.indices_char = dict((i, c) for i, c in enumerate(all_chars))

        with open(os.path.join(datadir, 'vocab.pkl'), 'wb') as f:
            pickle.dump(all_chars, f)

        print('Vectorization...')
        self.data = self.vectorize(text)

    def vectorize(self, text):
        indices = list(map(lambda char: self.char_indices[char], text))
        return np.array(indices, dtype=np.int32)

    def unvectorize(self, vector):
        chars = map(lambda index: self.indices_char[index], vector.tolist())
        return ''.join(chars)
