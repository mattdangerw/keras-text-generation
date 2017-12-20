# -*- coding: utf-8 -*-

from __future__ import print_function
from collections import Counter

import numpy as np

from utils import print_red
from utils import word_tokenize, word_detokenize

class Vectorizer:
    """
    Transforms text to vectors of integer numbers representing in text tokens
    and back. Handles word and character level tokenization.
    """
    def __init__(self, text, word_tokens, pristine_input, pristine_output):
        self.word_tokens = word_tokens
        self._pristine_input = pristine_input
        self._pristine_output = pristine_output

        tokens = self._tokenize(text)
        print('corpus length:', len(tokens))
        token_counts = Counter(tokens)
        # Sort so most common tokens come first in our vocabulary
        tokens = [x[0] for x in token_counts.most_common()]
        self._token_indices = {x: i for i, x in enumerate(tokens)}
        self._indices_token = {i: x for i, x in enumerate(tokens)}
        self.vocab_size = len(tokens)
        print('vocab size:', self.vocab_size)

    def _tokenize(self, text):
        if not self._pristine_input:
            text = text.lower()
        if self.word_tokens:
            if self._pristine_input:
                return text.split()
            return word_tokenize(text)
        return text

    def _detokenize(self, tokens):
        if self.word_tokens:
            if self._pristine_output:
                return ' '.join(tokens)
            return word_detokenize(tokens)
        return ''.join(tokens)

    def vectorize(self, text):
        """Transforms text to a vector of integers"""
        tokens = self._tokenize(text)
        indices = []
        for token in tokens:
            if token in self._token_indices:
                indices.append(self._token_indices[token])
            else:
                print_red('Ignoring unrecognized token:', token)
        return np.array(indices, dtype=np.int32)

    def unvectorize(self, vector):
        """Transforms a vector of integers back to text"""
        tokens = [self._indices_token[index] for index in vector.tolist()]
        return self._detokenize(tokens)
