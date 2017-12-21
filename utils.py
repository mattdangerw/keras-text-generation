# -*- coding: utf-8 -*-

from __future__ import print_function
import random
import re

import colorama
import numpy as np

colorama.init()


def print_green(*args, **kwargs):
    """Prints green text to terminal"""
    print(colorama.Fore.GREEN, end='')
    print(*args, **kwargs)
    print(colorama.Style.RESET_ALL, end='')


def print_cyan(*args, **kwargs):
    """Prints cyan text to terminal"""
    print(colorama.Fore.CYAN, end='')
    print(*args, **kwargs)
    print(colorama.Style.RESET_ALL, end='')


def print_red(*args, **kwargs):
    """Prints red text to terminal"""
    print(colorama.Fore.RED, end='')
    print(*args, **kwargs)
    print(colorama.Style.RESET_ALL, end='')


def sample_preds(preds, temperature=1.0):
    """
    Samples an unnormalized array of probabilities. Use temperature to
    flatten/amplify the probabilities.
    """
    preds = np.asarray(preds).astype(np.float64)
    # Add a tiny positive number to avoid invalid log(0)
    preds += np.finfo(np.float64).tiny
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def word_tokenize(text):
    """
    Basic word tokenizer based on the Penn Treebank tokenization script, but
    setup to handle multiple sentences. Newline aware, i.e. newlines are
    replaced with a specific token. You may want to consider using a more robust
    tokenizer as a preprocessing step, and using the --pristine-input flag.
    """
    regexes = [
        # Starting quotes
        (re.compile(r'(\s)"'), r'\1 “ '),
        (re.compile(r'([ (\[{<])"'), r'\1 “ '),
        # Punctuation
        (re.compile(r'([:,])([^\d])'), r' \1 \2'),
        (re.compile(r'([:,])$'), r' \1 '),
        (re.compile(r'\.\.\.'), r' ... '),
        (re.compile(r'([;@#$%&])'), r' \1 '),
        (re.compile(r'([?!\.])'), r' \1 '),
        (re.compile(r"([^'])' "), r"\1 ' "),
        # Parens and brackets
        (re.compile(r'([\]\[\(\)\{\}\<\>])'), r' \1 '),
        # Double dashes
        (re.compile(r'--'), r' -- '),
        # Ending quotes
        (re.compile(r'"'), r' ” '),
        (re.compile(r"([^' ])('s|'m|'d) "), r"\1 \2 "),
        (re.compile(r"([^' ])('ll|'re|'ve|n't) "), r"\1 \2 "),
        # Contractions
        (re.compile(r"\b(can)(not)\b"), r' \1 \2 '),
        (re.compile(r"\b(d)('ye)\b"), r' \1 \2 '),
        (re.compile(r"\b(gim)(me)\b"), r' \1 \2 '),
        (re.compile(r"\b(gon)(na)\b"), r' \1 \2 '),
        (re.compile(r"\b(got)(ta)\b"), r' \1 \2 '),
        (re.compile(r"\b(lem)(me)\b"), r' \1 \2 '),
        (re.compile(r"\b(mor)('n)\b"), r' \1 \2 '),
        (re.compile(r"\b(wan)(na)\b"), r' \1 \2 '),
        # Newlines
        (re.compile(r'\n'), r' \\n ')
    ]

    text = " " + text + " "
    for regexp, substitution in regexes:
        text = regexp.sub(substitution, text)
    return text.split()


def word_detokenize(tokens):
    """
    A heuristic attempt to undo the Penn Treebank tokenization above. Pass the
    --pristine-output flag if no attempt at detokenizing is desired.
    """
    regexes = [
        # Newlines
        (re.compile(r'[ ]?\\n[ ]?'), r'\n'),
        # Contractions
        (re.compile(r"\b(can)\s(not)\b"), r'\1\2'),
        (re.compile(r"\b(d)\s('ye)\b"), r'\1\2'),
        (re.compile(r"\b(gim)\s(me)\b"), r'\1\2'),
        (re.compile(r"\b(gon)\s(na)\b"), r'\1\2'),
        (re.compile(r"\b(got)\s(ta)\b"), r'\1\2'),
        (re.compile(r"\b(lem)\s(me)\b"), r'\1\2'),
        (re.compile(r"\b(mor)\s('n)\b"), r'\1\2'),
        (re.compile(r"\b(wan)\s(na)\b"), r'\1\2'),
        # Ending quotes
        (re.compile(r"([^' ]) ('ll|'re|'ve|n't)\b"), r"\1\2"),
        (re.compile(r"([^' ]) ('s|'m|'d)\b"), r"\1\2"),
        (re.compile(r'[ ]?”'), r'"'),
        # Double dashes
        (re.compile(r'[ ]?--[ ]?'), r'--'),
        # Parens and brackets
        (re.compile(r'([\[\(\{\<]) '), r'\1'),
        (re.compile(r' ([\]\)\}\>])'), r'\1'),
        (re.compile(r'([\]\)\}\>]) ([:;,.])'), r'\1\2'),
        # Punctuation
        (re.compile(r"([^']) ' "), r"\1' "),
        (re.compile(r' ([?!\.])'), r'\1'),
        (re.compile(r'([^\.])\s(\.)([\]\)}>"\']*)\s*$'), r'\1\2\3'),
        (re.compile(r'([#$]) '), r'\1'),
        (re.compile(r' ([;%:,])'), r'\1'),
        # Starting quotes
        (re.compile(r'(“)[ ]?'), r'"')
    ]

    text = ' '.join(tokens)
    for regexp, substitution in regexes:
        text = regexp.sub(substitution, text)
    return text.strip()


def find_random_seeds(text, num_seeds=50, max_seed_length=50):
    """Heuristic attempt to find some good seed strings in the input text"""
    lines = text.split('\n')
    # Take a random sampling of lines
    if len(lines) > num_seeds * 4:
        lines = random.sample(lines, num_seeds * 4)
    # Take the top quartile based on length so we get decent seed strings
    lines = sorted(lines, key=len, reverse=True)
    lines = lines[:num_seeds]
    # Split on the first whitespace before max_seed_length
    return [line[:max_seed_length].rsplit(None, 1)[0] for line in lines]


def shape_for_stateful_rnn(data, batch_size, seq_length, seq_step):
    """
    Reformat our data vector into input and target sequences to feed into our
    RNN. Tricky with stateful RNNs.
    """
    # Our target sequences are simply one timestep ahead of our input sequences.
    # e.g. with an input vector "wherefore"...
    # targets:   h e r e f o r e
    # predicts   ^ ^ ^ ^ ^ ^ ^ ^
    # inputs:    w h e r e f o r
    inputs = data[:-1]
    targets = data[1:]

    # We split our long vectors into semi-redundant seq_length sequences
    inputs = _create_sequences(inputs, seq_length, seq_step)
    targets = _create_sequences(targets, seq_length, seq_step)

    # Make sure our sequences line up across batches for stateful RNNs
    inputs = _batch_sort_for_stateful_rnn(inputs, batch_size)
    targets = _batch_sort_for_stateful_rnn(targets, batch_size)

    # Our target data needs an extra axis to work with the sparse categorical
    # crossentropy loss function
    targets = targets[:, :, np.newaxis]
    return inputs, targets

def _create_sequences(vector, seq_length, seq_step):
    # Take strips of our vector at seq_step intervals up to our seq_length
    # and cut those strips into seq_length sequences
    passes = []
    for offset in range(0, seq_length, seq_step):
        pass_samples = vector[offset:]
        num_pass_samples = pass_samples.size // seq_length
        pass_samples = np.resize(pass_samples,
                                 (num_pass_samples, seq_length))
        passes.append(pass_samples)
    # Stack our sequences together. This will technically leave a few "breaks"
    # in our sequence chain where we've looped over are entire dataset and
    # return to the start, but with large datasets this should be neglegable
    return np.concatenate(passes)

def _batch_sort_for_stateful_rnn(sequences, batch_size):
    # Now the tricky part, we need to reformat our data so the first
    # sequence in the nth batch picks up exactly where the first sequence
    # in the (n - 1)th batch left off, as the RNN cell state will not be
    # reset between batches in the stateful model.
    num_batches = sequences.shape[0] // batch_size
    num_samples = num_batches * batch_size
    reshuffled = np.zeros((num_samples, sequences.shape[1]), dtype=np.int32)
    for batch_index in range(batch_size):
        # Take a slice of num_batches consecutive samples
        slice_start = batch_index * num_batches
        slice_end = slice_start + num_batches
        index_slice = sequences[slice_start:slice_end, :]
        # Spread it across each of our batches in the same index position
        reshuffled[batch_index::batch_size, :] = index_slice
    return reshuffled
