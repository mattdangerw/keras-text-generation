# -*- coding: utf-8 -*-

from __future__ import print_function
import colorama
import numpy as np
import re

colorama.init()


def print_green(*args, **kwargs):
    print(colorama.Fore.GREEN, end='')
    print(*args, **kwargs)
    print(colorama.Style.RESET_ALL, end='')


def print_cyan(*args, **kwargs):
    print(colorama.Fore.CYAN, end='')
    print(*args, **kwargs)
    print(colorama.Style.RESET_ALL, end='')


def print_red(*args, **kwargs):
    print(colorama.Fore.RED, end='')
    print(*args, **kwargs)
    print(colorama.Style.RESET_ALL, end='')


# Samples an unnormalized array of probabilities. Use temperature to
# flatten/amplify the probabilities.
def sample_preds(preds, temperature=1.0):
    preds = np.asarray(preds).astype(np.float64)
    # Add a tiny positive number to avoid invalid log(0)
    preds += np.finfo(np.float64).tiny
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# Basic word tokenizer based on the Penn Treebank tokenization script, but
# setup to handle multiple sentences. Newline aware, i.e. newlines are replaced
# with a specific token. You may want to consider using a more robust tokenizer
# as a preprocessing step, and using the --pristine-input flag.
def word_tokenize(text):
    REGEXES = [
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
    for regexp, substitution in REGEXES:
        text = regexp.sub(substitution, text)
    return text.split()


# A hueristic attempt to undo the Penn Treebank tokenization above. Pass the
# --pristine-output flag if no attempt at detokenizing is desired.
def word_detokenize(tokens):
    REGEXES = [
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
    for regexp, substitution in REGEXES:
        text = regexp.sub(substitution, text)
    return text.strip()
