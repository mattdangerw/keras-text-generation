from __future__ import print_function
import colorama
import numpy as np
import re
import sys

colorama.init()

def print_green(*args, **kwargs):
    sys.stdout.write(colorama.Fore.GREEN)
    return print(*args, colorama.Style.RESET_ALL, **kwargs)

def print_red(*args, **kwargs):
    sys.stdout.write(colorama.Fore.RED)
    return print(*args, colorama.Style.RESET_ALL, **kwargs)

def sample_preds(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Tokenization/string cleaning.
# Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data
def clean_str(string):
    string = re.sub(r"[^가-힣A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()
