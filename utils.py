from __future__ import print_function
import colorama
import numpy as np
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
