Keras Text Generation
=====================

Recurrent neural network (RNN) text generation using Keras. Generating text with
neural networks [is fun](http://karpathy.github.io/2015/05/21/rnn-effectiveness/),
and there are a ton of projects and standalone scripts to do it.

This project does not provide any groundbreaking features over what it already
out there, but attempts to be a good, well documented place to start playing
with text generation within the Keras framework. It handles the nitty-gritty
details of loading a text corpus and feeding it into a Keras model.

Supports both a character-level model and a word-level model (with
tokenization). Supports saving a model and model metadata to disk for later
sampling. Supports using a validation set. Uses stateful RNNs within Keras for
more efficient sampling.

Requirements
------------

- Keras 2.0
- Colorama 0.3

Quick start
-----------

```shell
pip install tensorflow-gpu # Or tensorflow or Theano
pip install keras colorama
# Train on the included Shakespeare corpus with default parameters
python train.py
# Sample the included Shakespeare corpus with default parameters
python samply.py
# Train with long samples, more layers, more epochs, and live sampling
python train.py --seq-length 100 --num-layers 4 --num-epochs 100 --live-sample
# Sample with a random seed for 500 characters and more random output
python sample.py --length 500 --temperature 2.0
# Train on a new dataset with a word level model and larger embedding
python train.py --data-dir ~/datasets/twain --word-tokens --embedding-size 128
# Sample new dataset with a custom seed
python sample.py --data-dir ~/datasets/twain --seed "History doesn't repeat itself, but"
```

Usage
-----

There are two invokable scripts, `train.py` and `sample.py`, which should be run
in succession. Each operates on a data directory whose contents are as follows:

- **input.txt**, input text corpora. Required by `train.py`
- **validate.txt**, optional validation text corpora. Used in `train.py`
- **model.h5**, keras model weights. Created by `train.py` and required by `sample.py`
- **model.pkl**, model metadata. Created by `train.py` and required by `sample.py`

The `input.txt` file should contain whatever texts you would like to train the
RNN on, concatenated into a single file. The text processing is by default
newline aware, so if you files contain hard wrapped prose, you may want to
remove the wrapping newlines. The `validate.txt` file should be formatted
similarly to the input.txt. It is totally optional, but useful to monitor
for overfitting, etc.

There are two main modes to process the input--a character-level model and
a word-level model. Under the character level model, we will simply lowercase
the input text and feed it into the RNN character by character. Under the word
level model, the input text will be split into individual word tokens and each
token will be given a separate value before being fed into the RNN. Word will be
tokenized roughly following the Penn Treebank approach. By default we will
heuristically attempt to "detokenize" the text after sampling, but this can be
disabled with `--pristine-output`.

### train.py

- **--data-dir**, type=string, default=data/tinyshakespeare. The data directory
  containing an `input.txt` file.
- **--live-sample**, type=flag. Sample the model after every epoch. Very
  useful if you want to quickly iterate on a new approach.
- **--word-tokens**, type=flag. Whether to model the RNN at a word level or a
  a character level.
- **--pristine-input**, type=flag. For character models, do not lowercase the
  text corpora before feeding it into the RNN. For word models, do not attempt
  fancy tokenization. You can pass this this to use your own tokenizer as a
  preprocessing step. Implies `--pristine-output`.
- **--pristine-output**, type=flag. For word models, do not attempt to
  "detokenize" the output.
- **--embedding-size**, type=int, default=64. Size of the embedding layer.
  This can be much lower when using the character level model, and bigger under
  the word level model.
- **--rnn-size**, type=int, default=128. Number of LSTM cells in each RNN
  layer.
- **--num-layers**, type=int, default=2. Number of layers in the RNN.
- **--batch-size**, type=int, default=32. Batch size, i.e. how many samples
  to process in parallel during training.
- **--seq-length**, type=int, default=50. We will split the input text into
  into individual samples of length `--seq-length` before feeding them into the
  RNN. The model will be bad at learning dependencies in the text longer then
  `--seq-length`.
- **--seq-step**, type=int, default=25. We grab samples from the input text
  semi redundantly every `--seq-step` characters (or words). For example, a
  `--seq-length` of 50 and `--seq-step` of 25 would pull each character in the
  text into two separate samples offset from each other by 25 characters.
- **--num-epochs**, type=int, default=50. Number of epochs, i.e. passes over
  the entire dataset during training.

### sample.py

- **--data-dir**, type=string, default=data/tinyshakespeare. The data directory
  containing `model.h5` and `model.pkl` files generated by `train.py`.
- **--seed**, type=string, default=None. Seed string for sampling. If no seed
  is supplied we will grab a random seed from the input text.
- **--length**, type=int, default=1000. Length of the sample to generate. For
  the word level model this is length in words.
- **--diversity**, type=float, default=1.0. Sampling diversity. A diversity
  of < 1.0 will make take conservative guesses from the RNN when generating
  text. A diversity of > 1.0 will make riskier choices when generating text.

FAQ
---

**Why not just use [char-rnn-tensorflow](https://github.com/sherjilozair/char-rnn-tensorflow) or [word-rnn-tensorflow](https://github.com/hunkim/word-rnn-tensorflow)?**

If your goal is just computational speed or low memory footprint, go with those
projects! Pretty much the appeal here is using Keras. If you want an easy
declarative framework to try new approaches, this is a good place to start.

There are also a few additional features here, such as fancier word tokenization
and support for a hold out validation set, that may be of use depending on your
application.

**Can we add a command line flag for a different optimizer, RNN cell, etc.?**

Most of command line flags exposed are to work with different datasets of
varying sizes. If you want to change the structure of the RNN, just change the
code. That's where Keras excels.

**Can I use a different tokenization scheme for my word level model?**

Yep! Pass the `--pristine-input` flag and use a fancier tokenizer as a
preprocessing step. Tokens will be formed by calling `text.split()` on the
input.
