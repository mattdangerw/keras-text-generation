Keras Text Generation
=====================

RNN text generation using python and Keras. Supports both word and character
level models.

Requirements
------------
 - Keras 2.0
 - Colorama 0.3

Quick start
-----------
```shell
pip install tensorflow-gpu # Or tensorflow or Theano
pip install keras colorama
python train.py
python samply.py
```


Usage
-----

There are two invokable scripts, `train.py` and `sample.py`, which should be run in succession. Each operates on a
`data-dir` whose contents are as follows:
 - **input.txt**, input text corpora. Required by train.py
 - **model.h5**, keras model weights. Created by train.py and required by sample.py
 - **model.pkl**, model metadata. Created by train.py and required by sample.py

#### train.py

 - **--data-dir**, type=string, default=`data/nietzsche`. The data directory
   containing an `input.txt` file.
 - **--word-tokens**, type=flag, default=`False`. Whether to model the rnn at
   word level or char level.
 - **--pristine-input**, type=flag, default=`False`. Do not lowercase or attempt
   fancy. tokenization of input.
 - **--pristine-output**, type=flag, default=`False`. Do not detokenize output.
   Only applies is `--word-tokens` has been set.
 - **--embedding-size**, type=int, default=`32`. Size of the embedding layer.
 - **--rnn-size**, type=int, default=`128`. Number of LSTM cells in each RNN
   layer.
 - **--num-layers**, type=int, default=`2`. Number of layers in the RNN.
 - **--batch-size**, type=int, default=`100`. Minibatch size.
 - **--seq-length**, type=int, default=`50`. Training sequence length.
 - **--seq-step**, type=int, default=`10`. How often to pull a training sequence
   from the data.
 - **--num-epochs**, type=int, default=`50`. Number of epochs.
 - **--skip-sampling**, type=flag, default=`False`. Skip the live sampling stage
   of training.

#### sample.py

 - **--data-dir**, type=string, default=`data/nietzsche`. The data directory
   containing a `model.h5` and `model.pkl` file.
 - **--seed**, type=string, default=`None`. Seed string for sampling.
 - **--length**, type=int, default=`1000`. Length of the sample to generate.
 - **--diversity**, type=float, default=`1.0`. Sampling diversity.


FAQ
---

##### Why not just use [char-rnn-tensorflow](https://github.com/sherjilozair/char-rnn-tensorflow) or [word-rnn-tensorflow](https://github.com/hunkim/word-rnn-tensorflow)?

If your goal is just computational speed or low memory footprint, go with those
projects! Pretty much the appeal here is using Keras. If you want an easy
declarative language to try new approaches, this is a good place to start.


##### Can we add a command line flag for a different optimizer, RNN cell, etc.?

Most of command line flags exposed are to work with different datasets of
varying sizes. If you want to change the structure of the RNN, just change the
code! That's where Keras excels.

##### Can I use a different tokenization scheme for my word level model?

Yep! Pass the `--pristine-input` flag and use a fancier tokenizer as a
preprocessing step. Tokens will be formed by calling `text.split()` on the
input.

TODO
----

 - requirements.txt?
 - limit vocab size, UNK?? could be useful for mem footprint
 - look into embeddings more
 - lint
