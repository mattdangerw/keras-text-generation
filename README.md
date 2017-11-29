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

TODO!

FAQ
---

##### Why not just use char-rnn-tensorflow or word-rnn-tensorflow?

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
