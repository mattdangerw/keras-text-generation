# -*- coding: utf-8 -*-

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from model import MetaModel, save


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-dir', type=str, default='data/tinyshakespeare',
                        help='data directory containing input.txt')
    parser.add_argument('--live-sample', action='store_true',
                        help='live sample the model after each epoch')
    parser.add_argument('--word-tokens', action='store_true',
                        help='whether to model the rnn at word level or char level')
    parser.add_argument('--pristine-input', action='store_true',
                        help='do not lowercase or attempt fancy tokenization of input')
    parser.add_argument('--pristine-output', action='store_true',
                        help='do not detokenize output (word-tokens only)')
    parser.add_argument('--embedding-size', type=int, default=64,
                        help='size of the embedding')
    parser.add_argument('--rnn-size', type=int, default=128,
                        help='size of RNN layers')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='number of layers in the RNN')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='minibatch size')
    parser.add_argument('--seq-length', type=int, default=50,
                        help='training sequence length')
    parser.add_argument('--seq-step', type=int, default=25,
                        help='how often to pull a training sequence from the data')
    parser.add_argument('--num-epochs', type=int, default=50,
                        help='number of epochs')
    args = parser.parse_args()

    model = MetaModel()
    model.train(**vars(args))
    save(model, args.data_dir)


if __name__ == '__main__':
    main()
