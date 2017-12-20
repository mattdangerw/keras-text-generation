# -*- coding: utf-8 -*-

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from model import load


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-dir', type=str, default='data/tinyshakespeare',
                        help='data directory containing input.txt')
    parser.add_argument('--seed', type=str, default=None,
                        help='seed string for sampling')
    parser.add_argument('--length', type=int, default=1000,
                        help='length of the sample to generate')
    parser.add_argument('--diversity', type=float, default=1.0,
                        help='Sampling diversity')
    args = parser.parse_args()

    model = load(args.data_dir)
    del args.data_dir
    model.sample(**vars(args))


if __name__ == '__main__':
    main()
