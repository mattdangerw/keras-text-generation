import argparse

from model import load

def main():
    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-dir', type=str, default='data/nietzsche',
                        help='data directory containing input.txt')
    parser.add_argument('--seed', type=str, default=' ',
                        help='seed string for sampling')
    parser.add_argument('--length', type=int, default=400,
                        help='length of the sample to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature')
    args = parser.parse_args()
    train(args)

def train(args):
    model = load(args.data_dir)
    print(model.sample(args))

if __name__ == '__main__':
    main()
