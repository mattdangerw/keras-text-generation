import argparse

from model import Model, save

def main():
    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-dir', type=str, default='data/nietzsche',
                        help='data directory containing input.txt')
    parser.add_argument('--embedding-size', type=int, default=32,
                        help='size of the embedding')
    parser.add_argument('--rnn-size', type=int, default=128,
                        help='size of RNN hidden state')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='number of layers in the RNN')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='minibatch size')
    parser.add_argument('--seq-length', type=int, default=50,
                        help='training sequence length')
    parser.add_argument('--seq-step', type=int, default=10,
                        help='how often to pull a training sequence from the data')
    parser.add_argument('--num-epochs', type=int, default=50,
                        help='number of epochs')
    parser.add_argument('--skip-sampling', action='store_true',
                        help='skip the live sampling stage of training')
    args = parser.parse_args()
    train(args)

def train(args):
    model = Model()
    model.train(args)
    save(model, args.data_dir)

if __name__ == '__main__':
    main()
