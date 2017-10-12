print('Reading input...')
path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
text = open(path).read().lower()
print('corpus length:', len(text))

all_chars = sorted(list(set(text)))
vocab_size = len(all_chars)
print('total chars:', vocab_size)
char_indices = dict((c, i) for i, c in enumerate(all_chars))
indices_char = dict((i, c) for i, c in enumerate(all_chars))

print('Vectorization...')
indices = list(map(lambda char: char_indices[char], text))
data = np.array(indices, dtype=np.int32)
