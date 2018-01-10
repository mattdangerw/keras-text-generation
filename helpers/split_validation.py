"""
python split_validation.py raw.txt input.txt validate.txt

Takes an input file and splits it into a training text file and validation text
file of distinc text.

Usually just chunking off a large piece of text at the end of corpora is a bad
idea, as the subject matter and style of the text may have changed
significantly. You may rapidly appear to overfit your validation set when still
learning interesting patterns in your train set.

On the otherhand, just lopping off random strings of characters ad hoc from your
training data may ruin the continuity of the text and hurt the quality of your
data.

This script attempts a middle ground. Split out large enough chunks of text that
the continuity of both the training and validation texts are preserved. But make
enough slices in your input text to get a good sampling of the entire corpora.
In an additional attempt to preserve continuity, we will only split text into
a validation set at the end of a paragrah.
"""

from __future__ import print_function, division
import sys
import re

# Constants to modify, too lazy to expose these as arguments...

# Fraction of the data to make into validation data. As the incisions are only
# at the end of paragraphs, the actual split will not perfectly match the
# desired split.
VALIDATION_SPLIT = 0.1
# A regex to match valid inicision points. By default we only match blank lines
# so will only split on new paragraphs. Change to r'.*' to split on any newline.
INCISION_POINT = r'\n'
# The size in number of characters we would like our validations incisions to
# be. A smaller slice size means more incisions.
VALIDATION_SLICE_SIZE = 5000

def main():
    if len(sys.argv) < 4:
        print('Usage: python', sys.argv[0],
              'input_path train_path validate_path')
        sys.exit(1)
    input_path = sys.argv[1]
    train_path = sys.argv[2]
    validate_path = sys.argv[3]

    print('Input:', input_path)
    print('Train:', train_path)
    print('Validate:', validate_path)

    # Track number of incisions
    # Track actual split fraction
    with open(input_path, 'r') as input_file, \
         open(train_path, 'w') as train_file, \
         open(validate_path, 'w') as validate_file:
        incising = False
        read = 0
        train_split = (1.0 - VALIDATION_SPLIT)
        train_slice_size = VALIDATION_SLICE_SIZE * train_split / VALIDATION_SPLIT
        switch = train_slice_size
        num_incisions = 0
        validation_total_size = 0
        for line in input_file:
            read += len(line)
            if incising:
                validate_file.write(line)
                validation_total_size += len(line)
            else:
                train_file.write(line)
            if read > switch and re.match(INCISION_POINT, line):
                incising = not incising
                if incising:
                    num_incisions += 1
                    switch += VALIDATION_SLICE_SIZE
                else:
                    switch += train_slice_size
        print('Num incisions', num_incisions)
        print('Actual validation split:', validation_total_size / read)

if __name__ == '__main__':
    main()
