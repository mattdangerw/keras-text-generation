"""
python remove_hardwrap.py raw.txt fixed.txt

Simple script to remove newlines and extra whitespace from hard wrapping text.
By default the trained models in this project will attempt to learn where to
place newlines. This is useful for input such as Shakespeare, but hard wrapping
in prose is just for convinient viewing and adds no semantic value. You will
likely get better results with the wrapping removed.

Assumes any consective non whitespace lines should be joined with a space (it
is part of a hard wrapped paragraph), and lines seperated by blank lines should
be joined with a double newline (they are in seperate paragraphs).

If that's not suited to your corpus use a fancier script!

----- For example
Lorem ipsum dolor sit amet,
consectetur adipiscing elit,
sed do eiusmod tempor.

Ut enim
ad minim veniam, quis nostrud exercitation ullamco laboris nisi
ut.

Duis aute irure dolor in reprehenderit in voluptate velit esse cillum
dolore eu.
----- becomes
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor.

Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut.

Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu.
-----
"""

from __future__ import print_function
import sys


def main():
    if len(sys.argv) < 3:
        print('Usage: python', sys.argv[0], 'input_path output_path')
        sys.exit(1)
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    print('Input:', input_path)
    print('Output:', output_path)

    with open(input_path, 'r') as input_file, \
         open(output_path, 'w') as output_file:
        seperator = ''
        for line in input_file:
            stripped = line.strip()
            if stripped:
                # Write out our last seperator
                output_file.write(seperator)
                output_file.write(stripped)
                # If line is non empty, join the next line with a space
                seperator = ' '
            else:
                # If line is empty, we are at a paragraph break. Join the next
                # line with a double return
                seperator = '\n\n'

if __name__ == '__main__':
    main()
