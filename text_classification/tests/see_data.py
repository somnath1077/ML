import hashlib
import re

from text_classification.utils import input_file


def bag_of_words():
    bag = {}

    regex_pattern = b'[><,:;.^%#!@\[\]~`\'\"=+{}?\-\\ ]+'
    with open(input_file, 'rb') as f:
        for line in f:
            cleaned_line = line.strip()
            words = re.split(regex_pattern, cleaned_line)
            for word in words:
                if word in bag:
                    bag[word] += 1
                else:
                    bag[word] = 1
    for word, count in bag.items():
        print('word = ', word, ' count = ', count)
    print("Number of words = {}".format(len(bag.keys())))

def hash_words():
    bag = {}

    with open(input_file, 'rb') as f:
        for line in f:
            cleaned_line = line.strip()
            hsh = hashlib.sha1(cleaned_line)
            if hsh in bag:
                bag[hsh] += 1
            else:
                bag[hsh] = 1
    for word, count in bag.items():
        print('word = ', word, ' count = ', count)

def see():
    with open(input_file, 'rb') as f:
        count = 1
        min = 100000
        lx = None
        for line in f:
            print(count, len(line))
            if min > len(line):
                min = len(line)
            if len(line) == 169:
                lx, num = line, count
            count += 1

        print("Minimum line length = {}".format(min))
        print(lx, num)

if __name__ == '__main__':
    bag_of_words()
    # hash_words()