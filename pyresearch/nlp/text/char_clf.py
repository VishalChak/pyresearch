

from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os

def find_files(path):
    return glob.glob(path)

print(find_files('/home/vishal/datasets/data/names/*.txt'))

import unicodedata
import string

all_letters = string.ascii_letters +".,;'"
n_letter = len(all_letters)


### unicode string to plain ASCII

def unicodeToAscii(s):
    return ''.join( c for c in unicodedata.normalize('NFD', s)
    if unicodedata.category(c)!= 'Mn' and c in all_letters
    )

print(unicodeToAscii('Ślusàrski'))