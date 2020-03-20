import os

def load_names(path):
    with open(path) as text_file:
        return text_file.read().splitlines()




