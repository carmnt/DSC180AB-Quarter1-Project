#!/usr/bin/env python

import sys
import json
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
import random

from etl import get_data

def main(targets):
    if 'data' in targets:
        with open('data-params.json') as fh:
            data_params = json.load(fh)
        get_data(**data_params)

if __name__ == '__main__':
    args = sys.argv[1:]

def mask(seq, mask=0.15):
    '''Masks percentage of positions on a dna sequence'''
    seq_len = len(seq)
    num_mask = int(seq_len * mask)
    positions_to_mask = random.sample(range(seq_len), num_mask)

    # Masks positions on sequence
    seq_list = list(seq)
    for pos in positions_to_mask:
        seq_list[pos] = '?'

    masked_sequence = ''.join(seq_list)
    return masked_sequence