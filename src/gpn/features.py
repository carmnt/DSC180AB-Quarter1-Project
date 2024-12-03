import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Embedding
import torch.nn.functional as F
import random

from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, normalizers
from tokenizers.trainers import BpeTrainer

def tokenizer(seq):
    """
    Tokenizes nucleotides in a single DNA sequence.
    """
    mapping = {'a': 1, 'c': 2, 'g': 3, 't': 4}

    # Convert DNA strand to a tensor
    tensor_representation = torch.tensor([mapping[base] for base in seq.lower()])
    return tensor_representation

def get_features_labels(data):
    """
    Applies tokenizer to entire dataset and one-hot encodes labels.
    Returns GPN model input(X) and output labels(y) as (X, y).
    """
    X = torch.stack((data.seq.apply(tokenizer)).tolist())
    y = F.one_hot(X, num_classes=5).float()
    return X, y

class MaskLayer(nn.Module):
    def __init__(self, mask_percent=0.15):
        super().__init__()
        self.mask_percent = mask_percent

    def forward(self, x, training=False):
        if training:
            random_mask = torch.rand(x.shape, device=x.device) > self.mask_percent
            random_mask = random_mask.to(x.dtype)
            mask_output = x * random_mask
            return mask_output
        return x

class TransposeLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        return x

class ConvolutionLayer(nn.Module):
    def __init__(self, dilation, hidden_size=512):
        super().__init__()
        self.conv = nn.Sequential(
            TransposeLayer(),
            nn.Conv1d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                padding="same",
                kernel_size=9,
                dilation=dilation # DILATION
            ),
            TransposeLayer(),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, x):
        x = x + self.conv(x)
        x = x + self.ffn(x)
        return x

class ConvolutionalBlocks(nn.Module):
    def __init__(self, num_layers=1, hidden_size=512):
        super().__init__()
        dilations = get_dilation_schedule()
        self.layers = nn.ModuleList(
            [ConvolutionLayer(dilation=dilations[i]) for i in range(num_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def get_dilation_schedule(max=32, base=2, cycle=6):
    return [min(max, base ** (i % cycle)) for i in range(25)]