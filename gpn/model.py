import torch
import torch.nn as nn
from torch.nn import Embedding

class GPN(nn.Module):
    def __init__(self, num_layers=25):
        super().__init__()
        self.mask = MaskLayer()
        self.embed = Embedding(5, 512)
        self.conv = ConvolutionalBlocks(num_layers=num_layers)
        self.nuc_logits = nn.Linear(512, 5)
        self.softmax = nn.Softmax()

    def forward(self, input, training=False):
        x = input
        x = self.mask(x, training=training)
        x = self.embed(x)
        x = self.conv(x)
        x = self.nuc_logits(x)
        output = self.softmax(x)
        return output
    
class MaskLayer(nn.Module):
    def __init__(self, mask_percent=0.15):
        super().__init__()
        self.mask_percent = mask_percent

    def forward(self, x, training=False):
        if training:
            random_mask = torch.rand(x.shape) > self.mask_percent
            random_mask = random_mask.astype(x.dtype)
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
                dilation=dilation
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
    def __init__(self, num_layers=25):
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