import torch
import torch.nn.functional as F

from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, normalizers
from tokenizers.trainers import BpeTrainer

def tokenizer(seq):
    """
    Tokenizes nucleotides in a single DNA sequence.
    """
    dataset = ["acgt"]
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.Lowercase()
    trainer = BpeTrainer(vocab_size=1)

    tokenizer.train_from_iterator(dataset, trainer=trainer, length=len(dataset))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    return tokenizer.encode(seq)

def get_features_labels(data):
    """
    Applies tokenizer to entire dataset and one-hot encodes labels.
    Returns GPN model input(X) and output labels(y) as (X, y).
    """
    X = torch.stack((data.seq.apply(tokenizer).apply(torch.tensor) + 1).tolist())
    y = F.one_hot(X, num_classes=5)
    return X, y