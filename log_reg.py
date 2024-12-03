import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns

import torch
import torch.nn as nn
from torch.nn import Embedding
import torch.nn.functional as F
import random

from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, normalizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
import gffpandas.gffpandas as gffpd

from Bio import SeqIO, bgzf
from Bio.Seq import Seq
import gzip
from joblib import Parallel, delayed
import multiprocessing as mp
from tqdm.auto import tqdm
import bioframe as bf
import more_itertools

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import cross_val_predict, LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

tqdm.pandas()

DEFINED_SYMBOLS = np.frombuffer("ACGTacgt".encode("ascii"), dtype="S1")
UNMASKED_SYMBOLS = np.frombuffer("ACGT".encode("ascii"), dtype="S1")

def load_fasta(path, subset_chroms=None):
    with gzip.open(path, "rt") if path.endswith(".gz") else open(path) as handle:
        genome = pd.Series(
            {
                rec.id: str(rec.seq)
                for rec in SeqIO.parse(handle, "fasta")
                if subset_chroms is None or rec.id in subset_chroms
            }
        )
    return genome


def save_fasta(path, genome):
    with bgzf.BgzfWriter(path, "wb") if path.endswith(".gz") else open(
        path, "w"
    ) as handle:
        SeqIO.write(genome.values(), handle, "fasta")


# Some standard formats
def load_table(path):
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif "csv" in path:
        df = pd.read_csv(path)
    elif "tsv" in path:
        df = pd.read_csv(path, sep="\t")
    elif "vcf" in path:
        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            comment="#",
            usecols=[0, 1, 3, 4],
            dtype={0: str},
        ).rename(columns={0: "chrom", 1: "pos", 3: "ref", 4: "alt"})
    elif "gtf" in path or "gff" in path:
        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            comment="#",
            dtype={"chrom": str},
            names=[
                "chrom",
                "source",
                "feature",
                "start",
                "end",
                "score",
                "strand",
                "frame",
                "attribute",
            ],
        )
        df.start -= 1
    df.chrom = df.chrom.astype(str)
    return df

class Genome:
    def __init__(self, path, subset_chroms=None):
        self._genome = load_fasta(path, subset_chroms=subset_chroms)

    def get_seq(self, chrom, start, end, strand="+"):
        seq = self._genome[chrom][start:end]
        if strand == "-":
            seq = str(Seq(seq).reverse_complement())
        return seq

    def get_nuc(self, chrom, pos, strand="+"):
        # pos is assumed to be 1-based as in VCF
        seq = self._genome[chrom][pos - 1]
        if strand == "-":
            seq = str(Seq(seq).reverse_complement())
        return seq

    def filter_chroms(self, chroms):
        self._genome = self._genome[chroms]

    def get_seq_fwd_rev(self, chrom, start, end):
        seq_fwd = self.get_seq(chrom, start, end)
        seq_rev = str(Seq(seq_fwd).reverse_complement())
        return seq_fwd, seq_rev

    def get_all_intervals(self):
        return pd.DataFrame(
            [
                {"chrom": chrom, "start": 0, "end": len(seq)}
                for chrom, seq in self._genome.items()
            ]
        )

    def get_intervals_matching_symbols(self, symbols):
        def get_intervals_matching_symbols_chrom(chrom):
            complete_interval = pd.DataFrame(
                {"chrom": [chrom.name], "start": [0], "end": [len(chrom.seq)]}
            )
            intervals = pd.DataFrame(
                dict(
                    start=np.where(
                        ~np.isin(
                            np.frombuffer(chrom.seq.encode("ascii"), dtype="S1"),
                            symbols,
                        )
                    )[0]
                )
            )
            if len(intervals) > 0:
                intervals["chrom"] = chrom.name
                intervals["end"] = intervals.start + 1
                intervals = bf.merge(intervals).drop(columns="n_intervals")
                return bf.subtract(complete_interval, intervals)
            return complete_interval

        return pd.concat(
            self._genome.rename("seq")
            .to_frame()
            .progress_apply(
                get_intervals_matching_symbols_chrom,
                axis=1,
            )
            .values,
            ignore_index=True,
        )

    def get_defined_intervals(self):
        return self.get_intervals_matching_symbols(DEFINED_SYMBOLS)

    def get_unmasked_intervals(self):
        return self.get_intervals_matching_symbols(UNMASKED_SYMBOLS)

def get_interval_windows(interval, window_size, step_size, add_rc):
    windows = pd.DataFrame(
        dict(start=np.arange(interval.start, interval.end - window_size + 1, step_size))
    )
    windows["end"] = windows.start + window_size
    windows["chrom"] = interval.chrom
    windows = windows[["chrom", "start", "end"]]  # just re-ordering
    windows["strand"] = "+"
    if add_rc:
        windows_neg = windows.copy()  # TODO: this should be optional
        windows_neg.strand = "-"
        return pd.concat([windows, windows_neg], ignore_index=True)
    return windows

def get_seq(intervals, genome):
    intervals["seq"] = intervals.progress_apply(
        lambda i: genome.get_seq(i.chrom, i.start, i.end, i.strand),
        axis=1,
    )
    return intervals

############################### LOADING DATA ###########################################

gtf = load_table('raw/annotations.gtf.gz')
repeats = pd.read_csv('input/repeats.bed.gz', sep="\t").rename(columns=dict(genoName="chrom", genoStart="start", genoEnd="end"))[["chrom", "start", "end"]]

repeats.chrom = repeats.chrom.str.replace("Chr", "")
repeats = bf.merge(repeats).drop(columns="n_intervals")
repeats["feature"] = "Repeat"
gtf = pd.concat([gtf, repeats], ignore_index=True)

gtf_intergenic = bf.subtract(gtf.query('feature=="chromosome"'), gtf[gtf.feature.isin(["gene", "ncRNA_gene", "Repeat"])])
gtf_intergenic.feature = "intergenic"
gtf = pd.concat([gtf, gtf_intergenic], ignore_index=True)

gtf_exon = gtf[gtf.feature=="exon"]
gtf_exon["transcript_id"] = gtf_exon.attribute.str.split(";").str[0].str.split(":").str[-1]

def get_transcript_introns(df_transcript):
            df_transcript = df_transcript.sort_values("start")
            exon_pairs = more_itertools.pairwise(df_transcript.loc[:, ["start", "end"]].values)
            introns = [[e1[1], e2[0]] for e1, e2 in exon_pairs]
            introns = pd.DataFrame(introns, columns=["start", "end"])
            introns["chrom"] = df_transcript.chrom.iloc[0]
            return introns

gtf_introns = gtf_exon.groupby("transcript_id").apply(get_transcript_introns).reset_index().drop_duplicates(subset=["chrom", "start", "end"])
gtf_introns["feature"] = "intron"
gtf = pd.concat([gtf, gtf_introns], ignore_index=True)
gtf.to_parquet('output/annotation.expanded.parquet', index=False)

annotation = pd.read_parquet('output/annotation.expanded.parquet')
annotation[['start', 'end', 'feature', 'chrom']]

features_of_interest = [
    "intergenic",
    'CDS',
    'intron',
    'three_prime_UTR',
    'five_prime_UTR',
    "ncRNA_gene",
    "Repeat",
]

gene_data = Genome('output/genome.fa.gz')
gene_data.get_seq(1, 1006, 1106)

WINDOW_SIZE = 512
STEP_SIZE = 256
PRIORITY_ASSEMBLIES = [
    "GCF_000001735.4",  # Arabidopsis thaliana
    "GCF_000309985.2",  # Brassica rapa
]
splits = ["train", "validation", "test"]
EMBEDDING_WINDOW_SIZE = 100
CHROMS = ["1", "2", "3", "4", "5"]
NUCLEOTIDES = list("ACGT")

DEFINED_SYMBOLS = np.frombuffer("ACGTacgt".encode("ascii"), dtype="S1") #Genome Function
UNMASKED_SYMBOLS = np.frombuffer("ACGT".encode("ascii"), dtype="S1") #Genome Function

def filter_length(intervals, min_interval_len):
    return intervals[intervals.end - intervals.start >= min_interval_len]

def make_windows(intervals, window_size, step_size, add_rc=False):
    return pd.concat(
        intervals.progress_apply(
            lambda interval: get_interval_windows(
                interval, window_size, step_size, add_rc
            ),
            axis=1,
        ).values,
        ignore_index=True,
    )

#Defining Genome Class
class Genome:
    def __init__(self, path, subset_chroms=None):
        self._genome = load_fasta(path, subset_chroms=subset_chroms)

    def get_seq(self, chrom, start, end, strand="+"):
        seq = self._genome[chrom][start:end]
        if strand == "-":
            seq = str(Seq(seq).reverse_complement())
        return seq

    def get_nuc(self, chrom, pos, strand="+"):
        # pos is assumed to be 1-based as in VCF
        seq = self._genome[chrom][pos - 1]
        if strand == "-":
            seq = str(Seq(seq).reverse_complement())
        return seq

    def filter_chroms(self, chroms):
        self._genome = self._genome[chroms]



    def get_seq_fwd_rev(self, chrom, start, end):
        seq_fwd = self.get_seq(chrom, start, end)
        seq_rev = str(Seq(seq_fwd).reverse_complement())
        return seq_fwd, seq_rev

    def get_all_intervals(self):
        return pd.DataFrame(
            [
                {"chrom": chrom, "start": 0, "end": len(seq)}
                for chrom, seq in self._genome.items()
            ]
        )

    def get_intervals_matching_symbols(self, symbols):
        def get_intervals_matching_symbols_chrom(chrom):
            complete_interval = pd.DataFrame(
                {"chrom": [chrom.name], "start": [0], "end": [len(chrom.seq)]}
            )
            intervals = pd.DataFrame(
                dict(
                    start=np.where(
                        ~np.isin(
                            np.frombuffer(chrom.seq.encode("ascii"), dtype="S1"),
                            symbols,
                        )
                    )[0]
                )
            )
            if len(intervals) > 0:
                intervals["chrom"] = chrom.name
                intervals["end"] = intervals.start + 1
                intervals = bf.merge(intervals).drop(columns="n_intervals")
                return bf.subtract(complete_interval, intervals)
            return complete_interval

        return pd.concat(
            self._genome.rename("seq")
            .to_frame()
            .progress_apply(
                get_intervals_matching_symbols_chrom,
                axis=1,
            )
            .values,
            ignore_index=True,
        )

    def get_defined_intervals(self):
        return self.get_intervals_matching_symbols(DEFINED_SYMBOLS)

    def get_unmasked_intervals(self):
        return self.get_intervals_matching_symbols(UNMASKED_SYMBOLS)

genome = load_fasta('raw/genome.raw.fa.gz')
id_mapping = pd.read_csv('input/id_mapping.tsv', sep="\t", header=None, index_col=0).squeeze('columns')

updated_genome = pd.Series(
    {id_mapping.get(chrom, chrom): seq for chrom, seq in genome.items()}
)

def save_fasta(path, genome):
    with gzip.open(path, "wt") if path.endswith(".gz") else open(path, "w") as handle:
        for chrom, seq in genome.items():
            handle.write(f">{chrom}\n{seq}\n")

save_fasta('output/genome.fa.gz', updated_genome)

genome = Genome('output/genome.fa.gz')

gtf = pd.read_parquet('output/annotation.expanded.parquet')
genome = Genome('output/genome.fa.gz')
genome.filter_chroms(["1", "2", "3", "4", "5"])
defined_intervals = genome.get_defined_intervals()
defined_intervals = filter_length(defined_intervals, WINDOW_SIZE)
windows = make_windows(defined_intervals, WINDOW_SIZE, EMBEDDING_WINDOW_SIZE)
windows.rename(columns={"start": "full_start", "end": "full_end"}, inplace=True)

windows["start"] = (windows.full_start+windows.full_end)//2 - EMBEDDING_WINDOW_SIZE//2
windows["end"] = windows.start + EMBEDDING_WINDOW_SIZE

features_of_interest = [
    "intergenic",
    'CDS',
    'intron',
    'three_prime_UTR',
    'five_prime_UTR',
    "ncRNA_gene",
    "Repeat",
]

for f in features_of_interest:
    print(f)
    windows = bf.coverage(windows, gtf[gtf.feature==f])
    windows.rename(columns=dict(coverage=f), inplace=True)
        
windows = windows[(windows[features_of_interest]==EMBEDDING_WINDOW_SIZE).sum(axis=1)==1]
windows["Region"] = windows[features_of_interest].idxmax(axis=1)
windows.drop(columns=features_of_interest, inplace=True)

windows.rename(columns={"start": "center_start", "end": "center_end"}, inplace=True)
windows.rename(columns={"full_start": "start", "full_end": "end"}, inplace=True)
print(windows)
windows.to_parquet('output/embedding/windows.parquet', index=False)

windows = pd.read_parquet("output/embedding/windows.parquet")

data = windows.copy()
data = data[['chrom', 'strand', 'center_start','center_end','Region']]
genome = Genome('output/genome.fa.gz')

data['seq'] = [genome.get_seq(row.chrom, row.center_start, row.center_end, row.strand) for row in data.itertuples()]

skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
X = data['seq']
y = data['Region']
skf.get_n_splits(X, y)

kfolds = skf.split(X, y)
train, test = next(kfolds)

x_train, y_train = X.iloc[train], y.iloc[train]
x_test, y_test = X.iloc[test], y.iloc[test]

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

class GPN(nn.Module):
    def __init__(self, num_layers=1):
        super().__init__()
        self.mask = MaskLayer()
        self.embed = Embedding(5, 512)
        self.conv = ConvolutionalBlocks(num_layers=num_layers)
        self.nuc_logits = nn.Linear(512, 5)

    def forward(self, input, training=False):
        """
        input: (B, 512) a tensor with values in range [1,4]

        returns: output - (B, 512, 5) a tensor of prob.
        """
        x = input
        x = self.mask(x, training=True)
        x = self.embed(x)
        x = self.conv(x)
        output = self.nuc_logits(x)
        return output

    def get_embeddings(self, input):
        """
        input: (B, 100)

        returns: (B, 512)
        """
        x = input
        x = self.embed(x)
        x = self.conv(x)
        output = x.mean(axis=1) # if not batches, axis=0
        return output

from torch.utils.data import Dataset, DataLoader
torch.manual_seed(42)

class DNADataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        features, labels = get_features_labels(pd.DataFrame([row]))
        return features.squeeze(0), labels.squeeze(0)

torch.manual_seed(42)

BATCH_SIZE = 256

embedding = data[['seq']]

dataset = DNADataset(embedding)

dataloader = DataLoader(
    dataset=dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

features_batch, _ = next(iter(dataloader))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

model = GPN()
model.load_state_dict(torch.load('best_model_v3.pth', weights_only=True))

model.to(device)
embeddings_list = []
for x, _ in tqdm(dataloader):
    x = x.to(device)
    embeddings_list.append(model.get_embeddings(x).detach().cpu().numpy())

all_embed = np.concatenate(embeddings_list, axis=0)

skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
X = all_embed
y = data[['chrom', 'Region']]

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import LeaveOneGroupOut

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("linear", LogisticRegressionCV(
        random_state=42, verbose=True, max_iter=1000,
        class_weight="balanced", n_jobs=-1
        )
    ),
])

preds = cross_val_predict(
    clf, X, y['Region'], groups=y.chrom,
    cv=LeaveOneGroupOut(), verbose=True,
)
pd.DataFrame({"pred_Region": preds}).to_parquet('output/classification/model.parquet', index=False)

