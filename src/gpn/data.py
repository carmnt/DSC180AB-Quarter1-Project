#Helper functions for data loading
from Bio import SeqIO, bgzf
from Bio.Seq import Seq
import gzip
from joblib import Parallel, delayed
import multiprocessing as mp
from tqdm.auto import tqdm
import bioframe as bf
import more_itertools
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import torch
import torch.nn as nn
from torch.nn import Embedding
import torch.nn.functional as F
import random
from torch.utils.data import Dataset, DataLoader
from .features import get_features_labels


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
    with gzip.open(path, "wt") if path.endswith(".gz") else open(path, "w") as handle:
        for chrom, seq in genome.items():
            handle.write(f">{chrom}\n{seq}\n")


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

def get_transcript_introns(df_transcript):
            df_transcript = df_transcript.sort_values("start")
            exon_pairs = more_itertools.pairwise(df_transcript.loc[:, ["start", "end"]].values)
            introns = [[e1[1], e2[0]] for e1, e2 in exon_pairs]
            introns = pd.DataFrame(introns, columns=["start", "end"])
            introns["chrom"] = df_transcript.chrom.iloc[0]
            return introns

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
    
class DNADataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        features, labels = get_features_labels(pd.DataFrame([row]))
        return features.squeeze(0), labels.squeeze(0)