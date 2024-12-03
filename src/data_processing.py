import numpy as np
import pandas as pd
import bioframe as bf
from sklearn.model_selection import StratifiedKFold
from .gpn.data import load_table, load_fasta, get_transcript_introns, Genome, save_fasta, filter_length, make_windows
import os

def ensure_directory(path):
    """Ensure that the destination directory exists."""
    os.makedirs(path, exist_ok=True)
    
def process_data():
    ensure_directory('output')
    ensure_directory('output/embedding')
    ensure_directory('output/classification')
    ensure_directory('output/figures')
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

    gtf_introns = gtf_exon.groupby("transcript_id").apply(get_transcript_introns).reset_index().drop_duplicates(subset=["chrom", "start", "end"])
    gtf_introns["feature"] = "intron"
    gtf = pd.concat([gtf, gtf_introns], ignore_index=True)
    gtf.to_parquet('output/annotation.expanded.parquet', index=False)
    print('Created output/annotation.expanded.parquet')
    annotation = pd.read_parquet('output/annotation.expanded.parquet')

    features_of_interest = [
        "intergenic",
        'CDS',
        'intron',
        'three_prime_UTR',
        'five_prime_UTR',
        "ncRNA_gene",
        "Repeat",
    ]

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

    genome = load_fasta('raw/genome.raw.fa.gz')
    id_mapping = pd.read_csv('input/id_mapping.tsv', sep="\t", header=None, index_col=0).squeeze('columns')

    updated_genome = pd.Series(
        {id_mapping.get(chrom, chrom): seq for chrom, seq in genome.items()}
    )

    save_fasta('output/genome.fa.gz', updated_genome)
    print('created output/genome.fa.gz')
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
    ensure_directory('output/embedding')
    windows.to_parquet('output/embedding/windows.parquet', index=False)
    print('Created output/embedding/windows.parquet')
    print('Data processed successfully')

# windows = pd.read_parquet("output/embedding/windows.parquet")

# data = windows.copy()
# data = data[['chrom', 'strand', 'center_start','center_end','Region']]
# genome = Genome('output/genome.fa.gz')

# data['seq'] = [genome.get_seq(row.chrom, row.center_start, row.center_end, row.strand) for row in data.itertuples()]

# X = data['seq']
# y = data['Region']

# skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

# X = data['seq']
# y = data['Region']
# skf.get_n_splits(X, y)

# kfolds = skf.split(X, y)
# train, test = next(kfolds)

# x_train, y_train = X.iloc[train], y.iloc[train]
# x_test, y_test = X.iloc[test], y.iloc[test]

# from torch.utils.data import Dataset, DataLoader








