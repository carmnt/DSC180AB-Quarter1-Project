import pandas as pd
import numpy as np
from .gpn.data import Genome, DNADataset
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
import torch
from .gpn.model import GPN
import tqdm
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_predict, LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def get_pred_regions():
    BATCH_SIZE = 256
    
    windows = pd.read_parquet("output/embedding/windows.parquet")

    data = windows.copy()
    data = data[['chrom', 'strand', 'center_start','center_end','Region']]
    genome = Genome('output/genome.fa.gz')

    data['seq'] = [genome.get_seq(row.chrom, row.center_start, row.center_end, row.strand) for row in data.itertuples()]

    X = data['seq']
    y = data['Region']

    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

    X = data['seq']
    y = data['Region']
    skf.get_n_splits(X, y)

    kfolds = skf.split(X, y)
    train, test = next(kfolds)

    x_train, y_train = X.iloc[train], y.iloc[train]
    x_test, y_test = X.iloc[test], y.iloc[test]

    torch.manual_seed(42)

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

    model.load_state_dict(torch.load('output/best_model.pth', weights_only=True))

    model.to(device)

    embeddings_list = []
    for x, _ in tqdm(dataloader):
        x = x.to(device)
        embeddings_list.append(model.get_embeddings(x).detach().cpu().numpy())

    all_embed = np.concatenate(embeddings_list, axis=0)

    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    X = all_embed
    y = data[['chrom', 'Region']]

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
