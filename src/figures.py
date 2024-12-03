import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

def create_figures():
    windows = pd.read_parquet('output/embedding/windows.parquet')
    classification_model = pd.read_parquet('output/classification/model.parquet')
    windows['pred_Region'] = classification_model.pred_Region.values
    region_renaming = {
        "intergenic": "Intergenic",
        "intron": "Intron",
        "ncRNA_gene": "ncRNA",
        "five_prime_UTR": "5' UTR",
        "three_prime_UTR": "3' UTR",
    }
    windows.Region = windows.Region.replace(region_renaming)
    windows.pred_Region = windows.pred_Region.replace(region_renaming)
    windows

    value_counts = windows['Region'].value_counts()
    plt.figure(figsize=(8, 6))
    value_counts.plot(kind='bar', color='skyblue')

    plt.title('Distribution of Genomic Regions')
    plt.xlabel('Region')
    plt.ylabel('Frequency')

    plt.xticks(rotation=45);

    plt.savefig(f"output/figures/bar_plot.png", bbox_inches='tight')
    print('created output/figures/bar_plot.png')

    regions = windows.Region.value_counts().index.values

    # Make sure Repeat goes last
    if "Repeat" in regions:
        regions = regions[regions!="Repeat"].tolist() + ["Repeat"]
    regions

    ConfusionMatrixDisplay.from_predictions(
        windows.Region, windows.pred_Region, normalize='true', labels=regions,
        values_format=".0%", im_kw=dict(vmin=0, vmax=1),
    )
    plt.title('Genomic Region Prediction Confusion Matrix')
    plt.xticks(rotation=45);
    plt.savefig(f"output/figures/conf_matrix.svg", bbox_inches="tight")
    print('created output/figures/conf_matrix.svg')
    plt.savefig(f"output/figures/conf_matrix.png", bbox_inches='tight')
    print('created output/figures/conf_matrix.png')


