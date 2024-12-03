import pandas as pd

import time
import torch
import torch.nn as nn

from datasets import load_dataset

from torch.utils.data import Dataset, DataLoader
from .features import gpn_get_features_labels
from .model import GPN

BATCH_SIZE = 512

class DNADataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        features, labels = gpn_get_features_labels(pd.DataFrame([row]))
        return features.squeeze(0), labels.squeeze(0)


def run_one_epoch():
    train_data = load_dataset("songlab/genomes-brassicales-balanced-v1", split="train")
    valid_data = load_dataset("songlab/genomes-brassicales-balanced-v1", split="validation")

    # Convert data to dataframe
    all_train = pd.DataFrame(train_data)
    all_valid = pd.DataFrame(valid_data)

    # Find most common chromosome
    most_common_chrom = all_train.chrom.value_counts(normalize=True).sort_values(ascending=False).index[0]

    train = all_train[all_train.chrom == most_common_chrom].reset_index(drop=True)[:500]
    valid = all_valid[all_valid.chrom == most_common_chrom].reset_index(drop=True)[:500]

    combined_df = pd.concat([train, valid], ignore_index=True)
    train = combined_df.iloc[:int(0.8 * len(combined_df))]
    valid = combined_df.iloc[int(0.8 * len(combined_df)):]

    train_dataset = DNADataset(train)
    valid_dataset = DNADataset(valid)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    
    #train.to_csv('train.csv', index=False)
    #valid.to_csv('valid.csv', index=False)
    print("Training and validation ready")
    start_time = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = GPN()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(GPN())
    model = model.to(device)

    epochs = 1

    best_vloss = float('inf')

    model = GPN().to(device)

    # setup loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # create training and valid loop
    for epoch in range(epochs):
        print(f'Epoch: {epoch+1}\n-------')

        ### training
        model.train()
        running_train_loss, running_train_acc = 0, 0
        seq_processed = 0

        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)

            # forward pass
            y_pred = model(X)

            # reshape to fit cross entropy format
            y_pred = y_pred.reshape((-1, 5))
            y = y.reshape((-1, 5))
            seq_processed += y_pred.shape[0]

            # calculate loss
            loss = loss_fn(y_pred, y)
            running_train_loss += loss
            running_train_acc += (torch.argmax(y_pred, dim=-1) == torch.argmax(y, dim=-1)).float().sum()

            # optimizer zero grad
            optimizer.zero_grad()

            # loss backward
            loss.backward()

            # optimizer step
            optimizer.step()

            # print
            if batch % 100 == 99:
                print(f'Looked at {(batch + 1)}/{len(train_dataloader)} batches | Train loss: {running_train_loss/(batch+1)} | Train accuracy: {running_train_acc/seq_processed}')

        train_loss = running_train_loss/len(train_dataloader)
        train_acc = running_train_acc/seq_processed
        print(f'Train loss: {train_loss}')
        print(f'Train accuracy: {train_acc}')

        model.eval()
        running_vloss, running_vacc = 0, 0
        seq_processed_valid = 0

        with torch.no_grad():
            for batch, (X_valid, y_valid) in enumerate(valid_dataloader):
                X_valid, y_valid = X_valid.to(device), y_valid.to(device)

                y_valid_pred = model(X_valid)

                y_valid_pred = y_valid_pred.reshape((-1, 5))
                y_valid = y_valid.reshape((-1, 5))
                seq_processed_valid += y_valid_pred.shape[0]

                vloss = loss_fn(y_valid_pred, y_valid)
                running_vloss += vloss
                running_vacc += (torch.argmax(y_valid_pred, dim=-1) == torch.argmax(y_valid, dim=-1)).float().sum()
    
        valid_loss = running_vloss/len(valid_dataloader)
        valid_acc = running_vacc/seq_processed_valid
        print(f'Valid loss: {valid_loss}')
        print(f'Valid accuracy: {valid_acc}')

        if valid_loss < best_vloss:
            best_vloss = valid_loss
            torch.save(model.state_dict(), 'output/best_model.pth')
            print(f"saved best model with loss: {best_vloss}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"GPN training time elapsed: {elapsed_time:.2f} seconds")