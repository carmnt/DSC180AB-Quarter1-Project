#!/usr/bin/env python

import sys
import json
import pandas as pd
import os

from src.etl import run_etl
from src.data_processing import process_data
from src.train import run_train
from src.logistic_regression import get_pred_regions
from src.figures import create_figures

if __name__ == "__main__":
    try:
        run_etl()
        process_data()
        run_train()
        get_pred_regions()
        create_figures()
    except Exception as e:
        print(f"An error occurred during the process: {e}")