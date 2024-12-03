#!/usr/bin/env python

import sys
import json
import pandas as pd
import os

from src.etl import run_etl
from src.data_processing import process_data

if __name__ == "__main__":
    try:
        run_etl()
        process_data()
    except Exception as e:
        print(f"An error occurred during the ETL process: {e}")