#!/usr/bin/env python

import sys
import json
import pandas as pd
import os

from src.figures import create_figures

if __name__ == "__main__":
    try:
        create_figures()
    except Exception as e:
        print(f"An error occurred during the process: {e}")