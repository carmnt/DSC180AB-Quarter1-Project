# GPN

## Important Files

**run.py-** main script to execute GPN pipeline. Integrates model, feature extraction, and data handling for complete workflow.

**requirements.txt-** text file to install all required dependencies.

### notebook (folder)

**visualization.ipynb -** contains a Jupyter notebook to specifically run example visualization code.

### src

**data_processing.py -** contains code to process raw data into cleaned data table with gene annotation and sequences.

**etl.py -** - contains code to download the raw and input folder data files from Github and websites.

**train.py-** contains a function to run a training/validation loop.

#### gpn (folder)

**data.py -** contains helper functions for data processing.

**model.py -** contains Genomic Pre-trained Network implementation.

**features.py-** contains feature extraction utilities for DNA sequence data.

**training.py-** contains all necessary components to run a training (and validation) loop.

## Running Code

**Note:** We recommend having a decent GPU with a good amount of VRAM (at least 11 GB) for the training portion as the training runs significantly slower when running with CPU + RAM.

Make sure to have all required dependencies installed before running. After cloning repo, type "pip install -r requirements.txt" into your terminal. Then, input “python run.py” or “python run.py > [file_name].txt” (replace ‘file_name’ with desired name) if you’d like a text log file.
