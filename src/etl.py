import os
import requests
import subprocess
from datasets import load_dataset
import pandas as pd


# Define constants
URLS = {
    "genome": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/735/GCF_000001735.4_TAIR10.1/GCF_000001735.4_TAIR10.1_genomic.fna.gz",
    "annotations": "https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-55/gff3/arabidopsis_thaliana/Arabidopsis_thaliana.TAIR10.55.gff3.gz"
}

FILE_PATHS = {
    "genome": "raw/genome.raw.fa.gz",
    "annotations": "raw/annotations.gtf.gz"
}

# GitHub Repository Details
GITHUB_API_BASE = "https://api.github.com/repos"
REPO_OWNER = "songlab-cal"
REPO_NAME = "gpn"
BRANCH = "5da5958cc261af349d804b92e3506ab8cadced46"  # Commit SHA from your permalink
DIRECTORY_PATH = "analysis/arabidopsis/input"  # Path to the directory to download
DESTINATION = "input"  # Local directory to save files

def ensure_directory(path):
    """Ensure that the destination directory exists."""
    os.makedirs(path, exist_ok=True)
    
def fetch_directory_contents(owner, repo, path, branch):
    """
    Fetch the contents of a directory from the GitHub API.
    Returns a list of files and their raw URLs.
    """
    url = f"{GITHUB_API_BASE}/{owner}/{repo}/contents/{path}?ref={branch}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise RuntimeError(f"Failed to fetch directory contents: {response.status_code}, {response.text}")

def download_file(url, save_path):
    """Download a file from a given URL and save it to a specified path."""
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        print(f"Saved to {save_path}")
    else:
        raise RuntimeError(f"Failed to download {url}. HTTP status code: {response.status_code}")
def download_github_directory(owner, repo, path, branch, destination):
    """Download a directory from GitHub."""
    ensure_directory(destination)
    contents = fetch_directory_contents(owner, repo, path, branch)
    for item in contents:
        if item["type"] == "file":
            file_url = item["download_url"]
            file_path = os.path.join(destination, item["name"])
            download_file(file_url, file_path)
        elif item["type"] == "dir":
            subdir_path = os.path.join(destination, item["name"])
            download_github_directory(owner, repo, item["path"], branch, subdir_path)

def run_etl():
    """Run the ETL process: Download and save all required files."""
    ensure_directory("raw")
    for key in URLS:
        download_file(URLS[key], FILE_PATHS[key])
    download_github_directory(REPO_OWNER, REPO_NAME, DIRECTORY_PATH, BRANCH, DESTINATION)
    print("ETL process completed successfully.")

