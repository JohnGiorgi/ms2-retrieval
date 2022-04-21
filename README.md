
# MS2 Retrieval

A set of preliminary retrieval experiments on the [MS2 dataset](https://arxiv.org/abs/2104.06486).

## Installation

This repository requires Python 3.8 or later.

Clone the repository and install from source using [Poetry](https://python-poetry.org/):

```bash
# Install poetry for your system: https://python-poetry.org/docs/#installation
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python

# Clone and move into the repo
git clone https://github.com/JohnGiorgi/ms2-retrieval
cd ms2-retrieval

# Install the package with poetry
poetry install
```

## Usage

Expects a filepath to a local copy of the MS2 dataset, e.g. `"path/to/ms2_data"`.

```bash
# Preprocess the data
ms2-retrieval create-examples "path/to/ms2_data" "out"

# Create the FAISS index
ms2-retrieval create-index "out/to_index.jsonl" "out"

# Search the index and score the results
ms2-retrieval search-and-score "out/examples.jsonl" "out/index"
```