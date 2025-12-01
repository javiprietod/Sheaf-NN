# Sheaf-NN
A repository benchmarking the State-of-the-Art (SOTA) in Sheaf Neural Networks against the SOTA in Graph Neural Networks.

## Overview

Sheaf Neural Networks represent an emerging paradigm in geometric deep learning that leverages the mathematical framework of cellular sheaves to model complex relational data on graphs. This repository provides a benchmarking framework to compare the performance of Sheaf Neural Networks against traditional Graph Neural Networks (GNNs).

## Features

- Benchmarking framework for comparing Sheaf Neural Networks and Graph Neural Networks
- Integration with Open Graph Benchmark (OGB) datasets
- Built on PyTorch and PyTorch Geometric for efficient graph-based computations

## Requirements

- Python >= 3.10
- PyTorch >= 2.5.1
- PyTorch Geometric >= 2.0.2
- OGB (Open Graph Benchmark) 1.3.6
- NumPy >= 1.16.0
- pandas >= 0.24.0
- scikit-learn >= 0.20.0

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/javiprietod/Sheaf-NN.git
   cd Sheaf-NN
   ```

2. Create a virtual environment and install dependencies using `uv`:
   ```bash
   uv venv
   uv sync
   ```

## Usage

The jupyter notebook in which we showcase the benchmarking is `comparative.ipynb`. 