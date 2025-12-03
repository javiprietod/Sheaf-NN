# Sheaf-NN

This repository provides an implementation of Cooperative Sheaf Neural Networks (CSNNs) as introduced by Ribeiro et al. (2025). It includes baseline comparisons with a MixHop Graph Neural Network (MH-GNN) on the WebKB dataset and a visual study of separating capacity on a synthetic dataset.

<!-- ## Installation

To install the required dependencies, run:

```bash
uv sync
``` -->

<!-- https://docs.astral.sh/uv/getting-started/installation/#standalone-installer -->
<!-- Refer also to the installation of uv-->

## Installation

To install the required dependencies, use the `uv` package manager (see [uv documentation](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer) for installation instructions). Once `uv` is installed, run:

```bash
uv sync
```

## Usage

To run the main script for training and evaluating the CSNN model, execute:

```bash
python main.py
```

The repository includes Jupyter notebooks demonstrating the performance of CSNNs on the WebKB dataset and a synthetic dataset. You can find them in the `notebooks` directory:

- `notebooks/webkb`: Demonstrates CSNNs on the WebKB dataset.
- `notebooks/synthetic`: Visualizes the separating power of SNNs vs GNNs on a synthetic dataset.

## References

- Abu-El-Haija, S., et al. (2020). "MixHop: Higher-Order Graph Convolutional Architectures via Sparsified Neighborhood Mixing." [arXiv:1905.00067](https://arxiv.org/abs/1905.00067)
- Bodnar, C., et al. (2023). "Neural Sheaf Diffusion: A Topological Perspective on Heterophily and Oversmoothing in GNNs." [arXiv:2202.04579](https://arxiv.org/abs/2202.04579)
- Ribeiro, A., et al. (2025). "Cooperative Sheaf Neural Networks." [arXiv:2507.00647](https://arxiv.org/abs/2507.00647)
