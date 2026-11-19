# STHEN: Spatio-temporal Decoupled Heterogeneous Expert Network

Official implementation of **"Spatio-temporal Decoupled Heterogeneous Expert Network for Traffic Flow Prediction"** (under review, IEEE T-ITS).

STHEN is a data-driven macroscopic traffic prediction model that handles both **grid** (region-level inflow/outflow) and **graph** (sensor network) data with a single architecture:

- **Spatio-temporal Decoupled Encoder** with a non-overlapping patching strategy: linear-complexity attention and a unified interface for cross-modal inputs.
- **Multi-head Dense Heterogeneous Attention (MDHA)**: every head keeps the full feature dimension, building dense heterogeneous subspaces.
- **Alternating Mixed Heterogeneous Feed-Forward Network (AMHFN)**: alternates feature-wise and token-wise interactions with residual gating.
- **Heterogeneous Expert Network (HEN)**: clustering-based soft routing assigns each spatial unit to specialized experts.

> Note: the model classes in the code are named `HASTFormer` for historical reasons; STHEN refers to the same model.

This repository contains the **model implementation only**. Training, data-loading and evaluation code are not included.

## Repository layout

```
STHEN/
├── Models/Our/          # STHEN and ablation variants
│   ├── model.py              # STHEN
│   ├── model_Att.py          # ablation: attention variant
│   ├── model_FFN.py          # ablation: feed-forward variant
│   ├── model_Expert.py       # ablation: expert-routing variant
│   ├── model_S.py            # ablation: spatial-only variant
│   └── model_T.py            # ablation: temporal-only variant
├── model_configs.py     # per-dataset architecture configurations
├── example.py           # instantiate the model and run a dummy forward pass
├── data/                # benchmark data shipped with this repository
└── docs/DATASETS.md     # dataset provenance
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick start

```bash
python example.py
```

`example.py` builds STHEN from `model_configs.py` for a grid dataset (BikeNYC) and a graph dataset (PEMS03), prints the parameter count, and runs a forward pass on random tensors.

## Data

See `data/README.md` for the benchmark files shipped with this repository and `docs/DATASETS.md` for their provenance. Data files are redistributed as originally released; no preprocessing code is included.

## Citation

```bibtex
@article{yin2026sthen,
  author  = {Yin, Jie and Li, Donghuan and Lu, Keke},
  title   = {Spatio-temporal Decoupled Heterogeneous Expert Network for Traffic Flow Prediction},
  journal = {IEEE Transactions on Intelligent Transportation Systems},
  year    = {2026},
  note    = {Under review}
}
```

## License

This project is released under the [MIT License](LICENSE). Third-party datasets remain under the terms of their original providers (see `docs/DATASETS.md`).