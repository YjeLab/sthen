# STHEN: Spatio-temporal Decoupled Heterogeneous Expert Network

Official implementation of **"Spatio-temporal Decoupled Heterogeneous Expert Network for Traffic Flow Prediction"** (under review, IEEE T-ITS).

STHEN is a data-driven macroscopic traffic prediction model that handles both **grid** (region-level inflow/outflow) and **graph** (sensor network) data with a single architecture:

- **Spatio-temporal Decoupled Encoder** with a non-overlapping patching strategy: linear-complexity attention and a unified interface for cross-modal inputs.
- **Multi-head Dense Heterogeneous Attention (MDHA)**: every head keeps the full feature dimension, building dense heterogeneous subspaces.
- **Alternating Mixed Heterogeneous Feed-Forward Network (AMHFN)**: alternates feature-wise and token-wise interactions with residual gating.
- **Heterogeneous Expert Network (HEN)**: clustering-based soft routing assigns each spatial unit to specialized experts.

> Note: the model classes in the code are named `HASTFormer` for historical reasons; STHEN refers to the same model.

This repository contains the main pipeline only, from **dataset loading to model training**. Evaluation, visualization, and case-study scripts are not included.

## Repository layout

```
STHEN-release/
├── main.py               # entry point (Train / Test)
├── requirements.txt
├── Models/Our/           # STHEN and ablation variants
├── Train/                # training / validation loop
├── Utils/                # dataset configs, data loading and shared tools
├── data/                 # where datasets are placed (see data/README.md)
└── docs/                 # dataset provenance and layout
```

## Installation

```bash
conda create -n sthen python=3.10 -y
conda activate sthen
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Data preparation

Place each dataset under `Data/<Name>/` as described in `data/README.md`. Dataset sources and provenance are documented in `docs/DATASETS.md`.

## Training

```bash
# single-step grid prediction
python main.py --exp_type Train --Data_name TaxiBJ

# multi-step grid prediction
python main.py --exp_type Train --Data_name BikeNYC2

# graph prediction
python main.py --exp_type Train --Data_name PEMS04

# ablation variants
python main.py --exp_type Train --Data_name TaxiBJ --model_variants OurModel_Expert
```

Run `python main.py --help` for the full list of arguments.

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
