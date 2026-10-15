# Dataset provenance

The model was evaluated on 10 public benchmarks (8 grid + 2 graph). This
repository does not redistribute the data files; obtain them from the original
providers or the benchmark releases listed below, then convert them into the
layout described in `data/README.md`.

## Grid datasets

| Name | City | Source / original release | Notes |
|------|------|---------------------------|-------|
| TaxiBJ | Beijing | DeepST / ST-ResNet releases (Zhang et al.) | taxi inflow/outflow, 32x32 grid, 30 min, veh/30 min |
| TaxiCQ | Chongqing | ACFM release (Liu et al., AAAI 2018) | taxi inflow/outflow, 25x20 grid, 30 min |
| BikeNYC | New York | ST-ResNet release (NYC14_M16x8_T60_NewEnd.h5) | bike inflow/outflow, 16x8 grid, 60 min |
| BikeNYC2 | New York | released with deep flow-prediction baselines (e.g., DeepSTN+) | bike inflow/outflow, 16x8 grid, 30 min |
| TaxiNYC2 | New York | released with deep flow-prediction baselines | taxi inflow/outflow, 16x12 grid, 30 min |
| TDrive | Beijing | Microsoft T-Drive trajectory dataset (processed into grid flows) | 32x32 grid, 60 min |
| TrafficTJ | Tianjin | MC-STL release (Zhang et al., CIKM 2023) | region speed, 24x30 grid, 5 min, km/h |
| TrafficZZ | Zhengzhou | MC-STL release (Zhang et al., CIKM 2023) | region speed, 26x26 grid, 5 min, km/h |

## Graph datasets

| Name | Area | Source | Notes |
|------|------|--------|-------|
| PEMS03 | California | California Performance Measurement System (PeMS), benchmark release by ASTGCN (Guo et al., AAAI 2019) | 358 sensors, 5 min, flow in veh/5 min |
| PEMS04 | California | PeMS, benchmark release by ASTGCN | 307 sensors, 5 min, flow/occupancy/speed; flow is the target |

## Redistribution notes

- PeMS data are collected by the California DOT and are publicly available;
  the npz benchmark files above are widely redistributed by public benchmark
  suites (e.g., LibCity, BasicTS).
- TDrive is provided by Microsoft Research; download it from the official page
  and apply your own grid aggregation.
- The sliding-window npy conversions used by this repository are deterministic
  preprocessing steps; keep value_maxs / value_mins together with the windows
  so that de-normalization is reproducible.
