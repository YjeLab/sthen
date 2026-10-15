# Data layout

The benchmark files distributed with this repository are stored under
`data/`. Copy (or symlink) the converted windows into `Data/<Name>/`, which is
where the training script reads from.

## Shipped with the repository

| Dataset | File(s) in this repo | Conversion |
|---------|----------------------|------------|
| PEMS03 | `data/graph/PEMS03/PEMS03.npz` | `python data/convert/convert_pems_npz_to_windows.py --name PEMS03 --npz data/graph/PEMS03/PEMS03.npz` |
| PEMS04 | `data/graph/PEMS04/PEMS04.npz` | `python data/convert/convert_pems_npz_to_windows.py --name PEMS04 --npz data/graph/PEMS04/PEMS04.npz` |
| BikeNYC | `data/grid/BikeNYC/NYC14_M16x8_T60_NewEnd.h5` (ST-ResNet release) or `data/grid/BikeNYC/BikeNYC_short.zip` | `python data/convert/build_bikenyc.py --h5 data/grid/BikeNYC/NYC14_M16x8_T60_NewEnd.h5` |
| BikeNYC2 | `data/grid/BikeNYC2/BikeNYC2_short.zip` | unzip, then `python data/convert/convert_json_to_npy.py --json BikeNYC2_short.json --out Data/BikeNYC2` |
| TaxiNYC2 | `data/grid/TaxiNYC2/TaxiNYC2_short.zip` | unzip, then `python data/convert/convert_json_to_npy.py --json TaxiNYC2_short.json --out Data/TaxiNYC2` |
| TrafficTJ | `data/grid/TrafficTJ/TrafficTJ_short.zip` | unzip, then `python data/convert/convert_json_to_npy.py --json TrafficTJ_short.json --out Data/TrafficTJ` |
| TrafficZZ | `data/grid/TrafficZZ/TrafficZZ_short.zip` (LZMA-compressed) | unzip, then `python data/convert/convert_json_to_npy.py --json TrafficZZ_short.json --out Data/TrafficZZ` |

## Not redistributed here (too large; obtain from the original providers)

- **TaxiBJ**: 2.7 GB sliding-window archive in our local setup; obtain the
  original ST-ResNet release (Zhang et al.) and adapt
  `data/convert/convert_json_to_npy.py` accordingly.
- **TaxiCQ**: obtain from the ACFM release (Liu et al., AAAI 2018).
- **TDrive**: obtain from the Microsoft T-Drive page and aggregate the
  trajectories into the 32x32 grid.

## Normalization convention

The `.npy` windows store **raw values**. The training code normalizes them to
[0, 1] internally (`value_maxs` / `value_mins`), and metrics are reported after
de-normalization.

## File format

Each `train.npy` / `val.npy` / `test.npy` stores sliding windows of shape
`[num_windows, T, C, H, W]` for grid data or `[num_windows, T, N, C]` for graph
data, where the first `T-1` frames of a window are the model input and the last
frame is the prediction target.
