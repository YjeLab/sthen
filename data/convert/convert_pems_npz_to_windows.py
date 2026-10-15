# -*- coding: utf-8 -*-
"""Convert a PeMS benchmark .npz file into the sliding-window .npy files used
by the training code.

Usage:
    python convert_pems_npz_to_windows.py --name PEMS04 --npz data/graph/PEMS04/PEMS04.npz

Only the flow channel (channel 0) is used as the prediction target. The series
is split chronologically into 60% train / 20% val / 20% test and each part is
cut into sliding windows of `--window` steps with stride 1.
"""
import argparse
import os

import numpy as np


def create_windows(data, window_size, stride):
    num_windows = (data.shape[0] - window_size) // stride + 1
    windows = np.zeros((num_windows, window_size, data.shape[1], data.shape[2]))
    for i in range(num_windows):
        windows[i] = data[i * stride:i * stride + window_size]
    return windows


def main():
    ap = argparse.ArgumentParser(description='Convert a PeMS .npz file to sliding-window .npy files')
    ap.add_argument('--name', required=True, help='dataset name, e.g. PEMS04')
    ap.add_argument('--npz', required=True, help='path to the benchmark .npz file')
    ap.add_argument('--out', default=None, help='output directory (default: Data/<name>)')
    ap.add_argument('--window', type=int, default=24, help='sliding-window size')
    ap.add_argument('--ratios', type=float, nargs=3, default=[0.6, 0.2, 0.2],
                    help='chronological split ratios for train/val/test')
    args = ap.parse_args()
    out = args.out or os.path.join('Data', args.name)

    data = np.load(args.npz)['data'][:, :, 0:1]  # [T, N, 1] flow channel

    l = data.shape[0]
    train_end = int(l * args.ratios[0])
    val_end = int(l * (args.ratios[0] + args.ratios[1]))
    train_data, val_data, test_data = data[:train_end], data[train_end:val_end], data[val_end:]
    print('train min/max:', train_data.min(), train_data.max())

    print('creating sliding windows...')
    train_windows = create_windows(train_data, args.window, 1)
    val_windows = create_windows(val_data, args.window, 1)
    test_windows = create_windows(test_data, args.window, 1)
    print('train:', train_windows.shape, 'val:', val_windows.shape, 'test:', test_windows.shape)

    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, 'train.npy'), train_windows)
    np.save(os.path.join(out, 'val.npy'), val_windows)
    np.save(os.path.join(out, 'test.npy'), test_windows)
    print('saved to', out)


if __name__ == '__main__':
    main()
