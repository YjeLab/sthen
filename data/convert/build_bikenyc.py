# -*- coding: utf-8 -*-
"""Build the BikeNYC sliding-window dataset from the original ST-ResNet .h5
release (NYC14_M16x8_T60_NewEnd.h5).

The series is split chronologically with the last 30*24 steps held out for
testing, then cut into sliding windows of `--window` steps (6 input frames +
1 target frame) with stride 1.

Usage:
    python build_bikenyc.py --h5 data/grid/BikeNYC/NYC14_M16x8_T60_NewEnd.h5 --out Data/BikeNYC
"""
import argparse
import os

import h5py
import numpy as np


def create_windows(data, window_size, stride):
    num_windows = (data.shape[0] - window_size) // stride + 1
    return np.array([data[i:i + window_size] for i in range(num_windows)])


def main():
    ap = argparse.ArgumentParser(description='Build BikeNYC windows from the ST-ResNet .h5 file')
    ap.add_argument('--h5', required=True, help='path to NYC14_M16x8_T60_NewEnd.h5')
    ap.add_argument('--out', default='Data/BikeNYC', help='output directory')
    ap.add_argument('--window', type=int, default=7, help='sliding-window size (6 input + 1 target)')
    ap.add_argument('--test-steps', type=int, default=30 * 24, help='number of final steps held out for testing')
    args = ap.parse_args()

    with h5py.File(args.h5, 'r') as f:
        data = f['data'][:]  # [T, C, H, W]
    print('series shape:', data.shape)

    train_data = data[:-args.test_steps]
    test_data = data[-args.test_steps:]

    train_windows = create_windows(train_data, args.window, 1)
    test_windows = create_windows(test_data, args.window, 1)
    print('train:', train_windows.shape, 'max/min:', train_windows.max(), train_windows.min())
    print('test:', test_windows.shape)

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, 'train.npy'), train_windows)
    np.save(os.path.join(args.out, 'test.npy'), test_windows)
    print('saved to', args.out)


if __name__ == '__main__':
    main()
