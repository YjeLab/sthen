# -*- coding: utf-8 -*-
"""Convert a *_short.json dataset into the train/val/test .npy files used by
the training code.

The *_short.json files shipped in data/grid/<Name>/ contain the sliding
windows under the keys X_train / X_val / X_test (and optionally Y_*).

Usage:
    python convert_json_to_npy.py --json data/grid/TrafficTJ/TrafficTJ_short.json --out Data/TrafficTJ
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser(description='Convert a *_short.json dataset to .npy windows')
    ap.add_argument('--json', required=True, help='path to the *_short.json file')
    ap.add_argument('--out', required=True, help='output directory, e.g. Data/TrafficTJ')
    args = ap.parse_args()

    with open(args.json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    keys = list(data.keys())
    print('keys:', keys)

    os.makedirs(args.out, exist_ok=True)
    for split in ('X_train', 'X_val', 'X_test'):
        if split not in data:
            print('missing key:', split)
            continue
        arr = np.array(data[split][0])
        name = split.replace('X_', '').lower() + '.npy'
        np.save(os.path.join(args.out, name), arr)
        print(name, arr.shape)
    print('done')


if __name__ == '__main__':
    main()
