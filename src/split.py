"""Create the stratified 70/15/15 splits for seeds 42, 7, 123 and save them to one CSV.

Usage:  python split.py --master master.csv --out splits.csv
        python split.py --data /path/to/data --out splits.csv   (builds the table from folders)
master.csv / folders: data/DATASET/{glaucoma,normal}/*.jpg ; columns path,label,dataset.
"""
import argparse, os
import pandas as pd
from sklearn.model_selection import train_test_split

SEEDS = [42, 7, 123]
SOURCES = ['ACRIMA', 'ORIGA', 'RIM-ONE_DL']      # training sources; every other dataset (G1020) is test-only
EXT = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')


def build_master(data):
    rows = []
    for ds in sorted(os.listdir(data)):
        for cls, label in [('normal', 0), ('glaucoma', 1)]:
            folder = os.path.join(data, ds, cls)
            if os.path.isdir(folder):
                rows += [dict(path=os.path.join(folder, f), label=label, dataset=ds)
                         for f in sorted(os.listdir(folder)) if f.lower().endswith(EXT)]
    return pd.DataFrame(rows)


def split_source(d, seed):
    trval, te = train_test_split(d, test_size=0.15, stratify=d.label, random_state=seed)
    tr, va = train_test_split(trval, test_size=0.15 / 0.85, stratify=trval.label, random_state=seed)
    return tr, va, te


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--master'); ap.add_argument('--data'); ap.add_argument('--out', default='splits.csv')
    a = ap.parse_args()
    df = pd.read_csv(a.master) if a.master else build_master(a.data)
    parts = []
    for seed in SEEDS:
        for ds, d in df.groupby('dataset', sort=False):
            if ds in SOURCES:
                for name, part in zip(['train', 'val', 'test'], split_source(d, seed)):
                    parts.append(part.assign(seed=seed, subset=name))
            else:
                parts.append(d.assign(seed=seed, subset='external'))
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(a.out, index=False)
    print(out.groupby(['seed', 'dataset', 'subset']).label.agg(['size', 'sum']).rename(columns={'sum': 'glaucoma'}))
