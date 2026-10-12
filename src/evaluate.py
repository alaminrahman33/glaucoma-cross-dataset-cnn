"""Report results: primary 3x3 (ACRIMA/ORIGA/RIM-ONE DL) and separate external G1020 analysis.
All numbers are mean +/- sample SD (ddof=1) over seeds 42, 7, 123. Nothing is averaged over images.

Usage:  python evaluate.py --results results
"""
import argparse, os
from itertools import product
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns

SEEDS = [42, 7, 123]
MODELS = ['resnet50', 'densenet121', 'mobilenet_v2', 'efficientnet_b0', 'inception_v3']
PRIMARY, EXTERNAL = ['ACRIMA', 'ORIGA', 'RIM-ONE_DL'], 'G1020'
METRICS = ['auc', 'sens', 'spec', 'acc']
COLS = ['seed', 'model', 'source', 'target', 'n', 'val_auc', 'thr'] + METRICS


def summarize(df, keys, cols):
    g = df.groupby(keys, sort=False)
    n = g.seed.nunique()
    assert (n == len(SEEDS)).all() and (g.size() == n).all(), f'{keys}: need exactly one row per seed'
    out = g[cols].mean().add_suffix('_mean').join(g[cols].std(ddof=1).add_suffix('_sd')).join(n.rename('n_seeds'))
    for c in cols: out[f'{c}_mean_sd'] = [f'{m:.3f} ± {s:.3f}' for m, s in zip(out[f'{c}_mean'], out[f'{c}_sd'])]
    return out.reset_index()


def heatmaps(cell, stat, fname, title, cmap, vmin, vmax, out):
    fig, axes = plt.subplots(1, len(MODELS), figsize=(4.6 * len(MODELS), 4.4))
    for ax, m in zip(axes, MODELS):
        mat = cell[cell.model == m].pivot(index='source', columns='target', values=stat).reindex(index=PRIMARY, columns=PRIMARY)
        sns.heatmap(mat, annot=True, fmt='.3f', vmin=vmin, vmax=vmax, cmap=cmap, linewidths=0.5, ax=ax)
        ax.set_title(m); ax.set_xlabel('test dataset'); ax.set_ylabel('train dataset')
    fig.suptitle(title, y=1.02); plt.tight_layout(); plt.savefig(f'{out}/{fname}', dpi=200, bbox_inches='tight'); plt.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--results', default='results'); out = ap.parse_args().results
    # ---- load + validate (only the three expected files) ----
    frames = []
    for s in SEEDS:
        d = pd.read_csv(f'{out}/results_seed{s}.csv')
        assert set(COLS) <= set(d.columns), f'seed {s}: missing columns'
        assert d.seed.unique().tolist() == [s], f'seed {s}: file contains another seed'
        frames.append(d[COLS])
    allr = pd.concat(frames, ignore_index=True)
    assert not allr.duplicated(['seed', 'model', 'source', 'target']).any(), 'duplicate rows'
    need = pd.DataFrame(list(product(SEEDS, MODELS, PRIMARY, PRIMARY + [EXTERNAL])), columns=['seed', 'model', 'source', 'target'])
    missing = need.merge(allr, how='left', indicator=True).query("_merge == 'left_only'")
    assert missing.empty, f'missing results:\n{missing}'
    prim = allr[allr.source.isin(PRIMARY) & allr.target.isin(PRIMARY)].copy()
    ext = allr[allr.source.isin(PRIMARY) & (allr.target == EXTERNAL)].copy()
    prim['setting'] = np.where(prim.source == prim.target, 'in-domain', 'cross-source')

    # ---- primary 3x3: per cell ----
    cell = summarize(prim, ['model', 'source', 'target'], METRICS)
    cell.to_csv(f'{out}/primary_3x3_mean_sd.csv', index=False, encoding='utf-8-sig')
    # ---- primary AUC drop: per seed first (3 diagonal vs 6 off-diagonal cells), then mean/SD ----
    lvl = prim.groupby(['model', 'seed', 'setting'], as_index=False)[METRICS].mean()
    wide = lvl.pivot(index=['model', 'seed'], columns='setting', values='auc').reset_index()
    wide = wide.rename(columns={'in-domain': 'in_domain_auc', 'cross-source': 'cross_source_auc'})
    wide['auc_drop'] = wide.in_domain_auc - wide.cross_source_auc
    wide.to_csv(f'{out}/primary_auc_drop_per_seed.csv', index=False)
    summ = summarize(wide, ['model'], ['in_domain_auc', 'cross_source_auc', 'auc_drop'])
    summ.to_csv(f'{out}/primary_model_summary_mean_sd.csv', index=False, encoding='utf-8-sig')
    print('\nPRIMARY (mean ± SD over 3 seeds)')
    print(summ[['model', 'in_domain_auc_mean_sd', 'cross_source_auc_mean_sd', 'auc_drop_mean_sd']].to_string(index=False))
    # ---- sensitivity / specificity at the stored validation threshold ----
    ssa = summarize(lvl, ['model', 'setting'], METRICS)
    ssa.to_csv(f'{out}/primary_sens_spec_by_model_mean_sd.csv', index=False, encoding='utf-8-sig')
    print('\nSENS / SPEC / ACC')
    print(ssa[['model', 'setting', 'sens_mean_sd', 'spec_mean_sd', 'acc_mean_sd']].to_string(index=False))

    # ---- external G1020 (separate): AUC and drop = source in-domain AUC - G1020 AUC, per seed ----
    ext_tab = summarize(ext, ['model', 'source'], METRICS)
    ext_tab.to_csv(f'{out}/external_G1020_mean_sd.csv', index=False, encoding='utf-8-sig')
    ind = prim[prim.setting == 'in-domain'][['seed', 'model', 'source', 'auc']].rename(columns={'auc': 'in_domain_auc'})
    ed = ind.merge(ext[['seed', 'model', 'source', 'auc']].rename(columns={'auc': 'g1020_auc'}),
                   on=['seed', 'model', 'source'], validate='one_to_one')
    ed['external_drop'] = ed.in_domain_auc - ed.g1020_auc
    ed.to_csv(f'{out}/external_G1020_drop_per_seed.csv', index=False)
    ed_tab = summarize(ed, ['model', 'source'], ['in_domain_auc', 'g1020_auc', 'external_drop'])
    ed_tab.to_csv(f'{out}/external_G1020_drop_mean_sd.csv', index=False, encoding='utf-8-sig')
    print('\nEXTERNAL G1020')
    print(ed_tab[['model', 'source', 'g1020_auc_mean_sd', 'external_drop_mean_sd']].to_string(index=False))

    # ---- figures ----
    heatmaps(cell, 'auc_mean', 'primary_auc_mean_heatmaps.png', 'Mean AUC over seeds 42, 7, 123 (diagonal = in-domain)',
             'viridis', 0.5, 1.0, out)
    heatmaps(cell, 'auc_sd', 'primary_auc_sd_heatmaps.png', 'SD of AUC over seeds (ddof=1)',
             'magma_r', 0.0, max(0.05, float(cell.auc_sd.max())), out)
    print('\nsaved tables and figures to', out)
