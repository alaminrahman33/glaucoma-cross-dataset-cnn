######## Reporting Cell ####
import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from itertools import product
# ---- 5.0  Fixed definitions (nothing here changes the experiment) -------------------
EXPECTED_SEEDS = [42, 7, 123]
SEED_FILES = {s: f'{OUT}/results_seed{s}.csv' for s in EXPECTED_SEEDS}   # no glob
REQ_COLS = ['seed', 'model', 'source', 'target', 'setting', 'n', 'val_auc', 'thr',
            'auc', 'sens', 'spec', 'acc']
MODELS   = ['resnet50', 'densenet121', 'mobilenet_v2', 'efficientnet_b0', 'inception_v3']
PRIMARY  = ['ACRIMA', 'ORIGA', 'RIM-ONE_DL']      # folder names used in data/
EXTERNAL = 'G1020'                                # test-only external dataset
METRICS  = ['auc', 'sens', 'spec', 'acc']
LABEL = {'ACRIMA': 'ACRIMA', 'ORIGA': 'ORIGA', 'RIM-ONE_DL': 'RIM-ONE DL', 'G1020': 'G1020',
         'resnet50': 'ResNet50', 'densenet121': 'DenseNet121', 'mobilenet_v2': 'MobileNetV2',
         'efficientnet_b0': 'EfficientNet-B0', 'inception_v3': 'InceptionV3'}
RANK = {k: i for i, k in enumerate(MODELS + PRIMARY + [EXTERNAL] + ['in-domain', 'cross-source'])}
def check(cond, msg):
    if not cond: raise ValueError('REPORTING CHECK FAILED: ' + msg)
def pm(m, s, d=3):                         # "0.912 ± 0.021"
    return f'{m:.{d}f} ± {s:.{d}f}'
def sort_rows(df):
    cols = [c for c in ['model', 'source', 'target', 'setting'] if c in df.columns]
    return df.sort_values(cols, key=lambda s: s.map(RANK)).reset_index(drop=True)
def save(df, name):
    df.to_csv(f'{OUT}/{name}', index=False, encoding='utf-8-sig')
    print('saved', name, df.shape)
def summarize(df, keys, cols=METRICS):
    """mean and sample SD (ddof=1) ACROSS SEEDS. Every group must hold exactly one row per seed,
    so no image-level or repeated-row averaging can happen."""
    g = df.groupby(keys, sort=False)
    out = g[cols].mean().add_suffix('_mean').join(g[cols].std(ddof=1).add_suffix('_sd'))
    out['n_seeds'] = g['seed'].nunique()
    check((out.n_seeds == len(EXPECTED_SEEDS)).all(), f'{keys}: a group does not have all 3 seeds')
    check((g.size() == out.n_seeds).all(), f'{keys}: more than one row per seed in a group')
    for c in cols:
        out[f'{c}_mean_sd'] = [pm(m, s) for m, s in zip(out[f'{c}_mean'], out[f'{c}_sd'])]
    return out.reset_index()
# ---- 5.1  Load ONLY the three expected files and validate them ----------------------
frames = []
for s, f in SEED_FILES.items():
    if not os.path.isfile(f): raise FileNotFoundError(f'Missing results file for seed {s}: {f}')
    try: d = pd.read_csv(f)
    except Exception as e: raise RuntimeError(f'Cannot read {f}: {e}')
    absent = [c for c in REQ_COLS if c not in d.columns]
    check(not absent, f'{f}: missing columns {absent}')
    check(len(d) > 0, f'{f} is empty')
    found = sorted(d['seed'].unique().tolist())
    check(found == [s], f'{f}: expected seed [{s}] but file contains {found}')
    frames.append(d[REQ_COLS].copy())
allr = pd.concat(frames, ignore_index=True)
for c in ['model', 'source', 'target', 'setting']:
    allr[c] = allr[c].astype(str).str.strip()
check(sorted(allr.seed.unique().tolist()) == sorted(EXPECTED_SEEDS), 'seed set is not {42, 7, 123}')
check(set(allr.model) <= set(MODELS), f'unknown models: {set(allr.model) - set(MODELS)}')
check((set(allr.source) | set(allr.target)) <= set(PRIMARY + [EXTERNAL]),
      f'unknown datasets: {(set(allr.source) | set(allr.target)) - set(PRIMARY + [EXTERNAL])}')
dups = allr[allr.duplicated(['seed', 'model', 'source', 'target'], keep=False)]
check(len(dups) == 0, f'duplicate (seed, model, source, target) rows:\n{dups}')
exp_set = np.where(allr.source == allr.target, 'in-domain', 'cross-domain')
check((allr.setting.values == exp_set).all(), "'setting' disagrees with source == target")
print('files loaded:', {s: int((allr.seed == s).sum()) for s in EXPECTED_SEEDS}, 'rows per seed')
# ---- 5.2  Scope split: PRIMARY 3x3 vs EXTERNAL G1020 (they can never mix) -----------
prim = allr[allr.source.isin(PRIMARY) & allr.target.isin(PRIMARY)].copy()
ext  = allr[allr.source.isin(PRIMARY) & (allr.target == EXTERNAL)].copy()
rest = allr.drop(index=prim.index.union(ext.index))
check(rest.source.eq(EXTERNAL).all(), 'unexpected rows outside primary/external scope')
check(~prim.source.eq(EXTERNAL).any() and ~prim.target.eq(EXTERNAL).any(), 'G1020 leaked into primary')
check(ext.target.eq(EXTERNAL).all() and ext.source.isin(PRIMARY).all(), 'external table is malformed')
print(f'primary rows {len(prim)} | external rows {len(ext)} | '
      f'ignored rows with G1020 as SOURCE {len(rest)} (should be 0; G1020 is test-only)')
# ---- 5.3  Missing-result detection (never silently ignored) -------------------------
expected = pd.DataFrame(list(product(EXPECTED_SEEDS, MODELS, PRIMARY, PRIMARY + [EXTERNAL])),
                        columns=['seed', 'model', 'source', 'target'])
chk = expected.merge(allr[['seed', 'model', 'source', 'target']], how='left', indicator=True)
missing = chk[chk['_merge'] == 'left_only'].drop(columns='_merge')
if len(missing):
    missing['setting'] = np.where(missing.source == missing.target, 'in-domain', 'cross-domain')
    print(missing[['model', 'source', 'target', 'setting', 'seed']].to_string(index=False))
    raise ValueError(f'{len(missing)} required result(s) missing (listed above): '
                     'finish those runs before reporting.')
for d_ in (prim, ext):
    d_[METRICS] = d_[METRICS].apply(pd.to_numeric, errors='raise').astype(float)
    check(d_[METRICS].notna().all().all(), 'NaN found in auc/sens/spec/acc')
check(len(prim) == 3 * 5 * 9 and len(ext) == 3 * 5 * 3, 'unexpected number of result rows')
print('completeness OK: 3 seeds x 5 models x (9 primary + 3 G1020) results present')
# ---- 5.4  PRIMARY 3x3: per-cell mean and SD across seeds ----------------------------
cell = sort_rows(summarize(prim, ['model', 'source', 'target']))
cell.insert(3, 'setting', np.where(cell.source == cell.target, 'in-domain', 'cross-source'))
save(cell[['model', 'source', 'target', 'setting', 'n_seeds',
           'auc_mean', 'auc_sd', 'auc_mean_sd']], 'primary_3x3_auc_mean_sd.csv')
save(cell[['model', 'source', 'target', 'setting', 'n_seeds'] +
          [f'{m}_{k}' for m in METRICS for k in ('mean', 'sd', 'mean_sd')]],
     'primary_sens_spec_mean_sd.csv')          # per-cell AUC, sens, spec, acc
# detailed train/test table (display version)
detail = cell[['model', 'source', 'target', 'setting', 'auc_mean_sd',
               'sens_mean_sd', 'spec_mean_sd', 'acc_mean_sd']].copy()
for c in ['model', 'source', 'target']: detail[c] = detail[c].map(LABEL)
detail.columns = ['Model', 'Source', 'Target', 'Setting', 'AUC Mean ± SD',
                  'Sensitivity Mean ± SD', 'Specificity Mean ± SD', 'Accuracy Mean ± SD']
print('\nPRIMARY train/test performance (3 seeds)'); print(detail.to_string(index=False))
# ---- 5.5  PRIMARY AUC DROP: seed-level FIRST, aggregate afterwards ------------------
lvl = prim.groupby(['model', 'seed', 'setting'], as_index=False)[METRICS].mean()
cnt = prim.groupby(['model', 'seed', 'setting']).size().unstack('setting')
check((cnt['in-domain'] == 3).all() and (cnt['cross-domain'] == 6).all(),
      'need 3 diagonal and 6 off-diagonal cells per model and seed')
wide = lvl.pivot(index=['model', 'seed'], columns='setting', values='auc').reset_index()
wide = wide.rename(columns={'in-domain': 'in_domain_auc', 'cross-domain': 'cross_source_auc'})
wide['auc_drop'] = wide.in_domain_auc - wide.cross_source_auc           # per seed
wide = wide.sort_values(['model', 'seed'], key=lambda s: s.map(RANK) if s.name == 'model' else s)
save(wide, 'primary_auc_drop_per_seed.csv')                              # audit trail
summ = sort_rows(summarize(wide, ['model'], cols=['in_domain_auc', 'cross_source_auc', 'auc_drop']))
save(summ, 'primary_model_summary_mean_sd.csv')
tab = pd.DataFrame({'Model': summ.model.map(LABEL), 'In-domain AUC': summ.in_domain_auc_mean_sd,
                    'Cross-source AUC': summ.cross_source_auc_mean_sd, 'AUC Drop': summ.auc_drop_mean_sd})
print('\nPRIMARY model summary (mean ± SD over 3 seeds)'); print(tab.to_string(index=False))
# ---- 5.6  Sensitivity / specificity / accuracy by model (seed-level, then across seeds) 
lvl2 = lvl.copy(); lvl2['setting'] = lvl2.setting.replace({'cross-domain': 'cross-source'})
ssa = sort_rows(summarize(lvl2, ['model', 'setting']))
save(ssa[['model', 'setting', 'n_seeds'] + [f'{m}_{k}' for m in METRICS for k in ('mean', 'sd', 'mean_sd')]],
     'primary_sens_spec_by_model_mean_sd.csv')
t2 = ssa[['model', 'setting', 'sens_mean_sd', 'spec_mean_sd', 'acc_mean_sd']].copy()
t2['model'] = t2.model.map(LABEL)
t2.columns = ['Model', 'Setting', 'Sensitivity', 'Specificity', 'Accuracy']
print('\nSensitivity / specificity / accuracy at the stored validation threshold'); print(t2.to_string(index=False))
# ---- 5.7  EXTERNAL G1020 (separate from everything above) ---------------------------
ext_tab = sort_rows(summarize(ext, ['model', 'source', 'target']))
save(ext_tab[['model', 'source', 'target', 'n_seeds'] +
             [f'{m}_{k}' for m in METRICS for k in ('mean', 'sd', 'mean_sd')]], 'external_G1020_mean_sd.csv')
ind = prim[prim.setting == 'in-domain'][['seed', 'model', 'source', 'auc']].rename(columns={'auc': 'in_domain_auc'})
g10 = ext[['seed', 'model', 'source', 'auc']].rename(columns={'auc': 'g1020_auc'})
ed = ind.merge(g10, on=['seed', 'model', 'source'], how='inner', validate='one_to_one')
check(len(ed) == len(ext), 'in-domain / G1020 rows could not be paired one-to-one')
ed['external_drop'] = ed.in_domain_auc - ed.g1020_auc                    # per seed
save(sort_rows(ed), 'external_G1020_drop_per_seed.csv')
ed_tab = sort_rows(summarize(ed, ['model', 'source'], cols=['in_domain_auc', 'g1020_auc', 'external_drop']))
save(ed_tab, 'external_G1020_drop_mean_sd.csv')
g_tab = pd.DataFrame({'Model': ed_tab.model.map(LABEL), 'Train Source': ed_tab.source.map(LABEL),
                      'G1020 AUC': ed_tab.g1020_auc_mean_sd, 'External Drop': ed_tab.external_drop_mean_sd})
print('\nEXTERNAL G1020 (mean ± SD over 3 seeds)'); print(g_tab.to_string(index=False))
# ---- 5.8  Heatmaps (primary 3x3 only) ----------------------------------------------
def draw_heatmaps(stat, fname, title, cmap, vmin, vmax, fmt):
    fig, axes = plt.subplots(1, len(MODELS), figsize=(4.6 * len(MODELS), 4.4))
    for ax, m in zip(axes, MODELS):
        mat = (cell[cell.model == m].pivot(index='source', columns='target', values=stat)
               .reindex(index=PRIMARY, columns=PRIMARY).rename(index=LABEL, columns=LABEL))
        sns.heatmap(mat, annot=True, fmt=fmt, vmin=vmin, vmax=vmax, cmap=cmap, linewidths=0.5,
                    cbar=True, ax=ax, annot_kws={'size': 11})
        ax.set_title(LABEL[m]); ax.set_xlabel('test dataset (target)'); ax.set_ylabel('train dataset (source)')
    fig.suptitle(title, y=1.02); plt.tight_layout()
    plt.savefig(f'{OUT}/{fname}', dpi=200, bbox_inches='tight'); plt.show()
draw_heatmaps('auc_mean', 'primary_auc_mean_heatmaps.png',
              'Primary 3x3: mean AUC across seeds 42, 7, 123 (diagonal = in-domain)',
              'viridis', 0.5, 1.0, '.3f')
draw_heatmaps('auc_sd', 'primary_auc_sd_heatmaps.png',
              'Primary 3x3: SD of AUC across seeds 42, 7, 123 (ddof=1)',
              'magma_r', 0.0, max(0.05, float(cell.auc_sd.max())), '.3f')
# ---- 5.9  Optional bar chart (replaces old Step 5.3) -------------------------------
x = np.arange(len(summ)); w = 0.38
plt.figure(figsize=(8, 4))
plt.bar(x - w / 2, summ.in_domain_auc_mean, w, yerr=summ.in_domain_auc_sd, capsize=3, label='In-domain')
plt.bar(x + w / 2, summ.cross_source_auc_mean, w, yerr=summ.cross_source_auc_sd, capsize=3, label='Cross-source')
plt.xticks(x, [LABEL[m] for m in summ.model], rotation=20); plt.ylim(0.5, 1.0); plt.ylabel('AUC')
plt.title('Primary 3x3: in-domain vs cross-source AUC (mean ± SD over 3 seeds)')
plt.legend(); plt.tight_layout(); plt.savefig(f'{OUT}/primary_in_vs_cross.png', dpi=200); plt.show()
