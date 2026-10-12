"""Fine-tune 5 ImageNet CNNs on each source dataset (train/val split only), then test each model
in-domain (held-out 15%) and on ALL images of every other dataset. One CSV per seed, resumable.

Usage:  python train.py --seed 42 --splits splits.csv --out results
        (repeat with --seed 7 and --seed 123)
"""
import argparse, copy, os, random, time
import numpy as np, pandas as pd, torch, torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

IMG, BATCH, EPOCHS, LR, PATIENCE = 224, 32, 15, 1e-4, 5            # same for every model
MODELS = ['resnet50', 'densenet121', 'mobilenet_v2', 'efficientnet_b0', 'inception_v3']
SOURCES = ['ACRIMA', 'ORIGA', 'RIM-ONE_DL']                         # G1020 is never a training source
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_AMP = DEVICE == 'cuda'

norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_tf = transforms.Compose([transforms.Resize((IMG, IMG)), transforms.RandomHorizontalFlip(),
                               transforms.RandomRotation(15), transforms.ColorJitter(0.2, 0.2),
                               transforms.ToTensor(), norm])
eval_tf = transforms.Compose([transforms.Resize((IMG, IMG)), transforms.ToTensor(), norm])


class FundusDS(Dataset):
    def __init__(self, frame, tf):
        self.paths, self.labels, self.tf = frame.path.tolist(), frame.label.values.astype('float32'), tf
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert('RGB')), torch.tensor(self.labels[i])


def make_loader(frame, tf, shuffle=False, drop_last=False):
    return DataLoader(FundusDS(frame, tf), batch_size=BATCH, shuffle=shuffle, num_workers=2, drop_last=drop_last)


def make_model(name):
    if name == 'resnet50':
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2); m.fc = nn.Linear(m.fc.in_features, 1)
    elif name == 'densenet121':
        m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        m.classifier = nn.Linear(m.classifier.in_features, 1)
    elif name == 'mobilenet_v2':
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
    elif name == 'efficientnet_b0':
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
    elif name == 'inception_v3':
        m = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        m.aux_logits = False; m.AuxLogits = None; m.fc = nn.Linear(m.fc.in_features, 1)
    else:
        raise ValueError(name)
    return m.to(DEVICE)


def seed_everything(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def predict(model, loader):
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=USE_AMP):
                out = model(x.to(DEVICE))
            ps.append(torch.sigmoid(out.float()).squeeze(1).cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def metrics(y, p, thr):
    tn, fp, fn, tp = confusion_matrix(y, (p >= thr).astype(int), labels=[0, 1]).ravel()
    return dict(auc=roc_auc_score(y, p), sens=tp / max(tp + fn, 1), spec=tn / max(tn + fp, 1),
                acc=(tp + tn) / len(y))


def train_model(name, tr, va, seed):
    """Best epoch chosen by validation AUC; threshold = Youden's J on validation. Test data never used."""
    seed_everything(seed)
    model = make_model(name)
    tl, vl = make_loader(tr, train_tf, shuffle=True, drop_last=True), make_loader(va, eval_tf)
    pos = float(tr.label.sum()); neg = len(tr) - pos
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    best_auc, best_state, bad = -1.0, None, 0
    for ep in range(EPOCHS):
        model.train(); tot = 0.0
        for x, y in tl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=USE_AMP):
                out = model(x).squeeze(1)
            loss = crit(out.float(), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * len(x)
        sched.step()
        auc = roc_auc_score(*predict(model, vl))
        print(f'  [{name}] epoch {ep + 1:02d} train_loss {tot / len(tr):.3f} val_auc {auc:.3f}')
        if auc > best_auc: best_auc, best_state, bad = auc, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE: break
    model.load_state_dict(best_state)
    yv, pv = predict(model, vl)
    fpr, tpr, th = roc_curve(yv, pv)
    return model, float(th[np.argmax(tpr - fpr)]), float(best_auc)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True, choices=[42, 7, 123])
    ap.add_argument('--splits', default='splits.csv'); ap.add_argument('--out', default='results')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sp = pd.read_csv(a.splits); sp = sp[sp.seed == a.seed]
    targets = SOURCES + sorted(set(sp.dataset) - set(SOURCES))      # sources + test-only (G1020)
    res_file = f'{a.out}/results_seed{a.seed}.csv'
    results = pd.read_csv(res_file).to_dict('records') if os.path.exists(res_file) else []
    done = {(r['model'], r['source']) for r in results}
    for src in SOURCES:
        d = sp[sp.dataset == src]
        tr, va, te = (d[d.subset == s] for s in ['train', 'val', 'test'])
        for name in MODELS:
            if (name, src) in done: continue
            t0 = time.time()
            print(f'=== seed {a.seed} | {name} on {src} (train={len(tr)}, val={len(va)}, test={len(te)}) ===')
            model, thr, val_auc = train_model(name, tr, va, a.seed)
            for tgt in targets:
                d_eval = te if tgt == src else sp[sp.dataset == tgt]   # cross: ALL images of the other dataset
                m = metrics(*predict(model, make_loader(d_eval, eval_tf)), thr)
                results.append(dict(seed=a.seed, model=name, source=src, target=tgt,
                                    setting='in-domain' if tgt == src else 'cross-domain',
                                    n=len(d_eval), val_auc=val_auc, thr=thr, **m))
                print(f'  -> {tgt:12s} AUC {m["auc"]:.3f} sens {m["sens"]:.2f} spec {m["spec"]:.2f}')
            pd.DataFrame(results).to_csv(res_file, index=False)        # save after every run
            del model; torch.cuda.empty_cache()
            print(f'  done in {(time.time() - t0) / 60:.1f} min\n')
