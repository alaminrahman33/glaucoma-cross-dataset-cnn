####### Model factory (the 5 pre-trained CNNs) #######

def make_model(name):
    if name == 'resnet50':
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Linear(m.fc.in_features, 1)
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
        m.aux_logits = False; m.AuxLogits = None   # turn off the auxiliary head (simplifies training)
        m.fc = nn.Linear(m.fc.in_features, 1)
    else:
        raise ValueError(name)
    return m.to(DEVICE)
print(sum(p.numel() for p in make_model('mobilenet_v2').parameters()) / 1e6, 'million parameters')
########## The Training Function ##########
def train_model(name, tr, va, seed, epochs=EPOCHS, patience=5):
    seed_everything(seed)
    model = make_model(name)
    tl = make_loader(tr, train_tf, shuffle=True, drop_last=True)
    vl = make_loader(va, eval_tf)
    pos = float(tr.label.sum()); neg = len(tr) - pos
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    best_auc, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train(); tot = 0.0
        for x, y in tl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=USE_AMP):
                out = model(x).squeeze(1)
            loss = crit(out.float(), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * len(x)
        sched.step()
        yv, pv = predict(model, vl)
        auc = roc_auc_score(yv, pv)
        print(f'  [{name}] epoch {ep+1:02d}  train_loss {tot/len(tr):.3f}  val_auc {auc:.3f}')
        if auc > best_auc:
            best_auc, best_state, bad = auc, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best_state)
    yv, pv = predict(model, vl)
    fpr, tpr, th = roc_curve(yv, pv)
    thr = float(th[np.argmax(tpr - fpr)])
    return model, thr, float(best_auc)
    ####### Run The Full Experiment#####
MODELS  = ['resnet50', 'densenet121', 'mobilenet_v2', 'efficientnet_b0', 'inception_v3']
SOURCES = ['ACRIMA', 'ORIGA', 'RIM-ONE_DL']   # training sources (G1020 is NOT a source)
TARGETS = SOURCES + ['G1020']                  # G1020 = external test-only target
assert set(TARGETS) <= set(df.dataset.unique()), 'a dataset folder is missing from master.csv'
res_file = f'{OUT}/results_seed{SEED}.csv'
results = pd.read_csv(res_file).to_dict('records') if os.path.exists(res_file) else []
done = {(r['model'], r['source']) for r in results}
for src in SOURCES:
    tr, va, te = split_source(df[df.dataset == src], SEED)
    for name in MODELS:
        if (name, src) in done: continue
        t0 = time.time()
        print(f'=== train {name} on {src} (train={len(tr)}, val={len(va)}, test={len(te)}) ===')
        model, thr, val_auc = train_model(name, tr, va, SEED)
        for tgt in TARGETS:
            d_eval = te if tgt == src else df[df.dataset == tgt]
            y, p = predict(model, make_loader(d_eval, eval_tf))
            m = metrics(y, p, thr)
            results.append(dict(seed=SEED, model=name, source=src, target=tgt,
                                setting='in-domain' if tgt == src else 'cross-domain',
                                n=len(d_eval), val_auc=val_auc, thr=thr, **m))
            print(f'  -> test on {tgt:12s} AUC {m["auc"]:.3f} sens {m["sens"]:.2f} spec {m["spec"]:.2f}')
        pd.DataFrame(results).to_csv(res_file, index=False)    # save after every run
        del model; torch.cuda.empty_cache()
        print(f'  done in {(time.time()-t0)/60:.1f} min\n')
