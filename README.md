# Cross-Dataset Generalization of ImageNet Pre-trained CNNs for Glaucoma Detection

Code to reproduce the experiments and tables of the paper:

> **Cross-Dataset Generalization of ImageNet Pre-trained CNNs for Glaucoma Detection from Fundus Images**
> Al-Amin Rahman, Saifuddin Ahmed, Resalat Ahmed
> Supervised by Salman Fazle Rabby
> Department of Electrical and Electronic Engineering, Sylhet Engineering College, Sylhet, Bangladesh (affiliated with Shahjalal University of Science & Technology)
>
> **Paper:** [PASTE LINK HERE: arXiv / TechRxiv / Zenodo DOI]

> **Note:** This paper corrects and supersedes our earlier preprint, *"Automated Glaucoma Detection Using OCT and Fundus Images with Convolutional Neural Network"* (January 2023). The earlier work evaluated on a single data source; this study uses a cross-dataset protocol with multiple seeds.

---

## 1. What this project is

Many deep-learning papers on glaucoma report high AUC, but they test on images from the same dataset used for training (same camera, same hospital, same labeling habits). A screening tool in practice will see images from other devices and populations.

This project measures how much performance is lost in that situation. We fine-tune five standard ImageNet pre-trained CNNs under one fixed protocol:

- **Models:** ResNet50, DenseNet121, MobileNetV2, EfficientNet-B0, InceptionV3 (all from `torchvision`, no custom design)
- **Source/target datasets (primary 3×3 analysis):** ACRIMA, ORIGA-light, RIM-ONE DL
- **Test-only dataset:** G1020 (never a source in the main experiments)
- **Seeds:** 42, 7, 123 (results are mean ± SD over seeds)
- **Protocol:** train on one dataset (70% train / 15% validation / 15% test, stratified), test in-domain on the 15% test part and cross-source on the **complete** other datasets

Besides the baseline, the paper tests four follow-up questions:

1. **Threshold recalibration:** can 10% labeled target images repair the decision threshold?
2. **Multi-source training:** does training on two datasets together help on a third?
3. **Image normalization:** gray-world, Ben Graham, CLAHE and LAB reference normalization.
4. **Stronger color augmentation** as a control.

### Main findings (from the paper)

| Quantity (mean over 5 models, 3 seeds) | Value |
|---|---|
| In-domain AUC | 0.913 |
| Cross-source AUC | 0.702 |
| AUC drop | 0.211 ± 0.030 (95% CI 0.135 to 0.286) |
| Sensitivity at the source threshold (in-domain → cross-source) | 0.831 → 0.726 |
| Specificity at the source threshold (in-domain → cross-source) | 0.870 → 0.506 |
| Recalibration on 10% of target images (cross-source) | sensitivity 0.726 → 0.609, specificity 0.506 → 0.670, balanced accuracy 0.616 → 0.639 |
| Multi-source training (held-out AUC) | 0.695 (single-source: 0.702) |
| LAB reference normalization (cross-source AUC) | 0.755 (paired gain 0.053; 95% CI 0.046 to 0.060), drop 0.155 |
| Strong color augmentation (cross-source AUC) | 0.754 |
| Ben Graham / CLAHE / gray-world (cross-source AUC) | 0.733 / 0.712 / 0.696 |
| AUC on G1020 (45 baseline evaluations) | 0.458 to 0.529, mean 0.493 (chance level) |
| AUC when trained and tested inside G1020 | 0.523 ± 0.044 (also near chance) |

No architecture was clearly more robust than the others after correction for multiple comparisons. Normalization and color augmentation reduce the drop but do not remove it. See the paper for details and limitations.

---

## 2. Repository structure

```
.
├── README.md
├── requirements.txt
├── LICENSE
├── notebooks/
│   └── glaucoma_cross_dataset.ipynb     # full baseline pipeline (Colab notebook)
├── scripts/                              # [ADD] disc cropping, G1020 vCDR analysis, recalibration,
│                                         #       multi-source, normalization (see Section 7)
└── results/                              # CSV files and figures produced by the notebook
```

> Adjust this tree to match what you actually upload.

---

## 3. Requirements

The code was run on **Google Colab (free tier, NVIDIA T4 GPU)** with Python 3 and PyTorch.

| Library | Used for |
|---|---|
| `torch`, `torchvision` | models (ImageNet weights), training |
| `scikit-learn` | train/val/test split, AUC, ROC, confusion matrix |
| `numpy`, `pandas` | data handling, results tables |
| `Pillow` | image loading |
| `opencv-python` | CLAHE, Gaussian blur (Ben Graham), CIELAB conversion |
| `matplotlib`, `seaborn` | figures and heatmaps |
| `grad-cam` (`pytorch-grad-cam`) | Grad-CAM (optional) |

Install locally with:

```bash
pip install -r requirements.txt
```

On Colab, most of these are already installed. Only Grad-CAM needs an extra install: `!pip install -q grad-cam`.

**Versions used:** [FILL IN: run `import torch, torchvision, sklearn, cv2; print(torch.__version__, torchvision.__version__, sklearn.__version__, cv2.__version__)` in your Colab notebook and paste the result here.]

---

## 4. Datasets: download and folder layout

The datasets are **not included** in this repository. Each has its own license and terms of use. Download them yourself from the original sources and cite the original papers (see Section 9).

| Dataset | Images (glaucoma / normal) | Role | Where to get it |
|---|---|---|---|
| ACRIMA | 705 (396 / 309) | source, target | [ADD LINK] |
| RIM-ONE DL | 485 (172 / 313) | source, target | [ADD LINK] |
| ORIGA-light | 650 (168 / 482) | source, target | [ADD LINK] |
| G1020 | 1020 (296 / 724) | test only | [ADD LINK] |

### Where to place the files

Create this folder tree in your Google Drive (names exactly, no spaces):

```
MyDrive/
└── glaucoma/
    ├── raw/                  <- downloaded datasets, unsorted, as downloaded
    │   ├── ACRIMA/
    │   ├── RIM-ONE_DL/
    │   ├── ORIGA/            <- contains OrigaList.csv and Images_Cropped/
    │   └── G1020/            <- contains G1020.csv and Images/
    ├── data/                 <- created by the notebook (sorted into glaucoma/normal folders)
    └── results/              <- created by the notebook (CSV files, figures)
```

How labels are read (check these against your downloaded copies, since file and column names can differ):

- **ACRIMA:** glaucoma images have `_g_` in the file name.
- **RIM-ONE DL:** already in `glaucoma/` and `normal/` folders (if it has `training_set` / `test_set` sub-folders, copy both into the same folders; the original split is not used).
- **ORIGA-light:** labels in `OrigaList.csv` (column `Glaucoma`, 1 = glaucoma); one image version is used for all images (`Images_Cropped`).
- **G1020:** labels in `G1020.csv` (columns `imageID`, `binaryLabels`).

After running the sorting cells, the notebook prints image counts. They should match the table above. If they do not (for example 0 glaucoma images), the label rule is wrong; fix it before continuing.

---

## 5. How to reproduce the baseline tables

### Fixed settings (identical for all models)

| Setting | Value |
|---|---|
| Input size | 224 × 224 RGB (resized without keeping aspect ratio), ImageNet mean/std normalization |
| Augmentation (train only) | horizontal flip, rotation up to 15°, brightness/contrast jitter 0.2 |
| Optimizer | AdamW, learning rate 1e-4, weight decay 1e-4 |
| Schedule | cosine annealing over 15 epochs, max 15 epochs |
| Batch size | 32 |
| Loss | binary cross-entropy with positive-class weight = (normal / glaucoma) in the training split |
| Precision | mixed precision (AMP) on GPU |
| Model selection | best validation-AUC epoch, early stopping after 5 epochs without improvement |
| Decision threshold | Youden's J on the **validation** set |
| Fine-tuning | all layers; last layer replaced by one output unit; InceptionV3 auxiliary head removed |

Test images are never used to choose any setting.

### Notebook order

Open `notebooks/glaucoma_cross_dataset.ipynb` in Google Colab, set **Runtime → Change runtime type → T4 GPU**, and run the cells in this order:

| # | Step | What it does |
|---|---|---|
| 1 | Setup | `!nvidia-smi` and mount Google Drive |
| 2 | Configuration cell | imports, paths, hyper-parameters, `SEED` (**re-run every session**) |
| 3 | Sorting helper + dataset blocks | copy images into `data/DATASET/{glaucoma,normal}` for ACRIMA, RIM-ONE DL, ORIGA, G1020 |
| 4 | Master table | builds `master.csv`; check the printed counts |
| 5 | Look at images (optional) | saves `sample_images.png` |
| 6 | Dataset class, split function, model factory, metrics, training function | definitions only |
| 7 | Sanity check (optional) | trains one model on one dataset |
| 8 | **Full experiment** | trains 5 models × 3 sources and tests each on 3 sources + G1020 |
| 9 | **Reporting cell** | builds all tables and figures from the three seed files |
| 10 | Grad-CAM (optional) | qualitative maps for ResNet50 trained on ACRIMA |

### The three seeds

Run the full experiment (step 8) **three times**, once per seed:

1. Set `SEED = 42` in the configuration cell → run steps 2 and 6 → run step 8 → writes `results/results_seed42.csv`
2. Set `SEED = 7`, re-run steps 2 and 6, run step 8 → writes `results/results_seed7.csv`
3. Set `SEED = 123`, re-run steps 2 and 6, run step 8 → writes `results/results_seed123.csv`

Each file has 60 rows (5 models × 3 sources × 4 targets). Results are saved after every training run, so if Colab disconnects, re-run the configuration and function cells and then step 8 again: finished runs are skipped.

When all three files exist, run the reporting cell (step 9). It reads **only** these three files, checks that nothing is missing or duplicated, and stops with a clear error otherwise.

### Training time

On a free Colab **T4** GPU, one training run takes about **2 to 6 minutes**. One seed (15 training runs) takes about **1 to 1.5 hours**, so the full baseline experiment with three seeds takes roughly **3 to 4.5 hours**. The follow-up experiments in Section 7 train many more networks (see the table there) and take correspondingly longer. Keep the Colab tab open.

If you see `CUDA out of memory`, restart the session and set `BATCH = 16` for all models. If reading from Drive is slow, copy `data/` to the local Colab disk and rebuild `master.csv`.

### Outputs (saved in `results/`)

| File | Contents |
|---|---|
| `primary_3x3_auc_mean_sd.csv` | AUC mean ± SD per model and train/test cell (source of the 3×3 matrices, paper Table III) |
| `primary_model_summary_mean_sd.csv` | in-domain AUC, cross-source AUC and AUC drop per model (paper Table II) |
| `primary_sens_spec_by_model_mean_sd.csv` | sensitivity, specificity, accuracy per model (paper Table IV) |
| `primary_sens_spec_mean_sd.csv` | all metrics per train/test cell |
| `external_G1020_mean_sd.csv`, `external_G1020_drop_mean_sd.csv` | G1020 external results (paper Table V) |
| `primary_auc_drop_per_seed.csv`, `external_G1020_drop_per_seed.csv` | seed-level values behind the drops |
| `primary_auc_mean_heatmaps.png`, `primary_auc_sd_heatmaps.png` | 3×3 AUC heatmaps (paper Fig. 3) |
| `primary_in_vs_cross.png` | in-domain vs cross-source bar chart (paper Fig. 4) |
| `gradcam.png` | Grad-CAM example maps (paper Fig. 6, optional) |

> [CHECK] File names above come from the earlier README. Update them if your notebook now writes different names, and add the output files of the follow-up experiments (Section 7).

### How results are aggregated

For each model and seed, the in-domain AUC is the mean of the 3 diagonal cells of the 3×3 matrix and the cross-source AUC is the mean of the 6 off-diagonal cells. The AUC drop is computed **per seed first**; only then are the mean and sample SD (n − 1) taken over the three seeds. Confidence intervals use the t distribution with two degrees of freedom. G1020 is reported separately and is never part of the 3×3 averages.

> GPU computations are not perfectly deterministic, so small differences from the paper's numbers (in the third decimal place or so) are normal.

---

## 6. Expected results (for checking your run)

Baseline model summary from the paper (mean ± SD over three seeds):

| Model | In-domain AUC | Cross-source AUC | AUC drop |
|---|---|---|---|
| ResNet50 | 0.917 ± 0.009 | 0.683 ± 0.019 | 0.234 ± 0.021 |
| DenseNet121 | 0.913 ± 0.009 | 0.696 ± 0.034 | 0.217 ± 0.038 |
| MobileNetV2 | 0.918 ± 0.024 | 0.680 ± 0.018 | 0.238 ± 0.040 |
| EfficientNet-B0 | 0.924 ± 0.015 | 0.739 ± 0.002 | 0.185 ± 0.017 |
| InceptionV3 | 0.891 ± 0.033 | 0.712 ± 0.026 | 0.179 ± 0.058 |
| All models | 0.913 ± 0.018 | 0.702 ± 0.014 | 0.211 ± 0.030 |

Mean AUC by source and target (mean over five models and three seeds):

| Train \ Test | ACRIMA | ORIGA | RIM-ONE DL |
|---|---|---|---|
| ACRIMA | 0.998 ± 0.002 | 0.702 ± 0.007 | 0.660 ± 0.014 |
| ORIGA | 0.742 ± 0.024 | 0.808 ± 0.048 | 0.758 ± 0.019 |
| RIM-ONE DL | 0.635 ± 0.028 | 0.716 ± 0.007 | 0.932 ± 0.023 |

Normalization and augmentation (all models, mean ± SD over three seeds):

| Condition | In-domain AUC | Cross-source AUC | AUC drop |
|---|---|---|---|
| Baseline | 0.913 ± 0.018 | 0.702 ± 0.014 | 0.211 ± 0.030 |
| Gray-world | 0.920 ± 0.016 | 0.696 ± 0.008 | 0.224 ± 0.023 |
| Ben Graham | 0.910 ± 0.019 | 0.733 ± 0.009 | 0.177 ± 0.027 |
| CLAHE | 0.915 ± 0.025 | 0.712 ± 0.014 | 0.203 ± 0.037 |
| LAB reference | 0.910 ± 0.018 | 0.755 ± 0.011 | 0.155 ± 0.028 |
| Strong color augmentation | 0.912 ± 0.013 | 0.754 ± 0.007 | 0.158 ± 0.019 |

---

## 7. Additional analyses in the paper

| Analysis | Paper section | What it does | Networks trained |
|---|---|---|---|
| G1020 label check | III-F, IV-D | vertical, horizontal and area cup-to-disc ratio from the G1020 masks (AUC of each as a one-feature classifier); five networks also trained and tested inside G1020 (stratified 70/15/15, seeds 42, 7, 123). Reported separately, never part of any average. | 15 |
| Grad-CAM | III-G | ResNet50 trained on ACRIMA, last residual block, eight test images. Qualitative only. | – |
| Threshold recalibration | III-H, IV-F | New Youden threshold chosen on 5%, 10%, 20% or 30% of the target images (stratified), evaluated on the rest; 200 random splits at 10% (100 at the other sizes), averaged within each seed. The baseline networks were trained again with the same settings and seeds to save per-image scores. | 45 (re-run) |
| Multi-source training | III-I, IV-G | Train on the pooled training parts of two datasets, test on the third. A control arm subsamples the pool to the size of one source (size-matched). | 5 models × 3 targets × 3 seeds per arm |
| Normalization and color augmentation | III-J, IV-H | Gray-world, Ben Graham, CLAHE, LAB reference, strong color augmentation, each trained on each source with the same settings. G1020 stays test-only. | 225 |

Implementation details worth knowing when you re-run these:

- **Image cropping (ORIGA and G1020):** a square window centered on the disc mask supplied with the dataset, side = 2.5 × disc size. Where there was no usable mask, automatic disc detection was used. Crops were saved at 512 × 512 pixels. ACRIMA and RIM-ONE DL were used as downloaded. [ADD: put the cropping script in `scripts/`.]
- **Preprocessing order (normalization experiments):** resize to 224 × 224 → normalization → flip → rotation → color jitter → tensor → ImageNet standardization. In validation and testing: resize → normalization → tensor → ImageNet standardization. No label is used in any normalization.
- **Retina pixels:** a pixel counts as retina if max(R, G, B) > 15. Gray-world and LAB statistics are computed on retina pixels only (all pixels if 10% or fewer qualify).
- **Ben Graham:** 4I − 4(Gσ * I) + 128 per RGB channel, σ = 224/30.
- **CLAHE:** L channel of 8-bit CIELAB, clip limit 2.0, 8 × 8 tiles.
- **LAB reference:** per-channel CIELAB mean and SD mapped to reference values. The references are the average per-image means and SDs of the **source training images only**, recomputed for every source and seed.
- **Strong color augmentation:** brightness, contrast and saturation factors drawn between 0.6 and 1.4, hue shift up to ±0.1 (training only).

> [ADD: list the script or notebook for each row above, for example `scripts/recalibration.py`, `scripts/multi_source.py`, `scripts/normalization.py`, `scripts/g1020_vcdr.py`.]

---

## 8. Limitations

- Only three source datasets; the ORIGA and RIM-ONE DL test splits contain about 25 glaucoma images each, so in-domain estimates are noisy.
- Three seeds give rough SDs and wide confidence intervals; pairwise model comparisons are exploratory.
- No patient identifiers are available, so images of both eyes of one patient may appear in both the training and test parts of the same dataset. This can inflate the in-domain AUC but not the cross-source AUC.
- Labeling criteria differ between datasets and were not harmonized. Hyper-parameters were fixed, not tuned per model.
- G1020 gave chance-level AUC in every condition, including networks trained and tested inside it, so it cannot be used to measure transfer here. The cause may be label quality or information that is not visible at the optic disc.
- The recalibration experiment is a best case: calibration and evaluation images come from the same dataset and the calibration labels are assumed to be available.
- Normalization and color augmentation were each tested alone. They were not combined with pooling or recalibration, and no method that uses unlabeled target images (domain adaptation) was tested.

See the paper (Section V) for the full discussion.

---

## 9. Citation

If you use this code, please cite the paper:

```bibtex
@misc{rahman2026crossdataset,
  title   = {Cross-Dataset Generalization of ImageNet Pre-trained CNNs for Glaucoma Detection from Fundus Images},
  author  = {Rahman, Al-Amin and Ahmed, Saifuddin and Ahmed, Resalat},
  year    = {2026},
  note    = {Preprint. [ADD LINK / DOI]}
}
```

Please also cite the original dataset papers:

- ACRIMA: Diaz-Pinto et al., *CNNs for automatic glaucoma assessment using fundus images: An extensive validation*, BioMed. Eng. OnLine, 2019.
- RIM-ONE DL: Fumero Batista et al., *RIM-ONE DL: A unified retinal image database for assessing glaucoma using deep learning*, Image Anal. Stereol., 2020.
- ORIGA-light: Zhang et al., *ORIGA-light: An online retinal fundus image database for glaucoma analysis and research*, EMBC, 2010.
- G1020: Bajwa et al., *G1020: A benchmark retinal fundus image dataset for computer-aided glaucoma detection*, IJCNN, 2020.

---

## 10. License and contact

Code released under the **MIT License** (see `LICENSE`). The datasets are **not** covered by this license; follow each dataset's own terms.

Contact: Al-Amin Rahman (alaminrahman.eee.33@gmail.com) · ORCID: [YOUR ORCID] · [GitHub profile link]
