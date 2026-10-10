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

Many deep-learning papers on glaucoma report AUC above 0.95, but they test on images from the same dataset used for training (same camera, same hospital, same labeling habits). A screening tool in practice will see images from other devices and populations.

This project measures how much performance is lost in that situation. We fine-tune five standard ImageNet pre-trained CNNs under one fixed protocol:

- **Models:** ResNet50, DenseNet121, MobileNetV2, EfficientNet-B0, InceptionV3 (all from `torchvision`, no custom design)
- **Source/target datasets (primary 3x3 analysis):** ACRIMA, ORIGA-light, RIM-ONE DL
- **External test-only dataset:** G1020 (never used for training)
- **Seeds:** 42, 7, 123 (results are mean ± SD over seeds)
- **Protocol:** train on one dataset (70% train / 15% validation / 15% test, stratified), test in-domain on the 15% test part and cross-source on the **complete** other datasets

### Main findings (from the paper)

| Quantity (mean over 5 models, 3 seeds) | Value |
|---|---|
| In-domain AUC | 0.921 |
| Cross-source AUC | 0.701 |
| AUC drop | 0.219 ± 0.015 (95% CI 0.183 to 0.256) |
| Mean specificity at the source threshold (in-domain → cross-source) | 0.89 → 0.49 |
| AUC on G1020 (all 45 evaluations) | 0.46 to 0.53 (chance level) |

No architecture was clearly more robust than the others after correction for multiple comparisons. See the paper for details and limitations.

---

## 2. Repository structure

```
.
├── README.md
├── requirements.txt
├── LICENSE
├── notebooks/
│   └── glaucoma_cross_dataset.ipynb     # full pipeline (Colab notebook)
├── scripts/                              # [ADD] disc-cropping script, G1020 vCDR analysis (see Section 7)
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
| `matplotlib`, `seaborn` | figures and heatmaps |
| `grad-cam` (`pytorch-grad-cam`) | Grad-CAM (optional) |

Install locally with:

```bash
pip install -r requirements.txt
```

On Colab, most of these are already installed. Only Grad-CAM needs an extra install: `!pip install -q grad-cam`.

**Versions used:** [FILL IN: run `import torch, torchvision, sklearn; print(torch.__version__, torchvision.__version__, sklearn.__version__)` in your Colab notebook and paste the result here.]

---

## 4. Datasets: download and folder layout

The datasets are **not included** in this repository. Each has its own license and terms of use. Download them yourself from the original sources and cite the original papers (see Section 9).

| Dataset | Images (glaucoma / normal) | Where to get it |
|---|---|---|
| ACRIMA | 705 (396 / 309) | [ADD LINK] |
| RIM-ONE DL | 485 (172 / 313) | [ADD LINK] |
| ORIGA-light | 650 (168 / 482) | [ADD LINK] |
| G1020 (test-only) | 1020 (296 / 724) | [ADD LINK] |

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

## 5. How to reproduce the tables

### Fixed settings (identical for all models)

| Setting | Value |
|---|---|
| Input size | 224 × 224 RGB, ImageNet mean/std normalization |
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

On a free Colab **T4** GPU, one training run takes about **2 to 6 minutes**. One seed (15 training runs) takes about **1 to 1.5 hours**, so the full experiment with three seeds takes roughly **3 to 4.5 hours**. Keep the Colab tab open.

If you see `CUDA out of memory`, restart the session and set `BATCH = 16` for all models. If reading from Drive is slow, copy `data/` to the local Colab disk and rebuild `master.csv`.

### Outputs (saved in `results/`)

| File | Contents |
|---|---|
| `primary_3x3_auc_mean_sd.csv` | AUC mean ± SD per model and train/test cell (source of the 3×3 matrices) |
| `primary_model_summary_mean_sd.csv` | in-domain AUC, cross-source AUC and AUC drop per model (paper Table II) |
| `primary_sens_spec_by_model_mean_sd.csv` | sensitivity, specificity, accuracy per model (paper Table IV) |
| `primary_sens_spec_mean_sd.csv` | all metrics per train/test cell |
| `external_G1020_mean_sd.csv`, `external_G1020_drop_mean_sd.csv` | G1020 external results (paper Table V) |
| `primary_auc_drop_per_seed.csv`, `external_G1020_drop_per_seed.csv` | seed-level values behind the drops |
| `primary_auc_mean_heatmaps.png`, `primary_auc_sd_heatmaps.png` | 3×3 AUC heatmaps (paper Fig. 2) |
| `primary_in_vs_cross.png` | in-domain vs cross-source bar chart (paper Fig. 3) |
| `gradcam.png` | Grad-CAM example maps (paper Fig. 5, optional) |

### How results are aggregated

For each model and seed, the in-domain AUC is the mean of the 3 diagonal cells of the 3×3 matrix and the cross-source AUC is the mean of the 6 off-diagonal cells. The AUC drop is computed **per seed first**; only then are the mean and sample SD (n − 1) taken over the three seeds. G1020 is reported separately and is never part of the 3×3 averages.

> GPU computations are not perfectly deterministic, so small differences from the paper's numbers (in the third decimal place or so) are normal.

---

## 6. Expected results (for checking your run)

Model summary from the paper (mean ± SD over three seeds):

| Model | In-domain AUC | Cross-source AUC | AUC drop |
|---|---|---|---|
| ResNet50 | 0.914 ± 0.006 | 0.677 ± 0.025 | 0.237 ± 0.019 |
| DenseNet121 | 0.919 ± 0.008 | 0.710 ± 0.030 | 0.209 ± 0.022 |
| MobileNetV2 | 0.932 ± 0.027 | 0.679 ± 0.016 | 0.253 ± 0.027 |
| EfficientNet-B0 | 0.936 ± 0.023 | 0.731 ± 0.007 | 0.204 ± 0.029 |
| InceptionV3 | 0.902 ± 0.035 | 0.709 ± 0.005 | 0.193 ± 0.040 |
| All models | 0.921 ± 0.016 | 0.701 ± 0.006 | 0.219 ± 0.015 |

---

## 7. Additional analyses in the paper

- **Image cropping:** [ADD: describe exactly how ORIGA and G1020 images were cropped around the optic disc and saved at 512 × 512, and add the script to `scripts/`.]
- **G1020 label check (vCDR):** the vertical cup-to-disc ratio computed from the G1020 disc and cup masks (paper Section III-F, Fig. 4). [ADD: script or notebook in `scripts/`.]
- **Grad-CAM:** step 10 of the notebook (ResNet50 trained on ACRIMA, last residual block, eight test images). Qualitative only.

---

## 8. Limitations

- Only three source datasets; the ORIGA and RIM-ONE DL test splits contain about 25 glaucoma images each, so in-domain estimates are noisy.
- Three seeds give rough SDs and wide confidence intervals; pairwise model comparisons are exploratory.
- No patient identifiers are available, so images of both eyes of one patient may appear in both the training and test parts of the same dataset. This can inflate the in-domain AUC but not the cross-source AUC.
- Labeling criteria differ between datasets and were not harmonized. Hyper-parameters were fixed, not tuned per model.

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

Contact: [YOUR NAME] ([YOUR EMAIL]) · ORCID: [YOUR ORCID] · [GitHub profile link]
