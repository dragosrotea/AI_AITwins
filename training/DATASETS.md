# DriveMate dataset inventory

Use `data/download_datasets.sh` to prepare datasets. Store all downloaded files under `data/raw/`; those files are excluded from git. Before a training run, inspect the downloaded folder structure and a few filenames because layouts vary.

| Purpose | Source | Licence or terms | Approximate size | Download method |
|---|---|---|---:|---|
| Eye-state training | [MRL Eye Dataset](http://mrl.cs.vsb.cz/eyedataset) | Subject to the MRL dataset licence/page terms; verify acceptance before use | Known to be large (not recorded here until inspected) | Manual download or approved mirror (`mrl`) |
| Yawn training/evaluation | [YawDD](https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset) | IEEE DataPort terms; a free account is required | Varies by subset; not recorded until downloaded | Manual sign-up/download (`yawdd`) |
| Expression training | [FER2013](https://www.kaggle.com/datasets/msambare/fer2013) | Verify the Kaggle dataset licence and terms at download time | Not recorded here until downloaded | Kaggle CLI (`fer2013`) |
| Distraction/drowsiness evaluation | [DMD](https://dmd.vicomtech.org/#Download) | Creative Commons Attribution-NonCommercial-NoDerivatives 4.0; approved access terms apply | Access-controlled; RGB subset availability may vary | Manual access request (`dmd`) |
| Age/face experiments | [Age Detection - Human Faces 18-60](https://www.kaggle.com/datasets/trainingdatapro/age-detection-human-faces-18-60-years) | CC BY-NC-ND 4.0 according to Kaggle metadata | 357,965,008 bytes (metadata) | Kaggle CLI (`age_faces`) |
| Distracted-driving visual data | [SynDD1 / PMC9730022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9730022/) | Dataset licence agreement PDF supplied with the data; review before use | Varies by subset; not recorded here until downloaded | Manual Mendeley download (`syndd1`) |

## Access and integrity rules

- Do not commit datasets or model files larger than 50 MB.
- Do not store camera frames, recordings, or labels outside local training data; delete teammate validation recordings after CV-T11.
- Do not train or report results as calibrated for an individual. Split all subject/person-labelled data by person.
- The Kaggle user must accept the age-dataset terms and agree that CC BY-NC-ND 4.0 permits the planned research use.
- YawDD and DMD require manual sign-up/access. Scripts must not bypass their terms or download credentials.
- The PMC article's SynDD1 record is hosted by Mendeley: [ptcp7rp3wb/4](https://data.mendeley.com/datasets/ptcp7rp3wb/4). Download the included licence agreement before use.

## Local layout

```text
data/raw/
  mrl_eye/
  yawdd/
  fer2013/
  dmd/
  age_detection_faces/
  syndd1/
```

Inspect these directories after download:

```bash
find data/raw -maxdepth 2 -type f | sort | head -20
find data/raw -mindepth 1 -maxdepth 1 -type d | sort
```
