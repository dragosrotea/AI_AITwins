# DriveMate dataset inventory

Store local dataset copies under `data/raw/`; those files are excluded from git. Before a training run, inspect the downloaded folder structure and a few filenames because layouts vary.

| Purpose | Source | Licence or terms | Local status and size | Download method |
|---|---|---|---:|---|
| Eye-state training | [MRL Eye Dataset](http://mrl.cs.vsb.cz/eyedataset) | Follow the terms published by MRL; verify permitted use before training | Not present; the legacy URL currently redirects to the MRL homepage | Manual download after reviewing the official page (`mrl`) |
| Yawn training/evaluation | [YawDD](https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset) | IEEE DataPort terms; the dataset page describes research use and restricts public screenshots based on participant permissions | Not present in this checkout | Free IEEE DataPort account and manual download (`yawdd`) |
| Expression training | [FER2013](https://www.kaggle.com/datasets/msambare/fer2013) | Verify the dataset licence and Kaggle terms at download time | Not present in this checkout | Kaggle CLI after account and licence review (`fer2013`) |
| Distraction/drowsiness evaluation | [DMD](https://dmd.vicomtech.org/#Download) | Creative Commons Attribution-NonCommercial-NoDerivatives 4.0; approved access terms apply | Present locally; approximately 5.5 GiB | Manual access request for new copies (`dmd`) |
| Age/face experiments | [Age Detection - Human Faces 18-60](https://www.kaggle.com/datasets/trainingdatapro/age-detection-human-faces-18-60-years) | CC BY-NC-ND 4.0 according to Kaggle metadata | 357,965,008 bytes (metadata) | Kaggle CLI (`age_faces`) |
| Distracted-driving visual data | [SynDD1 / PMC9730022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9730022/) | Dataset licence agreement PDF supplied with the data; review before use | Varies by subset; not recorded here until downloaded | Manual Mendeley download (`syndd1`) |

## Access and integrity rules

- Do not commit datasets or model files larger than 50 MB.
- Do not store camera frames, recordings, or labels outside local training data; delete teammate validation recordings after CV-T11.
- Do not train or report results as calibrated for an individual. Split all subject/person-labelled data by person.
- The Kaggle user must accept the age-dataset terms and agree that CC BY-NC-ND 4.0 permits the planned research use.
- YawDD and DMD require manual sign-up/access. Scripts must not bypass their terms or download credentials.
- The PMC article's SynDD1 record is hosted by Mendeley: [ptcp7rp3wb/4](https://data.mendeley.com/datasets/ptcp7rp3wb/4). Download the included licence agreement before use.

Use `data/download_datasets.sh status` to check local availability. The script
downloads the MediaPipe Face Landmarker asset from Google's official model
storage and FER2013 through the Kaggle CLI after `ACCEPT_FER2013_TERMS=yes`;
it prints manual steps for MRL, YawDD, and DMD rather than automating account
or access-controlled downloads. Review FER2013 terms before accepting them.

The current checkout also has `age_detection` (approximately 342 MiB) and
`emotion_detection` (approximately 142 MiB) under `data/raw/`. The DMD copy is
approximately 5.5 GiB; inspect its exact annotation and video layout before
training because the local subset is not the complete dataset.

## Local layout

```text
data/raw/
  mrl_eye/
  yawdd/
  fer2013/
  dmd/
  age_detection/
  emotion_detection/
  syndd1/
```

Inspect these directories after download:

```bash
find data/raw -maxdepth 2 -type f | sort | head -20
find data/raw -mindepth 1 -maxdepth 1 -type d | sort
```

## Eye-state model (CV-T07)

After downloading and extracting the MRL Eye Dataset to `data/raw/mrl_eye/`,
inspect the printed subject directories and sample filenames, then train and
validate on held-out subjects:

```bash
python3 -m pip install -r training/requirements.txt
python3 -m training.eye_state.train
```

The trainer uses the filename's subject and eye-state fields for an
approximately 80/20 person-level split, uses the same grayscale 64x64
preprocessor as the live camera, exports `models/eye_state.onnx`, and prints
held-out closed-eye precision, recall, model size, and CPU inference latency.
It refuses to claim metrics when MRL data is missing or the split lacks both
eye classes.

Reproduce the eye model's held-out precision, recall, F1, and CPU latency table
without retraining:

```bash
python3 -m training.evaluate
```

The evaluator uses the same deterministic subject split and the configured
closed-eye threshold. It reports eye-state metrics only; yawn and expression
metrics are added when those later models are trained.

## Training the four-model pipeline

The current local folders are `age_detection`, `emotion_detection`, and `dmd`.
Install the requirements in `training/requirements.txt`, then run:

```bash
python3 -m pip install -r training/requirements.txt
python3 -m training.train_models
```

The trainer uses pretrained MobileNetV3-Large weights, first fitting the
classifier head with a frozen backbone and then fine-tuning the backbone at a
lower learning rate. Mild crop, rotation, flip, and lighting augmentation are
applied only to training data. It tracks validation macro precision, recall,
and F1 per class using a stratified holdout split from `train/`; the supplied
`test/` folder is used only for the final report. The public image folders do
not identify people, so these metrics are not person-independent. The trainer
saves the best validation macro-F1 checkpoint, reduces the learning rate when
validation F1 plateaus, and stops early when the score stops improving.
Use `--tasks emotion` (or any subset of the four tasks) to train without
overwriting unrelated models. Use `--init-from-dir models/trained` to fine-tune
the current bundle; the trainer checks task, class, and tensor compatibility,
and retains the starting checkpoint unless validation macro-F1 improves. Age
and emotion use their provided `train/` and
`test/` image folders. Age outputs the provided five age bands; emotion outputs
the seven directory labels. It extracts every annotated frame from each DMD
annotation's `face_camera` stream only, resizes those frames to 224x224, and
trains:

- Drowsiness as independent multi-label probabilities for each `eyes_state/`,
  `yawning/`, and `blinks/` annotation.
- Distraction as independent multi-label probabilities for each
  `driver_actions/` annotation.

The DMD validation split is chronological per recording to avoid placing
neighboring frames in both splits. The supplied DMD subset has very few
recordings, so this is not subject-independent validation and should not be
treated as a generalization estimate.

The current age folder has 150 images total (125 train, 25 test), and the
current DMD folder has one drowsiness recording and three distraction
recordings. Age and DMD metrics from these files are exploratory only and are
not sufficient evidence of reliable driver-safety performance.

By default, extracted frame cache is written to `data/cache/dmd/`, model bundles
and `training_report.json` to `models/trained/`. Both locations are excluded
from version control. Adjust training cost with `--epochs`, `--batch-size`,
`--workers`, and `--device`; the DMD frame step is every frame.

Run all four saved models on one webcam or face-camera video. The same incoming
frame is resized and passed through all four models:

```bash
python3 -m backend.cv.model_pipeline --camera 0
python3 -m backend.cv.model_pipeline --video path/to/face_camera.mp4
```

The pipeline emits newline-delimited JSON predictions to stdout. DMD models
were trained only on face-camera streams; do not substitute DMD hands, body, or
mosaic videos.
