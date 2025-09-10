# Neural Lyric Transcriber

Lyrics transcription from **clean vocal stems** using deep learning.  
A curriculum learning extension to gradually add background music was planned but **not included** in this submission.

<p align="left">
  <a href="https://colab.research.google.com/github/gideon-szamet/neural-lyric-transcriber/blob/main/notebooks/01_Data_preparations.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open 01_Data_preparations in Colab"> 01_Data_preparations
  </a><br>
  <a href="https://colab.research.google.com/github/gideon-szamet/neural-lyric-transcriber/blob/main/notebooks/02_First_Model_CNN_BiLSTM_CTC.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open 02_First_Model_CNN_BiLSTM_CTC in Colab"> 02_First_Model_CNN_BiLSTM_CTC
  </a><br>
  <a href="https://colab.research.google.com/github/gideon-szamet/neural-lyric-transcriber/blob/main/notebooks/03_Second_Model_Transformer_CTC.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open 03_Second_Model_Transformer_CTC in Colab"> 03_Second_Model_Transformer_CTC
  </a><br>
  <a href="https://colab.research.google.com/github/gideon-szamet/neural-lyric-transcriber/blob/main/notebooks/04_Third_Model_Whisper_finetune.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open 04_Third_Model_Whisper_finetune in Colab"> 04_Third_Model_Whisper_finetune
  </a>
</p>


## Overview
This project explores three successive approaches to lyric transcription from clean vocal stems.
Each stage reflects lessons learned from prior limitations:

1) **CNN–BiLSTM with CTC** on log-Mel spectrograms — quickly overfitted, unable to generalize.  
2) **Transformer CTC (Wav2Vec2)** pipeline — underfitted despite augmentation, failed to converge meaningfully.  
3) **OpenAI Whisper fine-tuning** — stable and effective, producing the final working model.

A curriculum extension to progressively include mixed audio was planned but not executed within the project scope.

## Features
- Whisper fine-tuning with evaluation (WER/CER via `jiwer`) on **vocal stems**
- CNN–BiLSTM–CTC and Wav2Vec2–CTC exploration with training/eval code
- Reproducible notebooks under `notebooks/`
- GPU-friendly; works in Google Colab or locally

## Reports

[![View PDF](https://img.shields.io/badge/View-PDF-blue)](reports/DL_Final%20Project%20Summary_Gideon_Szamet.pdf)
[![View PPTX](https://img.shields.io/badge/View-PPTX-orange)](reports/Final_DL_project_Gideon_Szamet.pptx)



## Project Structure
```
neural-lyric-transcriber/
├─ notebooks/
│  ├─ 01_Data_preperations.ipynb
│  ├─ 02_First_Model_CNN_BiLSTM_CTC.ipynb
│  ├─ 03_Second_Model_Transformer_CTC.ipynb
│  └─ 04_Third_Model_Whisper_finetune.ipynb
├─ reports/
│  ├─ DL_Final Project Summary_Gideon_Szamet.pdf
│  └─ Final_DL_project_Gideon_Szamet.pptx
├─ data/
│  └─ processed/           # CSV metadata tracked in Git
├─ requirements.txt
├─ .gitignore
└─ README.md
```

## Data
This repository does **not** include the raw MUSDB18 audio, due to size and licensing restrictions.  
Instead, it provides the **processed metadata (CSVs)** needed to reproduce experiments and run the notebooks.

- **Dataset**: MUSDB18 (150 multitrack songs on Zenodo) 
- **Lyrics extension**: MUSDB18 aligned lyrics (time-aligned annotations on Zenodo)

### Dataset access
MUSDB18 consists of 150 full tracks (~10h) with stems (vocals, bass, drums, other).

- Official info: https://sigsep.github.io/datasets/musdb.html  
- MUSDB18 (original, access request on Zenodo): https://zenodo.org/records/1117372  
- MUSDB18-HQ (uncompressed WAV on Zenodo): https://zenodo.org/records/3338373  
- MUSDB18 lyrics extension (aligned, on Zenodo): https://zenodo.org/records/3989267

Notes:
1. This project originally used the MUSDB18 tracks; it can also be tested on MUSDB18-HQ.
2. You must obtain the dataset yourself and accept its license/terms. This repo never stores audio or model weights; the notebooks read MUSDB18 from your local filesystem (or Google Drive).


### Local setup
1. Download MUSDB18 and the aligned lyrics extension from Zenodo.
2. Place them under `data/raw/` in your local clone (e.g., `data/raw/train/`, `data/raw/test/`, `data/raw/lyrics_extension/`).
3. Processed CSVs with segment-level metadata (e.g., `train_segments_chunked.csv`, `test_segments_chunked.csv`) are already included in `data/processed/`.

### Reproducing preprocessing
Run `notebooks/01_Data_preperations.ipynb` to:
- Parse MUSDB18 metadata and aligned lyrics
- Split audio into lyric-aligned chunks
- Apply augmentation (pitch shift, time stretch)
- Generate processed CSVs

This setup ensures reproducibility without redistributing large audio files.



## Quickstart
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```


Open the notebooks in notebooks/ and follow the cells.
If using Colab, mount Drive and point DATA_ROOT (or equivalent) to your data/raw/ folder.


---

Optional local folders you may create (ignored by Git): 
```
data/ # manifests, small CSVs you generate (no raw MUSDB18 here)
checkpoints/ # saved model weights
outputs/ # predictions, metrics, plots
logs/ # training logs
hf_cache/ # huggingface cache (alternative to ~/.cache/huggingface)
```

