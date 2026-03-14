# EndoClean

**EndoClean** is a fully automated deep learning framework for **Boston Bowel Preparation Scale (BBPS)** scoring from full-length colonoscopy videos.

This repository accompanies the manuscript:

> [*EndoClean: A Hybrid Deep Learning Framework for Automated Full-Video Boston Bowel Preparation Scale Assessment*](https://www.mdpi.com/2306-5354/13/3/294)

---

## Features

- Fully automated BBPS scoring from colonoscopy videos
- Video-level assessment (no manual frame or segment selection)
- Segment-wise (right / transverse / left colon) and total BBPS scores
- Expert-level agreement with senior endoscopists
- Conservative scoring to reduce overestimation in ambiguous cases

---

## Pipeline

EndoClean consists of three main components:

1. **Frame Selection**
   - Filters non-diagnostic frames (blur, stool obstruction, poor illumination)

2. **Anatomical Segmentation**
   - CNN-based frame classification
   - Hidden Markov Model (HMM) for temporal consistency
   - Automatic detection of BBPS segment transition points

3. **BBPS Scoring & Aggregation**
   - Frame-level BBPS score prediction (0–3)
   - Rule-based aggregation within anatomical segments
   - Final segmental and total BBPS scores

---

## Output

For each colonoscopy video, EndoClean outputs:

- BBPS score for:
  - Right colon
  - Transverse colon
  - Left colon
- Total BBPS score
---

## Repository Structure

```text
EndoClean/
├── documents/
│   ├──emission_map.csv
│   └──transfer_map.csv
├── ile_detection.py        # Starting point detection
├── infering.py             # BBPS-scoring and colon segmentation
├── merge.py                # Serialize the results
├── summary.py              # Summarize the colon preparation
├── pipeline.py
└── README.md
```
---

## Data Availability

Clinical colonoscopy videos and images are **not included** due to patient privacy and institutional regulations.
Pretrained model [checkpoints](https://drive.google.com/drive/folders/1rdxYLJZz-Dk6O62yg_qPEBH6g3jbWwvw?usp=sharing) are available.
---

## Citation

If you use this repository in your research, please cite the corresponding [paper](https://www.mdpi.com/2306-5354/13/3/294):
```
@article{zhu2026endoclean,
  title={EndoClean: A Hybrid Deep Learning Framework for Automated Full-Video Boston Bowel Preparation Scale Assessment},
  author={Zhu, Yan and Li, Si-Yuan and Fu, Pei-Yao and Zhang, Zhen and Wang, Shuo and Li, Quan-Lin and Zhou, Ping-Hong},
  journal={Bioengineering},
  volume={13},
  number={3},
  pages={294},
  year={2026},
  publisher={MDPI}
}
```
