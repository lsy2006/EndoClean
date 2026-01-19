# EndoClean

**EndoClean** is a fully automated deep learning framework for **Boston Bowel Preparation Scale (BBPS)** scoring from full-length colonoscopy videos.

This repository accompanies the manuscript:

> *An Artificial Intelligence System for Colonoscopy: Evaluation of a New Approach for the Boston Bowel Preparation Scale*

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
├── model/
│   ├──BBPS_90.pth
│   ├──IS.pth
│   └──VJacc9462.pyh
├── ile_detection.py        # Starting point detection
├── infering.py             # BBPS-scoring and colon segmentation
├── merge.py                # Serialize the results
├── summary.py              # Summarize the colon preparation
├── pipeline.py
└── README.md