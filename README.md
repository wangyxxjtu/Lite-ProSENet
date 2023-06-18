# Lite-ProSENet
The officie pytorch code of our paper [Multimodal Learning for Non-small Cell Lung Cancer Prognosis](https://arxiv.org/pdf/2211.03280.pdf)

## Introduction
The Illustration of PCNet:

<img src="https://github.com/wangyxxjtu/Lite-ProSENet/master/figures/framework.png" width="845" alt="workflow" />

### Overview

This paper focuses on the task of survival time analysis for lung cancer. Although much progress has been made in this problem in recent years, the performance of existing methods is still far from satisfactory. Traditional and some deep learning- based survival time analyses for lung cancer are mostly based on textual clinical information such as staging, age, histology, etc. Unlike existing methods that predicting on the single modality, we observe that a human clinician usually takes multimodal data\ such as text clinical data and visual scans to estimate survival time. Motivated by this, in this work, we contribute a smart cross-modality network for survival analysis network named Lite-ProSENet that simulates a human’s manner of decision making. To be specific, Lite-ProSENet is a two-tower architecture that takes the clinical data and the CT scans as inputs to create the survival prediction. The textural tower is responsible for modelling the clinical data, we build a light transformer using multi-head self-attention as our textural tower. The visual tower (namely ProSENet) is responsible
for extracting features from the CT scans. The backbone of ProSENet is a 3D Resnet that works together with several repeatable building block named 3D-SE Resblock for a compact feature extraction. Our 3D-SE Resblock is composed of a 3D channel “Squeeze-and-Excitation” (SE) block and a temporal SE block. The purpose of 3D-SE Resblock is to adaptively select the valuable features from CT scans. Besides, to further filter out the redundant information among CT scans, we develop a simple but effective frame difference mechanism that takes the performance of our model to the new state- of-the-art. Extensive experiments were conducted using data from 422 NSCLC patients from The Cancer Imaging Archive (TCIA). The results show that our Lite-ProSENet outperforms favorably again all comparison methods and achieves the new state of the art with the 89.3% on concordance.


## Usage
### Data Prepration
We considered 422 NSCLC patients from TCIA to assess the proposed framework, download from [here](https:)

### 🌻 Training and testing
```
sh train.sh
python test.py
```

## Citation
If you think this repo useful, please cite
```
@article{DBLP:journals/corr/abs-2211-03280,
  author       = {Yujiao Wu and
                  Yaxiong Wang and
                  Xiaoshui Huang and
                  Fan Yang and
                  Sai Ho Ling and
                  Steven Weidong Su},
  title        = {Multimodal Learning for Non-small Cell Lung Cancer Prognosis},
  journal      = {CoRR},
  volume       = {abs/2211.03280},
  year         = {2022},
  url          = {https://doi.org/10.48550/arXiv.2211.03280},
  doi          = {10.48550/arXiv.2211.03280},
  eprinttype    = {arXiv},
  eprint       = {2211.03280},
  timestamp    = {Wed, 09 Nov 2022 17:33:26 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2211-03280.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
