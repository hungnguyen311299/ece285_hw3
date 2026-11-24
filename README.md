# ECE 285 – Homework 3

A diffusion model trained on the 5,179 histopathology images from the Kaggle Scientific Image Classification Dataset, at 64×64.

## Data

Download [`rushilprajapati/data-final`](https://www.kaggle.com/datasets/rushilprajapati/data-final) from Kaggle and keep only the `Histopathology` folder. The result should be:

```
data/
└── Histopathology/
    ├── image_0.png
    ├── image_0 (2).png
    └── ...              (5,179 PNG files, 256×256)
```

## Setup

Python 3.10+ with a CUDA GPU. Run from the repository root:

```bash
pip install -r requirements.txt
python main.py      # train for 200 epochs and select the checkpoint by validation FID
python figures.py   # report figures (run after main.py)