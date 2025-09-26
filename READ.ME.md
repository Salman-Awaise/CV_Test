### &nbsp;		Color Constancy CNN (Jupyter Notebook)



##### Overview:



This repository contains a Jupyter Notebook implementation of a compact CNN for color constancy (estimating the scene illuminant and enabling white-balance correction). The approach is inspired by Bianco et al. (CVPR 2015), but trains image-level (not patch-pooled) with an angular/cosine loss and evaluates against Gray-World and White-Patch-99 baselines.





##### Dataset:



\- SimpleCube++ (train/test images + per‑image ground‑truth illuminants).

\- Expected layout (can be adapted in the notebook):

&nbsp; SIMPLECUBE++/

&nbsp;   train/PNG/ , train/gt.csv

&nbsp;   test/PNG/  , test/gt.csv





##### Required libraries:



from IPython import get\_ipython

import IPython.display as ipd

import PIL.Image as PILImage

import torch.optim as optim

from torch.optim.lr\_scheduler import ReduceLROnPlateau

import os, glob, csv, random, re

from typing import Tuple, Dict

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random\_split

from torchvision import transforms





##### How to run:



1\) Open Jupyter and load the notebook: `lol 1.ipynb`.

2\) Set the dataset path (either set env var SIMPLECUBE\_ROOT or edit the path cell).

3\) Run cells top‑to‑bottom: data → loaders → model → training → evaluation → baselines.

4\) Outputs are written to ./results/





##### Key outputs:



\- illum\_cnn\_best.pt (best model weights)

\- preds\_test.csv (per‑image predictions + angular error)

\- loss\_curve.png , angle\_curve.png

\- angle\_hist.png , angle\_cdf.png

\- baselines\_bar.png , baselines\_cdf.png , baselines\_overlay.png

\- wb\_preview\_\*.png , grids\_\*.png



##### Notes:



\- GPU recommended but not required.

\- The notebook is image‑level (not patch‑pooled).

