import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input/nvidiaapex/repository/NVIDIA-apex-39e153a"))

# Installing Nvidia Apex
! pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ../input/nvidiaapex/repository/NVIDIA-apex-39e153a

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import pkg_resources
import seaborn as sns
import time
import scipy.stats as stats
import gc
import re
import operator 
import sys
from sklearn import metrics
from sklearn import model_selection
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from nltk.stem import PorterStemmer
from sklearn.metrics import roc_auc_score
%load_ext autoreload
%autoreload 2
%matplotlib inline
from tqdm import tqdm, tqdm_notebook
import os
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import warnings
warnings.filterwarnings(action='once')
import pickle
from apex import amp
import shutil
