import gc
import os
from pathlib import Path
import random
import sys

from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy as sp

import matplotlib.pyplot as plt
import seaborn as sns

#plotly
from plotly import tools, subplots
import plotly.offline as py
import plotly.grapth_objs as go
import plotly.express as px
import plotly.figure_factory as ff
import plotly.io as pio
pio.templates.default = "plotly_dark"

#models
from sklearn import preprocessing
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

#setup
pd.set_option('max_columns', 50)

