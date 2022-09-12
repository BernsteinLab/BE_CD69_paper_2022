import time
import os
import subprocess
import sys
import re
import argparse
import collections
import gzip
import math
import shutil
#import matplotlib.pyplot as plt
import wandb
import numpy as np
import time
from datetime import datetime
import random

#import seaborn as sns
#%matplotlib inline
#import logging
#from silence_tensorflow import silence_tensorflow

import tensorflow as tf
import sonnet as snt

import tensorflow.experimental.numpy as tnp
import tensorflow_addons as tfa
from tensorflow import strings as tfs
from tensorflow.keras import mixed_precision
from scipy.stats.stats import pearsonr  
from scipy.stats.stats import spearmanr  

os.environ['XLA_FLAGS']='--xla_dump_to=/tmp/generated'
os.environ['TF_XLA_FLAGS']='--tf_xla_auto_jit=2'

from scipy import stats

import enformer as enformer
