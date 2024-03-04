
import argparse, sys, os, io, base64, pickle, json, math, random
from packaging import version
import os.path as op
import time, errno
from easydict import EasyDict as edict
from skimage.feature import hog as hog_feature
from datetime import datetime
from tqdm import tqdm
import inspect
from fairscale.nn.misc import checkpoint_wrapper

import numpy as np
import torch as T
import torchvision as TV
import torch.distributed as DIST

import cv2
from PIL import Image

import transformers
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

from toolz.sandbox import unzip
from collections import defaultdict
from torch.utils.data import ConcatDataset
from datetime import timedelta
try:
    import wandb
    os.environ['WANDB_API_KEY'] = 'ea195323245622a43bcd1a710f1390276aba5bc5'
    WANDB_ENABLE = int(os.environ.get('WANDB_ENABLE', 1))
    print('WANDB_ENABLE:', WANDB_ENABLE)
except:
    print("failed to load wandb")
    WANDB_ENABLE = 0
