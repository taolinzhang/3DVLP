import os
import sys
from easydict import EasyDict
from .config import CONF as CONF_BASE
import copy

CONF = copy.deepcopy(CONF_BASE)
print('Using Captioning Config')

# output
CONF.PATH.OUTPUT = os.path.join(CONF.PATH.BASE, "outputs/exp_joint")

# train
CONF.TRAIN = EasyDict()
CONF.TRAIN.MIN_DES_LEN = 5
CONF.TRAIN.MAX_DES_LEN = 36
CONF.TRAIN.SEED = 42
CONF.TRAIN.OVERLAID_THRESHOLD = 0.5
CONF.TRAIN.MIN_IOU_THRESHOLD = 0.25
CONF.TRAIN.NUM_BINS = 6

# eval
CONF.EVAL = EasyDict()
CONF.EVAL.MIN_IOU_THRESHOLD = 0.5
