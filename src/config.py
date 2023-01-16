from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


from easydict import EasyDict as edict


config = edict()
config.slice_stack = 1
config.num_classes = 3
config.CutWindow = [-150, 350]
config.batch_size = 4