#!/usr/bin/env python2
# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.

import argparse
import logging
import os
import random
import sys

# Add path for DIGITS package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import digits.config
digits.config.load_config()
from digits import utils, log

logger = logging.getLogger('digits.tools.inference')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Calculate mAP tool - DIGITS')

    ### Positional arguments

    parser.add_argument('model',
            help='Path to model prototxt file')
    parser.add_argument('val',
            help='Path to validation DB')
    parser.add_argument('snapshot',
            help='Path to model snapshot')

    parser.add_argument('-g', '--gpu',
            type=int,
            default=None,
            help='GPU to use (as in nvidia-smi output, default: None)',
            )

    args = vars(parser.parse_args())

    logger.info('Model: %s', args["model"])
    logger.info('Val DB: %s', args["val"])
    logger.info('Snapshot: %s', args["snapshot"])

    # do something here
    mAP = random.random()

    logger.info('MAP = %.2f' % mAP)



