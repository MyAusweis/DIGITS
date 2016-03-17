# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os.path
import re
import shutil
import sys
import tempfile

import digits
from digits.task import Task
from digits.utils import subclass, override

PATH_TO_INFER_SCRIPT = '/home/lyeager/code/dlar/digits-detector/scripts/infer.py'

@subclass
class ComputeMAPTask(Task):
    """
    A task to compute the mAP of a model
    """

    def __init__(self, network, weights, val_dir, **kwargs):
        """
        Arguments:
        network -- path to model prototxt
        weights -- path to model snapshot
        val_dir -- path to validation directory
        """
        # memorize parameters
        self.network = network
        self.weights = weights
        self.val_dir = val_dir

        # resources
        self.gpu = None

        # output data
        self.mAP = None

        super(ComputeMAPTask, self).__init__(**kwargs)

    @override
    def name(self):
        return 'Compute mAP'

    @override
    def process_output(self, line):
        float_exp = '(NaN|[-+]?[0-9]*\.?[0-9]+(e[-+]?[0-9]+)?)'

        match = re.match(r'.*P-R Moderate mAP %s' % float_exp, line, flags=re.IGNORECASE)
        if match:
            value = float(match.group(1))
            self.mAP = value

        print line
        return True

    @override
    def offer_resources(self, resources):
        reserved_resources = {}
        # we need one CPU resource from compute_task_pool
        cpu_key = 'compute_task_pool'
        if cpu_key not in resources:
            return None
        for resource in resources[cpu_key]:
            if resource.remaining() >= 1:
                reserved_resources[cpu_key] = [(resource.identifier, 1)]
                # we reserve the first available GPU, if there are any
                gpu_key = 'gpus'
                if resources[gpu_key]:
                    for resource in resources[gpu_key]:
                        if resource.remaining() >= 1:
                            self.gpu = int(resource.identifier)
                            reserved_resources[gpu_key] = [(resource.identifier, 1)]
                            break
                return reserved_resources
        return None

    @override
    def before_run(self):
        self.tempdir = tempfile.mkdtemp()

    @override
    def task_arguments(self, resources, env):
        args = [sys.executable,
                PATH_TO_INFER_SCRIPT,
                '--val-path', self.val_dir,
                '--model-def', self.network,
                '--weights', self.weights,
                '--results-dir', self.tempdir,
                ]

        return args

    @override
    def task_environment(self, resources):
        if self.gpu is not None:
            return {'CUDA_VISIBLE_DEVICES': str(self.gpu)}
        else:
            return {}

    @override
    def after_run(self):
        shutil.rmtree(self.tempdir)
