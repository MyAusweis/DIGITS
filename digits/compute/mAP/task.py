# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os.path
import re
import sys

import digits
from digits.task import Task
from digits.utils import subclass, override

@subclass
class ComputeMAPTask(Task):
    """
    A task to compute the mAP of a model
    """

    def __init__(self, model, val, snapshot, **kwargs):
        """
        Arguments:
        model      -- path to model prototxt
        val        -- path to validation DB
        snapshot   -- path to model snapshot
        """
        # memorize parameters
        self.model = model
        self.val = val
        self.snapshot = snapshot

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

        timestamp, level, message = self.preprocess_output_digits(line)
        if not message:
            return False

        match = re.match(r'MAP = %s' % float_exp, message, flags=re.IGNORECASE)
        if match:
            value = float(match.group(1))
            self.mAP = value
            return True

        return False

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
    def task_arguments(self, resources, env):

        args = [sys.executable,
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(digits.__file__))), 'tools', 'calculate-map.py'),
            self.model,
            self.val,
            self.snapshot
            ]

        if self.gpu is not None:
            args.append('--gpu=%d' % self.gpu)

        return args


