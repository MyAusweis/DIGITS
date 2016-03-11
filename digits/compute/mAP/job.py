# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from .task import ComputeMAPTask
from digits.job import Job
from digits.utils import subclass, override

@subclass
class ComputeMAPJob(Job):
    """
    A Job that computes the MAP of an object detection model snapshot
    """

    def __init__(self, model, val, snapshot, **kwargs):
        """
        Arguments:
        model      -- path to model prototxt
        val        -- path to validation DB
        snapshot   -- path to model snapshot
        """
        super(ComputeMAPJob, self).__init__(username    = "compute_agent",
                                            name        = "Compute mAP",
                                            **kwargs)
        # create mAP task
        self.tasks.append(ComputeMAPTask(model = model, val = val, snapshot = snapshot, job_dir = self.dir()))

    def get_data(self):
        """Return mAP"""
        task = self.tasks[0]
        return task.mAP
