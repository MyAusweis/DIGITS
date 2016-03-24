# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from .tasks import ObjectDetectionInferenceTask

import digits.frameworks
from digits.config import config_value
from digits.task import Task
from digits.job import Job
from digits.utils import subclass, override

@subclass
class InferenceJob(Job):
    """
    A Job that exercises the forward pass of a neural network
    """

    def __init__(self, model, images, epoch, layers, **kwargs):
        """
        Arguments:
        model   -- job object associated with model to perform inference on
        images  -- list of image paths to perform inference on
        epoch   -- epoch of model snapshot to use
        layers  -- layers to import ('all' or 'none')
        """
        super(InferenceJob, self).__init__(**kwargs)

        # get handle to framework object
        fw_id = model.train_task().framework_id
        fw = digits.frameworks.get_framework_by_id(fw_id)

        if model.dataset.is_detectnet() and config_value('digits_detector_root'):
            # create object detection inference task
            # (only supported with Caffe)
            self.tasks.append(ObjectDetectionInferenceTask(
                job_dir   = self.dir(),
                model     = model,
                images    = images,
                epoch     = epoch))
            if layers and layers != 'none':
                # Create a separate task for calculating visualizations
                self.using_separate_task_for_vis = True
                self.tasks.append(fw.create_inference_task(
                    job_dir   = self.dir(),
                    model     = model,
                    images    = images,
                    epoch     = epoch,
                    layers    = layers))

        else:
            # create inference task
            self.tasks.append(fw.create_inference_task(
                job_dir   = self.dir(),
                model     = model,
                images    = images,
                epoch     = epoch,
                layers    = layers))

    @override
    def __getstate__(self):
        fields_to_save = ['_id', '_name']
        full_state = super(InferenceJob, self).__getstate__()
        state_to_save = {}
        for field in fields_to_save:
            state_to_save[field] = full_state[field]
        return state_to_save

    def inference_task(self):
        """Return the first and only Task for this job"""
        return [t for t in self.tasks if isinstance(t, Task)][0]

    @override
    def __setstate__(self, state):
        super(InferenceJob, self).__setstate__(state)

    def get_data(self):
        """Return inference data"""
        if hasattr(self, 'using_separate_task_for_vis') and self.using_separate_task_for_vis:
            # XXX GTC Demo
            inference_task = self.tasks[0]
            vis_task = self.tasks[1]
            return (
                inference_task.inference_inputs,
                inference_task.inference_outputs,
                vis_task.inference_layers)

        task = self.inference_task()
        return task.inference_inputs, task.inference_outputs, task.inference_layers

    @override
    def is_read_only(self):
        """
        Returns True if this job cannot be edited
        """
        return True

