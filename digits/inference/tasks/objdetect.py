# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import csv
import glob
import os.path
import re
import shutil
import sys
import tempfile

from google.protobuf import text_format
import PIL.Image

import digits
from digits import utils
from digits.config import config_value
from digits.task import Task
from digits.utils import subclass, override
from .inference import InferenceTask

import caffe_pb2

@subclass
class ObjectDetectionInferenceTask(Task):
    """
    A task to retrieve object detections
    """

    def __init__(self, model, images, epoch, **kwargs):
        """
        Arguments:
        model  -- trained model to perform inference on
        images -- list of images to perform inference on
        epoch  -- model snapshot to use
        """
        # memorize parameters
        self.model = model
        self.images = images
        self.epoch = epoch

        # resources
        self.gpu = None

        # generated data
        self.inference_inputs = None
        self.inference_outputs = None
        self.inference_layers = []

        # Get data shape from deploy file
        tmp_network = caffe_pb2.NetParameter()
        with open(self.model.path(self.model.train_task().deploy_file)) as infile:
            text_format.Merge(infile.read(), tmp_network)
        if tmp_network.input_shape:
            self.data_shape = tmp_network.input_shape[0].dim
        else:
            self.data_shape = tmp_network.input_dim[:4]

        super(ObjectDetectionInferenceTask, self).__init__(**kwargs)

    @override
    def name(self):
        return 'Object Detection Task'

    @override
    def process_output(self, line):
        print line
        return True

    @override
    def offer_resources(self, resources):
        reserved_resources = {}
        # we need one CPU resource from compute_task_pool
        cpu_key = 'inference_task_pool'
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
        # copy all images to temp directory
        for image_path in self.images:
            image = utils.image.load_image(image_path)
            image = utils.image.resize_image(
                image,
                self.data_shape[2]-1, # height
                self.data_shape[3]-1, # width
                channels=self.data_shape[1],
                resize_mode='squash',
            )
            image = PIL.Image.fromarray(image)
            new_path = os.path.join(
                self.tempdir,
                os.path.basename(image_path),
            )
            image.save(
                new_path,
                format='PNG', # XXX regardless of filename
            )

    @override
    def task_arguments(self, resources, env):
        path_to_infer_script = os.path.join(config_value('digits_detector_root'), 'scripts', 'infer.py')
        train_task = self.model.train_task()
        deploy_file = train_task.path(train_task.deploy_file)
        # find snapshot
        snapshots = train_task.snapshots
        if not self.epoch:
            epoch = snapshots[-1][1]
            weights_file = self.snapshots[-1][0]
        else:
            for snapshot_file, snapshot_epoch in snapshots:
                if snapshot_epoch == self.epoch:
                    weights_file = snapshot_file
                    break
        if weights_file is None:
            raise Exception('snapshot not found for epoch "%s"' % epoch)
        # prepare arguments
        args = [sys.executable,
                path_to_infer_script,
                '--input-files', self.tempdir,
                '--model-def', deploy_file,
                '--weights', weights_file,
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
        detections = {}

        # get network input dimensions
        db_task = self.model.train_task().dataset.analyze_db_tasks()[0]
        height = db_task.image_height
        width = db_task.image_width
        channels = db_task.image_channels

        # parse CSV files
        csv_files = glob.glob(os.path.join(self.tempdir,'*detections*.csv'))
        for csv_file in csv_files:
            with open(csv_file) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fname = row['image_file']
                    if fname not in detections:
                        detections[fname] = []
                    pred_bb = [[float(row['pred_xl']), float(row['pred_yt'])], [float(row['pred_xr']), float(row['pred_yb'])]]
                    pred = [float(row['confidence']), pred_bb]
                    detections[fname].append(pred)

        # now return inference data in expected format
        self.inference_inputs = {'ids': [], 'data': []}
        self.inference_outputs = {'bboxes': []}
        for idx, fname in enumerate(self.images):
            self.inference_inputs['ids'].append(idx)
            image = utils.image.load_image(fname)

            image = utils.image.resize_image(
                image,
                self.data_shape[2], # height
                self.data_shape[3], # width
                channels=self.data_shape[1],
                resize_mode='squash',
            )

            base_name = os.path.basename(fname)
            if os.path.basename(base_name) in detections:
                self.inference_outputs['bboxes'].append(detections[base_name])
                # draw bounding boxes
                for bbox in detections[base_name]:
                    image = utils.image.add_bboxes_to_image(image, [bbox[1]], width=2)
            else:
                # no detection for this file
                self.inference_outputs['bboxes'].append([])
            self.inference_inputs['data'].append(image)

        shutil.rmtree(self.tempdir)
