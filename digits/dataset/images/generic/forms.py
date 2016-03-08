# Copyright (c) 2015-2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os
import os.path

import wtforms
from wtforms import validators

from ..forms import ImageDatasetForm
from digits import utils
from digits.utils.forms import validate_required_iff

class GenericImageDatasetForm(ImageDatasetForm):
    """
    Defines the form used to create a new GenericImageDatasetJob
    """

    # Use a SelectField instead of a HiddenField so that the default value
    # is used when nothing is provided (through the REST API)
    method = wtforms.SelectField(u'Dataset type',
            choices = [
                ('prebuilt', 'Prebuilt'),
                ],
            default='prebuilt',
            )

    def validate_lmdb_path(form, field):
        if not field.data:
            pass
        else:
            # make sure the filesystem path exists
            if not os.path.exists(field.data) or not os.path.isdir(field.data):
                raise validators.ValidationError('Folder does not exist')

    def validate_file_path(form, field):
        if not field.data:
            pass
        else:
            # make sure the filesystem path exists
            if not os.path.exists(field.data) or not os.path.isfile(field.data):
                raise validators.ValidationError('File does not exist')

    ### Method - prebuilt

    prebuilt_train_images = wtforms.StringField('Training Images',
            validators=[
                validate_required_iff(method='prebuilt'),
                validate_lmdb_path,
                ]
            )
    prebuilt_train_labels = wtforms.StringField('Training Labels',
            validators=[
                validate_lmdb_path,
                ]
            )
    prebuilt_val_images = wtforms.StringField('Validation Images',
            validators=[
                validate_lmdb_path,
                ]
            )
    prebuilt_val_labels = wtforms.StringField('Validation Labels',
            validators=[
                validate_lmdb_path,
                ]
            )

    # Can't use a BooleanField here because HTML doesn't submit anything
    # for an unchecked checkbox. Since we want to use a REST API and have
    # this default to True when nothing is supplied, we have to use a
    # SelectField
    force_same_shape = utils.forms.SelectField('Enforce same shape',
            choices = [
                (1, 'Yes'),
                (0, 'No'),
                ],
            coerce = int,
            default = 1,
            tooltip = 'Check that each entry in the database has the same shape (can be time-consuming)'
            )

    prebuilt_mean_file = utils.forms.StringField('Mean Image',
            validators=[
                validate_file_path,
                ],
            tooltip = "Path to a .binaryproto file on the server"
            )

    # XXX GTC demo

    is_drivenet_data = utils.forms.BooleanField(
        'DriveNet data',
        tooltip='Check if this data should be flagged for use with DriveNet',
    )
    drivenet_val_labels_dir = utils.forms.StringField(
        'Validation labels directory',
        validators=[
            validate_required_iff(is_drivenet_data=True),
        ],
        tooltip='The original KITTI-formatted label textfiles are needed for mAP calculations',
    )
    def validate_drivenet_val_labels_dir(form, field):
        if not field.data:
            pass
        else:
            # make sure the filesystem path exists
            if not os.path.exists(field.data) or not os.path.isdir(field.data):
                raise validators.ValidationError('Directory does not exist')

            # Cryptic command to get the last part of the path
            # Necessary in case there's a trailing slash
            last_part = os.path.basename(os.path.dirname(os.path.join(field.data, '')))
            if last_part != 'label_2':
                if 'label_2' in os.listdir(field.data):
                    field.data = os.path.join(field.data, 'label_2')
                else:
                    raise validators.ValidationError("Can't find label2/ directory nearby")

