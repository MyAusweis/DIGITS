# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os

from . import config_option
from . import prompt

class DigitsDetectorOption(config_option.Option):
    @staticmethod
    def config_file_key():
        return 'digits_detector_root'

    @classmethod
    def prompt_title(cls):
        return 'DigitsDetector'

    @classmethod
    def prompt_message(cls):
        return 'Where is the Digits Detector project installed?'

    def optional(self):
        return False

    def suggestions(self):
        suggestions = []
        return suggestions

    @staticmethod
    def is_path():
        return True

    @classmethod
    def validate(cls, value):
        if not value:
            return value

        # Find the infer.py script
        value = os.path.abspath(value)
        if not os.path.isdir(value):
            raise config_option.BadValue('"%s" is not a directory' % value)
        expected_path = os.path.join(value, 'scripts', 'infer.py')
        if not os.path.exists(expected_path):
            raise config_option.BadValue('scripts/infer.py not found at "%s"' % value)
        return value

    @classmethod
    def validate_version(cls, executable):
        """
        Utility for checking the caffe version from within validate()
        Throws BadValue

        Arguments:
        executable -- path to a caffe executable
        """
        # Currently DIGITS don't have any restrictions on version, so no need to implement this.
        pass

    def apply(self):
        pass
