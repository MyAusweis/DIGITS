#!/bin/bash
# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
set -e

LOCAL_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
ROOT_DIR=$( dirname "$(dirname "$LOCAL_DIR")")

set -x

export CAFFE_ROOT=~/caffe
if $TEST_TORCH; then
    export TORCH_ROOT=~/torch
fi
# Disable OpenMP multi-threading
export OMP_NUM_THREADS=1
export OPENBLAS_MAIN_FREE=1

cd $ROOT_DIR
./digits-test -v

