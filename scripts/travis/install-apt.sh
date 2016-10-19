#!/bin/bash
# Copyright (c) 2015-2016, NVIDIA CORPORATION.  All rights reserved.
set -e
set -x

sudo apt-get update -q

if $RUN_TESTS; then
    sudo apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        cython \
        git \
        graphviz \
        libboost-filesystem-dev \
        libboost-python-dev \
        libboost-system-dev \
        libboost-thread-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopenblas-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python-dev \
        python-flask \
        python-gevent \
        python-gevent-websocket \
        python-gflags \
        python-h5py \
        python-matplotlib \
        python-mock \
        python-nose \
        python-numpy \
        python-opencv \
        python-pil \
        python-pip \
        python-protobuf \
        python-psutil \
        python-pydot \
        python-requests \
        python-scipy \
        python-six \
        python-skimage

elif $LINT_CHECK; then
    sudo apt-get install -y --no-install-recommends \
        python-flake8

elif $DEB_BUILD; then
    sudo apt-get install -y --no-install-recommends \
        debhelper \
        devscripts \
        dput
fi
