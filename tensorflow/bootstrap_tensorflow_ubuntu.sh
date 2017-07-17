#!/bin/bash

# This bootstrap script will install tensorflow (without GPU support) on
# Ubuntu Server 16.04 running on AWS EC2.  This was tested on a t2.micro
# so YMMV.

apt-get upgrade && apt-get update

apt-get install -y python3-pip python3-dev

# NOTE: These alias commands don't work in bootstrap script??
alias python=python3
alias pip=pip3


# NOTE:  Put a check in here to determine if we've got an NVDIA GPU, if so,
# install tensorflow-gpu, else, install tensorflow

pip3 install tensorflow
