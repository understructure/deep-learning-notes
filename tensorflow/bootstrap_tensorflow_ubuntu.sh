#!/bin/bash

# This bootstrap script will install tensorflow (without GPU support) on
# Ubuntu Server 16.04 running on AWS EC2.  This was tested on a t2.micro
# so YMMV.

apt-get upgrade && apt-get update

apt-get install -y python3-pip python3-dev

# NOTE: These alias commands don't work in bootstrap script??
echo "alias python=python3" >> /home/ubuntu/.profile
echo "alias pip=pip3" >> /home/ubuntu/.profile

# NOTE:  Put a check in here to determine if we've got an NVDIA GPU, if so,
# install tensorflow-gpu, else, install tensorflow

pip3 install tensorflow
pip3 install pandas
pip3 install matplotlib seaborn

apt-get install -y python3-tk

apt-get install -y awscli

aws s3 cp s3://ustruct-code/tensorflow/trainer.py .
aws s3 cp s3://ustruct-code/tensorflow/distributed_tf.sh .

chmod a+x distributed_tf.sh
chmod +x trainer.py


