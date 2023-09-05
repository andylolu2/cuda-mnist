#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <dist_dir>"
    exit 1
elif [ -d $1 ]; then
    echo "Error: $1 already exists"
    exit 1
fi 

DIST_DIR=$1
mkdir -p ${DIST_DIR}

wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -O ${DIST_DIR}/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -O ${DIST_DIR}/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -O ${DIST_DIR}/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -O ${DIST_DIR}/t10k-labels-idx1-ubyte.gz

cd ${DIST_DIR}

gzip -d train-images-idx3-ubyte.gz
gzip -d train-labels-idx1-ubyte.gz
gzip -d t10k-images-idx3-ubyte.gz
gzip -d t10k-labels-idx1-ubyte.gz