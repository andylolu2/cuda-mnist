#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <dist_dir>"
    exit 1
fi 

DIST_DIR=$1
mkdir -p ${DIST_DIR}

for file in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz  t10k-labels-idx1-ubyte.gz; do 
    link=https://web.archive.org/web/20220331130319/https://yann.lecun.com/exdb/mnist/${file}
    wget ${link} --no-check-certificate -O ${DIST_DIR}/${file}
    gzip -d ${DIST_DIR}/${file}
done
