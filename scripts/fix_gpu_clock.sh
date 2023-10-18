#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <clock_mhz>"
    exit 1
fi

clock=$1

nvidia-smi -i 0 -pm 1
nvidia-smi -i 0 --lock-gpu-clocks=$clock
