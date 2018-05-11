#!/bin/bash

docker pull djpetti/isl-gazecapture

nvidia-docker run --rm -v `pwd`:/server_dir --net=host \
    djpetti/isl-gazecapture bash -c \
    "cd /server_dir && /usr/bin/python server_main.py $1"
