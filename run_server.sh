#!/bin/bash

docker pull djpetti/isl-gazecapture

nvidia-docker run --rm -v `pwd`:/server_dir -p 6219:6219 \
    djpetti/isl-gazecapture bash -c \
    "cd /server_dir && /usr/bin/python server_main.py $1"
