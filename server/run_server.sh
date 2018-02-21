#!/bin/bash

docker pull djpetti/isl-gazecapture

nvidia-docker run --rm -v `pwd`:/server_dir -p 6219:6219 \
    djpetti/isl-gazecapture /usr/bin/python /server_dir/main.py
