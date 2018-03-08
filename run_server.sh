#!/bin/bash

docker pull djpetti/docker-keras

nvidia-docker run --rm -v `pwd`:/server_dir -p 6219:6219 \
    docker-keras /usr/bin/python /server_dir/server_main.py
