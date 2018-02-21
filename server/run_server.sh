#!/bin/bash

docker pull djpetti/rpinets-tensorflow

nvidia-docker run --rm -v `pwd`:/server_dir \
    djpetti/rpinets-tensorflow /usr/bin/python /server_dir/main.py
