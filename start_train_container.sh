#!/bin/bash

docker pull djpetti/isl-gazecapture

nvidia-docker run -ti --rm --net=host \
  -v `pwd`:/root/isl_gazecapture \
  -v /var/training_data/:/training_data djpetti/isl-gazecapture \
  /bin/bash
