#!/usr/bin/python


import argparse
import logging


def _configure_logging():
  """ Configure logging handlers. """
  # Cofigure root logger.
  root = logging.getLogger()
  root.setLevel(logging.DEBUG)
  file_handler = logging.FileHandler("itracker_train.log")
  file_handler.setLevel(logging.DEBUG)
  stream_handler = logging.StreamHandler()
  stream_handler.setLevel(logging.WARNING)
  formatter = logging.Formatter("%(name)s@%(asctime)s: " + \
      "[%(levelname)s] %(message)s")

  file_handler.setFormatter(formatter)
  stream_handler.setFormatter(formatter)

  root.addHandler(file_handler)
  root.addHandler(stream_handler)

# Some modules need a logger to be configured immediately.
_configure_logging()


from six.moves import cPickle as pickle
import json
import os
import sys

import keras.applications as applications
import keras.backend as K
from keras.models import Model, load_model
import keras.layers as layers
import keras.optimizers as optimizers
import keras.regularizers as regularizers
from keras.backend.tensorflow_backend import set_session

import tensorflow as tf

import cv2

import numpy as np

from itracker.common import config, custom_data_loader
from itracker.training import pipelines


batch_size = 32
# Shape of the raw images from the dataset.
raw_shape = (400, 400, 3)

# How many batches to run between testing intervals.
train_interval = 160
# How many batches to run during testing.
test_interval = 24

# Learning rates to set.
learning_rates = [0.001, 0.0001, 0.00001]
# How many iterations to train for at each learning rate.
iterations = [100000, 100000, 100000]
# How many iterations to validate for.
valid_iters = 124

# Learning rate hyperparameters.
momentum = 0.9

# Where to save the network.
save_file = "eye_model.hd5"
synsets_save_file = "synsets.pkl"
# Location of the dataset files.
dataset_base = \
    "/training_data/daniel/gazecap_tfrecords/gazecapture_%s.tfrecord"
train_dataset_file = dataset_base % ("train")
test_dataset_file = dataset_base % ("test")
valid_dataset_file = dataset_base % ("val")

# L2 regularizer for weight decay.
l2_reg = regularizers.l2(0.0005)

# Configure GPU VRAM usage.
tf_config = tf.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 1.0
session = tf.Session(config=tf_config)
set_session(session)


def distance_metric(y_true, y_pred):
  """ Calculates the euclidean distance between the two labels and the
  predictions.
  Args:
    y_true: The true labels.
    y_pred: The predictions.
  Returns:
    The element-wise euclidean distance between the labels and the predictions.
  """
  diff = y_true - y_pred
  sqr = K.square(diff)
  total = K.sum(sqr, axis=1)
  return K.sqrt(total)

def train_section(model, learning_rate, iters, labels):
  """ Trains for a number of iterations at one learning rate.
  Args:
    model: The model to train.
    learning_rate: The learning rate to train at.
    iters: Number of iterations to train for.
    labels: Tensor for the labels.
  Returns:
    Training loss and testing accuracy for this section. """
  print "\nTraining at %f for %d iters.\n" % (learning_rate, iters)

  # Set the learning rate.
  opt = optimizers.SGD(lr=learning_rate, momentum=momentum)
  model.compile(optimizer=opt, loss=distance_metric, metrics=[distance_metric],
                target_tensors=[labels])

  training_loss = []
  testing_acc = []

  for i in range(0, iters / train_interval):
    # Train the model.
    history = model.fit(epochs=1, steps_per_epoch=train_interval)

    training_loss.extend(history.history["loss"])
    logging.info("Training loss: %s" % (history.history["loss"]))

    loss, accuracy = model.evaluate(steps=test_interval)

    logging.info("Loss: %f, Accuracy: %f" % (loss, accuracy))
    testing_acc.append(accuracy)

    # Save the trained model.
    model.save_weights(save_file)

  return (training_loss, testing_acc)

def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("train_dataset",
                      help="The location of the training dataset.")
  parser.add_argument("test_dataset",
                      help="The location of the testing dataset.")

  parser.add_argument("-v", "--valid_dataset",
                      help="The location of the validation dataset.")
  parser.add_argument("-m", "--model",
                      help="Existing model to load. Necessary if validating.")

  args = parser.parse_args()

  return args

def train(args):
  """
  Runs the training procedure.
  Args:
    args: The parsed CLI arguments. """
  # Create the training and testing pipelines.
  face_size = config.FACE_SHAPE[:2]
  eye_size = config.EYE_SHAPE[:2]
  builder = pipelines.PipelineBuilder(raw_shape, face_size, batch_size,
                                      eye_size=eye_size)

  input_tensors = builder.build_pipeline(args.train_dataset, args.test_dataset)
  data_tensors = input_tensors[:4]
  label_tensor = input_tensors[4]

  # Create the model.
  net = config.NET_ARCH(config.FACE_SHAPE, eye_shape=config.EYE_SHAPE,
                        data_tensors=data_tensors)
  model = net.build()
  load_model = args.model
  if load_model:
    logging.info("Loading pretrained model '%s'." % (load_model))
    model.load_weights(load_model)

  # Create a coordinator and run queues.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord, sess=session)

  training_acc = []
  training_loss = []
  testing_acc = []

  # Train at each learning rate.
  for lr, iters in zip(learning_rates, iterations):
    loss, acc = train_section(model, lr, iters, label_tensor)

    training_loss.extend(loss)
    testing_acc.extend(acc)

  coord.request_stop()
  coord.join(threads)

  print "Saving results..."
  results_file = open("gazecapture_results.json", "w")
  json.dump((training_loss, testing_acc, training_acc), results_file)
  results_file.close()

def validate(args):
  """ Validates an existing model.
  Args:
    args: Parsed CLI arguments. """
  if not args.valid_dataset:
    raise ValueError("--valid_dataset must be specified.")

  # Create the validation pipeline.
  face_size = config.FACE_SHAPE[:2]
  eye_size = config.EYE_SHAPE[:2]
  builder = pipelines.PipelineBuilder(raw_shape, face_size, batch_size,
                                      eye_size=eye_size)

  input_tensors = builder.build_valid_pipeline(args.valid_dataset)
  data_tensors = input_tensors[:4]
  label_tensor = input_tensors[4]

  input_tensors = build_valid_pipeline(args)
  data_tensors = input_tensors[:4]
  label_tensor = input_tensors[4]

  # Create the model.
  net = config.NET_ARCH(config.FACE_SHAPE, eye_shape=config.EYE_SHAPE,
                        data_tensors=data_tensors)
  model = net.build()
  load_model = args.model
  if not load_model:
    # User did not tell us which model to validate.
    raise ValueError("--model must be specified.")
  logging.info("Loading pretrained model '%s'." % (load_model))
  model.load_weights(load_model)

  # Compile the model. The learning settings don't really matter, since we're
  # not training.
  opt = optimizers.SGD(lr=0.001, momentum=0.9)
  model.compile(optimizer=opt, loss=distance_metric, metrics=[distance_metric],
                target_tensors=[label_tensor])

  # Create a coordinator and run queues.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord, sess=session)

  testing_acc = []

  # Validate.
  for _ in range(0, valid_iters):
    loss, accuracy = model.evaluate(steps=test_interval)

    logging.info("Loss: %f, Accuracy: %f" % (loss, accuracy))
    testing_acc.append(accuracy)

  print "Total accuracy: %f" % (np.mean(testing_acc))

  coord.request_stop()
  coord.join(threads)

def main():
  args = parse_args()

  if args.valid_dataset:
    validate(args)
  else:
    train(args)

if __name__ == "__main__":
  main()
