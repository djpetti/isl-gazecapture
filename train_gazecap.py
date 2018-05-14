#!/usr/bin/python


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

import keras.backend as K
from keras.models import Model, load_model
import keras.layers as layers
import keras.optimizers as optimizers
import keras.regularizers as regularizers
from keras.backend.tensorflow_backend import set_session

import tensorflow as tf

import cv2

import numpy as np

from itracker.common import config
from pipeline import data_loader, preprocess, keras_utils


batch_size = 64
# Shape of the input to the network.
input_shape = (224, 224, 3)
# Shape of the raw images from the dataset.
raw_shape = (400, 400, 3)

# How many batches to run between testing intervals.
train_interval = 20
# How many batches to run during testing.
test_interval = 3

# Learning rates to set.
learning_rates = [0.0001]
# How many iterations to train for at each learning rate.
iterations = [20000]

# Learning rate hyperparameters.
momentum = 0.9

# Where to save the network.
save_file = "eye_model_finetuned.hd5"
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

def fuse_loaders(train_loader, train_pipelines, test_loader, test_pipelines):
  """ Fuses the outputs from the training and testing loaders.
  Args:
    train_loader: The training loader.
    train_pipelines: The pipelines associated with the train loader.
    test_loader: The testing loader.
    test_pipelines: The pipelines associated with the test loader.
  Returns:
    The fused outputs, in the same order as the pipeline inputs, with the labels
    at the end. """
  train_data = train_loader.get_data()
  train_labels = train_loader.get_labels()
  test_data = test_loader.get_data()
  test_labels = test_loader.get_labels()

  # Extract the corresponding outputs for the pipelines.
  train_outputs = []
  for pipeline in train_pipelines:
    train_outputs.append(train_data[pipeline])
  # Add the labels too.
  train_outputs.append(train_labels)

  test_outputs = []
  for pipeline in test_pipelines:
    test_outputs.append(test_data[pipeline])
  test_outputs.append(test_labels)

  # Fuse the outputs.
  return keras_utils.fuse_loaders(train_outputs, test_outputs)

def add_train_stages(loader):
  """ Convenience function to configure train loader.
  Args:
    loader: The DataLoader to configure.
  Returns:
    A tuple of the pipelines created for the loader. """
  pipeline = loader.get_pipeline()

  # Extract eye crops.
  extract_stage = preprocess.EyeExtractionStage()
  leye, reye, face = pipeline.add(extract_stage)

  # Extract face mask.
  mask_stage = preprocess.FaceMaskStage()
  mask, face = face.add(mask_stage)

  # Random cropping.
  crop_stage = preprocess.RandomCropStage((390, 390))
  face_crop_stage = preprocess.RandomCropStage((360, 360))
  leye.add(crop_stage)
  reye.add(crop_stage)
  face.add(face_crop_stage)

  # Random adjustments.
  brightness_stage = preprocess.RandomBrightnessStage(50)
  contrast_stage = preprocess.RandomContrastStage(0.9, 1.4)
  hue_stage = preprocess.RandomHueStage(0.1)
  saturation_stage = preprocess.RandomSaturationStage(0.9, 1.1)
  grayscale_stage = preprocess.GrayscaleStage()

  leye.add(brightness_stage)
  leye.add(contrast_stage)
  leye.add(grayscale_stage)

  reye.add(brightness_stage)
  reye.add(contrast_stage)
  reye.add(grayscale_stage)

  face.add(brightness_stage)
  face.add(contrast_stage)
  face.add(hue_stage)
  face.add(saturation_stage)

  # Normalization and final sizing.
  norm_stage = preprocess.NormalizationStage()
  output_size = input_shape[:2]
  resize_stage = preprocess.ResizeStage(output_size)
  leye.add(norm_stage)
  reye.add(norm_stage)
  face.add(norm_stage)

  leye.add(resize_stage)
  reye.add(resize_stage)
  face.add(resize_stage)

  # Build the loader graph.
  loader.build()

  return (leye, reye, face, mask)

def add_test_stages(loader):
  """ Convenience function to configure test and validation loaders.
  Args:
    loader: The DataLoader to configure.
  Returns:
    A tuple of the pipelines created for the loader. """
  pipeline = loader.get_pipeline()

  # Extract eye crops.
  extract_stage = preprocess.EyeExtractionStage()
  leye, reye, face = pipeline.add(extract_stage)

  # Extract face mask.
  mask_stage = preprocess.FaceMaskStage()
  mask, face = face.add(mask_stage)

  # Take the central crops.
  crop_stage = preprocess.CenterCropStage(0.975)
  face_crop_stage = preprocess.CenterCropStage(0.9)
  leye.add(crop_stage)
  reye.add(crop_stage)
  face.add(face_crop_stage)

  # Grayscale.
  grayscale_stage = preprocess.GrayscaleStage()
  leye.add(grayscale_stage)
  reye.add(grayscale_stage)

  # Normalization and final sizing.
  norm_stage = preprocess.NormalizationStage()
  output_size = input_shape[:2]
  resize_stage = preprocess.ResizeStage(output_size)
  leye.add(norm_stage)
  reye.add(norm_stage)
  face.add(norm_stage)

  leye.add(resize_stage)
  reye.add(resize_stage)
  face.add(resize_stage)

  # Build the loader graph.
  loader.build()

  return (leye, reye, face, mask)


def build_pipeline():
  """ Builds the preprocessing pipeline.
  Returns:
    The fused output nodes from the loaders, in order: leye, reye, face, grid,
    dots. """
  train_loader = data_loader.TrainDataLoader(train_dataset_file, batch_size,
                                             raw_shape)
  test_loader = data_loader.TestDataLoader(test_dataset_file, batch_size,
                                           raw_shape)

  train_pipelines = add_train_stages(train_loader)
  test_pipelines = add_test_stages(test_loader)

  return fuse_loaders(train_loader, train_pipelines,
                      test_loader, test_pipelines)

def build_valid_pipeline():
  """ Builds the preprocessing pipeline for the validation split.
  Returns:
    The leye, reye, face, grid, and dots nodes for the validation loader. """
  valid_loader = data_loader.ValidDataLoader(valid_dataset_file, batch_size,
                                             raw_shape)

  valid_pipelines = add_test_stages(valid_loader)

  # Extract the associated output nodes.
  data = valid_loader.get_data()
  nodes = []
  for pipeline in valid_pipelines:
    nodes.append(data[pipeline])
  nodes.append(valid_loader.get_labels())

  return nodes

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

def main(load_model=None):
  """
  Args:
    load_model: A pretrained model to load, if specified. """
  # Create the training and testing pipelines.
  input_tensors = build_pipeline()
  data_tensors = input_tensors[:4]
  label_tensor = input_tensors[4]

  # Create the model.
  eye_shape = (input_shape[0], input_shape[1], 1)
  net = config.NET_ARCH(input_shape, eye_shape=eye_shape,
                        data_tensors=data_tensors)
  model = net.build()
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

def validate(load_model, iters):
  """ Validates an existing model.
  Args:
    load_model: The model to load.
    iters: How many iterations to validate for. """
  # Create the validation pipeline.
  input_tensors = build_valid_pipeline()
  data_tensors = input_tensors[:4]
  label_tensor = input_tensors[4]

  # Create the model.
  eye_shape = (input_shape[0], input_shape[1], 1)
  net = config.NET_ARCH(input_shape, eye_shape=eye_shape,
                        data_tensors=data_tensors)
  model = net.build()
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
  for _ in range(0, iters):
    loss, accuracy = model.evaluate(steps=test_interval)

    logging.info("Loss: %f, Accuracy: %f" % (loss, accuracy))
    testing_acc.append(accuracy)

  print "Total accuracy: %f" % (np.mean(testing_acc))

  coord.request_stop()
  coord.join(threads)


if __name__ == "__main__":
  main(load_model="eye_model_finetuned.hd5")
