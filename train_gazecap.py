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


<<<<<<< HEAD
# This forks a lot of processes, so we want to import it as soon as possible,
# when there is as little memory as possible in use.
from rpinets.myelin import data_loader

=======
>>>>>>> origin/feature/exp_architectures
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

from itracker.common import config
from pipeline import data_loader, preprocess, keras_utils


batch_size = 64
<<<<<<< HEAD
# How many batches to have loaded into VRAM at once.
load_batches = 5
# Shape of the input images.
image_shape = (400, 400, 3)
# Shape of the extracted patches.
patch_shape = (390, 390)
=======
>>>>>>> origin/feature/exp_architectures
# Shape of the input to the network.
input_shape = (224, 224, 3)
# Shape of the raw images from the dataset.
raw_shape = (400, 400, 3)

# How many batches to run between testing intervals.
train_interval = 20
# How many batches to run during testing.
test_interval = 3

# Learning rates to set.
learning_rates = [0.0001, 0.00001]
# How many iterations to train for at each learning rate.
<<<<<<< HEAD
iterations = [51864, 300000]
=======
iterations = [100000, 100000]
>>>>>>> origin/feature/exp_architectures

# Learning rate hyperparameters.
momentum = 0.9

# Where to save the network.
save_file = "eye_model_finetuned.hd5"
synsets_save_file = "synsets.pkl"
# Location of the dataset files.
<<<<<<< HEAD
dataset_files = "/training_data/gazecap_myelin/dataset"
# Location of the cache files.
cache_dir = "/training_data/gazecap_myelin"

# Validation data.
valid_dataset_files = "/training_data/gazecap_myelin_val/dataset"
valid_cache_dir = "/training_data/gazecap_myelin_val"
# Fine-tuning data.
ft_dataset_files = "/training_data/gazecap_myelin/dataset"
ft_cache_dir = "/training_data/gazecap_myelin"
=======
dataset_base = \
    "/training_data/daniel/gazecap_tfrecords/gazecapture_%s.tfrecord"
train_dataset_file = dataset_base % ("train")
test_dataset_file = dataset_base % ("test")
valid_dataset_file = dataset_base % ("val")
>>>>>>> origin/feature/exp_architectures

# L2 regularizer for weight decay.
l2_reg = regularizers.l2(0.0005)

# Configure GPU VRAM usage.
<<<<<<< HEAD
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1.0
set_session(tf.Session(config=config))


def create_bitmask_images(bboxes):
  """ Creates the bitmask images from the bbox points.
  Args:
    bboxes: A matrix of bounding boxes, where each box is a row vector of x, y,
            width, and height.
  Returns:
    The generated bitmask image. """
  # Scale to mask size.
  bboxes *= 25
  # It's one-indexed in the dataset.
  bboxes = np.round(bboxes).astype(np.int8) - 1

  bboxes = np.clip(bboxes, 0, 25)

  # Create the background.
  frames = np.zeros((bboxes.shape[0], 25, 25))
=======
tf_config = tf.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 1.0
session = tf.Session(config=tf_config)
set_session(session)
>>>>>>> origin/feature/exp_architectures


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
<<<<<<< HEAD
    The built network, ready to train. """
  trainable = not fine_tune

  left_eye_input = layers.Input(shape=input_shape, name="left_eye_input")
  right_eye_input = layers.Input(shape=input_shape, name="right_eye_input")
  # The face crop gets resized on-the-fly.
  face_shape = (patch_shape[0], patch_shape[1], input_shape[2])
  face_input = layers.Input(shape=face_shape, name="face_input")
  grid_input = layers.Input(shape=(25, 25), name="grid_input")

  left_eye_floats = K.cast(left_eye_input, "float32")
  right_eye_floats = K.cast(right_eye_input, "float32")
  face_floats = K.cast(face_input, "float32")
  grid_floats = K.cast(grid_input, "float32")

  # Resize face.
  scale_layer = layers.Lambda(lambda x: tf.image.resize_images(x, (224, 224)))
  face_scaled = scale_layer(face_floats)

  # Convert everything to grayscale.
  gray_layer = layers.Lambda(lambda x: rgb_to_grayscale(x))
  left_eye_gray = gray_layer(left_eye_floats)
  right_eye_gray = gray_layer(right_eye_floats)

  # Get pretrained VGG model for use as a base.
  vgg = applications.vgg19.VGG19(include_top=False,
                                 input_tensor=face_scaled)
  vgg_out = vgg.outputs[0]

  # Freeze all layers in VGG.
  for layer in vgg.layers:
    layer.trainable = False

  # Shared eye layers.
  conv_e1 = layers.Conv2D(144, (11, 11), strides=(4, 4), activation="relu",
                          kernel_regularizer=l2_reg, trainable=trainable)
  pool_e1 = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))
  norm_e1 = layers.BatchNormalization(trainable=trainable)

  pad_e2 = layers.ZeroPadding2D(padding=(2, 2))
  conv_e2 = layers.Conv2D(384, (5, 5), activation="relu",
                          kernel_regularizer=l2_reg, trainable=trainable)
  pool_e2 = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))
  norm_e2 = layers.BatchNormalization(trainable=trainable)

  pad_e3 = layers.ZeroPadding2D(padding=(1, 1))
  conv_e3 = layers.Conv2D(576, (3, 3), activation="relu",
                          kernel_regularizer=l2_reg, trainable=trainable)

  conv_e4 = layers.Conv2D(64, (1, 1), activation="relu",
                          kernel_regularizer=l2_reg, trainable=trainable)
  flatten_e4 = layers.Flatten()

  # Left eye stack.
  leye_conv_e1 = conv_e1(left_eye_gray)
  leye_pool_e1 = pool_e1(leye_conv_e1)
  leye_norm_e1 = norm_e1(leye_pool_e1)

  leye_pad_e2 = pad_e2(leye_norm_e1)
  leye_conv_e2 = conv_e2(leye_pad_e2)
  leye_pool_e2 = pool_e2(leye_conv_e2)
  leye_norm_e2 = norm_e2(leye_pool_e2)

  leye_pad_e3 = pad_e3(leye_norm_e2)
  leye_conv_e3 = conv_e3(leye_pad_e3)

  leye_conv_e4 = conv_e4(leye_conv_e3)
  leye_flatten_e4 = flatten_e4(leye_conv_e4)

  # Right eye stack.
  reye_conv_e1 = conv_e1(right_eye_gray)
  reye_pool_e1 = pool_e1(reye_conv_e1)
  reye_norm_e1 = norm_e1(reye_pool_e1)

  reye_pad_e2 = pad_e2(reye_norm_e1)
  reye_conv_e2 = conv_e2(reye_pad_e2)
  reye_pool_e2 = pool_e2(reye_conv_e2)
  reye_norm_e2 = norm_e2(reye_pool_e2)

  reye_pad_e3 = pad_e3(reye_norm_e2)
  reye_conv_e3 = conv_e3(reye_pad_e3)

  reye_conv_e4 = conv_e4(reye_conv_e3)
  reye_flatten_e4 = flatten_e4(reye_conv_e4)

  # Concatenate eyes and put through a shared FC layer.
  eye_combined = layers.Concatenate()([reye_flatten_e4, leye_flatten_e4])
  eye_combined_drop = layers.Dropout(0.5)(eye_combined)
  fc_e1 = layers.Dense(128, activation="relu",
                       kernel_regularizer=l2_reg)(eye_combined_drop)

  # Face layers.
  face_flatten_f4 = layers.Flatten()(vgg_out)
  face_flatten_drop = layers.Dropout(0.5)(face_flatten_f4)

  face_fc1 = layers.Dense(128, activation="relu",
                          kernel_regularizer=l2_reg,
                          trainable=trainable)(face_flatten_drop)
  face_fc2 = layers.Dense(64, activation="relu",
                          kernel_regularizer=l2_reg)(face_fc1)

  # Face grid.
  grid_flat = layers.Flatten()(grid_floats)
  grid_fc1 = layers.Dense(256, activation="relu",
                          kernel_regularizer=l2_reg,
                          trainable=trainable)(grid_flat)
  grid_fc2 = layers.Dense(128, activation="relu",
                          kernel_regularizer=l2_reg,
                          trainable=trainable)(grid_fc1)

  # Concat everything and put through a final FF layer.
  all_concat = layers.Concatenate()([fc_e1, face_fc2, grid_fc2])
  all_fc1 = layers.Dense(128, activation="relu",
                         kernel_regularizer=l2_reg)(all_concat)
  all_fc2 = layers.Dense(2, kernel_regularizer=l2_reg)(all_fc1)

  # Build the model.
  model = Model(inputs=[left_eye_input, right_eye_input, face_input,
                        grid_input],
                outputs=all_fc2)
  model.summary()

  return model

def process_data(face_data, labels):
  """ Helper that performs all the pre-processing on the input data.
=======
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
>>>>>>> origin/feature/exp_architectures
  Args:
    loader: The DataLoader to configure.
  Returns:
<<<<<<< HEAD
    The converted left eye crops, right eye crops, face crops, face grids, and
    ground-truth dot locations. """
  # Process raw label names.
  dot_data, leye_data, reye_data, mask_bboxes = convert_labels(labels)
  # Randomly flip some images for data augmentation.
  dot_data, leye_data, reye_data, mask_bboxes, face_data = \
      maybe_flip(dot_data, leye_data, reye_data, mask_bboxes, face_data)
  # Generate masks.
  mask_data = create_bitmask_images(mask_bboxes)
  # Extract left and right eye crops.
  leye_crops, reye_crops = extract_eye_crops(face_data, leye_data, reye_data)

  return (leye_crops, reye_crops, face_data, mask_data, dot_data)

def train_section(model, data, learning_rate, iters):
=======
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
>>>>>>> origin/feature/exp_architectures
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

<<<<<<< HEAD
  print "Mean accuracy: %f" % (np.mean(testing_acc))

  data.exit_gracefully()

def fine_tune(load_model, ft_lrs):
  """ Fine-tunes the model.
  Args:
    load_model: The model to load for fine-tuning.
    ft_lrs: List of tuples of learning rates and iteration counts for
            fine-tuning. """
  model = build_network()
  logging.info("Loading pretrained model '%s'." % (load_model))
  model.load_weights(load_model)

  data = data_loader.DataManagerLoader(batch_size, load_batches, image_shape,
                                       ft_cache_dir, ft_dataset_files,
                                       patch_shape=patch_shape,
                                       pca_stddev=50,
                                       patch_flip=False,
                                       raw_labels=True)

  if os.path.exists(synsets_save_file):
    logging.info("Loading existing synsets...")
    data.load(synsets_save_file)
=======
  print "Total accuracy: %f" % (np.mean(testing_acc))
>>>>>>> origin/feature/exp_architectures

  coord.request_stop()
  coord.join(threads)


if __name__ == "__main__":
  main(load_model="eye_model_finetuned.hd5")
