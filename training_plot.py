#!/usr/bin/python


import argparse
import json
import re
import sys

import matplotlib.pyplot as plot

import numpy as np


""" Utility for plotting data collected during training runs. """


def plot_training(loss, testing_acc, iter_step):
  """ Plots the training results.
  Args:
    loss: The list of loss data.
    testing_acc: The list of testing accuracy data.
    iter_step: How many iterations elapsed between each logging interval. """
  # Make three subplots that share the same time axis.
  fig, time_axes = plot.subplots(2, sharex=True)

  testing_acc = np.repeat(testing_acc, len(loss) / len(testing_acc))
  # Cut the last bit so they're the same shape.
  loss = loss[:len(testing_acc)]

  # Compute x values.
  x_values = range(0, len(loss))
  x_values = np.multiply(x_values, iter_step)

  # Plot everything.
  time_axes[0].plot(x_values, loss)
  time_axes[1].plot(x_values, testing_acc)

  # One x label at the bottom.
  time_axes[1].set_xlabel("Iterations")
  # Y lables for each graph.
  time_axes[0].set_ylabel("Loss")
  time_axes[1].set_ylabel("Testing Accuracy")

  fig.tight_layout()
  plot.show()

def average_filter(data, window):
  """ Uses a sliding-window average filter to reduce data noise. """
  if window == 1:
    # No averaging.
    return data

  averaged = []
  for i in range(0, len(data) - window):
    sample = np.mean(data[i:(i + window)])
    averaged.append(sample)

  return averaged

def load_log(log_file):
  """ Parses data from a log file instead of the JSON dump.
  Args:
    log_file: The file to load data from.
  Returns:
    List of the testing_loss, training loss, testing accuracy, and
                training accuracy. """
  testing_loss = []
  training_loss = []
  testing_acc = []
  training_acc = []

  lines = log_file.read().split("\n")
  for line in lines:
    if "Training loss" in line:
      # This line contains the training loss and accuracy.
      numbers = re.findall("\d\.\d+", line)
      loss, acc = [float(num) for num in numbers]

      training_loss.append(loss)
      training_acc.append(acc)

    if "Testing loss" in line:
      # This line contains the testing loss and accuracy.
      numbers = re.findall("\d\.\d+", line)
      loss, acc = [float(num) for num in numbers]

      testing_loss.append(loss)
      testing_acc.append(acc)

  return testing_loss, training_loss, testing_acc, training_acc


def main():
  parser = argparse.ArgumentParser("Analyze training data logs.")
  parser.add_argument("data_file", help="The data file to analyze.")
  parser.add_argument("-l", "--log_file", action="store_true",
                      help="Analyze log file instead of JSON dump.")
  parser.add_argument("-i", "--interval", default=1, type=int,
                      help="Number of iterations between training logs.")
  parser.add_argument("-f", "--filter_interval", default=1, type=int,
                      help="Window size for average filtering.")
  args = parser.parse_args()

  # Load the logged data.
  log_file = file(args.data_file)
  if not args.log_file:
    loss, test_acc, _ = json.load(log_file)
  else:
    # Parse from the log file.
    _, loss, test_acc, _ = load_log(log_file)
  log_file.close()

  # Average filtering.
  loss = average_filter(loss, args.filter_interval)
  test_acc = average_filter(test_acc, args.filter_interval)

  plot_training(loss, test_acc, args.interval)


if __name__ == "__main__":
  main()
