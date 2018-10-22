#!/usr/bin/python


import argparse

from itracker.training import experiment

import logging_config


def parse_args():
  """ Builds a parser for CLI arguments.
  Returns:
    The parser it built. """
  parser = argparse.ArgumentParser()

  parser.add_argument("train_dataset",
                      help="The location of the training dataset.")
  parser.add_argument("test_dataset",
                      help="The location of the testing dataset.")

  parser.add_argument("-v", "--valid_dataset",
                      help="The location of the validation dataset.")
  parser.add_argument("-m", "--model",
                      help="Existing model to load. Necessary if validating.")
  parser.add_argument("-o", "--output", default="eye_model.hd5",
                      help="Where to save the trained model.")
  parser.add_argument("--autoencoder", action="store_true",
      help="Specifies that we want to evaluate the autoencoder.")

  parser.add_argument("--batch_size", type=int, default=32,
                      help="Examples in each batch.")
  parser.add_argument("--testing_interval", type=int, default=4,
                      help="No. of training iterations to run before testing.")
  parser.add_argument("--learning_rate", type=float, default=0.001,
                      help="The initial learning rate.")
  parser.add_argument("--momentum", type=float, default=0.9,
                      help="The initial momentum.")
  parser.add_argument("--training_steps", type=int, default=40,
                      help="Num. of batches to run for each training iter.")
  parser.add_argument("--testing_steps", type=int, default=24,
                      help="Num. of batches to run for each testing iter.")
  parser.add_argument("--valid_iters", type=int, default=124,
                      help="How many iterations to validate for.")

  return parser

def main():
  logging_config.configure_logging()
  parser = parse_args()

  # Create and start the experiment.
  my_experiment = experiment.Experiment(parser)
  my_experiment.run()


if __name__ == "__main__":
  main()
