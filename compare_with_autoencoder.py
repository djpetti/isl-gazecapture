#!/usr/bin/python


import argparse

import cv2

import numpy as np

from itracker.server import gaze_predictor
from itracker.common import phone_config
from itracker.common.network import autoencoder


class Comparator(object):
  """ Class that compares images using the autoencoder. """

  def __init__(self, model_file, phone_data):
    """
    Args:
      model_file: The model file to use.
      phone_data: The configuration data for the phone we are using. """
    # Create the gaze predictor.
    self.__predictor = gaze_predictor.GazePredictor(autoencoder.Autoencoder,
                                                    model_file, phone_data,
                                                    drop_stale=False)

  def compare(self, image1, image2):
    """ Compares two images using the autoencoder.
    Args:
      image1: The first image to compare.
      image2: The second image to compare.
    Returns:
      The vector distance between the two encodings. """
    self.__predictor.process_image(image1, 0)
    self.__predictor.process_image(image2, 0)

    # Get the results.
    prediction1, _, _ = self.__predictor.predict_gaze()
    prediction2, _, _ = self.__predictor.predict_gaze()
    if (prediction1 is None or prediction2 is None):
      raise RuntimeError("Failed to detect faces in input.")
    _, _, encoding1 = prediction1
    _, _, encoding2 = prediction2

    # Get the vector distance.
    return np.linalg.norm(encoding1 - encoding2)


def main():
  parser = argparse.ArgumentParser( \
      description="Compares images using the autoencoder.")
  parser.add_argument("model_file", help="The saved autoencoder weights.")
  parser.add_argument("phone_file",
                      help="The file containing phone config data.")
  parser.add_argument("image_file_1", help="The first image to compare.")
  parser.add_argument("image_file_2", help="The second image to compare.")

  args = parser.parse_args()

  phone = phone_config.PhoneConfig(args.phone_file)
  comparator = Comparator(args.model_file, phone)

  # Load the images.
  image1 = cv2.imread(args.image_file_1)
  image2 = cv2.imread(args.image_file_2)

  # Compare.
  distance = comparator.compare(image1, image2)
  print "Image distance: %f" % float(distance)


if __name__ == "__main__":
  main()
