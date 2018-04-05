#!/usr/bin/python

import argparse

import cv2

import numpy as np


def generate_board(width, height, width_px, height_px):
  """ Generates a calibration image for a paticular screen with 1x1 cm squares.
  Args:
    width: The width of the screen, in cm.
    height: The height of the screen, in cm.
    width_px: The width of the screen, in px.
    height_px: The height of the screen, in px.
  Returns:
    Generated checkerboard image. """
  # Compute the number of pixels per square.
  square_width = int(width_px / width)
  square_height = int(height_px / height)

  # Create background.
  image = np.ones((height_px, width_px))
  num_squares_y = int(height)
  num_squares_x = int(width)

  for r in range(0, num_squares_y):
    start_c = r % 2

    for c in range(start_c, num_squares_x, 2):
      start_y = r * square_height
      start_x = c * square_width

      # Create black square.
      square = np.zeros((square_height, square_width))
      image[start_y:(start_y + square_height),
            start_x:(start_x + square_width)] = square

  # Multiply by 255 so it's black and white.
  image *= 255

  return image

def main():
  parser = argparse.ArgumentParser("Generate calibration images for a screen.")
  parser.add_argument("width_cm", type=float,
                      help="The width of the screen in cm.")
  parser.add_argument("height_cm", type=float,
                      help="The height of the screen in cm.")
  parser.add_argument("width_px", type=int,
                      help="The width of the screen in pixels.")
  parser.add_argument("height_px", type=int,
                      help="The height of the screen in pixels.")
  args = parser.parse_args()

  print "Generating calibration image..."
  board = generate_board(args.width_cm, args.height_cm, args.width_px,
                         args.height_px)

  # Save the image.
  cv2.imwrite("calibration_image.png", board)

if __name__ == "__main__":
  main()
