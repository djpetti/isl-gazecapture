#!/usr/bin/python


import argparse
import logging

import cv2

import server


# TODO (daniel) Logging to a file eventually.
logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
                    level=logging.INFO)


def demo():
  """ A demo version of the server that displays received images on-screen. """
  my_server = server.Server(6219)
  my_server.wait_for_client()

  while True:
    frame = my_server.read_next_frame()
    if frame is None:
      # Connection closed, wait for another.
      cv2.destroyAllWindows()
      my_server.wait_for_client()
      continue

    cv2.imshow("test", frame)
    cv2.waitKey(1)

def main():
  parser = argparse.ArgumentParser( \
      description="Run the gaze estimation server.")
  parser.add_argument("-d", "--demo", action="store_true",
                      help="Run a demo server that displays received images.")
  args = parser.parse_args()

  if args.demo:
    # Run the demo.
    demo()
  else:
    raise NotImplementedError("Full server is not yet implemented.")


if __name__ == "__main__":
  main()
