#!/usr/bin/python


import argparse
import logging

import cv2

from itracker.common import config, phone_config
from itracker.server.gaze_predictor import GazePredictor
from itracker.server import server


# TODO (daniel) Logging to a file eventually.
logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
                    level=logging.INFO)


def demo(port):
  """ A demo version of the server that displays received images on-screen, and
  sends fake results back.
  Args:
    port: The port to run the demo server on. """
  my_server = server.Server(port)
  my_server.wait_for_client()

  while True:
    frame, sequence_num = my_server.read_next_frame()
    if frame is None:
      # Connection closed, wait for another.
      cv2.destroyAllWindows()
      my_server.wait_for_client()
      continue

    logging.info("Sequence number: %d" % (sequence_num))

    # Send a response for a fake prediction.
    my_server.send_response((0.0, 0.0), sequence_num)

    cv2.imshow("test", frame)
    cv2.waitKey(1)

def predict_forever(model_file, port, phone, display_crops=False):
  """ Runs the server process and prediction pipeline forever.
  Args:
    model_file: The model file to use for predictions.
    port: The port for the server to listen on.
    phone: The configuration for the phone we are using.
    display_crops: If true, it will display the crops for the images it
                   receives. Useful for debugging. """
  predictor = GazePredictor(config.NET_ARCH, model_file, phone,
                            display=display_crops)
  my_server = server.Server(port)

  while True:
    # Wait for a client to connect.
    my_server.wait_for_client()

    # Start server-handling processes.
    receive_process = server.ReceiveProcess(my_server, predictor)
    send_process = server.SendProcess(my_server, predictor)

    # Wait for the client to diconnect.
    receive_process.wait_for_disconnect()
    send_process.wait_for_disconnect()

def main():
  parser = argparse.ArgumentParser( \
      description="Run the gaze estimation server.")
  parser.add_argument("phone_config",
                      help="Path to configuration for phone we are using.")
  parser.add_argument("-d", "--demo", action="store_true",
                      help="Run a demo server that displays received images.")
  parser.add_argument("-c", "--display_crops", action="store_true",
                      help="Displays crops for the images it receives.")
  args = parser.parse_args()

  # Load the phone configuration.
  phone = phone_config.PhoneConfig(args.phone_config)

  if args.demo:
    # Run the demo.
    demo(config.SERVER_PORT)
  else:
    predict_forever(config.MODEL_FILE, config.SERVER_PORT, phone,
                    display_crops=args.display_crops)


if __name__ == "__main__":
  main()
