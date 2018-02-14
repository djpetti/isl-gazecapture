#!/usr/bin/python


import socket
import time

import cv2


# Port to connect to the server on.
PORT = 6219


def main():
  cam = cv2.VideoCapture(0)

  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  sock.connect(("", PORT))

  while True:
    # Capture and encode an image.
    ret, image = cam.read()
    if not ret:
      raise RuntimeError("Image capture failed.")
    ret, encoded = cv2.imencode(".jpg", image)

    # Send the image.
    sock.sendall(encoded)

    time.sleep(0.03)

  sock.close()


if __name__ == "__main__":
  main()
