#!/usr/bin/python


import socket
import struct
import time

import numpy as np

import cv2


# Port to connect to the server on.
PORT = 6219


def main():
  cam = cv2.VideoCapture(0)

  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  sock.connect(("", PORT))

  seq = 0
  for i in range(0, 10):
    # Capture and encode an image.
    ret, image = cam.read()
    if not ret:
      raise RuntimeError("Image capture failed.")
    ret, encoded = cv2.imencode(".jpg", image)
    size = struct.pack("I", len(encoded))
    sequence_num = struct.pack("B", seq)

    seq += 1
    seq = seq % 255

    # Send the image.
    sock.sendall(size)
    sock.sendall(encoded)
    sock.sendall(sequence_num)

    time.sleep(0.03)

  while True:
    # Wait for a response.
    resp = sock.recv(1)
    print resp


  sock.close()


if __name__ == "__main__":
  main()
