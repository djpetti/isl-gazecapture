#!/usr/bin/python


import json
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
  while True:
    # Capture and encode an image.
    ret, image = cam.read()
    if not ret:
      raise RuntimeError("Image capture failed.")
    ret, encoded = cv2.imencode(".jpg", image)
    size = struct.pack("I", socket.htonl(len(encoded)))
    sequence_num = struct.pack("B", seq)

    seq += 1
    seq = seq % 255

    start_time = time.time()

    # Send the image.
    sock.sendall(size)
    sock.sendall(encoded)
    sock.sendall(sequence_num)

    response = ""
    while True:
      # Wait for a response.
      char = sock.recv(1)
      response += char

      if char == "{":
        # Start of JSON message.
        response = char
      elif char == "}":
        # End of JSON message.
        break

    turnaround_time = time.time() - start_time
    print "Turnaround time: %f" % (turnaround_time)

    print json.loads(response)


  sock.close()


if __name__ == "__main__":
  main()
