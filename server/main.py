#!/usr/bin/python

import cv2

import server


def main():
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


if __name__ == "__main__":
  main()
