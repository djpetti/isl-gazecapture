# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 15:49:10 2017

@author: morepoch
"""
import cv2
import numpy as np
from threading import Thread
import landmark_detection as ld
import time
class VideoCapture(object):
    
    def __init__(self, src=0):
        # initialize the video camera stream and read the first frame from the stream
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()   
        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False
       
    def start(self):
        # start the thread to read frames from the video stream
    		Thread(target=self.update, args=()).start()
    		return self
    def isOpened(self):
        return self.stream.isOpened()
    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()
    def read(self):
        # return the frame most recently read
		   return self.grabbed, self.frame
    def release(self):   
        self.stopped = True 
        self.stream.release()
        
 
class VideoLmksCapture(object):
    """
    Detect landmarks in thread
    """
    def __init__(self, src=0):
        # initialize the video camera stream and read the first frame from the stream
        self.stream = cv2.VideoCapture(src)
        
        (self.grabbed, self.frame) = self.stream.read()   
        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False
        # landmark detection engine
        self.engine = ld.LandmarkDetection()
        self.pts = np.zeros((51, 2))
    def start(self):
        # start the thread to read frames from the video stream
    		Thread(target=self.update, args=()).start()
    		return self
    def isOpened(self):
        return self.stream.isOpened()
    def update(self):
        # keep looping infinitely until the thread is stopped
        ffp_flag = 1
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
            # otherwise, read the next frame from the stream
            #(self.grabbed, temp) = self.stream.read()
            (self.grabbed, temp) = self.stream.read()
            if not self.grabbed:
                continue
            self.frame = np.fliplr(temp)
            if  ffp_flag > 0:
                self.pts, ffp_flag, self.confidence = self.engine.ffp_detect(self.frame)
            else:
                self.pts, ffp_flag, self.confidence = self.engine.ffp_track(self.frame, self.pts)
                
    def read(self):
        # return the frame most recently read
		   return self.grabbed, self.frame, self.pts, self.confidence
    def release(self):
        self.stopped = True  
        self.stream.release()