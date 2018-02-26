# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 17:53:40 2016

@author: morepoch

Main interface for landmark detection 
"""

import landmark_detection as ld
import numpy as np
from misc import loadmat, crop_eye
import matplotlib.pyplot as plt
import cv2
import time
from helpers import draw_str
from VideoCapture import VideoCapture, VideoLmksCapture
engine = ld.LandmarkDetection()
pose_engine = ld.PoseEstimation()

def test_single_image():
    """
    Test on a single image
    """
    matdata = loadmat('test1_img.mat')
    img = matdata['im']
    bbox = engine.face_detect(img)
    pts1, f1, c1 = engine.ffp_detect(img)
    pts2, f2, c2 = engine.ffp_track(img, pts1)
    plt.imshow(img, cmap='gray')
    plt.plot(pts1[:, 0], pts1[:, 1], 'g.')
    return bbox, pts1, pts2, c1, c2
    
def test_online(filename):
    """
    Test online detection and tracking
    """
    cap = VideoCapture(filename)
    
    retval = cap.isOpened()
    if not retval:
        print("Cannot open camera/video!")
    else: # start the video capturing thread
        cap.start()
    ffp_flag = True
    fps_queue = []
    fps_queue_size = 20
    while(retval):
        # Capture frame-by-frame
        tStart = time.time()
        ret, img = cap.read()
        if not ret:
            break
        # mirror image
        img = np.fliplr(img)
        # landmark detection tracking
        if  ffp_flag > 0:
            pts, ffp_flag, confidence = engine.ffp_detect(img)
        else:
            pts, ffp_flag, confidence = engine.ffp_track(img, pts)
            #pts, ffp_flag, confidence = engine.ffp_detect(img)
        tEnd = time.time()
        if np.sum(pts) == 0: # failed detectio, continue
            continue
        ## uncomment for eye cropping
        #eye_left = crop_eye(img, pts[19, :], pts[22, :])
        #eye_right = crop_eye(img, pts[25, :], pts[28, :])
        #gaze = f(eye_left) # potential gaze estimation
        vis = img.copy()
        if ffp_flag == 0:
            num_pts = pts.shape[0]
            if vis.ndim < 3:
                vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2RGB)
            for idx in np.arange(num_pts):
                cv2.circle(vis, (int(pts[idx,0].round()), int(pts[idx,1].round())), 2, (0,255,0),  -1)

        fps= 1.0/(tEnd - tStart)
        if len(fps_queue) < fps_queue_size:
            fps_queue.append(fps)
        else:
            fps_queue[:-1] = fps_queue[1:]
            fps_queue[-1] = fps
        tStart = tEnd
        draw_str(vis, (20, 20), 'fps: %3.1f, flag = %d, confidence = %4.2f' % (np.median(fps_queue), ffp_flag, confidence))
        #print fps
        cv2.imshow('ffp_detection',vis)
        if 0xFF & cv2.waitKey(5) == 27:
            break
        #return img
    cap.release()
    cv2.destroyAllWindows()

    
test_online(0)
