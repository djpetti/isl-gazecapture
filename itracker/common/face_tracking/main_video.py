# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 17:53:40 2016

@author: morepoch

Main interface for landmark detection 
"""


import cv2
import time
from helpers import draw_str
import glob

    
def test_online(scene, camera, savename='test.avi'):
    """
    Test online detection and tracking
    """
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(savename, fourcc, 25.0, (640,480))
    cap_scene = cv2.VideoCapture(scene)
    cap = cv2.VideoCapture(camera)
    
    retval = cap.isOpened() and cap_scene.isOpened()
    if not retval:
        print("Cannot open camera/video!")
    cv2.namedWindow('scene', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('scene', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    while(retval):
        # Capture frame-by-frame
        tStart = time.time()
        ret1, img1 = cap.read()
        ret2, img2 = cap_scene.read()
        if not (ret1 and ret2):
            break

        
        
        out.write(img1)        
        tEnd = time.time()
        fps= 1.0/(tEnd - tStart)
        tStart = tEnd
        draw_str(img2, (20, 20), 'fps: %3.1f' % (fps))
        #print fps
        cv2.imshow('scene', img2)
        if 0xFF & cv2.waitKey(5) == 27:
            break
    cap.release()
    cap_scene.release()
    cv2.destroyAllWindows()
    
#bbox, pts1, pts2, c1, c2 = test_single_image()
scene_path = 'F:\Dropbox\TobiiGlasses\sample_actions\\'
scene_videos = glob.glob(scene_path + '*.avi')
test_online(scene_videos[3], 0)    

    

