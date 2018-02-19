# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 22:14:39 2016

@author: morepoch

Helper functions
"""

from .misc import loadmat
import numpy as np
import cv2


def draw_str(dst, target, s):
    x, y = target
    #cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 0), thickness = 3)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 2.0, (25, 15, 25), thickness=2)
    
def transform_svm_mat2file(filename):
    """
    Transform the svm model in .mat to .file
    """
    model = loadmat(filename)
    text_file = open(filename[:-4], "w")
    text_file.write("solver_type L2R_LR\n")
    text_file.write("nr_class %d\n" % model['svmmodel']['nr_class'])
    text_file.write("label 1 0\n")
    text_file.write("nr_feature %d\n" % model['svmmodel']['nr_feature'])
    text_file.write("bias %d\n" % model['svmmodel']['bias'])
    text_file.write("w \n")
    for idx in np.arange(model['svmmodel']['w'].shape[0]):       
        text_file.write("%f\n" % model['svmmodel']['w'][idx])
    text_file.close()

def lbp(X, neighbors=8, radius=1.0):
    X = np.asanyarray(X)
    ysize, xsize = X.shape

    # calculate sample points on circle with radius
    sample_points = np.array([[-1, -1, -1, 0, 0, 1, 1, 1], [-1, 0, 1, -1, 1, -1, 0, 1]], dtype=float).T
    sample_points *= radius
    # find boundaries of the sample points
    miny=min(sample_points[:,0])
    maxy=max(sample_points[:,0])
    minx=min(sample_points[:,1])
    maxx=max(sample_points[:,1])
    # calculate block size, each LBP code is computed within a block of size bsizey*bsizex
    blocksizey = np.ceil(max(maxy,0)) - np.floor(min(miny,0)) + 1
    blocksizex = np.ceil(max(maxx,0)) - np.floor(min(minx,0)) + 1
    # coordinates of origin (0,0) in the block
    origy =  (0 - np.floor(min(miny,0))).astype(int)
    origx =  (0 - np.floor(min(minx,0))).astype(int)
    # calculate output image size
    dx = (xsize - blocksizex + 1).astype(int)
    dy = (ysize - blocksizey + 1).astype(int)
    # get center points
    
    C = np.asarray(X[origy:origy+dy,origx:origx+dx], dtype=np.uint8)
    result = np.zeros((dy,dx), dtype=np.uint32)
    for i,p in enumerate(sample_points):
        # get coordinate in the block
        y,x = p + (origy, origx)
        # Calculate floors, ceils and rounds for the x and y.
        fx = np.floor(x)
        fy = np.floor(y)
        cx = np.ceil(x).astype(int)
        cy = np.ceil(y).astype(int)
        # calculate fractional part    
        ty = y - fy
        tx = x - fx
        # calculate interpolation weights
        w1 = (1 - tx) * (1 - ty)
        w2 =      tx  * (1 - ty)
        w3 = (1 - tx) *      ty
        w4 =      tx  *      ty
        # calculate interpolated image
        fx = np.round(fx).astype(int)
        fy = np.round(fy).astype(int)
        N = w1*X[fy:fy+dy,fx:fx+dx]
        N += w2*X[fy:fy+dy,cx:cx+dx]
        N += w3*X[cy:cy+dy,fx:fx+dx]
        N += w4*X[cy:cy+dy,cx:cx+dx]
        # update LBP codes        
        D = N >= C
        result += ((1<<i)*D).astype('uint32')
    return result, np.histogram(result, 256)[0]
        
def compute_nloc_optim(s):
    """
    given a 1-d sequence, compute the accumulated amplitude
    and number of local minimum or maximum
    """
    count_max = 0
    count_min = 0
    pre = 0
    cur = 1
    n = len(s)
    ampl = []
    
    while cur < n:
        
        if cur < n and s[cur] > s[cur-1]:
            # find maximum
            while cur < n and s[cur] >= s[cur-1]:
                cur += 1
            fur = cur
            cur -= 1
            # find 
            while fur < n and s[fur-1] >= s[fur]:
                fur += 1
            fur -= 1
            
            temp = 2*s[cur] - s[fur] - s[pre]
            if fur - cur < n // 2.5 and cur - pre < n // 2.5 and temp > 0.15:                
                ampl.append(temp)
                count_max += 1
            pre = cur
            cur = max([fur, cur+1])
        else:
            while cur < n and s[cur] <= s[cur-1]:
                cur += 1
            fur = cur
            cur -= 1
            # find 
            while fur < n and s[fur-1] <= s[fur]:
                fur += 1
            fur -= 1
            
            temp = -2*s[cur] + s[fur] + s[pre]
            if fur - cur < n // 2.5 and cur - pre < n // 2.5 and temp > 0.15:
                
                ampl.append(temp)
                count_min += 1
            pre = cur
            cur = max([fur, cur+1]) 
    if len(ampl) > 0:
        fea = np.array(ampl)
        fea = np.sum(fea[fea > 0.0])
    else:
        fea = 0.
    return fea, count_max, count_min
