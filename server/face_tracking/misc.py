# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 22:19:29 2016

@author: kang wang
"""

import scipy.io as sio
import scipy
import numpy as np
import matplotlib.pyplot as plt
import cv2
def rodrigues(r):
    def S(n):
        Sn = np.array([[0,-n[2],n[1]],[n[2],0,-n[0]],[-n[1],n[0],0]])
        return Sn
    theta = scipy.linalg.norm(r)
    if theta > 1e-30:
        n = r/theta
        Sn = S(n)
        R = np.eye(3) + np.sin(theta)*Sn + (1-np.cos(theta))*np.dot(Sn, Sn)
    else:
        Sr = S(r)
        theta2 = theta**2
        R = np.eye(3) + (1-theta2/6.)*Sr + (.5-theta2/24.)*np.dot(Sr, Sr)
    return np.mat(R)
def ind2sub(array_shape, ind):
    rows = (ind // array_shape[1])
    cols = (ind % array_shape[1]) # or numpy.mod(ind.astype('int'), array_shape[1])
    return (rows, cols)
def savemat(filename, data):
    """
    directly call sio.savemat
    """
    sio.savemat(filename, data)
def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
        elif isinstance(dict[key], np.ndarray) :
            new_dict = {}
            for idx in range(len(dict[key])):
                item = dict[key][idx]
                if isinstance(item, sio.matlab.mio5_params.mat_struct):
                    new_dict[idx] = _todict(item)
            if new_dict:
                dict[key] = new_dict
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

def crop_eye(img, rc, lc, aspect_ratio=(5, 3), pad_ratio=(4, 1)):
    """
    Crop eye image given two eye corners
    input:
        img: face or whole image
        rc: right eye corner from person (left in image)
        lc: left eye corner
        aspect_ratio: width:height
        pad_ratio: pad the left and right region outside eye corners. 
                   eye width : pad width
    return:
        cropped eye image or None(exceed boundary)
    """
    ec = (rc + lc) / 2.
    ec_width = abs(rc[0]- lc[0])
    pad_width = ec_width/pad_ratio[0]*pad_ratio[1]
    eye_width = np.round(ec_width + 2*pad_width)
    eye_width = eye_width + (4 - eye_width % 4) # multiple of 4
    eye_height = eye_width  / aspect_ratio[0] * aspect_ratio[1]
    origin = np.array([ec[0] - eye_width/2., ec[1] - eye_height/2.]).round().astype(int)
    eye_width = eye_width.astype(int)
    eye_height = eye_height.astype(int)
    #print(origin)
    if origin[0] >= 0 and origin[1] >= 0 and \
       origin[0]+eye_width < img.shape[1] and \
       origin[1]+eye_height < img.shape[0]:   
        return img[origin[1]:origin[1]+eye_height, origin[0]:origin[0]+eye_width], origin
    else:
        return None, None
    
def crop_face(img, pts, pad_ratio=(10, 1)):
    """
    Crop face image given 51 facial landmarks
    input:
        img:  whole image
        pts: 51 landmarks
        pad_ratio: pad the left and right region outside eye corners. 
                   face width : pad width
    return:
        cropped eye image or None(exceed boundary)
    """
    center = np.mean(pts, 0)
    size = np.max([np.max(pts[:, 0]) - np.min(pts[:, 0]), np.max(pts[:, 1]) - np.min(pts[:, 1])])
    pad_size = size/pad_ratio[0]*pad_ratio[1]
    size = np.round(size + 2*pad_size)
  
    origin = np.array([center[0] - size/2., center[1] - size/2.]).round().astype(int)
    size = size.astype(int)

    if origin[0] >= 0 and origin[1] >= 0 and \
       origin[0] + size < img.shape[1] and \
       origin[1] + size< img.shape[0]:   
        return img[origin[1]:origin[1]+size, origin[0]:origin[0]+size], origin
    else:
        return None, None
def crop_face_warp(img, pts, anchor=np.array([(290, 200),(350, 200),(320, 270)]), 
                             pad_ratio=(10, 1)):
    """
    Crop face image given 51 facial landmarks, warp face images give 3 anchor
    points, two eye centers and one mouth centers.
    input:
        img:  whole image
        pts: 51 landmarks
        anchor: 3 anchor points
        pad_ratio: pad the left and right region outside eye corners. 
                   face width : pad width
    return:
        cropped eye image or None(exceed boundary)
    """
    import cv2
    if np.sum(pts) == 0:
        return None, None
    eye_right = np.mean(pts[19:25, :], 0)
    eye_left = np.mean(pts[25:31, :], 0)
    mouth = np.mean(pts[32:51, :], 0)
    X = np.array([eye_right, eye_left, mouth])
    X = np.concatenate((X, np.ones((3, 1))), 1)
    mapMatrix = np.linalg.lstsq(X, anchor)[0]
    dst = cv2.warpAffine(img, mapMatrix.T, dsize=img.shape) 
    ygap = np.round((anchor[2, 1] - anchor[0, 1])/2).astype(int)
    xgap = (((anchor[2, 1] - anchor[0, 1]) + 2*ygap - (anchor[1, 0] - anchor[0, 0])) / 2).astype(int)
   # print(xgap, ygap)
    face = dst[anchor[0, 1]-ygap:anchor[2, 1]+ygap, anchor[0, 0]-xgap:anchor[1, 0]+xgap]
    return face
class Draw():
    """
    Draw arbitray shapes, images, texts, canvas
    """
    def draw_eye_lmks(self, eye, pts, name='', text=False):
        """ 
        draw 27 eye landmarks on eye image
        """
        plt.figure()
        plt.imshow(eye, cmap='gray')
        shape = pts.reshape((-1, 2))
        for idx in range(10):
            plt.plot(shape[idx, 0], shape[idx, 1], 'b.', markersize=12)
            if text:
                plt.text(shape[idx, 0], shape[idx, 1], str(idx))
        for idx in range(10, pts.shape[0]):
            plt.plot(shape[idx, 0], shape[idx, 1], 'g.', markersize=8)
            if text:
                plt.text(shape[idx, 0], shape[idx, 1], str(idx))
        
        plt.plot(shape[-1, 0], shape[-1, 1], 'r*', markersize=12)
        if text:
            plt.text(shape[-1, 0], shape[-1, 1], str(pts.shape[0]))
        plt.title(name)
    def draw_face_with_gaze(self, face, gaze):
        plt.figure(figsize=(6, 10))
        face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
        plt.imshow(face, cmap='gray')
        plt.title('%5.0f, %5.0f'%(gaze[0], gaze[1]), fontsize=12)
        
    def draw_grid_eye_lmks(self, eye, pts, name='', n_grid=10):
        """
        draw grid of eye and corresponding landmarks
        """
        n_image = eye.shape[0]
        n_dim = (36, 60)
        assert n_image == n_grid**2, 'Please provide %d images!'%(n_grid**2)
        
        canvas = np.zeros((n_dim[0]*n_grid, n_dim[1]*n_grid))
        eyelid = np.zeros((0, 2))
        iris = np.zeros((0, 2))
        pupil = np.zeros((0, 2))
        for idx in range(n_image):
            row, col = ind2sub((n_grid, n_grid), idx)
            canvas[row*n_dim[0]:(row+1)*n_dim[0], col*n_dim[1]:(col+1)*n_dim[1]] = eye[idx, :].reshape(n_dim)
            shape = pts[idx, :].reshape((-1, 2))
            shape = shape + np.array([col*n_dim[1], row*n_dim[0]]).reshape((1, -1))
            eyelid = np.concatenate((eyelid, shape[:10, :]))
            iris = np.concatenate((iris, shape[10:-1, :]))
            pupil = np.concatenate((pupil, shape[-1, :][None, :]))
        print(canvas.shape)   
        plt.imshow(canvas, cmap='gray')
        plt.plot(eyelid[:, 0], eyelid[:, 1], 'b.', markersize=12) 
        plt.plot(iris[:, 0], iris[:, 1], 'g.', markersize=8)
        plt.plot(pupil[:, 0], pupil[:, 1], 'r*', markersize=12)

  