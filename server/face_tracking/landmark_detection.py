# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 22:10:48 2016

@author: kang wang

Implement a landmark detection class.
"""

import cv2
import numpy as np
from liblinearutil import predict, load_model
from .misc import loadmat
from .helpers import transform_svm_mat2file, lbp
import itertools
import os
#%%


def draw_ffp(image, pts):
    """ draw facial landmark points on image """
    for idx in np.arange(pts.shape[0]):
        cv2.circle(image, (int(pts[idx,0].round()), int(pts[idx,1].round())), 2, (0,0,255),  -1)
def draw_pupil(image, pts):
    """ draw pupil position on image """
    cv2.circle(image, (int(pts[0][0].round()), int(pts[0][1].round())), 2, (0,255,0),  -1)

def calw_more_points(pts_init, pts_mean):
    """Used for ffp tracking, given initial pts, update current pts"""
    num_pts = pts_init.shape[0]
    X_mat = np.zeros((2*num_pts, 4))
    for idx in np.arange(num_pts):
        X_mat[2*idx, :] = [pts_init[idx, 0], -pts_init[idx, 1], 1, 0]
        X_mat[2*idx+1, :] = [pts_init[idx, 1], pts_init[idx, 0], 0, 1]
    b_mean = pts_mean.reshape(-1, 1)
    x_final, dummy_1, dummy_2, dummy_3 = np.linalg.lstsq(X_mat, b_mean)
    return x_final
#%%
class PoseEstimation():
    """
    Estimate head pose given 2D landmarks
    """
    def __init__(self):
        ## construct 3D deformable head model
        module_path = os.path.dirname(os.path.realpath(__file__))
        self.model_path = os.path.join(module_path, 'model')
        data = loadmat(os.path.join(self.model_path, 'PCA_51pts.mat'))
        self.DM = {'mu': data['mm'].reshape((-1, 1)), 'coeff': data['coeff']}
        self.num_pts = data['mm'].shape[0] // 3
        self.num_comp = data['coeff'].shape[1] 
        
    def weakIterative_Occlusion(self, pts2d, n_iter=50):
        """
        compute deformable model and head pose simultaneously
        """
        alpha = np.zeros((self.num_comp, 1))
        angle_old = np.zeros((3, 1))
        for idx in range(n_iter):
            # estimate pose given fixed face shape
            pts3d = (self.DM['mu'] + np.dot(self.DM['coeff'], alpha)).reshape((-1, 3))
            
            angle, lambda1, lambda2, T, M = self.weakLinear_Occlusion(pts2d, pts3d)
            # solve deformable coefficient alpha
            
            alpha = self.weakLinear_Alpha(pts2d, angle, lambda1, lambda2, T)
            if np.max(np.abs(angle - angle_old)) / np.pi * 180 < 0.1:
                break
            else:
                angle_old = angle
        return angle
                
    def weakLinear_Occlusion(self, pts2d, pts3d):
        """
        solve pose given 2d and 3d correspondence
        """
        SIGMA = np.concatenate((pts3d.reshape((-1, 3)), np.ones((self.num_pts, 1))), axis=1)
        u = pts2d[:, 0].reshape((-1, 1))
        v = pts2d[:, 1].reshape((-1, 1))
        temp = np.dot(np.linalg.pinv(np.dot(SIGMA.T, SIGMA)), SIGMA.T)
        Qu = np.dot(temp, u)
        Qv = np.dot(temp, v)
        M = np.concatenate((Qu[:3].T, Qv[:3].T))
        T = np.array((Qu[-1], Qv[-1])).reshape((-1, 1))
        #pts2d_e = np.dot(M, pts3d.T).T + T.T
        angle, lambda1, lambda2, R = self.getAnglesM(M)
        return angle, lambda1, lambda2, T, M
    def getRfromAng_roll_pitch_roll(self, angle):
        """
        angle to matrix
        """
        ang1, ang2, ang3 = angle
        R1 = np.array(((np.cos(ang2), 0, np.sin(ang2)), (0, 1, 0), (-np.sin(ang2), 0, np.cos(ang2))))
        R2 = np.array(((1, 0, 0), (0, np.cos(ang1), -np.sin(ang1)),  (0, np.sin(ang1), np.cos(ang1))))
        R3 = np.array(( (np.cos(ang3), -np.sin(ang3), 0),  (np.sin(ang3), np.cos(ang3), 0), (0, 0, 1)))
        R = np.dot(R1, np.dot(R2, R3))
        return R.astype(np.float64)
    def getAnglesR_roll_pitch_yaw(self, R):
        if np.abs(R[0, 2] < 0.1) and np.abs(R[2, 2]) < 0.1:
            ang1 = np.sign(-R[2, 1]) * np.pi / 2
            ang2 = 0
            ang3 = np.arctan(R[2, 0] / R[2, 1])
        elif np.abs(R[2, 2] < 0.1) and np.abs(R[0, 2]) > 0.1:
            ang2 = np.sign(R[0, 2]) * np.pi / 2
            cosbeta = np.sqrt(R[1, 0]**2 + R[1, 1]**2)
            
            ang1 = np.arctan(-R[1, 2] / cosbeta)
            ang3 = np.arctan(R[1, 0] / R[1, 1])
        elif np.abs(R[2, 2]) < 0.1:
            ang3 = np.sign(R[1, 0])*np.pi/2
            cosbeta = np.sqrt(R[1, 0]**2+R[1, 1]**2)
            ang1 = np.atan(-R[1, 2] / cosbeta)
            ang2 = np.atan(R[0, 2] /R[2, 2])
        else:
            ang3 = np.arctan(R[1, 0] / R[1, 1]);
            cosbeta = np.sqrt(R[1, 0]**2 + R[1, 1]**2);
            ang1 = np.arctan(-R[1, 2] / cosbeta);
            ang2 = np.arctan(R[0, 2]/ R[2, 2]);

        return np.array((ang1, ang2, ang3)).reshape((-1, 1))
            
    def getAnglesM(self, M):
        m1 = M[0, :]
        m2 = M[1, :]
        lambda1 = np.linalg.norm(m1)
        lambda2 = np.linalg.norm(m2)
        if lambda1 != 0 and lambda2 != 0 and lambda1 and lambda2:
            m1 = m1 / lambda1
            m2 = m2 / lambda2
        m3 = np.cross(m1, m2)
        R = np.array((m1, m2, m3))
        if not np.isnan(R[0][0]):
            U, S, V = np.linalg.svd(R)
            R = np.dot(U, V)
            angle = self.getAnglesR_roll_pitch_yaw(R)
        else:
            angle = np.zeros((3, 1))
        return angle, lambda1, lambda2, R
    
    def weakLinear_Alpha(self, pts2d, angle, lambda1, lambda2, T):
        """
        solve deformable coefficients
        """
        R = self.getRfromAng_roll_pitch_roll(angle)
        M = R[:2, :]
        M[0, :] *= lambda1
        M[1, :] *= lambda2
        
        SIGMA = self.DM['coeff']
        mm = self.DM['mu'].reshape((-1, 3)).T
        pts_t = np.array(pts2d - np.dot(M, mm).T - T.T).reshape((-1, 2))
        S = np.zeros((0, self.num_comp))
        for idx in range(self.num_pts):
            S = np.concatenate((S, np.dot(M, SIGMA[idx*3:(idx+1)*3, :])))
        
        U = pts_t.reshape((-1, 1))
        alpha = np.dot(np.linalg.pinv(S), U)
        return alpha
        
#%%
class LandmarkDetection():
    """ Landmark Detection Class """

    def __init__(self):
        
        ## load SIFT feature extractor
        if cv2.__version__[0] == '2':
            self.sift_extractor = cv2.DescriptorExtractor_create("SIFT")
        else:
            self.sift_extractor = cv2.xfeatures2d.SIFT_create()
        module_path = os.path.dirname(os.path.realpath(__file__))
        self.model_path = os.path.join(module_path, 'model')
        ## construct face detector
        self.face_detector = {}
        self.face_detector['detector'] = cv2.CascadeClassifier(os.path.join(self.model_path, 'haarcascade_frontalface_alt2.xml'))
        self.face_detector['minNeighbors'] = 4
        self.face_detector['minSize'] = (20, 20)
        # confidence LBP
        transform_svm_mat2file(os.path.join(self.model_path, 'confidence_face_LBP.mat'))
        self.face_detector['confidence_LBP'] = {'model':load_model(os.path.join(self.model_path, 'confidence_face_LBP')), 
                                            'thre': 0.01}                                       
        # confidence SIFT
        matdata = loadmat(os.path.join(self.model_path, 'meanImg_sing_20.mat'))
        kpt_small = []
        kpt_small.append(cv2.KeyPoint(20, 20, 3.1, -1, 1, 0, 1))
        descriptors_mean_small = np.zeros((1, 0))
        for idx_pt in np.arange(17, 68):
            _, temp = self.sift_extractor.compute(matdata['meanImg'][idx_pt]['roi'], kpt_small)
            descriptors_mean_small = np.concatenate((descriptors_mean_small, temp.reshape((1, -1))), axis=1)
        # Lower the confidence, higher the tolerance.  
        # For this version, we can set different threhsolds for detection and tracking phases. 
        # For near-frontal face, the confidence is about 0.65. 
        # For non-frontal faces the confidence is about 0.55. 
        self.face_detector['confidence_SIFT'] = {'descriptor':descriptors_mean_small, 
                                                 'thre_detect': 0.56,  
                                                 'thre_track': 0.46}     
        ## construct landmark detector/tracker
        self.face_lmks_model = {}

        prior = loadmat(os.path.join(self.model_path, 'Prior.mat'))
        mm = prior['mm'].reshape(-1, 2)
        self.face_lmks_model['mm'] = mm[range(17, 68), :].flatten()
        

        self.face_lmks_model['num_pts'] = 51 
        self.face_lmks_model['num_iter'] = 4
        self.face_lmks_model['norm_width'] = 200
        self.face_lmks_model['margin'] = 50

        self.face_lmks_model['para_detect'] = {}
        self.face_lmks_model['para_track'] = {}

        for it in range(self.face_lmks_model['num_iter']):
            x = loadmat(os.path.join(self.model_path, 'Detect_it%d.mat'%(it+1)))
            self.face_lmks_model['para_detect'][it] = x
            x = loadmat(os.path.join(self.model_path, 'Tracking_it%d.mat'%(it+1)))
            self.face_lmks_model['para_track'][it] = x

        ## face motion parameters
        self.face_motion = {'queue_size': 60,
                            'threshold': 10,
                            'queue':100*np.random.normal(size=(60, self.face_lmks_model['num_pts']*2))}

        ## construct eye detector
        self.eye_detector_SDM = {}
        self.eye_detector_SDM['norm_width'] = 25
        self.eye_detector_SDM['num_pts'] = 27
        self.eye_detector_SDM['num_iter'] = 3
        matdata = loadmat(os.path.join(self.model_path, 'leftPara_20160516.mat'))
        mmc = matdata['leftmm'].reshape((-1, 2))
        pair_set = np.array([[i, j] for i in np.arange(self.eye_detector_SDM['num_pts']) for j in np.arange(i+1, self.eye_detector_SDM['num_pts'])])
        pt1 = mmc[pair_set[:, 0], :]
        pt2 = mmc[pair_set[:, 1], :]
        mmshapefea1 = (pt1 - pt2).reshape((1, -1))
        mmshapefea2 = np.sqrt(np.sum((pt1 - pt2)**2, axis=1))
        self.eye_detector_SDM['pair_set'] = pair_set
        self.eye_detector_SDM['mmshapefea1'] = mmshapefea1
        self.eye_detector_SDM['mmshapefea2'] = mmshapefea2
        #for it in range(self.eye_detector_SDM['num_iter']):
        self.eye_detector_SDM['para'] = matdata['leftPara']
        self.eye_detector_SDM['mm'] = matdata['leftmm']
        
        
    def compute_confidence(self, img, pts, conf_model, threshold):
        """
        compute confidence of current estimated pts
        """
        mmc = self.face_lmks_model['mm'].reshape(-1, 2)
        w = calw_more_points(pts, mmc)
        theta = -np.arctan(w[1]/w[0])[0]
        center = np.array((img.shape[1]/2., img.shape[0]/2.))
        M = cv2.getRotationMatrix2D(tuple(center), theta/np.pi*180, 1)
        img_rotate = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        rotM = np.array(([np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]))
        
        pts_rotate = np.dot((pts-center.reshape((1, -1))), rotM) + \
                     np.array((img_rotate.shape[1]/2., img_rotate.shape[0]/2.)).reshape((1, -1))
        # compute sift fea
        tkp = []
        for idx in np.arange(pts.shape[0]):
            tkp.append(cv2.KeyPoint(pts_rotate[idx, 0], pts_rotate[idx, 1], 3.1, -1, 1, 0, 1))
        tkp, tdp = self.sift_extractor.compute(img_rotate, tkp)
        tdp = tdp.reshape(1, -1)/255.
        if np.linalg.norm(tdp) == 0:
            confidence = 0
        else:
            confidence = np.dot(tdp, conf_model.T) / \
                         (np.linalg.norm(tdp)*np.linalg.norm(conf_model))
            confidence = confidence[0][0]
        #print(conf_model)
        if confidence < threshold:
            flag_succ = 2
        else:
            flag_succ = 0
        return flag_succ, confidence
    def compute_face_size(self, pts):
        """
        Given facial landmarks, compute a normalized face size
        pts: 51 facial landmarks
        """
        mm = pts.mean(axis=0).reshape((1, -1))
        dis = np.sqrt(np.sum((pts - mm)**2, axis=1))
        return np.median(dis)
        
    def face_detect(self, img):
        """
        Detect face bounding box given image
        img: input image
        """
        # convert to gray
        if img.ndim > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # detect face      
        bboxes = self.face_detector['detector'].detectMultiScale(img,
                                                minNeighbors=self.face_detector['minNeighbors'], 
                                                minSize=self.face_detector['minSize'])
        if len(bboxes) == 0:
            #print('No face is detected')
            return np.zeros((0, 4))
        # else, select appropriate face
        # exclude very small bounding box
        index_face_size = (-bboxes[:, 2]).argsort() # descending order
        bboxes = bboxes[index_face_size, :]
        for idx in np.arange(1, bboxes.shape[0]):
            if bboxes[idx, 2] <= np.round(bboxes[0, 2]*0.3):
                bboxes = bboxes[:idx, :]
                break
            
        # compute confidence for each remaining bbox
        final_bboxes = np.zeros((0, 4))
        C = []
        for idx in np.arange(bboxes.shape[0]):
            bbox = bboxes[idx, :]
            im_cut = img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
            im_cut = cv2.resize(im_cut, (160, 160), interpolation=cv2.INTER_CUBIC)
            _, descriptor = lbp(im_cut)
            descriptor = descriptor.astype(float)/np.sum(descriptor)
            descriptor = list(descriptor)
            _, _, confidence = predict([0], [descriptor], self.face_detector['confidence_LBP']['model'], '-b 1 -q')
            if confidence[0][0] < self.face_detector['confidence_LBP']['thre']:
                continue
            C.append(confidence[0][0])
            final_bboxes = np.concatenate((final_bboxes, bbox.reshape((1, -1))))
        
            
                
        if final_bboxes.shape[0] == 0:
            return final_bboxes
        
        # choose largest and best one
        #index_face_size = (-final_bboxes[:, 2]).argsort() # descending order
        #final_bboxes = final_bboxes[index_face_size, :]
        #C = C[index_face_size]
        maxC = np.max(C)
        for idx in np.arange(final_bboxes.shape[0]):
            if C[idx] - maxC > -0.05:
                bbox = final_bboxes[idx, :].reshape((1, -1))
                break
        return bbox
        
    def ffp_detect(self, img):
        """
        Given the image, detect the facial landmark points
        args:
            img: input image
        return:
            x_est: (num_pts, 2)
            flag_succ: succesful detection flag
            confidence: condidence of detection
        """
        # convert to gray
        if img.ndim > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        # detect face first
        bbox = self.face_detect(img).flatten()
        num_pts = self.face_lmks_model['num_pts']
        norm_width = self.face_lmks_model['norm_width']
        num_iter = self.face_lmks_model['num_iter']
        if bbox.shape[0] == 0:
            pts = np.zeros((num_pts, 2))
            return pts, 2, 0
        

        # obtain normalized face image and bounding box
        face_scale = norm_width/bbox[2]
        img = cv2.resize(img, None, fx=face_scale, fy=face_scale, interpolation=cv2.INTER_CUBIC)        
        bbox_norm = (bbox*face_scale).round().astype(np.uint16)
        cut_x1 = max([0, bbox_norm[0] - self.face_lmks_model['margin']])
        cut_x2 = min([bbox_norm[0] + bbox_norm[2] + self.face_lmks_model['margin'], img.shape[1]-1])
        cut_y1 = max([0, bbox_norm[1] - self.face_lmks_model['margin']])
        cut_y2 = min([bbox_norm[1] + bbox_norm[3] + self.face_lmks_model['margin'], img.shape[0]-1])
        im_cut = img[cut_y1:cut_y2, cut_x1:cut_x2]
        bbox_cut = bbox_norm.copy()
        bbox_cut[0] = bbox_cut[0] - cut_x1 + 1
        bbox_cut[1] = bbox_cut[1] - cut_y1 + 1

        # detect facial landmarks with cascade framework
        for it in np.arange(num_iter):
            if it == 0:
                x0_norm = np.zeros((num_pts*2))
                x0_norm[0::2] = self.face_lmks_model['mm'][0::2] + bbox_cut[0] + bbox_cut[2]/2.0
                x0_norm[1::2] = self.face_lmks_model['mm'][1::2] + bbox_cut[1] + bbox_cut[3]/2.0
            # compute features
            temp = x0_norm.reshape(-1, 2)
            tkp = []
            for idx in range(temp.shape[0]):
                tkp.append(cv2.KeyPoint(temp[idx, 0], temp[idx, 1], 5.2, -1, 1, 0, 1))
            tkp, tdp = self.sift_extractor.compute(im_cut, tkp)
            tdp = tdp.reshape(1, -1)
            tdp = np.append(1, tdp/255.0)
            V_diff = np.dot(self.face_lmks_model['para_detect'][it]['R'], tdp)
            x0_norm = x0_norm + V_diff
            
        # confidence, evaluate the quality of facial landmark detection
        flag_succ, confidence =  self.compute_confidence(im_cut, x0_norm.reshape((-1, 2)), 
                                                         self.face_detector['confidence_SIFT']['descriptor'],
                                                         self.face_detector['confidence_SIFT']['thre_detect'])
        if flag_succ == 0:
            x0_norm = x0_norm.reshape((-1, 2))
            x_est = (x0_norm + np.array([cut_x1-1, cut_y1-1]).reshape((-1, 2)))/face_scale 
        else:
            x_est = np.zeros((num_pts, 2))
        return x_est.reshape((-1, 2)), flag_succ, confidence

    def ffp_track(self, img, pts_init):
        """
        Given the detected ffp for the last frame, detect the ffp for current frame.
        img: input image
        pts_init: detected ffp from last frame
        """
        num_iter = self.face_lmks_model['num_iter']
        num_pts = self.face_lmks_model['num_pts']  
        if img.ndim > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        mmc = self.face_lmks_model['mm'].reshape(-1, 2)
        w = calw_more_points(pts_init, mmc)
        face_scale = np.linalg.norm(w[0:2])
        img = cv2.resize(img, None, fx=face_scale, fy=face_scale, interpolation=cv2.INTER_CUBIC)
        pts_init = (pts_init*face_scale).round()
        
        cut_x1 = np.round(max([1, pts_init[:, 0].min() - self.face_lmks_model['margin']])).astype(int)
        cut_x2 = np.round(min([pts_init[:, 0].max() + self.face_lmks_model['margin'], img.shape[1]])).astype(int)
        cut_y1 = np.round(max([1, pts_init[:, 1].min() - self.face_lmks_model['margin']])).astype(int)
        cut_y2 = np.round(min([pts_init[:, 1].max() + self.face_lmks_model['margin'], img.shape[0]])).astype(int)
        im_cut = img[cut_y1:cut_y2, cut_x1:cut_x2]
        pts_init[:, 0] = pts_init[:, 0] - cut_x1 + 1
        pts_init[:, 1] = pts_init[:, 1] - cut_y1 + 1
        # detect facial landmarks with cascade framework
        for it in np.arange(num_iter):
            if it == 0:
                w = calw_more_points(mmc, pts_init)
                scale = np.linalg.norm(w[0:2])
                x0_norm = scale*mmc + np.array([w[2, 0], w[3, 0]])
                x0_norm = x0_norm.reshape(1, -1)
            # compute features
            temp = x0_norm.reshape(-1, 2)
            tkp = []
            for idx in range(temp.shape[0]):
                if it < 2:
                    tkp.append(cv2.KeyPoint(temp[idx, 0], temp[idx, 1], 5.2, -1, 1, 0, 1))
                else:
                    tkp.append(cv2.KeyPoint(temp[idx, 0], temp[idx, 1], 3.1, -1, 1, 0, 1))
            tkp, tdp = self.sift_extractor.compute(im_cut, tkp)
            tdp = tdp.reshape(1, -1)
            V_diff = np.dot(self.face_lmks_model['para_track'][it]['R'], np.append(1, tdp/255.0))
            x0_norm = x0_norm + V_diff
            
        # confidence, evaluate the quality of facial landmark tracking
        flag_succ, confidence =  self.compute_confidence(im_cut, x0_norm.reshape((-1, 2)), 
                                                         self.face_detector['confidence_SIFT']['descriptor'],
                                                         self.face_detector['confidence_SIFT']['thre_track'])
        if flag_succ == 0:
            x0_norm = x0_norm.reshape((-1, 2))
            x_est = (x0_norm + np.array([cut_x1-1, cut_y1-1]).reshape((-1, 2)))/face_scale 
        else:
            x_est = np.zeros((num_pts, 2))
        return x_est.reshape((-1, 2)), flag_succ, confidence
        
    def pupil_detect_single_eye(self, img, pts, eye='left'):
        """ Detect pupil position given image and detected ffp
            @img: input image
            @pts: detected ffp
            @eye: indicate left or right eye, default for left eye
        """
        if eye == 'left':
            idx_1, idx_2 = 19, 22
        else:
            idx_1, idx_2 = 25, 28

        eye_width = abs(pts[idx_1, 0] - pts[idx_2, 0])
        if eye_width < 10:
            return np.zeros((1, 2))
        scale = self.SINGLE_EYE_NORM_WIDTH/eye_width
        cut_x1 = max([1, pts[idx_1, 0] - self.face_margin])
        cut_x2 = min([pts[idx_2, 0] + self.face_margin, img.shape[1]-1])
        cut_y1 = max([1, pts[idx_1, 1] - self.face_margin])
        cut_y2 = min([pts[idx_2, 1] + self.face_margin, img.shape[0]-1])
        im_cut = img[round(cut_y1)-1:round(cut_y2), round(cut_x1)-1:round(cut_x2)]
        #interpolation = cv2.INTER_CUBIC
        im_norm = cv2.resize(im_cut, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        avg_eye = [(cut_x2 - cut_x1)/2*scale, (cut_y2 - cut_y1)/2*scale]
        x0_norm = np.zeros((1, 6))
        x0_norm[0::2] = self.para_pupil_detect['eyeMean'][0::2] + avg_eye[0]-1
        x0_norm[1::2] = self.para_pupil_detect['eyeMean'][1::2] + avg_eye[1]-1
        for it in np.arange(self.num_cascade_iter):
            temp = x0_norm.reshape(-1, 2)
            tkp = []
            for idx in range(temp.shape[0]):
                tkp.append(cv2.KeyPoint(temp[idx, 0], temp[idx, 1], 5, -1, 1, 0, 1))
            tkp, tdp = self.SIFT_EXTRACTOR.compute(im_norm, tkp)
            tdp = tdp.reshape(1, -1)
            V_diff = np.dot(self.para_pupil_detect['eyePara'][0, it][0], np.append(1, tdp/255.0))
            x0_norm = x0_norm + V_diff
        x0_norm_l = x0_norm.reshape((-1, 2))/scale + [(cut_x1 - 2), (cut_y1 - 2)]
        return x0_norm_l[0, :].reshape(1, -1)
    def pupil_detect_in_eyepatch(self, eye_patch, rc, lc):
        """ Detect pupil position on a given eye patch
            @eye_patch: input eyepatch
            @rc: right eye corner
            @lc: left eye corner
        """

        eye_width = abs(rc[0]- lc[0])
        if eye_width < 10:
            return np.zeros((1, 2))
        scale = self.SINGLE_EYE_NORM_WIDTH/eye_width

        #interpolation = cv2.INTER_CUBIC
        im_norm = cv2.resize(eye_patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        avg_eye = [eye_patch.shape[0]*scale, eye_patch.shape[1]*scale]

        x0_norm = np.zeros((1, 6))
        x0_norm[0::2] = self.para_pupil_detect['eyeMean'][0::2] + avg_eye[0]-1
        x0_norm[1::2] = self.para_pupil_detect['eyeMean'][1::2] + avg_eye[1]-1
        for it in np.arange(self.num_cascade_iter):
            temp = x0_norm.reshape(-1, 2)
            tkp = []
            for idx in range(temp.shape[0]):
                tkp.append(cv2.KeyPoint(temp[idx, 0], temp[idx, 1], 5, -1, 1, 0, 1))
            tkp, tdp = self.SIFT_EXTRACTOR.compute(im_norm, tkp)
            tdp = tdp.reshape(1, -1)
            V_diff = np.dot(self.para_pupil_detect['eyePara'][0, it][0], np.append(1, tdp/255.0))
            x0_norm = x0_norm + V_diff
        x0_norm = x0_norm.reshape((-1, 2))/scale
        return x0_norm[0, :]
    def pupil_detect_in_image(self, img, pts):
        """
        detect pupils in an image given landmarks
        """
        margin = 30
        def crop_eye(eye):
            if eye == 'left':
                idx_1, idx_2 = 19, 22
            else:
                idx_1, idx_2 = 25, 28
            origin = np.array((max([1, pts[idx_1, 0] - margin]), max([1, pts[idx_1, 1] - margin])))    
            cut_x1 = np.round(max([1, pts[idx_1, 0] - margin])).astype(int)
            cut_x2 = np.round(min([pts[idx_2, 0] + margin, img.shape[1]-1])).astype(int)
            cut_y1 = np.round(max([1, pts[idx_1, 1] - margin])).astype(int)
            cut_y2 = np.round(min([pts[idx_2, 1] + margin, img.shape[0]-1])).astype(int)
            im_cut = img[cut_y1-1:cut_y2, cut_x1-1:cut_x2]
            
            return im_cut, pts[idx_1, :] - origin, pts[idx_2, :] - origin, origin
        
        eye_left, rc_left, lc_left, origin_left = crop_eye('left')
        eye_right, rc_right, lc_right, origin_right = crop_eye('right')
        #print(eye_left.shape, rc_left, lc_left, origin_left)
        fea_left = self.eye_feature_detection_in_eyepatch(eye_left, rc_left, lc_left, 'left')
        fea_right = self.eye_feature_detection_in_eyepatch(eye_right, rc_right, lc_right, 'right')
        return fea_left[-1, :]+origin_left-1, fea_right[-1, :]+origin_right-1
        
    def eye_feature_detection_in_eyepatch(self, eye_patch, rc, lc, lr):
        """
        Detect eye features on a given eye patch
        args:
            eye_patch: input eye patch
            rc: right eye corner
            lc: left eye corner
            lr: left or right eye
        Return:
            Detected 27 eye features, including 16 iris contour points, 10 eyelid points and 1 pupil center
        """
        # left right
        if rc[0] < 0 or lc[0] < 0:
            return np.zeros((self.eye_detector_SDM['num_pts'], 2))
        mmshapefea1 = self.eye_detector_SDM['mmshapefea1']
        mmshapefea2 = self.eye_detector_SDM['mmshapefea2']
        para = self.eye_detector_SDM['para']
        mm = self.eye_detector_SDM['mm']
#        else:
#            mmshapefea1 = self.eye_lmks_model['mmshapefea1_right']
#            mmshapefea2 = self.eye_lmks_model['mmshapefea2_right']
#            para = self.eye_lmks_model['rightPara']
#            mm = self.eye_lmks_model['rightmm'] 
        # some constants
        norm_width = self.eye_detector_SDM['norm_width'] = 25
        num_pts = self.eye_detector_SDM['num_pts'] = 27
        
        eye_width = abs(rc[0]- lc[0])
        
        scale = norm_width / eye_width
        #print(scale)
        # matlab imresize default use bicubic
        im_norm = cv2.resize(eye_patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        avg_eye = (rc + lc)/2*scale;

        x0_norm = np.zeros((1, num_pts*2))

        x0_norm[0, 0::2] = mm[0::2] + avg_eye[0] 
        x0_norm[0, 1::2] = mm[1::2] + avg_eye[1] 
        c0 = np.ones((1, num_pts))
        for it in np.arange(3):
            temp = x0_norm.reshape(-1, 2)
            tkp = []
            for idx in range(temp.shape[0]):
                tkp.append(cv2.KeyPoint(temp[idx, 0], temp[idx, 1], 3, -1, 1, 0, 1))
            tkp, tdp = self.sift_extractor.compute(im_norm, tkp)
            F = tdp.reshape(1, -1)/255.
            xxtemp = temp
            all_combos = np.array(list(itertools.combinations(range(num_pts), 2)))
            pt1, pt2 = xxtemp[all_combos[:, 0], :], xxtemp[all_combos[:, 1], :]
            D = np.concatenate((pt1[:, 0] - pt2[:, 0], pt1[:, 1] - pt2[:, 1]), axis=0)
            D = D - mmshapefea1
            D = D/np.append(mmshapefea2, mmshapefea2)
            
            # update C
            C = np.dot(para[it]['T'], np.concatenate((np.ones((1,1)), F, D.reshape((1, -1))), axis=1).T).T
            c0 = np.min(np.max(c0 + C, 0).reshape((1, -1)), 0).reshape((1, -1))
            C = np.tile(c0, (128, 1))
            C = np.reshape(C, (1, 128*num_pts))
            F = np.multiply(F, C)

            V_diff = np.dot(para[it]['R'], np.concatenate((np.ones((1,1)), F, D.reshape((1, -1))), axis=1).T).T
            x0_norm = x0_norm + V_diff
        x0_norm = x0_norm.reshape((-1, 2))/scale
        return x0_norm
def test_pose_estimation():
    filename = 'F:\\Dropbox\\CISL\\matlab-code\\head_nodding_data_collection\\head_nodding_data1.avi'
    cap = cv2.VideoCapture(filename)
    ffp_flag = 1
    engine = LandmarkDetection()
    pose_engine = PoseEstimation()
    angle_all = np.zeros((0, 4))
    count = 0
    while cap.isOpened():
        ret, img = cap.read()
        count += 1
        print(count)
        if not ret or count >= 5000:
            break
        img = np.fliplr(img)
        #print(img.shape)
        if  ffp_flag > 0:
            pts, ffp_flag, confidence = engine.ffp_detect(img)
        else:
            pts, ffp_flag, confidence = engine.ffp_track(img, pts)
        if np.sum(pts) > 0:
            angle = pose_engine.weakIterative_Occlusion(pts)
            angle = np.concatenate((np.array(count).reshape((1, -1)), angle.reshape((1, -1))), axis=1)
            angle_all = np.concatenate((angle_all, angle))
    return angle_all    
    #angle_all = test_pose_estimation()
        
#print(inst.getRfromAng_roll_pitch_roll([1.,2.,3.]))
