# /bin/env python
# coding: utf-8

# DeepHand model files and inference codes are from: https://github.com/neccam/TF-DeepHand

from __future__ import print_function

import sys
import os
import json
import numpy as np
import scipy.io as sio
import re
import subprocess

#DeepHand features
import cv2
import tensorflow as tf
from deephand import DeepHand

import logging
import argparse

def get2d(openPoseV):
    retVal= np.zeros(shape=(int(len(openPoseV)/3),2))
    for i in range(retVal.shape[0]):
        retVal[i]=(openPoseV[i*3],openPoseV[i*3+1])
    return retVal


def featDeriv(feat, L=2):
    N,_ = feat.shape
    x = np.concatenate((np.ones(shape=(L,1)) * feat[0], feat, np.ones(shape=(L,1))*feat[N-1]))    
    dx = np.zeros(x.shape)
    ii = np.arange(0,N)+L
    S=0
    for k in range(-L, L+1):
        dx[ii]+=k*x[ii+k]
        S+=k*k
    dx/=S
    dx = dx[L:-L]
    return dx

def read_image(imageF):
    return cv2.imread(imageF)

def preprocess_image(img, mean):
    img = cv2.resize(img, (227, 227))
    img = img - mean
    img = img[1:225,1:225,:]
    img = np.expand_dims(img, axis=0)
    return img

def dirType(dir):
    if os.path.isdir(dir):
        return dir
    else:
        raise argparse.ArgumentTypeError(dir)

if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--log-level', default='INFO', help="log-level (INFO|WARN|DEBUG|FATAL|ERROR)")
    argparser.add_argument('-i', '--input_dirs', required=True, type=dirType, nargs='+', help='List of OpenPose output directory')

    argparser.add_argument('--deephand', default='deephand', type=dirType, help='DeepHand model path (must contain deephand_model.npy and onemilhands_mean.npy)')
    argparser.add_argument('--deephand_device', default='cpu:0', help='cpu:0 or gpu:0')

    argparser.add_argument('-c', '--crop_ratio', default=2.0, type=float, help='crop ratio size (1.5, 2.0, ...) ')

    args = argparser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=args.log_level)
    logger = logging.getLogger("DragonFly-ASLR-FeatureExtraction")

    if args.deephand:
        logger.info('Loading DeepHand model from ' + args.deephand)
        deephand_modelF = args.deephand+'/deephand_model.npy'
        deephand_meanF = args.deephand+'/onemilhands_mean.npy'
        deephand_inputNode_placeHolder = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
        deephand_inputNode_np = np.zeros((1, 224, 224, 3))
    
        deephand_model = DeepHand({'data': deephand_inputNode_placeHolder})
        deephand_mean = np.load(deephand_meanF)
        deephand_mean = np.transpose(deephand_mean, (1, 2, 0))    
        deephand_config=tf.ConfigProto(allow_soft_placement=True)
        deephand_config.gpu_options.allow_growth=True
        deephand_device = args.deephand_device

        sess = tf.Session(config=deephand_config)
        with tf.device('/'+deephand_device):
            deephand_model.load(deephand_modelF, sess)

        def getDeepHandFeat(imageF):
            img = read_image(imageF)
            deephand_inputNode_np[0, :, :, :] = preprocess_image(img, deephand_mean)
            loss1_fc, loss2_fc, pool5_7x7_s1 = sess.run([deephand_model.layers['loss1_fc'], 
                                                         deephand_model.layers['loss2_fc'], 
                                                         deephand_model.layers['pool5_7x7_s1']], 
                                                         feed_dict={deephand_inputNode_placeHolder: deephand_inputNode_np})
            return np.squeeze(loss1_fc), np.squeeze(loss2_fc), np.squeeze(pool5_7x7_s1)


    for d in args.input_dirs:
    
        jsonFs = sorted([f for f in os.listdir(d) if f.endswith('.json')])
        jpgFs = sorted([f for f in os.listdir(d) if f.endswith('.jpg') and not f.endswith('_LH.jpg') and not f.endswith('_RH.jpg') ])
        
        if len(jsonFs) != len(jpgFs) or len(jsonFs) < 2:
            logger.error('Number of jsonFs and jpgFs mismatch: ' + d + ' (' + str(len(jsonFs)) + ' != ' + str(len(jpgFs)) + ')' )
            continue

        logger.info('Processing directory: ' + d + ' (' + str(len(jsonFs)) + ' frames)')
    

        template=sio.loadmat('MITLLTemplate.mat')
    
        skel = np.zeros(shape=(len(jsonFs),25,2))
        normskel = np.zeros(skel.shape)
        skel_flipped = np.zeros(skel.shape) 
        normskel_flipped = np.zeros(skel.shape) 
    
        deephand_l_fc1 = np.zeros(shape=(len(jsonFs), 1024))
        deephand_l_fc2 = np.zeros(shape=(len(jsonFs), 1024))
        deephand_l_fc3 = np.zeros(shape=(len(jsonFs), 1024))
        deephand_r_fc1 = np.zeros(shape=(len(jsonFs), 1024))
        deephand_r_fc2 = np.zeros(shape=(len(jsonFs), 1024))
        deephand_r_fc3 = np.zeros(shape=(len(jsonFs), 1024))

        cnt=0
        skippedFrames = []

        for i, f in enumerate(jsonFs):
            logger.debug('Processing: ' + f)

            j = json.load(open(d+'/'+f, 'r'))

            if len(j['people'])<1:
                logger.error('Missing people annotation in jsonF: ' + f)
                skippedFrames.append(i)
                continue

            person = j['people'][0]
            pose = get2d(person['pose_keypoints_2d'])
            face = get2d(person['face_keypoints_2d'])
            leftHand = get2d(person['hand_left_keypoints_2d'])
            rightHand = get2d(person['hand_right_keypoints_2d'])
             
            logger.debug('Nose: ' + str(pose[0]))
            logger.debug('RWrist: ' + str(pose[4]))
            logger.debug('LWrist: ' + str(pose[7]))
     
            for prefix, hand in [('LH', leftHand), ('RH', rightHand)]:
     
                minXY = np.min(hand,0)
                maxXY = np.max(hand,0)
                dxy = np.round(args.crop_ratio * max(maxXY - minXY))
                bxy = np.floor(minXY - (np.array([dxy,dxy]) - (maxXY-minXY))/2)

                logger.debug(prefix + ': ' + str(hand))
                logger.debug(prefix+' Min: ' + str(minXY))
                logger.debug(prefix+' Max: ' + str(maxXY))
                logger.debug(prefix+' Bbox offset: ' + str(bxy) + ', Size: ' + str(dxy))

                cropped_imageF = d + '/' + jpgFs[i].replace('.jpg', '_' + prefix + '.jpg')

                if prefix == 'LH': # left hands needs to be horizontally flipped
                    command = ['convert', '-crop', '%dx%d+%d+%d' % (dxy, dxy, bxy[0], bxy[1]), d + '/' + jpgFs[i], '-resize', '227x227!', '-flop', cropped_imageF]
                else:
                    command = ['convert', '-crop', '%dx%d+%d+%d' % (dxy, dxy, bxy[0], bxy[1]), d + '/' + jpgFs[i], '-resize', '227x227!', cropped_imageF]
                logger.debug(command)        
                subprocess.call(command)

                fc1, fc2, fc3 = getDeepHandFeat(cropped_imageF)
                if prefix == 'LH':
                    deephand_l_fc1[i,:] = np.copy(fc1)
                    deephand_l_fc2[i,:] = np.copy(fc2)
                    deephand_l_fc3[i,:] = np.copy(fc3)
                else:
                    deephand_r_fc1[i,:] = np.copy(fc1)
                    deephand_r_fc2[i,:] = np.copy(fc2)
                    deephand_r_fc3[i,:] = np.copy(fc3)

            pose_norm = np.copy(pose)
            pose_norm -= pose_norm[1]
            if np.sum(pose_norm[0]**2) != 0:
                pose_norm /= np.sqrt(np.sum(pose_norm[0]**2))
            else:
                logger.error('pose_norm[0] all zero: ' + f + '\n' + str(pose_norm) + '\n' + str(pose_norm[0]))
                skippedFrames.append(i)
                continue

            pose_norm[pose==0] = 0
            pose_norm[:,1] = -pose_norm[:,1]
            logger.debug('Normalized LH: ' + str(pose_norm[4,:]))
            logger.debug('Normalized RH: ' + str(pose_norm[7,:]))    

            skel[i,11,:] = pose[4,:]
            skel[i,7,:] = pose[7,:]
            
            normskel[i,11,:] = pose_norm[4,:]
            normskel[i,7,:] = pose_norm[7,:]
    
            skel_flipped[i,11,:] = pose[7,:]
            skel_flipped[i,7,:] = pose[4,:]
    
            normskel_flipped[i,11,:] = pose_norm[7,:]
            normskel_flipped[i,7,:] = pose_norm[4,:]

            cnt+=1

        if cnt != len(jsonFs):
            logger.error('Count mismatch in json and result: ' + d)
            continue

        if np.all(normskel[:,7,:]==0) or np.all(normskel[:,11,:]==0):
            logger.error('Left or right hand all 0: ' + d)

            if np.all(normskel[:,7,:]==0): # LEFT
                # delete HAND IMAGES
                for i, f in enumerate(jsonFs):
                    prefix = 'LH'
                    cropped_imageF = d + '/' + jpgFs[i].replace('.jpg', '_' + prefix + '.jpg')
                    command = ['rm', cropped_imageF]
                    subprocess.call(command)
                # RESET DEEPHAND FEATURE TO 0
                deephand_l_fc1.fill(0)
                deephand_l_fc2.fill(0)
                deephand_l_fc3.fill(0)

            else: # np.all(normskel[:,11,:]==0), RIGHT
                # delete HAND IMAGES
                for i, f in enumerate(jsonFs):
                    prefix = 'RH'
                    cropped_imageF = d + '/' + jpgFs[i].replace('.jpg', '_' + prefix + '.jpg')
                    command = ['rm', cropped_imageF]
                    subprocess.call(command)
                # RESET DEEPHAND FEATURE TO 0
                deephand_r_fc1.fill(0)
                deephand_r_fc2.fill(0)
                deephand_r_fc3.fill(0)

        logger.debug('normskel_l: ' + str(normskel[:,7,:]))
        logger.debug('normskel_r: ' + str(normskel[:,11,:]))
    
        dnormskel_l = featDeriv(np.squeeze(normskel[:,7,:]))
        dnormskel_r = featDeriv(np.squeeze(normskel[:,11,:]))
        logger.debug('dnormskel_l: ' + str(dnormskel_l))
        logger.debug('dnormskel_r: ' + str(dnormskel_r))

        if np.count_nonzero(np.isnan(normskel)) > 0:
            logger.error(str(np.count_nonzero(np.isnan(normskel))) + ' NaNs at normskel')
            normskel[np.isnan(normskel)] = 0
        if np.count_nonzero(np.isnan(normskel_flipped)) > 0:
            logger.error(str(np.count_nonzero(np.isnan(normskel_flipped))) + ' NaNs at normskel_flipped')
            normskel_flipped[np.isnan(normskel_flipped)] = 0
        if np.count_nonzero(np.isnan(dnormskel_l)) > 0:
            logger.error(str(np.count_nonzero(np.isnan(dnormskel_l))) + ' NaNs at dnormskel_l')
            dnormskel_l[np.isnan(dnormskel_l)] = 0
        if np.count_nonzero(np.isnan(dnormskel_r)) > 0:
            logger.error(str(np.count_nonzero(np.isnan(dnormskel_r))) + ' NaNs at dnormskel_r')
            dnormskel_r[np.isnan(dnormskel_r)] = 0

        if np.count_nonzero(np.isnan(deephand_r_fc1)) > 0:
            logger.error(str(np.count_nonzero(np.isnan(deephand_r_fc1))) + ' NaNs at deephand_r_fc1')
            deephand_r_fc1[np.isnan(deephand_r_fc1)] = 0
        if np.count_nonzero(np.isnan(deephand_r_fc2)) > 0:
            logger.error(str(np.count_nonzero(np.isnan(deephand_r_fc2))) + ' NaNs at deephand_r_fc2')
            deephand_r_fc2[np.isnan(deephand_r_fc2)] = 0
        if np.count_nonzero(np.isnan(deephand_r_fc3)) > 0:
            logger.error(str(np.count_nonzero(np.isnan(deephand_r_fc3))) + ' NaNs at deephand_r_fc3')
            deephand_r_fc3[np.isnan(deephand_r_fc3)] = 0

        if np.count_nonzero(np.isnan(deephand_l_fc1)) > 0:
            logger.error(str(np.count_nonzero(np.isnan(deephand_l_fc1))) + ' NaNs at deephand_l_fc1')
            deephand_l_fc1[np.isnan(deephand_l_fc1)] = 0
        if np.count_nonzero(np.isnan(deephand_l_fc2)) > 0:
            logger.error(str(np.count_nonzero(np.isnan(deephand_l_fc2))) + ' NaNs at deephand_l_fc2')
            deephand_l_fc2[np.isnan(deephand_l_fc2)] = 0
        if np.count_nonzero(np.isnan(deephand_l_fc3)) > 0:
            logger.error(str(np.count_nonzero(np.isnan(deephand_l_fc3))) + ' NaNs at deephand_l_fc3')
            deephand_l_fc3[np.isnan(deephand_l_fc3)] = 0
        
        template['RecStrct']['feat'][0,0]['normskeleton'][0,0]['skel'][0,0] = np.copy(normskel.astype(np.float32))
        template['RecStrct']['feat'][0,0]['normskeleton'][0,0]['skel_flipped'][0,0] = np.copy(normskel_flipped.astype(np.float32))
        
        template['RecStrct']['feat'][0,0]['normskeleton'][0,0]['dskel'][0,0] = np.copy(normskel.astype(np.float32))
        template['RecStrct']['feat'][0,0]['normskeleton'][0,0]['dskel'][0,0][:,7,:] = np.copy(dnormskel_l.astype(np.float32))
        template['RecStrct']['feat'][0,0]['normskeleton'][0,0]['dskel'][0,0][:,11,:] = np.copy(dnormskel_r.astype(np.float32))
        
        template['RecStrct']['feat'][0,0]['normskeleton'][0,0]['dskel_flipped'][0,0] = np.copy(normskel.astype(np.float32))
        template['RecStrct']['feat'][0,0]['normskeleton'][0,0]['dskel_flipped'][0,0][:,7,:] = np.copy(dnormskel_r.astype(np.float32))
        template['RecStrct']['feat'][0,0]['normskeleton'][0,0]['dskel_flipped'][0,0][:,11,:] = np.copy(dnormskel_l.astype(np.float32))

        template['RecStrct']['feat'][0,0]['deephand'][0,0]['righthandfc1'][0,0] = np.copy(deephand_r_fc1.astype(np.float32))
        template['RecStrct']['feat'][0,0]['deephand'][0,0]['righthandfc2'][0,0] = np.copy(deephand_r_fc2.astype(np.float32))
        template['RecStrct']['feat'][0,0]['deephand'][0,0]['righthandfc3'][0,0] = np.copy(deephand_r_fc3.astype(np.float32))

        template['RecStrct']['feat'][0,0]['deephand'][0,0]['lefthandfc1'][0,0] = np.copy(deephand_l_fc1.astype(np.float32))
        template['RecStrct']['feat'][0,0]['deephand'][0,0]['lefthandfc2'][0,0] = np.copy(deephand_l_fc2.astype(np.float32))
        template['RecStrct']['feat'][0,0]['deephand'][0,0]['lefthandfc3'][0,0] = np.copy(deephand_l_fc3.astype(np.float32))


        sio.savemat(d+'.mat', template)
