##############################################################################
# Copyright (c) 2013, Konstantinos Makantasis
# All rights reserved.
#
# Distributed under the terms of the BSD Simplified License
#
##############################################################################

import cv2
import numpy as np
from skimage.filter.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte

def kmFindEdges(frame, minThresh, ratio, window):
    maxThresh = ratio*minThresh
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    edgeCanny = cv2.Canny(gray, minThresh, maxThresh)
    edgeCanny = cv2.convertScaleAbs(edgeCanny)
    ret, edgeCanny = cv2.threshold(edgeCanny, 128, 1, type=cv2.THRESH_BINARY)    
    
    grayBlurred = cv2.GaussianBlur(gray, (3,3), 0)
    sobelX = cv2.Sobel(grayBlurred,cv2.CV_16S,1,0,ksize=3)
    sobelX = cv2.convertScaleAbs(sobelX)
    sobelY = cv2.Sobel(grayBlurred,cv2.CV_16S,0,1,ksize=3)
    sobelY = cv2.convertScaleAbs(sobelY)    
    sobelXY = cv2.addWeighted(sobelX, 0.5, sobelY, 0.5, gamma=0)
    
    edgeImg = cv2.multiply(edgeCanny, sobelXY)    
    
    return edgeImg


def kmFindFrequency(frame):
    frameBlurred = cv2.GaussianBlur(frame, (3,3), 0)
    gray = cv2.cvtColor(frameBlurred, cv2.COLOR_BGR2GRAY)
    laplacianImg = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    laplacianImg = cv2.convertScaleAbs(laplacianImg)

    return laplacianImg    


def kmFindLines(frame):
    kernel = np.zeros((5,5))
    kernel[0,2] = 1
    kernel[1,2] = 2
    kernel[2,0] = 1
    kernel[2,1] = 2
    kernel[2,2] = 4
    kernel[2,3] = 2
    kernel[2,4] = 1
    kernel[3,2] = 2
    kernel[4,2] = 1
    kernel = kernel/16.0
    
    edgeImg = kmFindEdges(frame, 25, 3, 3);
    
    linesImg = cv2.filter2D(edgeImg, -1 , kernel, borderType=cv2.BORDER_DEFAULT )
    
    return linesImg
    
    
def kmFindColor(frame):
    labColorImg = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    colorImg = np.zeros((frame.shape))
    meanColor0 = labColorImg[:,:,0].mean()
    meanColor1 = labColorImg[:,:,1].mean()
    meanColor2 = labColorImg[:,:,2].mean()
    colorImg0 = np.absolute(labColorImg[:,:,0].astype(np.int) - meanColor0)
    colorImg1 = np.absolute(labColorImg[:,:,1].astype(np.int) - meanColor1)
    colorImg2 = np.absolute(labColorImg[:,:,2].astype(np.int) - meanColor2)
    colorImg = (colorImg0 + colorImg1 + colorImg2) / 3
    
    return colorImg.astype(np.uint8)
    

def kmFindEntropy(frame):
    frameBlurred = cv2.GaussianBlur(frame, (3,3), 0)
    gray = cv2.cvtColor(frameBlurred, cv2.COLOR_BGR2GRAY)
    entropyImg = entropy(img_as_ubyte(gray), disk(5))
    
    return entropyImg
    
    
def kmFeatureNormalize(frame):
    frame = (frame / float(frame.max())) * 255.0
    
    return frame.astype(np.uint8)