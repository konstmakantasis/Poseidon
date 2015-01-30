##############################################################################
# Copyright (c) 2013, Konstantinos Makantasis
# All rights reserved.
#
# Distributed under the terms of the BSD Simplified License
#
##############################################################################

import cv2
import numpy as np
import kmLowLevelFeatures
import kmBlockDivision

def kmPyramidFeatures(frame):    
    pyr1 = frame
    edgePyr1 = kmBlockDivision.kmBlockDivision(kmLowLevelFeatures.kmFindEdges(pyr1, 25, 3, 3), 4)
    laplacianPyr1 = kmBlockDivision.kmBlockDivision(kmLowLevelFeatures.kmFindFrequency(pyr1), 4)
    linesPyr1 = kmBlockDivision.kmBlockDivision(kmLowLevelFeatures.kmFindLines(pyr1), 4)
    colorPyr1 = kmBlockDivision.kmBlockDivision(kmLowLevelFeatures.kmFindColor(pyr1), 4)
    entropyPyr1 = kmBlockDivision.kmBlockDivision(kmLowLevelFeatures.kmFindEntropy(pyr1), 4)    
    
    pyr2 = cv2.pyrDown(pyr1)
    edgePyr2 = kmBlockDivision.kmBlockDivision(kmLowLevelFeatures.kmFindEdges(pyr2, 25, 3, 3), 4)
    laplacianPyr2 = kmBlockDivision.kmBlockDivision(kmLowLevelFeatures.kmFindFrequency(pyr2), 4)
    linesPyr2 = kmBlockDivision.kmBlockDivision(kmLowLevelFeatures.kmFindLines(pyr2), 4)
    colorPyr2 = kmBlockDivision.kmBlockDivision(kmLowLevelFeatures.kmFindColor(pyr2), 4)
    entropyPyr2 = kmBlockDivision.kmBlockDivision(kmLowLevelFeatures.kmFindEntropy(pyr2), 4)  
    
    pyr3 = cv2.pyrDown(pyr2)
    edgePyr3 = kmBlockDivision.kmBlockDivision(kmLowLevelFeatures.kmFindEdges(pyr3, 25, 3, 3), 4)
    laplacianPyr3 = kmBlockDivision.kmBlockDivision(kmLowLevelFeatures.kmFindFrequency(pyr3), 4)
    linesPyr3 = kmBlockDivision.kmBlockDivision(kmLowLevelFeatures.kmFindLines(pyr3), 4)
    colorPyr3 = kmBlockDivision.kmBlockDivision(kmLowLevelFeatures.kmFindColor(pyr3), 4)
    entropyPyr3 = kmBlockDivision.kmBlockDivision(kmLowLevelFeatures.kmFindEntropy(pyr3), 4)
    
    pyr4 = cv2.pyrDown(pyr3)
    edgePyr4 = kmBlockDivision.kmBlockDivision(kmLowLevelFeatures.kmFindEdges(pyr4, 25, 3, 3), 4)
    laplacianPyr4 = kmBlockDivision.kmBlockDivision(kmLowLevelFeatures.kmFindFrequency(pyr4), 4)
    linesPyr4 = kmBlockDivision.kmBlockDivision(kmLowLevelFeatures.kmFindLines(pyr4), 4)
    colorPyr4 = kmBlockDivision.kmBlockDivision(kmLowLevelFeatures.kmFindColor(pyr4), 4)
    entropyPyr4 = kmBlockDivision.kmBlockDivision(kmLowLevelFeatures.kmFindEntropy(pyr4), 4)
    
    edgePyrs = [edgePyr1] + [edgePyr2] + [edgePyr3] + [edgePyr4]
    laplacianPyrs = [laplacianPyr1] + [laplacianPyr2] + [laplacianPyr3] + [laplacianPyr4]
    linesPyrs = [linesPyr1] + [linesPyr2] + [linesPyr3] + [linesPyr4]
    colorPyrs = [colorPyr1] + [colorPyr2] + [colorPyr3] + [colorPyr4]
    entropyPyrs = [entropyPyr1] + [entropyPyr2] + [entropyPyr3] + [entropyPyr4]
    
    return edgePyrs, laplacianPyrs, linesPyrs, colorPyrs, entropyPyrs


def kmPyramidSummation(pyr1, pyr2, pyr3, pyr4):
    pyr2R = cv2.resize(pyr2, (pyr1.shape[1], pyr1.shape[0]), interpolation=cv2.INTER_LINEAR)
    pyr3R = cv2.resize(pyr3, (pyr1.shape[1], pyr1.shape[0]), interpolation=cv2.INTER_LINEAR)
    pyr4R = cv2.resize(pyr4, (pyr1.shape[1], pyr1.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    tempPyr = cv2.addWeighted(pyr4R, 0.25, pyr3R, 0.25, 0.0)
    tempPyr = cv2.addWeighted(pyr2R, 0.25, tempPyr, 1.0, 0.0)
    tempPyr = cv2.addWeighted(pyr1, 0.25, tempPyr, 1.0, 0.0)
    
    return tempPyr.astype(np.uint8)    


def kmLocalCues(edgePyrs, laplacianPyrs, linesPyrs, colorPyrs, entropyPyrs):
    edgeLocal = kmPyramidSummation(edgePyrs[0], edgePyrs[1], edgePyrs[2], edgePyrs[3])
    laplacianLocal = kmPyramidSummation(laplacianPyrs[0], laplacianPyrs[1], laplacianPyrs[2], laplacianPyrs[3])
    linesLocal = kmPyramidSummation(linesPyrs[0], linesPyrs[1], linesPyrs[2], linesPyrs[3])
    colorLocal = kmPyramidSummation(colorPyrs[0], colorPyrs[1], colorPyrs[2], colorPyrs[3])
    entropyLocal = kmPyramidSummation(entropyPyrs[0], entropyPyrs[1], entropyPyrs[2], entropyPyrs[3])
    
    return edgeLocal, laplacianLocal, linesLocal, colorLocal, entropyLocal
    
    
def kmGlobalCues(edgePyrs, laplacianPyrs, linesPyrs, colorPyrs, entropyPyrs):
    edgeGlobal = kmPyramidSummation(kmGMC(edgePyrs[0]), kmGMC(edgePyrs[1]), kmGMC(edgePyrs[2]), kmGMC(edgePyrs[3]))
    laplacianGlobal = kmPyramidSummation(kmGMC(laplacianPyrs[0]), kmGMC(laplacianPyrs[1]), kmGMC(laplacianPyrs[2]), kmGMC(laplacianPyrs[3]))
    linesGlobal = kmPyramidSummation(kmGMC(linesPyrs[0]), kmGMC(linesPyrs[1]), kmGMC(linesPyrs[2]), kmGMC(linesPyrs[3]))
    colorGlobal = kmPyramidSummation(kmGMC(colorPyrs[0]), kmGMC(colorPyrs[1]), kmGMC(colorPyrs[2]), kmGMC(colorPyrs[3]))
    entropyGlobal = kmPyramidSummation(kmGMC(entropyPyrs[0]), kmGMC(entropyPyrs[1]), kmGMC(entropyPyrs[2]), kmGMC(entropyPyrs[3]))
    
    return edgeGlobal, laplacianGlobal, linesGlobal, colorGlobal, entropyGlobal
    
  
def kmGMC(featureImg):
    globalImg = np.zeros((featureImg.shape[0], featureImg.shape[1]))
    for i in range(featureImg.shape[0]):
        for j in range(featureImg.shape[1]):
            pixelVal = featureImg[i,j]
            temp = np.absolute(featureImg.astype(np.int) - pixelVal)
            globalImg[i,j] = temp.mean()
    
    globalImg = (globalImg / globalImg.max()) * 255    
    
        
    return globalImg
            

def kmCSCues(edgePyrs, laplacianPyrs, linesPyrs, colorPyrs, entropyPyrs):
    edgeCS = kmPyramidSummation(kmCSMC(edgePyrs[0]), kmCSMC(edgePyrs[1]), kmCSMC(edgePyrs[2]), kmCSMC(edgePyrs[3]))
    laplacianCS = kmPyramidSummation(kmCSMC(laplacianPyrs[0]), kmCSMC(laplacianPyrs[1]), kmCSMC(laplacianPyrs[2]), kmCSMC(laplacianPyrs[3]))
    linesCS = kmPyramidSummation(kmCSMC(linesPyrs[0]), kmCSMC(linesPyrs[1]), kmCSMC(linesPyrs[2]), kmCSMC(linesPyrs[3]))
    colorCS = kmPyramidSummation(kmCSMC(colorPyrs[0]), kmCSMC(colorPyrs[1]), kmCSMC(colorPyrs[2]), kmCSMC(colorPyrs[3]))
    entropyCS = kmPyramidSummation(kmCSMC(entropyPyrs[0]), kmCSMC(entropyPyrs[1]), kmCSMC(entropyPyrs[2]), kmCSMC(entropyPyrs[3]))
    
    return edgeCS, laplacianCS, linesCS, colorCS, entropyCS

            
def kmCSMC(featureImg):
    csImg = np.zeros((featureImg.shape[0], featureImg.shape[1]))
    x = featureImg.shape[0] 
    y = featureImg.shape[1]
    windowSizeX = 0
    windowSizeY = 0
    for i in range(featureImg.shape[0]):
        if i < (x - i):
            windowSizeX = i
        else:
            windowSizeX = x - i
            
        for j in range(featureImg.shape[1]):
            if j < (y - j):
                windowSizeY = j
            else:
                windowSizeY = y - j
    
            pixelVal = featureImg[i,j]
            if i < 2 or j < 2 or i > x-2 or j > y-2:
                temp = np.absolute(featureImg.astype(np.int) - pixelVal)
                csImg[i,j] = temp.mean()
            else:
                tempImg = featureImg[i-windowSizeX:i+windowSizeX, j-windowSizeY:j+windowSizeY].astype(np.int)
                temp = np.absolute(tempImg - pixelVal)
                csImg[i,j] = temp.mean()
                
    
    csImg = (csImg / csImg.max()) * 255   

    return csImg
            
  
def kmAllCues(frame):
    edgePyrs, laplacianPyrs, linesPyrs, colorPyrs, entropyPyrs = kmPyramidFeatures(frame)
    
    edgeLocal, laplacianLocal, linesLocal, colorLocal, entropyLocal = kmLocalCues(edgePyrs, laplacianPyrs, linesPyrs, colorPyrs, entropyPyrs)    
    localCues = [edgeLocal] + [laplacianLocal] + [linesLocal] + [colorLocal] +[entropyLocal] 
    
    edgeGlobal, laplacianGlobal, linesGlobal, colorGlobal, entropyGlobal = kmGlobalCues(edgePyrs, laplacianPyrs, linesPyrs, colorPyrs, entropyPyrs)    
    globalCues = [edgeGlobal] + [laplacianGlobal] + [linesGlobal] + [colorGlobal] +[entropyGlobal]  
    
    edgeCS, laplacianCS, linesCS, colorCS, entropyCS = kmCSCues(edgePyrs, laplacianPyrs, linesPyrs, colorPyrs, entropyPyrs)    
    csCues = [edgeCS] + [laplacianCS] + [linesCS] + [colorCS] +[entropyCS] 
    
    return localCues, globalCues, csCues