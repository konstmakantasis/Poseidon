##############################################################################
# Copyright (c) 2013, Konstantinos Makantasis
# All rights reserved.
#
# Distributed under the terms of the BSD Simplified License
#
#
# Main script to run visual attention construction code
#
# DEPENDENCIES: 1) OpenCV
#               2) Numpy
#               3) scikit-image
#
# FEATURES MAPPING: localEdges = localCues[0]
#                   localFrequency = localCues[1]
#                   localLines = localCues[2]
#                   localColor = localCues[3]
#                   localEntropy = localCues[4]
# The same mapping holds for global and window descriptors
##############################################################################


import cv2
import kmCues
import kmLowLevelFeatures

cap = cv2.VideoCapture('data/8.mp4')
cap.set(0, 203000)

while(cap.isOpened()):
        
    ret, frame = cap.read()
    frame = frame[:,110:745,:]
    frame = cv2.resize(frame, (frame.shape[1]/4, frame.shape[0]/4), interpolation=cv2.INTER_LINEAR)
    
    localCues, globalCues, csCues = kmCues.kmAllCues(frame)    
       
    feature = localCues[0]
    feature = cv2.resize(feature, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    feature = kmLowLevelFeatures.kmFeatureNormalize(feature)    
   
    
    feature = cv2.applyColorMap(feature, cv2.COLORMAP_JET)
   
    cv2.imshow('Original Frame',frame)
    cv2.imshow('Visual Attention Map', feature)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()