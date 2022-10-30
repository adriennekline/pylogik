#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 14:23:31 2022

@author: ask4118
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt 

def dice_score(pred, true, k=1):
    """
    dice_score -- calculating a dice score between images
    Parameters
    ----------
    pred : array
       array of the predicted segmentation
    true : array
       array of the ground truth segmentation
    k : value
       value to perform matching on (default = 1)
    Returns
    dice : float
        score from 0 to 1 signifying the degree of overlap
    """
    intersection = np.sum(pred[true==k]) * 2.0
    dice = intersection / (np.sum(pred) + np.sum(true))
    return dice

def imshowpair(pred, true, color1 = (124,252,0), color2 = (255,0,252), show_fig = True):
    """
    imshow_pair -- creating image to show dice score
    Parameters
    ----------
    pred : array
       array of the predicted segmentation
    true : array
       array of the ground truth segmentation
    color1 : tuple
        First color to show unique values from first image
    color2 : tuple
        Second color to show unique values from second image
    Returns
    im_dice_pair : array
        3dimensional numpy array for dice score
    """
    # assert that input images are the same size
    assert pred.shape == true.shape, "input image sizes do not match" 
    
    # check type compatibility 
    if (pred.dtype == bool) and (true.dtype == bool):
        pred = pred
        true = true
    else: 
        try: 
            pred = pred.astype('bool')
            true = true.astype('bool')
        except:
            raise TypeError("Only logical types are allowed")
        
    # peform logical comparisons:
    bitwise_and = (pred == True) & (true == True) 
    bitwise_pred = (pred ==True) & (true == False)
    bitwise_true = (pred ==False) & (true == True)
        
    # convert logical type to uint8 to scale to 0-255
    overlap = bitwise_and.astype('uint8')*255
    pred_only = bitwise_pred.astype('uint8')*255
    true_only = bitwise_true.astype('uint8')*255
    
    #convert grey scale to RGB
    image_and = cv2.cvtColor(overlap, cv2.COLOR_GRAY2BGR)
    image_or_pred = cv2.cvtColor(pred_only, cv2.COLOR_GRAY2BGR)
    image_or_true = cv2.cvtColor(true_only, cv2.COLOR_GRAY2BGR)
    
    #reassign color to the 'or' components of image:
    image_or_pred_copy = image_or_pred.copy()
    image_or_true_copy = image_or_true.copy()
    image_or_pred_copy[np.all(image_or_pred_copy == (255,255,255), axis=-1)] = color1
    image_or_true_copy[np.all(image_or_true_copy == (255,255,255), axis=-1)] = color2
    
    # pass back array to user     
    im_dice_pair = image_or_pred_copy + image_or_true_copy + image_and
    
    if show_fig == True:
        plt.imshow(im_dice_pair)
    else:
        pass
    
    return im_dice_pair