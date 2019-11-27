#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 12:52:31 2019

@author: bharatforge
"""

import torch

ssd_model = None
utils = None
classes_to_labels = None
precision = 'fp32'

def get_model_utils():
    global ssd_model, utils, classes_to_labels
    if not ssd_model == None:
        return ssd_model , utils
    else:
        ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math=precision)
        ssd_model.to('cuda')
        #print(ssd_model.eval())
        
        utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
        classes_to_labels = utils.get_coco_object_dictionary()
        return ssd_model , utils
    
def get_detection(ssd_model, utils, img_path):
    X = utils.prepare_input(img_path)
    tensor = utils.prepare_tensor(X, precision == 'fp16')
    print(tensor)
    #detections_batch = ssd_model(tensor)
    #results_per_input = utils.decode_results(detections_batch)
    #best_results_per_input = [utils.pick_best(results, 0.40) for results in results_per_input]
    #print(type(best_results_per_input))
        
if __name__ == "__main__": 
    ssd_model , utils = get_model_utils()
    img = "/media/bharatforge/Ubuntu_data/vishal/datasets/VOC2012/JPEGImages/2007_000027.jpg"
    get_detection(ssd_model, utils, img)