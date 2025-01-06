import numpy as np
import matplotlib.pyplot as plt
import os
import torch


#load the csv file containing the loss data
data = np.loadtxt('model_loss_data/tensorboard.csv', delimiter=',', skiprows=1)

#The first column is the wall time, the second column is the step, the third column is the loss
#plot the loss as a function of the step
plt.plot(data[:,1], data[:,2])
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Loss vs Step During Training')
plt.show()




# def jaccard_index_with_threshold_and_rotation(predicted, ground_truth, threshold=0.01, max_angle_deg=30):
#     # Flatten the translation parts (elements [0:3, 3] of the 4x4 matrices)
#     # print("predicted", predicted.shape)
#     # print("ground_truth", ground_truth.shape)
#     pred_translation = predicted[:3, 3]
#     gt_translation = ground_truth[:3, 3]
#     # print("pred_translation", pred_translation)
#     # print("gt_translation", gt_translation)
#     # # Check translation difference
#     translation_diff = np.linalg.norm(pred_translation - gt_translation)
#     # print("translation_diff", translation_diff)

#     # Calculate the union for translation part
#     union = set(pred_translation).union(set(gt_translation))
#     # print("union", union)

#     # Intersection count starts with 0 and we'll increment it if both conditions (translation & rotation) are met
#     intersection = 0
#     rotation_diff = None
#     if translation_diff <= threshold:
#         # Extract 3x3 rotation matrices from both predicted and ground truth
#         pred_rotation = predicted[:3, :3]
#         gt_rotation = ground_truth[:3, :3]

#         # Compute the angular difference between the two rotations
#         rotation_diff = rotation_angle_difference(pred_rotation, gt_rotation)

#         # Check if the rotation difference is within 30 degrees
#         if rotation_diff <= max_angle_deg:
#             intersection += 1
#     # print('translation_diff', translation_diff, 'rotation_diff', rotation_diff)
#     return intersection