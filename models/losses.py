import torch
import numpy as np

import utils.utils as utils
from utils.visualization_utils import *

def control_point_l1_loss_better_than_threshold(pred_control_points,
                                                gt_control_points,
                                                confidence,
                                                confidence_threshold,
                                                device="cpu"):
    npoints = pred_control_points.shape[1]
    mask = torch.greater_equal(confidence, confidence_threshold)
    mask_ratio = torch.mean(mask)
    mask = torch.repeat_interleave(mask, npoints, dim=1)
    p1 = pred_control_points[mask]
    p2 = gt_control_points[mask]

    return control_point_l1_loss(p1, p2), mask_ratio


def accuracy_better_than_threshold(pred_success_logits,
                                   gt,
                                   confidence,
                                   confidence_threshold,
                                   device="cpu"):
    """
      Computes average precision for the grasps with confidence > threshold.
    """
    pred_classes = torch.argmax(pred_success_logits, -1)
    correct = torch.equal(pred_classes, gt)
    mask = torch.squeeze(torch.greater_equal(confidence, confidence_threshold),
                         -1)

    positive_acc = torch.sum(correct * mask * gt) / torch.max(
        torch.sum(mask * gt), torch.tensor(1))
    negative_acc = torch.sum(correct * mask * (1. - gt)) / torch.max(
        torch.sum(mask * (1. - gt)), torch.tensor(1))

    return 0.5 * (positive_acc + negative_acc), torch.sum(mask) / gt.shape[0]


def control_point_l1_loss(pred_control_points,
                          gt_control_points,
                          confidence=None,
                          confidence_weight=None,
                          device="cpu", dual_grasp=False):
    """
      Computes the l1 loss between the predicted control points and the
      groundtruth control points on the gripper.
    """
    #print('control_point_l1_loss', pred_control_points.shape,
    #      gt_control_points.shape)
    # print(pred_control_points.shape, gt_control_points.shape)

    # confidence = confidence.reshape(32, 2, 1).swapaxes(1, 2).reshape(32, 2)
    # print('confidence', confidence.shape)
    if confidence is not None:
        assert (confidence_weight is not None)
        if dual_grasp:
            ###Code snippet to test the error calculation
            ###Set the predicted control points to the ground truth control points and perform random perturbations that are scaled
            # pred_control_points = gt_control_points
            # pred_control_points = pred_control_points + torch.randn_like(pred_control_points) * 0.2
            # error = torch.sum(torch.abs(pred_control_points - gt_control_points), -1)
            # error = torch.mean(error, -1)
            # error *= confidence
            # error = torch.mean(error, -1)
            # print('error', torch.mean(error))
            # print(xd)
            ###End of code snippet

            error = torch.sum(torch.abs(pred_control_points - gt_control_points), -1)
            # print('error', error.shape)
            #Try to make the error bigger, make it inversely proportional to the downscaling factor of the meshes
            # error *= 2.5
            error = torch.mean(error, -1)
            error *= confidence
            #Take the mean of each pair
            error = torch.mean(error, -1)
            # print("final error mean", error)
            if torch.isnan(torch.mean(error)):
                print('error is nan')
            #split the confidence in half
            confidence = torch.mean(confidence, -1)
            confidence_term = torch.mean(
                torch.log(torch.max(
                    confidence,
                    torch.tensor(1e-10).to(device)))) * confidence_weight
            #check if the confidence term is nan
            if torch.isnan(confidence_term):
                print('confidence_term is nan')
        else:
            error = torch.sum(torch.abs(pred_control_points - gt_control_points), -1)
            print('error', error.shape)
            # print('error', error.shape)
            error = torch.mean(error, -1)
            # print('mean error', error.shape)
            # print('error', error.shape, "confidence", confidence.shape)
            error *= confidence
            confidence_term = torch.mean(
                torch.log(torch.max(
                    confidence,
                    torch.tensor(1e-10).to(device)))) * confidence_weight
        #print('confidence_term = ', confidence_term.shape)

    #print('l1_error = {}'.format(error.shape))
    if confidence is None:
        return torch.mean(error)
    else:
        return torch.mean(error), -confidence_term


def classification_with_confidence_loss(pred_logit,
                                        gt,
                                        confidence,
                                        confidence_weight,
                                        device="cpu"):
    """
      Computes the cross entropy loss and confidence term that penalizes
      outputing zero confidence. Returns cross entropy loss and the confidence
      regularization term.
    """
    # if len(pred_logit) == (2 * len(gt)):
    #     print('dual grasp')
    #     pred_logit = torch.tensor_split(pred_logit, 2, dim=0)
        
    #     pred_logit = pred_logit[0].add(pred_logit[1])
    #     #Devide all values by 2
    #     pred_logit = pred_logit / 2.0
    classification_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        pred_logit, gt)
    confidence_term = torch.mean(
        torch.log(torch.max(
            confidence,
            torch.tensor(1e-10).to(device)))) * confidence_weight

    return classification_loss, -confidence_term


def min_distance_loss(pred_control_points,
                      gt_control_points,
                      confidence=None,
                      confidence_weight=None,
                      threshold=None,
                      device="cpu"):
    """
    Computes the minimum distance (L1 distance)between each gt control point 
    and any of the predicted control points.

    Args: 
      pred_control_points: tensor of (N_pred, M, 4) shape. N is the number of
        grasps. M is the number of points on the gripper.
      gt_control_points: (N_gt, M, 4)
      confidence: tensor of N_pred, tensor for the confidence of each 
        prediction.
      confidence_weight: float, the weight for confidence loss.
    """
    pred_shape = pred_control_points.shape
    gt_shape = gt_control_points.shape
    #calc the loss for dual grasp
    if len(pred_shape) == 4:
        #split the tensor in 32,6,3 from 32,2,6,3
        pred_control_points1 = pred_control_points[:,0,:,:]
        pred_control_points2 = pred_control_points[:,1,:,:]
        gt_control_points1 = gt_control_points[:,0,:,:]
        gt_control_points2 = gt_control_points[:,1,:,:]
        error1 = pred_control_points1.unsqueeze(1) - gt_control_points1.unsqueeze(0)
        error2 = pred_control_points2.unsqueeze(1) - gt_control_points2.unsqueeze(0)
        error1 = torch.sum(torch.abs(error1), -1)  # L1 distance of error (N_pred, N_gt, M)
        error2 = torch.sum(torch.abs(error2), -1)  # L1 distance of error (N_pred, N_gt, M)
        error1 = torch.mean(error1, -1)  # average L1 for all the control points. (N_pred, N_gt)
        error2 = torch.mean(error2, -1)  # average L1 for all the control points. (N_pred, N_gt)
        min_distance_error1, closest_index1 = error1.min(0)  #[0]  # take the min distance for each gt control point. (N_gt)
        min_distance_error2, closest_index2 = error2.min(0)  #[0]  # take the min distance for each gt control point. (N_gt)
        if confidence is not None:
            confidence1 = confidence[:,0]
            confidence2 = confidence[:,1]
            selected_confidence1 = torch.nn.functional.one_hot(
                closest_index1,
                num_classes=closest_index1.shape[0]).float()
            selected_confidence1 *= confidence1
            selected_confidence1 = torch.sum(selected_confidence1, -1)  # N_gt
            min_distance_error1 *= selected_confidence1
            selected_confidence2 = torch.nn.functional.one_hot(
                closest_index2,
                num_classes=closest_index2.shape[0]).float()
            selected_confidence2 *= confidence2
            selected_confidence2 = torch.sum(selected_confidence2, -1)  # N_gt
            min_distance_error2 *= selected_confidence2
            confidence_term = torch.mean(
                torch.log(torch.max(
                    confidence,
                    torch.tensor(1e-4).to(device)))) * confidence_weight
            min_distance_error = torch.mean(min_distance_error1 + min_distance_error2)
        else:
            error = torch.mean(error1, error2)
            confidence_term = 0.
            min_distance_error = torch.mean(error)
    else:       


        if len(pred_shape) != 3:
            raise ValueError(
                "pred_control_point should have len of 3. {}".format(pred_shape))
        if len(gt_shape) != 3:
            raise ValueError(
                "gt_control_point should have len of 3. {}".format(gt_shape))
        if pred_shape != gt_shape:
            raise ValueError("shapes do no match {} != {}".format(
                pred_shape, gt_shape))

        # N_pred x Ngt x M x 3
        error = pred_control_points.unsqueeze(1) - gt_control_points.unsqueeze(0)
        error = torch.sum(torch.abs(error),
                        -1)  # L1 distance of error (N_pred, N_gt, M)
        error = torch.mean(
            error, -1)  # average L1 for all the control points. (N_pred, N_gt)

        min_distance_error, closest_index = error.min(
            0)  #[0]  # take the min distance for each gt control point. (N_gt)
        #print('min_distance_error', get_shape(min_distance_error))
        if confidence is not None:
            #print('closest_index', get_shape(closest_index))
            selected_confidence = torch.nn.functional.one_hot(
                closest_index,
                num_classes=closest_index.shape[0]).float()  # (N_gt, N_pred)
            selected_confidence *= confidence
            #print('selected_confidence', selected_confidence)
            selected_confidence = torch.sum(selected_confidence, -1)  # N_gt
            #print('selected_confidence', selected_confidence)
            min_distance_error *= selected_confidence
            confidence_term = torch.mean(
                torch.log(torch.max(
                    confidence,
                    torch.tensor(1e-4).to(device)))) * confidence_weight
        else:
            confidence_term = 0.

    return torch.mean(min_distance_error), -confidence_term


def min_distance_better_than_threshold(pred_control_points,
                                       gt_control_points,
                                       confidence,
                                       confidence_threshold,
                                       device="cpu"):
    error = torch.expand_dims(pred_control_points, 1) - torch.expand_dims(
        gt_control_points, 0)
    error = torch.sum(torch.abs(error),
                      -1)  # L1 distance of error (N_pred, N_gt, M)
    error = torch.mean(
        error, -1)  # average L1 for all the control points. (N_pred, N_gt)
    error = torch.min(error, -1)  # (B, N_pred)
    mask = torch.greater_equal(confidence, confidence_threshold)
    mask = torch.squeeze(mask, dim=-1)

    return torch.mean(error[mask]), torch.mean(mask)


def kl_divergence(mu, log_sigma, device="cpu", dual_grasp=False):
    """
      Computes the kl divergence for batch of mu and log_sigma.
    """
    # if dual_grasp:
    #     mu = torch.mean(mu, -1)
    #     log_sigma = torch.mean(log_sigma, -1)

    return torch.mean(
        -.5 * torch.sum(1. + log_sigma - mu**2 - torch.exp(log_sigma), dim=-1))


def confidence_loss(confidence, confidence_weight, device="cpu"):
    return torch.mean(
        torch.log(torch.max(
            confidence,
            torch.tensor(1e-10).to(device)))) * confidence_weight
