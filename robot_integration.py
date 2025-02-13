#!/usr/bin/env python3
from __future__ import print_function

import numpy as np
import argparse
import grasp_estimator
import torch
import sys
import os
import glob
import mayavi.mlab as mlab
# from utils.visualization_utils import *
from utils.visualization_utils_headless import *
import mayavi.mlab as mlab
from utils import utils
from data import DataLoader
import trimesh.transformations as tra
from scipy.spatial.transform import Rotation as R
import rospy 
import os 
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import geometry_msgs.msg
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseArray, Pose, PointStamped
from visualization_msgs.msg import Marker, MarkerArray
import time


import time
import roslib; roslib.load_manifest('race_basic_motion_control')
from race_basic_motion_control.srv import *
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory

class PointcloudParser():
    def __init__(self, pointcloud_topic, parser):
        self.pointcloud_topic = "\aggregated_point_cloud"
        self.pointcloud_sub = rospy.Subscriber(pointcloud_topic, PointCloud2, self.pointcloud_callback)
        self.pointcloud_data = None
        args = parser.parse_args()
        grasp_sampler_args = utils.read_checkpoint_args(args.grasp_sampler_folder)
        grasp_sampler_args.is_train = False
        num_objects_to_show = args.num_objects_to_show
        grasp_evaluator_args = utils.read_checkpoint_args(
            args.grasp_evaluator_folder)
        grasp_evaluator_args.continue_train = True
        self.estimator = grasp_estimator.GraspEstimator(grasp_sampler_args,
                                                grasp_evaluator_args, args)

    def pointcloud_callback(self, msg):
        self.pointcloud_data = msg
        rospy.loginfo("Received point cloud data")
        pc_data = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        #plot the point cloud
        pc_data = list(pc_data)
        pc_data = np.array(pc_data)
        #from the point cloud data, remove all points with a z value less than 0.03, as this is the table
        print("before removing low z-axis points", pc_data.shape)
        # draw_scene(pc_data, save_path='./demo/with_table.png')
        pc_data_no_table = pc_data[pc_data[:,2] > 0.03]
        #also remove all the points with a y value less than 0.4. This is to remove the robot torso and arms
        pc_data_no_table = pc_data_no_table[pc_data_no_table[:,1] > 0.4]
        print("after removing table and robot points", pc_data_no_table.shape)
        # draw_scene(pc_data_no_table, save_path='./demo/no_table.png')
        # #scale dow the pc with a factor of 0.3
        pc_data_no_table = pc_data_no_table * 0.3
        
        generated_grasps, generated_scores, downsampled_pc = self.estimator.generate_and_refine_grasps(
                    pc_data_no_table)

        draw_scene(downsampled_pc.cpu().numpy(), grasps=generated_grasps, save_path='./demo/grasps_with_unwanted_pairs.png')
        #randomly shuffle the grasps as most of the grasp pairs consist of very similar grasps
        # np.random.shuffle(generated_grasps)

        ###Remove pairs of grasps where both are on the same side of the object.
        ###This is done by remove pairs if both grasps have a positive x value or both have a negative x value
        new_grasps = []
        for idx in range(len(generated_grasps)):
            if idx % 2 == 0:
                grasp1 = generated_grasps[idx]
                grasp2 = generated_grasps[idx+1]
                #Calculate the distance between the two grasps
                distance = np.linalg.norm(grasp1[:3, 3] - grasp2[:3, 3])
                # print("Distance between grasps", distance)
                if distance > 0.025:
                    #Also check if the left grasp is not too much on the right side and the right grasp is not too much on the left side
                    if grasp1[0, 3] < grasp2[0, 3]:
                        left_grasp = grasp1
                        right_grasp = grasp2
                    else:
                        left_grasp = grasp2
                        right_grasp = grasp1

                    if not (left_grasp[0, 3] > 0.1 or right_grasp[0, 3] < -0.1):
                        #also remove grasps if the z value is less than 0.05
                        if grasp1[2, 3] > 0.10 and grasp2[2, 3] > 0.10:
                            new_grasps.append(grasp1)
                            new_grasps.append(grasp2)
        
        ###Only save grasp pairs that have a poitive x value for both grasps
        # new_grasps = []
        # for idx in range(len(generated_grasps)):
        #     if idx % 2 == 0:
        #         grasp1 = generated_grasps[idx]
        #         grasp2 = generated_grasps[idx+1]
        #         if grasp1[0, 3] < 0 and grasp2[0, 3] < 0:
        #             new_grasps.append(grasp1)
        #             new_grasps.append(grasp2)
        print("Number of grasps before removing pairs", len(generated_grasps))
        print("Number of grasps after removing pairs with similar grasps", len(new_grasps))

        # draw_scene(downsampled_pc.cpu().numpy(), grasps=new_grasps, save_path='./demo/grasps_removed_unwanted_pairs.png')
        generated_grasps = new_grasps
        
        # draw_scene(pc_data_no_table, grasps=generated_grasps, save_path='./demo/grasps_before_rescale.png')

        #scale the grasps back
        scale = 1/0.3
        S = np.diag([scale, scale, scale, 1])
        for i, grasp in enumerate(generated_grasps):
            #scale the grasps back
            generated_grasps[i] = S.dot(grasp)

        pc_data_no_table = pc_data_no_table * scale
        draw_scene(pc_data_no_table, grasps=generated_grasps, save_path='./demo/grasps_after_rescale.png')
        if len(generated_grasps) != 0:
            #publish the grasps to a topic
            # self.publish_grasps(generated_grasps[:10])
            # self.execute_grasp(generated_grasps[:10])
            self.execute_grasp(generated_grasps)


    def listener(self):
        rospy.init_node('pointcloud_listener', anonymous=True)
        rospy.spin()

    def publish_control_points(self, control_points_both, zero_points, mid_points):
        
        point_pub1_left = rospy.Publisher('/zeropoint_leftgripper', PointStamped, queue_size=10)
        point_pub2_left = rospy.Publisher('/midpoint_leftgripper', PointStamped, queue_size=10)
        point_pub3_left = rospy.Publisher('/end_pointleft_leftgripper', PointStamped, queue_size=10)
        point_pub4_left = rospy.Publisher('/end_pointright_leftgripper', PointStamped, queue_size=10)
        point_pub5_left = rospy.Publisher('/mid_pointleft_leftgripper', PointStamped, queue_size=10)
        point_pub6_left = rospy.Publisher('/mid_pointright_leftgripper', PointStamped, queue_size=10)
        control_points = control_points_both[0]
        zero_point = zero_points[0]
        mid_point = mid_points[0]
        #publish all the relevant control points to visualize in rviz and debug
        zeropoint = PointStamped()
        zeropoint.header.frame_id = "world_base_link"
        zeropoint.point.x = zero_point[0]
        zeropoint.point.y = zero_point[1]
        zeropoint.point.z = zero_point[2]
        point_pub1_left.publish(zeropoint)
        midpoint = PointStamped()
        midpoint.header.frame_id = "world_base_link"
        midpoint.point.x = mid_point[0]
        midpoint.point.y = mid_point[1]
        midpoint.point.z = mid_point[2]
        point_pub2_left.publish(midpoint)

        end_pointleft = PointStamped()
        end_pointleft.header.frame_id = "world_base_link"
        end_pointleft.point.x = control_points[4][0]
        end_pointleft.point.y = control_points[4][1]
        end_pointleft.point.z = control_points[4][2]
        point_pub3_left.publish(end_pointleft)

        end_pointright = PointStamped()
        end_pointright.header.frame_id = "world_base_link"
        end_pointright.point.x = control_points[5][0]
        end_pointright.point.y = control_points[5][1]
        end_pointright.point.z = control_points[5][2]
        point_pub4_left.publish(end_pointright)

        mid_pointleft = PointStamped()
        mid_pointleft.header.frame_id = "world_base_link"
        mid_pointleft.point.x = control_points[2][0]
        mid_pointleft.point.y = control_points[2][1]
        mid_pointleft.point.z = control_points[2][2]
        point_pub5_left.publish(mid_pointleft)

        mid_pointright = PointStamped()
        mid_pointright.header.frame_id = "world_base_link"
        mid_pointright.point.x = control_points[3][0]
        mid_pointright.point.y = control_points[3][1]
        mid_pointright.point.z = control_points[3][2]
        point_pub6_left.publish(mid_pointright)


        point_pub1_right = rospy.Publisher('/zeropoint_rightgripper', PointStamped, queue_size=10)
        point_pub2_right = rospy.Publisher('/midpoint_rightgripper', PointStamped, queue_size=10)
        point_pub3_right = rospy.Publisher('/end_pointleft_rightgripper', PointStamped, queue_size=10)
        point_pub4_right = rospy.Publisher('/end_pointright_rightgripper', PointStamped, queue_size=10)
        point_pub5_right = rospy.Publisher('/mid_pointleft_rightgripper', PointStamped, queue_size=10)
        point_pub6_right = rospy.Publisher('/mid_pointright_rightgripper', PointStamped, queue_size=10)

        control_points = control_points_both[1]
        zero_point = zero_points[1]
        mid_point = mid_points[1]
        zeropoint = PointStamped()
        zeropoint.header.frame_id = "world_base_link"
        zeropoint.point.x = zero_point[0]
        zeropoint.point.y = zero_point[1]
        zeropoint.point.z = zero_point[2]
        point_pub1_right.publish(zeropoint)
        midpoint = PointStamped()
        midpoint.header.frame_id = "world_base_link"
        midpoint.point.x = mid_point[0]
        midpoint.point.y = mid_point[1]
        midpoint.point.z = mid_point[2]
        point_pub2_right.publish(midpoint)

        end_pointleft = PointStamped()
        end_pointleft.header.frame_id = "world_base_link"
        end_pointleft.point.x = control_points[4][0]
        end_pointleft.point.y = control_points[4][1]
        end_pointleft.point.z = control_points[4][2]
        point_pub3_right.publish(end_pointleft)

        end_pointright = PointStamped()
        end_pointright.header.frame_id = "world_base_link"
        end_pointright.point.x = control_points[5][0]
        end_pointright.point.y = control_points[5][1]
        end_pointright.point.z = control_points[5][2]
        point_pub4_right.publish(end_pointright)

        mid_pointleft = PointStamped()
        mid_pointleft.header.frame_id = "world_base_link"
        mid_pointleft.point.x = control_points[2][0]
        mid_pointleft.point.y = control_points[2][1]
        mid_pointleft.point.z = control_points[2][2]
        point_pub5_right.publish(mid_pointleft)

        mid_pointright = PointStamped()
        mid_pointright.header.frame_id = "world_base_link"
        mid_pointright.point.x = control_points[3][0]
        mid_pointright.point.y = control_points[3][1]
        mid_pointright.point.z = control_points[3][2]
        point_pub6_right.publish(mid_pointright)
    def execute_grasp(self,grasps):
        rospy.wait_for_service('/behaviors_service')
        behaviors_service = rospy.ServiceProxy('/behaviors_service', behaviors)
        test_behavior = behaviorsRequest()
        test_behavior.robot='dual'
        # print("Move to initial pose result: ", result)
        time.sleep(5)
        print("Number of grasps: ", len(grasps))
        for idx in range(len(grasps)):
            test_behaviour_left = behaviorsRequest()
            test_behaviour_right = behaviorsRequest()
            test_behaviour_left.robot='left'
            test_behaviour_right.robot='right'
            test_behaviour_left.behavior='IK_move_to_pose_real_time'
            test_behaviour_right.behavior='IK_move_to_pose_real_time'
            # test_behavior.behavior='visualize_target_ee_pose'
            if idx % 2 == 0:
                test_behavior.behavior='go_to_initial_top_pose'
                result = behaviors_service(test_behavior)
                print("Grasp pair: ", (idx/2) + 1)
                grasp1 = grasps[idx]
                grasp2 = grasps[idx+1]
                control_points_both = []
                zero_points_both = []
                mid_points_both = []
                ###Code snippet to move the grasp pose slightly away from the object
                ###This is to compensate for the different gripper used when training
                #First do it for the first grasp
                og_control_points = utils.get_control_point_tensor(1, False)
                shape = og_control_points.shape
                ones = np.ones((shape[0], shape[1], 1), dtype=np.float32)
                control_points = np.concatenate((og_control_points, ones), -1)
                control_points = np.matmul(control_points, np.transpose([grasp1], (0, 2, 1)))
                control_points = control_points[:, :,:3]
                control_points = control_points[0]
                # print("Control points: ", control_points.shape)
                mid_point = np.zeros(3)
                mid_point[0] = (control_points[2][0] + control_points[3][0]) /2.0
                mid_point[1] = (control_points[2][1] + control_points[3][1]) /2.0
                mid_point[2] = (control_points[2][2] + control_points[3][2]) /2.0
                zero_point = control_points[0]
                # print("point 1: ", control_points[1])
                # print("point 2: ", control_points[2])

                control_points_both.append(control_points)
                zero_points_both.append(zero_point)
                mid_points_both.append(mid_point)

                # print("Mid point: ", mid_point)
                # print("Zero point: ", zero_point)
                x_diff = zero_point[0] - mid_point[0]
                y_diff = zero_point[1] - mid_point[1]
                z_diff = zero_point[2] - mid_point[2]
                # y_diff = mid_point[1] - zero_point[1]
                # z_diff = mid_point[2] - zero_point[2]
                # print("X diff: ", x_diff)
                # print("Y diff: ", y_diff)
                # print("Z diff: ", z_diff)
                delta = 0.190
                delta_initial = 0.250
                sum_of_diffs = abs(x_diff) + abs(y_diff) + abs(z_diff)
                x_change = (x_diff/sum_of_diffs) * delta
                # print("X change: ", x_change)
                x_change_initial = (x_diff/sum_of_diffs) * delta_initial
                # print("X_change initial: ", x_change_initial)
                y_change = (y_diff/sum_of_diffs) * delta
                y_change_initial = (y_diff/sum_of_diffs) * delta_initial
                z_change = (z_diff/sum_of_diffs) * delta
                z_change_initial = (z_diff/sum_of_diffs) * delta_initial
                # print("X change: ", x_change)
                # print("Y change: ", y_change)
                # print("Z change: ", z_change)
                delta_array = np.array([x_change, y_change, z_change])
                delta_array_initial = np.array([x_change_initial, y_change_initial, z_change_initial])
                grasp1_initial = np.copy(grasp1)
                grasp1[:3, 3] += delta_array
                grasp1_initial[:3, 3] += delta_array_initial

                #Now do it for the second grasp
                og_control_points = utils.get_control_point_tensor(1, False)
                shape = og_control_points.shape
                ones = np.ones((shape[0], shape[1], 1), dtype=np.float32)
                control_points = np.concatenate((og_control_points, ones), -1)
                control_points = np.matmul(control_points, np.transpose([grasp2], (0, 2, 1)))
                control_points = control_points[:, :,:3]
                control_points = control_points[0]

                mid_point = np.zeros(3)
                mid_point[0] = (control_points[2][0] + control_points[3][0]) /2.0
                mid_point[1] = (control_points[2][1] + control_points[3][1]) /2.0
                mid_point[2] = (control_points[2][2] + control_points[3][2]) /2.0

                zero_point = control_points[0]

                control_points_both.append(control_points)
                zero_points_both.append(zero_point)
                mid_points_both.append(mid_point)
                # print("Publishing control points")
                # self.publish_control_points(control_points_both, zero_points_both, mid_points_both)

                x_diff = zero_point[0] - mid_point[0]
                y_diff = zero_point[1] - mid_point[1]
                z_diff = zero_point[2] - mid_point[2]

                sum_of_diffs = abs(x_diff) + abs(y_diff) + abs(z_diff)
                x_change = (x_diff/sum_of_diffs) * delta
                x_change_initial = (x_diff/sum_of_diffs) * delta_initial
                y_change = (y_diff/sum_of_diffs) * delta
                y_change_initial = (y_diff/sum_of_diffs) * delta_initial
                z_change = (z_diff/sum_of_diffs) * delta
                z_change_initial = (z_diff/sum_of_diffs) * delta_initial

                delta_array = np.array([x_change, y_change, z_change])
                delta_array_initial = np.array([x_change_initial, y_change_initial, z_change_initial])
                grasp2_initial = np.copy(grasp2)
                grasp2[:3, 3] += delta_array
                grasp2_initial[:3, 3] += delta_array_initial
                ### End of code snippet

                #first do a rotation of 90 degrees around the z-axis
                R = tra.rotation_matrix(np.pi/2, [0, 0, 1])
                grasp1[:3, :3] = np.matmul(grasp1[:3, :3], R[:3, :3].T)
                grasp2[:3, :3] = np.matmul(grasp2[:3, :3], R[:3, :3].T)
                grasp1_initial[:3, :3] = np.matmul(grasp1_initial[:3, :3], R[:3, :3].T)
                grasp2_initial[:3, :3] = np.matmul(grasp2_initial[:3, :3], R[:3, :3].T)
                
                # #then do a rotation of 90 degrees around the x-axis
                R = tra.rotation_matrix(np.pi/2, [0, 1, 0])
                grasp1[:3, :3] = np.matmul(grasp1[:3, :3], R[:3, :3].T) 
                grasp2[:3, :3] = np.matmul(grasp2[:3, :3], R[:3, :3].T)
                grasp1_initial[:3, :3] = np.matmul(grasp1_initial[:3, :3], R[:3, :3].T)
                grasp2_initial[:3, :3] = np.matmul(grasp2_initial[:3, :3], R[:3, :3].T)
                #print the coordinates of the positoins of the grasp
                # print("Grasp 1: ", grasp1[:3, 3])

                # R = tra.rotation_matrix(-np.pi/2, [1, 0, 0])
                # grasp1[:3, :3] = np.matmul(grasp1[:3, :3], R[:3, :3].T) 
                # grasp2[:3, :3] = np.matmul(grasp2[:3, :3], R[:3, :3].T)

                
                
                #Determine which grasp is on the left and which is on the right
                if grasp1[0, 3] < grasp2[0, 3]:
                    left_grasp = grasp1
                    right_grasp = grasp2
                    left_grasp_initial = grasp1_initial
                    right_grasp_initial = grasp2_initial
                else:
                    left_grasp = grasp2
                    right_grasp = grasp1
                    left_grasp_initial = grasp2_initial
                    right_grasp_initial = grasp1_initial

                ###Individual arms
                #First go to the initial pre-grasp pose
                # test_behaviour_left.left_target_pose.pose.position.x = left_grasp_initial[0, 3] #+ 0.25
                # test_behaviour_left.left_target_pose.pose.position.y = left_grasp_initial[1, 3]
                # test_behaviour_left.left_target_pose.pose.position.z = left_grasp_initial[2, 3]
                # quaternion = self.rotation_matrix_to_quaternion(left_grasp_initial[:3, :3])
                # test_behaviour_left.left_target_pose.pose.orientation.x = quaternion[0]
                # test_behaviour_left.left_target_pose.pose.orientation.y = quaternion[1]
                # test_behaviour_left.left_target_pose.pose.orientation.z = quaternion[2]
                # test_behaviour_left.left_target_pose.pose.orientation.w = quaternion[3]
                # test_behaviour_left.behavior='IK_move_to_pose_real_time'
                # result_left = behaviors_service(test_behaviour_left)
                # print("Move to initial left pre-grasp pose result: ", result_left.result)
                # if not result_left.result:
                #     print("Initial left pre-grasp pose failed, moving to next pair")
                #     # time.sleep(1)
                #     continue
                # test_behaviour_right.right_target_pose.pose.position.x = right_grasp_initial[0, 3] #+ 0.25
                # test_behaviour_right.right_target_pose.pose.position.y = right_grasp_initial[1, 3]
                # test_behaviour_right.right_target_pose.pose.position.z = right_grasp_initial[2, 3]
                # quaternion = self.rotation_matrix_to_quaternion(right_grasp_initial[:3, :3])
                # test_behaviour_right.right_target_pose.pose.orientation.x = quaternion[0]
                # test_behaviour_right.right_target_pose.pose.orientation.y = quaternion[1]
                # test_behaviour_right.right_target_pose.pose.orientation.z = quaternion[2]
                # test_behaviour_right.right_target_pose.pose.orientation.w = quaternion[3]
                # test_behaviour_right.behavior='IK_move_to_pose_real_time'
                # result_right = behaviors_service(test_behaviour_right)
                # print("Move to initial right pre-grasp pose result: ", result_right.result)
                # if not result_right.result:
                #     print("Initial right pre-grasp pose failed, moving to next pair")
                #     # time.sleep(1)
                #     continue
                # time.sleep(1)

                # test_behaviour_left.left_target_pose.pose.position.x = left_grasp[0, 3]
                # test_behaviour_left.left_target_pose.pose.position.y = left_grasp[1, 3]
                # test_behaviour_left.left_target_pose.pose.position.z = left_grasp[2, 3]
                # # Convert the 3x3 rotation matrix to a quaternion
                # quaternion = self.rotation_matrix_to_quaternion(left_grasp[:3, :3])
                # test_behaviour_left.left_target_pose.pose.orientation.x = quaternion[0]
                # test_behaviour_left.left_target_pose.pose.orientation.y = quaternion[1]
                # test_behaviour_left.left_target_pose.pose.orientation.z = quaternion[2]
                # test_behaviour_left.left_target_pose.pose.orientation.w = quaternion[3]
                
                # # test_behaviour_left.behavior='visualize_target_ee_pose'
                # # behaviors_service(test_behaviour_left)
                
                # test_behaviour_left.behavior='IK_move_to_pose_real_time'
                # result_left = behaviors_service(test_behaviour_left)
                # print("Move to left grasp pose result: ", result_left.result)
                # if not result_left.result:
                #     print("Left grasp failed, moving to next pair")
                #     test_behavior.behavior='go_to_initial_pose'
                #     result = behaviors_service(test_behavior)
                #     # time.sleep(1)
                #     continue

                # test_behaviour_right.right_target_pose.pose.position.x = right_grasp[0, 3]
                # test_behaviour_right.right_target_pose.pose.position.y = right_grasp[1, 3]
                # test_behaviour_right.right_target_pose.pose.position.z = right_grasp[2, 3]
                # # Convert the 3x3 rotation matrix to a quaternion
                # quaternion = self.rotation_matrix_to_quaternion(right_grasp[:3, :3])
                # test_behaviour_right.right_target_pose.pose.orientation.x = quaternion[0]
                # test_behaviour_right.right_target_pose.pose.orientation.y = quaternion[1]
                # test_behaviour_right.right_target_pose.pose.orientation.z = quaternion[2]
                # test_behaviour_right.right_target_pose.pose.orientation.w = quaternion[3]

                # # test_behaviour_right.behavior='visualize_target_ee_pose'
                # # behaviors_service(test_behaviour_right)
                
                # test_behaviour_right.behavior='IK_move_to_pose_real_time'
                # result_right = behaviors_service(test_behaviour_right)

                # print("Move to right grasp pose result: ", result_right.result)
                # if not result_right.result:
                #     print("Right grasp failed, moving to next pair")
                #     test_behavior.behavior='go_to_initial_pose'
                #     result = behaviors_service(test_behavior)
                #     # time.sleep(1)
                #     continue
                # time.sleep(3)
                ### End of individual arms

                ###Dual arm

                test_behavior.robot='dual'
                test_behavior.behavior='IK_move_to_pose'
                #First go to the initial pre-grasp pose
                test_behavior.right_target_pose.pose.position.x = right_grasp_initial[0, 3] #+ 0.25
                test_behavior.right_target_pose.pose.position.y = right_grasp_initial[1, 3]
                test_behavior.right_target_pose.pose.position.z = right_grasp_initial[2, 3]
                # Convert the 3x3 rotation matrix to a quaternion
                # quaternion = self.quaternion_from_matrix(right_grasp)
                # quaternion = self.normalize_quaternion(quaternion)
                quaternion = self.rotation_matrix_to_quaternion(right_grasp_initial[:3, :3])
                test_behavior.right_target_pose.pose.orientation.x = quaternion[0]
                test_behavior.right_target_pose.pose.orientation.y = quaternion[1]
                test_behavior.right_target_pose.pose.orientation.z = quaternion[2]
                test_behavior.right_target_pose.pose.orientation.w = quaternion[3] 

                test_behavior.left_target_pose.pose.position.x = left_grasp_initial[0, 3] #- 0.25
                test_behavior.left_target_pose.pose.position.y = left_grasp_initial[1, 3]
                test_behavior.left_target_pose.pose.position.z = left_grasp_initial[2, 3]
                # Convert the 3x3 rotation matrix to a quaternion
                # quaternion = self.quaternion_from_matrix(left_grasp)
                # quaternion = self.normalize_quaternion(quaternion)
                quaternion = self.rotation_matrix_to_quaternion(left_grasp_initial[:3, :3])
                test_behavior.left_target_pose.pose.orientation.x = quaternion[0]
                test_behavior.left_target_pose.pose.orientation.y = quaternion[1]
                test_behavior.left_target_pose.pose.orientation.z = quaternion[2]
                test_behavior.left_target_pose.pose.orientation.w = quaternion[3]
                # test_behavior.behavior='visualize_target_ee_pose'
                # result = behaviors_service(test_behavior)
                result = behaviors_service(test_behavior)
                print("Move to initial pre-grasp pose result: ", result.result)
                if not result.result:
                    print("Initial pre-grasp pose failed, moving to next pair")
                    # time.sleep(1)
                    continue
                time.sleep(3)

                test_behavior.robot='dual'
                test_behavior.behavior='IK_move_to_pose'
                
                test_behavior.right_target_pose.pose.position.x = right_grasp[0, 3] #+ 0.25
                test_behavior.right_target_pose.pose.position.y = right_grasp[1, 3]
                test_behavior.right_target_pose.pose.position.z = right_grasp[2, 3]
                # Convert the 3x3 rotation matrix to a quaternion
                # quaternion = self.quaternion_from_matrix(right_grasp)
                # quaternion = self.normalize_quaternion(quaternion)
                quaternion = self.rotation_matrix_to_quaternion(right_grasp[:3, :3])
                test_behavior.right_target_pose.pose.orientation.x = quaternion[0]
                test_behavior.right_target_pose.pose.orientation.y = quaternion[1]
                test_behavior.right_target_pose.pose.orientation.z = quaternion[2]
                test_behavior.right_target_pose.pose.orientation.w = quaternion[3] 

                test_behavior.left_target_pose.pose.position.x = left_grasp[0, 3] #- 0.25
                test_behavior.left_target_pose.pose.position.y = left_grasp[1, 3]
                test_behavior.left_target_pose.pose.position.z = left_grasp[2, 3]
                # Convert the 3x3 rotation matrix to a quaternion
                # quaternion = self.quaternion_from_matrix(left_grasp)
                # quaternion = self.normalize_quaternion(quaternion)
                quaternion = self.rotation_matrix_to_quaternion(left_grasp[:3, :3])
                test_behavior.left_target_pose.pose.orientation.x = quaternion[0]
                test_behavior.left_target_pose.pose.orientation.y = quaternion[1]
                test_behavior.left_target_pose.pose.orientation.z = quaternion[2]
                test_behavior.left_target_pose.pose.orientation.w = quaternion[3]

                # test_behavior.behavior='visualize_target_ee_pose'
                # result = behaviors_service(test_behavior)
                result = behaviors_service(test_behavior)
                print("Move to grasp pose result: ", result.result)
                time.sleep(3)

                ###End of dual arm
                #If both grasps resulted in true
                # if result_left.result and result_right.result:
                #     print("Both grasps succeeded, attempting to close the gripper")
                #     #close the gripper
                #     test_behavior.behavior='close_gripper'
                #     result = behaviors_service(test_behavior)
                #     print("Close gripper result: ", result.result)
                #     time.sleep(3)
                #     #open the gripper
                #     test_behavior.behavior='open_gripper'
                #     result = behaviors_service(test_behavior)
                #     print("Open gripper result: ", result.result)
                    
                    #print all the components of result
                    # print(result)
                #return to initial pose
                #print all the components of result
                # print(result)
                # time.sleep(3)


        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            # test_behavior.behavior='go_to_initial_pose'
            # result = behaviors_service(test_behavior)
            rate.sleep()

    def publish_grasps(self,grasps):

        # Create a publisher for PoseArray messages (to visualize grasps in RViz)
        grasp_pub = rospy.Publisher("/grasp_markers", MarkerArray, queue_size=10)
        
        # Set the loop rate
        rate = rospy.Rate(10)  # 10 Hz

        marker_array = MarkerArray()
        num_of_pairs = len(grasps) // 2
        current_pair = 0
        # Add grasps to MarkerArray
        for i, grasp in enumerate(grasps):
            # Create a Marker (e.g., a cylinder) to represent the grasp
            if i % 2 == 0 and i != 0:
                current_pair += 1
            marker = Marker()
            marker.header.frame_id = "base_link1"  # Frame of reference
            marker.header.stamp = rospy.Time.now()
            marker.id = i
            marker.type = Marker.CYLINDER  # Use a cylinder to represent the grasp point
            marker.action = Marker.ADD
            #rotate the grasp by 90 degrees around the z-axis
            grasp = tra.rotation_matrix(-np.pi/2, [0, 0, 1]).dot(grasp)
            marker.pose.position.x = grasp[0, 3]
            marker.pose.position.y = grasp[1, 3]
            marker.pose.position.z = grasp[2, 3]
            
            # Convert the 3x3 rotation matrix to a quaternion
            quaternion = self.quaternion_from_matrix(grasp)
            marker.pose.orientation.x = quaternion[0]
            marker.pose.orientation.y = quaternion[1]
            marker.pose.orientation.z = quaternion[2]
            marker.pose.orientation.w = quaternion[3]

            # Set the scale (dimensions) of the cylinder
            marker.scale.x = 0.05  # Radius of the cylinder
            marker.scale.y = 0.05  # Radius of the cylinder
            marker.scale.z = 0.1  # Height of the cylinder

            # Set color (rgba)
            #give each grasp a different color
            colour_ranges = np.linspace(0, 1, int(len(grasps)/2.0))
            marker.color.r = colour_ranges[current_pair]
            marker.color.g = 1 - colour_ranges[current_pair]
            marker.color.b = 0.0
            marker.color.a = 1.0  # Full opacity

            # Add the marker to the MarkerArray
            marker_array.markers.append(marker)

        # Publish the PoseArray to the topic
        while not rospy.is_shutdown():
            grasp_pub.publish(marker_array)
            rate.sleep()
    def normalize_quaternion(self, q):
        """
        Normalize a quaternion.
        """
        norm = np.sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)
        if norm == 0:
            raise ValueError("Cannot normalize a zero quaternion.")
        return [qi / norm for qi in q]
    def rotation_matrix_to_quaternion(self, R):
        """
        Convert a 3x3 rotation matrix to a quaternion (w, x, y, z).
        
        Parameters:
            R (numpy.ndarray): 3x3 rotation matrix.
        
        Returns:
            numpy.ndarray: Quaternion (w, x, y, z).
        """
        trace = np.trace(R)
        
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2  # S = 4 * qw
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S = 4 * qx
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S = 4 * qy
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S = 4 * qz
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S

        return np.array([qx, qy, qz, qw])
    def quaternion_from_matrix(self, matrix):
        """
        Convert a 3x3 rotation matrix to a quaternion (x, y, z, w)
        This function assumes you have a proper 3x3 rotation matrix.
        """
        # Extract the elements of the rotation matrix
        m = matrix[:3, :3]
        qw = np.sqrt(1.0 + m[0, 0] + m[1, 1] + m[2, 2]) / 2.0
        qx = (m[2, 1] - m[1, 2]) / (4.0 * qw)
        qy = (m[0, 2] - m[2, 0]) / (4.0 * qw)
        qz = (m[1, 0] - m[0, 1]) / (4.0 * qw)
        
        return self.normalize_quaternion([qx, qy, qz, qw])

def make_parser():
    parser = argparse.ArgumentParser(
        description='6-DoF GraspNet Demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--grasp_sampler_folder',
                        type=str,
                        default='checkpoints/gan_pretrained/')
    parser.add_argument('--grasp_evaluator_folder',
                        type=str,
                        default='checkpoints/evaluator_pretrained/')
    parser.add_argument('--refinement_method',
                        choices={"gradient", "sampling"},
                        default='sampling')
    parser.add_argument('--refine_steps', type=int, default=25)

    parser.add_argument('--npy_folder', type=str, default='demo/data/')
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.8,
        help=
        "When choose_fn is something else than all, all grasps with a score given by the evaluator notwork less than the threshold are removed"
    )
    parser.add_argument(
        '--object_scale',
        type=float,
        default=0.2,
        help=
        "Sets the downscaled size of the object. The object is scaled to this size before being fed to the network"
    )
    parser.add_argument(
        '--choose_fn',
        choices={
            "all", "better_than_threshold", "better_than_threshold_in_sequence"
        },
        default='better_than_threshold',
        help=
        "If all, no grasps are removed. If better than threshold, only the last refined grasps are considered while better_than_threshold_in_sequence consideres all refined grasps"
    )
    parser.add_argument('--num_objects_to_show', type=int, default=5)
    parser.add_argument('--target_pc_size', type=int, default=1024)
    parser.add_argument('--num_grasp_samples', type=int, default=200)
    parser.add_argument(
        '--generate_dense_grasps',
        action='store_true',
        help=
        "If enabled, it will create a [num_grasp_samples x num_grasp_samples] dense grid of latent space values and generate grasps from these."
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=30,
        help=
        "Set the batch size of the number of grasps we want to process and can fit into the GPU memory at each forward pass. The batch_size can be increased for a GPU with more memory."
    )
    parser.add_argument('--train_data', action='store_true')
    opts, _ = parser.parse_known_args()
    if opts.train_data:
        parser.add_argument('--dataset_root_folder',
                            required=True,
                            type=str,
                            help='path to root directory of the dataset.')
    parser.add_argument("--dual_grasp",
                        action="store_true",
                        help="If enabled, the model will predict two grasps.")
    return parser
if __name__ == '__main__':
    behaviors_service = rospy.ServiceProxy('/behaviors_service', behaviors)
    test_behavior = behaviorsRequest()
    test_behavior.robot='dual'
    test_behavior.behavior='go_to_initial_pose'
    result = behaviors_service(test_behavior)
    parser = make_parser()
    pointcloud_parser = PointcloudParser("/aggregated_point_cloud", parser)
    pointcloud_parser.listener()