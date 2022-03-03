#!/usr/bin/env python 
# -*- coding: utf-8 -*-
 
import roslib;
import rospy  
import actionlib  
from actionlib_msgs.msg import *  
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped, Point, Quaternion, Twist  
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal  


rospy.init_node('move_test', anonymous=True)  
  

move_base = actionlib.SimpleActionClient("/robot_3/move_base", MoveBaseAction)  

rospy.loginfo("Waiting for move_base action server...")  


while move_base.wait_for_server(rospy.Duration(5.0)) == 0:
    rospy.loginfo("Connected to move base server")  


target = Pose(Point(0.0, 0.0 , 0.000), Quaternion(0.000, 0.000, 0.001, 0.000))  
goal = MoveBaseGoal()  
goal.target_pose.pose = target  
goal.target_pose.header.frame_id = 'map'  
goal.target_pose.header.stamp = rospy.Time.now()  

rospy.loginfo("Going to: " + str(target))  


move_base.send_goal(goal)  

 
finished_within_time = move_base.wait_for_result(rospy.Duration(100))   

 
if not finished_within_time:  
    move_base.cancel_goal()  
    rospy.loginfo("Timed out achieving goal")  
else:  
    state = move_base.get_state()  
    if state == GoalStatus.SUCCEEDED:  
        rospy.loginfo("Goal succeeded!")
    else:  
      rospy.loginfo("Goal failed！ ")  