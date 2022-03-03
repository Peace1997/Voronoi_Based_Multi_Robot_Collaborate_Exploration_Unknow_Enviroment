#! /usr/bin/env python

import rospy
import rospkg
from rospy.timer import sleep
import tf
from std_msgs.msg import String
from geometry_msgs.msg import Twist, Point, Quaternion
from math import cos, radians, copysign, sqrt, pow, pi, atan2
from tf.transformations import euler_from_quaternion

import threading
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState 

from gazebo_msgs.srv import SetModelState
from sensor_msgs.msg import LaserScan
#from kobuki_msgs.msg import BumperEvent
import time

import tensorflow
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, merge
from keras.layers.merge import Add, Concatenate
from keras.optimizers import Adam, get
import keras.backend as K
import gym
import numpy as np
import math
import random

from std_srvs.srv import Empty



import roslib;
import rospy  
import actionlib  
from actionlib_msgs.msg import *  
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped, Point, Quaternion, Twist  
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal  
from random import sample  
from math import pow, sqrt  

class InfoGetter(object):
    def __init__(self):
        #event that will block until the info is received
        self._event = threading.Event()
        #attribute for storing the rx'd message
        self._msg = None

    def __call__(self, msg):
        #Uses __call__ so the object itself acts as the callback
        #save the data, trigger the event
        self._msg = msg
        self._event.set()

    def get_msg(self, timeout=None):
        """Blocks until the data is rx'd with optional timeout
        Returns the received message
        """
        self._event.wait(timeout)
        return self._msg

class GameState:

    def __init__(self):
        self.talker_node = rospy.init_node('talker', anonymous=True)
        self.pose_ig = InfoGetter()
        self.collision_ig = InfoGetter()
        
        self.move_cmd = Twist()

        # tf
        self.tf_listener = tf.TransformListener()
        rospy.sleep(2)
        self.odom_frame = '/odom'
        self.base_frame = '/base_footprint'

        self.robot_name = ['tb3_0','tb3_1','tb3_2']
        self.position = {'tb3_0':Point(),'tb3_1':Point(),'tb3_2':Point()}

        self.rotation = {'tb3_0':0.0,'tb3_1':0.0,'tb3_2':0.0}
        
        # record target position
        self.record_info_node = {'tb3_0':[],'tb3_1':[],'tb3_2':[]}

        # set netx target position
        self.next_target_node = {'tb3_0':[],'tb3_1':[],'tb3_2':[]}

        # whether arrive information node 
        self.arr_info_node = {'tb3_0':False,'tb3_1':False,'tb3_2':False}

        

        # whether complete
        self.done = False

        # each robot share its own position max distacen
        self.communication_max_range = 6

        # Is there any information node point within the detection range
        self.detect_info_node = False

        # the value of laser_crashed_value is True when a collision is imminent,else False
        self.laser_crashed_value = {'tb3_0':False,'tb3_1':False,'tb3_2':False}
        
        #

        self.rate = rospy.Rate(10) # 100hz

        # crush default value
        self.crash_indicator = 0

        # observation_space and action_space
        self.state_num = 28 #685                 # when you change this value, remember to change the reset default function as well
        self.action_num = 2
        self.observation_space = np.empty(self.state_num)
        self.action_space = np.empty(self.action_num)


        self.laser_reward = 0

        # set turtlebot index in gazebo world
        self.model_index = 10 #25

        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
    
    def get_init_info_node(self):
        for name in self.robot_name:
            laser_ig = InfoGetter()
            rospy.Subscriber(name+'/scan', LaserScan, laser_ig)
            laser_msg = laser_ig.get_msg()
            self.laser_msg_range_max = laser_msg.range_max
            laser_values = laser_msg.ranges
            option_target_point = []
            for j in range(len(laser_values)):
                if laser_values[j] == np.inf:
                    theta = j*(laser_msg.angle_increment) + (math.pi/2 - laser_msg.angle_max)
                    option_target_point_x = self.position[name].x + (self.laser_msg_range_max * math.sin(theta) )
                    option_target_point_y = self.position[name].y - (self.laser_msg_range_max * math.cos(theta) )
                    option_target_point.append([option_target_point_x,option_target_point_y])
            option_target_point = self.voronoi_select_point(name,option_target_point)
            self.next_target_node[name] = random.choice(option_target_point)
            self.record_info_node[name].append(self.next_target_node[name])

    def reset(self):
        self.laser_crashed_value = {'tb3_0':False,'tb3_1':False,'tb3_2':False}
        self.rate.sleep()
        self.crash_indicator = 0

        # location initialization
        state_msg_0 = ModelState()    
        state_msg_0.model_name = 'tb3_0'
        state_msg_0.pose.position.x = -4.2
        state_msg_0.pose.position.y = -3.2 #random_turtlebot_y
        state_msg_0.pose.position.z = 0
        state_msg_0.pose.orientation.x = 0
        state_msg_0.pose.orientation.y = 0
        state_msg_0.pose.orientation.z = 0
        state_msg_0.pose.orientation.w = 0

        state_msg_1 = ModelState()    
        state_msg_1.model_name = 'tb3_1'
        state_msg_1.pose.position.x = -4.2
        state_msg_1.pose.position.y = -0.5 #random_turtlebot_y
        state_msg_1.pose.position.z = 0
        state_msg_1.pose.orientation.x = 0
        state_msg_1.pose.orientation.y = 0
        state_msg_1.pose.orientation.z = 0
        state_msg_1.pose.orientation.w = 0

        state_msg_2 = ModelState()    
        state_msg_2.model_name = 'tb3_2'
        state_msg_2.pose.position.x = -4.2
        state_msg_2.pose.position.y = 2.5 #random_turtlebot_y
        state_msg_2.pose.position.z = 0
        state_msg_2.pose.orientation.x = 0
        state_msg_2.pose.orientation.y = 0
        state_msg_2.pose.orientation.z = 0
        state_msg_2.pose.orientation.w = 0

        rospy.wait_for_service('/gazebo/reset_simulation')
        rospy.wait_for_service('/gazebo/set_model_state')

        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            set_state(state_msg_0)
            set_state(state_msg_1)
            set_state(state_msg_2)

            #resp_target = set_state(state_target_msg)

        except rospy.ServiceException as e:
            print ("Service call failed: %s" % e)

        # get position of robot
        for name in self.robot_name:
            self.set_tf(name)
            time.sleep(0.1)
            self.position[name],self.rotation[name] = self.get_odom(name)


        for name in self.robot_name:
            pub = rospy.Publisher(name+'/cmd_vel', Twist, queue_size=1)
            self.move_cmd.linear.x = 0
            self.move_cmd.angular.z = 0
            self.rate.sleep()
            pub.publish(self.move_cmd)
            self.rate.sleep()
            

        # get target node
        self.get_init_info_node()
        for name in self.robot_name:
            print(name,":target position",self.next_target_node[name][0],self.next_target_node[name][1])

        initial_state = np.ones(self.state_num)
        #initial_state[self.state_num-2] = 0
        initial_state[self.state_num-1] = 0
        initial_state[self.state_num-2] = 0
        initial_state[self.state_num-3] = 0
        initial_state[self.state_num-4] = 0

        self.rate.sleep()
        initial_state = [initial_state]*3
        return initial_state

    def set_tf(self,robot_name):
        try:
            self.tf_listener.waitForTransform(robot_name+self.odom_frame, robot_name+'/base_footprint', rospy.Time(), rospy.Duration(1.0))
            self.base_frame = robot_name + '/base_footprint'
        except (tf.Exception, tf.ConnectivityException, tf.LookupException):
            try:
                self.tf_listener.waitForTransform(robot_name+ self.odom_frame, robot_name+'/base_link', rospy.Time(), rospy.Duration(1.0))
                self.base_frame = robot_name + '/base_link'
            except (tf.Exception, tf.ConnectivityException, tf.LookupException):
                rospy.loginfo("Cannot find transform between odom and base_link or base_footprint")
                rospy.signal_shutdown("tf Exception")
    
    def get_odom(self,robot_name):
        try:
            (trans, rot) = self.tf_listener.lookupTransform(robot_name+ self.odom_frame, self.base_frame, rospy.Time(0))
            rotation = euler_from_quaternion(rot)

        except (tf.Exception, tf.ConnectivityException, tf.LookupException):
            rospy.loginfo("TF Exception")
            return
        return Point(*trans), rotation[2]
    
    def line_distance(self,position0_x,position0_y,position1_x,position1_y):
        return math.sqrt((position0_x - position1_x)**2 + (position0_y - position1_y)**2)

    
    def whether_detect_info_node(self):
        pass

    # the number of robot 
    def num_robot_site(self,robot_name):
        option_site = []
        for name in self.robot_name:
            if robot_name == name:
                pass
            else:
                if self.line_distance(self.position[robot_name].x,self.position[robot_name].y,self.position[name].x,self.position[name].y) <=6:
                    option_site.append(name)
        #print(option_site)
        return option_site
            
    # select new option_target_point through voronoi algorithm
    def voronoi_select_point(self,robot_name,option_target_point):
        option_site = self.num_robot_site(robot_name)
        voronoi_option_target_point = []
        if len(option_site)==0:
            return option_target_point
        for i in range(len(option_target_point)):
            j=0
            for name in option_site:
                distance = self.line_distance(self.position[name].x,self.position[name].y,option_target_point[i][0],option_target_point[i][1])
                if distance > self.laser_msg_range_max:
                    j+=1
                if distance < self.laser_msg_range_max:
                    pass
                if j== len(option_site):
                    voronoi_option_target_point.append(option_target_point[i])
        return voronoi_option_target_point

    def get_min_Omega_distance_point(self,robot_name,option_target_point):
        Omega = 0 # distance of d_ik and phi_ik
        min_Omega = np.inf
        for i in range(len(option_target_point)):
            Omega = 0.2*(self.line_distance(self.record_info_node[robot_name][0][0],self.record_info_node[robot_name][0][1],option_target_point[i][0],option_target_point[i][1])) \
                     + 0.8*(self.line_distance(self.position[robot_name].x, self.position[robot_name].y,option_target_point[i][0],option_target_point[i][1]))
            if Omega < min_Omega:
                min_Omega = Omega
                return option_target_point[i]
    
    def get_record_next_info_node(self,robot_name,option_target_point):
        if(self.arr_info_node[robot_name] == True):
            option_target_point = self.voronoi_select_point(robot_name,option_target_point) # Further select the next point through the Voronoi algorithm
            if len(option_target_point) == 0:
                return False
            else:
                self.next_target_node[robot_name] = self.get_min_Omega_distance_point(robot_name,option_target_point)
                print(robot_name,":target position",self.next_target_node[robot_name][0],self.next_target_node[robot_name][1])
                self.record_info_node[robot_name].append(self.next_target_node[robot_name])
                self.arr_info_node[robot_name] = False
                return True

    def game_step(self, robot_name):
        pub = rospy.Publisher(robot_name+'/cmd_vel', Twist, queue_size=1)
        move_base = actionlib.SimpleActionClient('move_base', MoveBaseAction)  

        laser_ig = InfoGetter()
        rospy.Subscriber(robot_name+'/scan', LaserScan, laser_ig)

        start_time = time.time()
        record_time = start_time


        self.base_frame = robot_name + '/base_footprint'

        self.position[robot_name], self.rotation[robot_name]= self.get_odom(robot_name)
        turtlebot_x_previous = self.position[robot_name].x
        turtlebot_y_previous = self.position[robot_name].y

        if self.line_distance(self.position[robot_name].x,self.position[robot_name].y,self.next_target_node[robot_name][0],self.next_target_node[robot_name][1])<0.5:
            self.arr_info_node[robot_name] = True
        else:
            self.arr_info_node[robot_name] = False

        
        # get list of optional target point
        #self.arr_info_node[robot_name] = True
        option_target_point = []
        theta= 0
        if self.arr_info_node[robot_name] == True:
            laser_msg = laser_ig.get_msg()
            laser_values = laser_msg.ranges
            for i in range(len(laser_values)):
                if laser_values[i] == np.inf:
                    theta = (i+1)*laser_msg.angle_increment
                    option_target_point_x = turtlebot_x_previous + (self.laser_msg_range_max * math.sin(theta))
                    option_target_point_y = turtlebot_y_previous - (self.laser_msg_range_max * math.cos(theta))
                    option_target_point.append([option_target_point_x,option_target_point_y])
            if len(option_target_point):
                self.get_record_next_info_node(robot_name,option_target_point)
            else:
                self.move_cmd.linear.x = 0
                self.move_cmd.angular.z = 0
                self.rate.sleep()
                self.laser_crashed_value[robot_name] =1 
                # prepare the normalized laser value and check if it is crash
        
        laser_msg = laser_ig.get_msg()
        laser_values = laser_msg.ranges
        for i in range(len(laser_values)):
            if (laser_values[i] < 0.3):
                self.laser_crashed_value[robot_name] = True
                #self.reset()
            if (laser_values[i] < 0.2):
                self.laser_crashed_value[robot_name] = True
                self.reset()
                break
        target_x = self.next_target_node[robot_name][0]
        target_y = self.next_target_node[robot_name][1]


        move_base.wait_for_server(rospy.Duration(5.0))
        
        # set target point
        target = Pose(Point(target_x, target_y, 0.000), Quaternion(0.000, 0.000, 0.000, 0.000))  
        goal = MoveBaseGoal()  
        goal.target_pose.pose = target  
        goal.target_pose.header.frame_id = 'map'  
        goal.target_pose.header.stamp = rospy.Time.now()

        rospy.loginfo("Going to: " + str(target))  

        # go target point 
        move_base.send_goal(goal)

        finished_within_time = move_base.wait_for_result(rospy.Duration(10))

        # whether get target point  
        if not finished_within_time:  
            move_base.cancel_goal()  
            rospy.loginfo("Timed out achieving goal")  
        else:  
            state = move_base.get_state()  
            if state == GoalStatus.SUCCEEDED:  
                rospy.loginfo("Goal succeeded!")
            else:  
                rospy.loginfo("Goal failed!")  

    

if __name__ == '__main__':
    try:
        sess = tensorflow.Session()
        K.set_session(sess)

        game_state = GameState()
        game_state.reset()
        game_state.game_step(game_state.robot_name[0])
  


    except rospy.ROSInterruptException:
        pass
