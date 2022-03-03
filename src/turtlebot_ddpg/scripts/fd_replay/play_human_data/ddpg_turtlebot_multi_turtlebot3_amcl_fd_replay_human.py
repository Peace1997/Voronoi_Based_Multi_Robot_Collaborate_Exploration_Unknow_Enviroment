#! /usr/bin/env python


## version 2: 
## 1, navigate the robot using a constant heading angle
## 2, add the ddpg neural network
## 3, 24 laser data and just heading
## 4, added potential collisions



## Command:
## roslaunch turtlebot_iros turtlebot_world.launch world_file:='/home/hanlin/catkin_ws/src/turtlebot/turtlebot_iros/modified_world.world'
## source ~/iros_env/bin/activate
## rosrun turtlebot_iros ddpg_turtlebot.py

import rospy
import rospkg
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
        #self.laser_ig = InfoGetter()
        self.collision_ig = InfoGetter()
        

        #self.pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        # self.pub0 = rospy.Publisher('/tb3_0/cmd_vel', Twist, queue_size=10)
        # self.pub1 = rospy.Publisher('/tb3_1/cmd_vel', Twist, queue_size=10)
        # self.pub2 = rospy.Publisher('/tb3_2/cmd_vel', Twist, queue_size=10)
        
        # self.move_cmd_0 = Twist()
        # self.move_cmd_1 = Twist()
        # self.move_cmd_2 = Twist()
        self.move_cmd = Twist()
        

        # self.move_cmd_0.linear.x = 0    
        # self.move_cmd_0.angular.z = 0

        # self.move_cmd_1.linear.x = 0
        # self.move_cmd_1.angular.z = 0

        # self.move_cmd_2.linear.x = 0
        # self.move_cmd_2.angular.z = 0


        # self.pose_info = rospy.Subscriber("/gazebo/model_states", ModelStates, self.pose_ig)
        # self.laser_info = rospy.Subscriber("/laserscan_filtered", LaserScan, self.laser_ig)
        # self.bumper_info = rospy.Subscriber("/mobile_base/events/bumper", BumperEvent, self.processBump)


        # self.laser_info_0 = rospy.Subscriber("tb3_0/scan", LaserScan, self.laser_ig)
        # self.laser_info_1 = rospy.Subscriber("tb3_1/scan", LaserScan, self.laser_ig)
        # self.laser_info_2 = rospy.Subscriber("tb3_2/scan", LaserScan, self.laser_ig)



        self.position_0 = Point()
        self.position_1 = Point()
        self.position_2 = Point()

        # tf
        self.tf_listener = tf.TransformListener()
        rospy.sleep(2)
        self.odom_frame = '/odom'
        self.base_frame = '/base_footprint'

        self.robot_0 = 'tb3_0'
        self.robot_1 = 'tb3_1'
        self.robot_2 = 'tb3_2'
        
        
        self.set_tf(self.robot_0)
        (self.position_0, self.rotation_0) = self.get_odom(self.robot_0)

        self.set_tf(self.robot_1)
        (self.position_1, self.rotation_1) = self.get_odom(self.robot_1)

        self.set_tf(self.robot_2)
        (self.position_2, self.rotation_2) = self.get_odom(self.robot_2)
        

  
        self.rate = rospy.Rate(100) # 100hz

        # Create a Twist message and add linear x and angular z values
        # self.move_cmd = Twist()
        # self.move_cmd.linear.x = 0.6 #linear_x
        # self.move_cmd.angular.z = 0.2 #angular_z

        # crush default value
        self.crash_indicator = 0

        # observation_space and action_space
        self.state_num = 28 #685                 # when you change this value, remember to change the reset default function as well
        self.action_num = 2
        self.observation_space = np.empty(self.state_num)
        self.action_space = np.empty(self.action_num)
        # self.state_input1_space =  np.empty(1)
        # self.state_input2_space =  np.empty(1)

        self.laser_reward = 0

        # record target position
        self.list_info_x_0= []
        self.list_info_y_0= []
        self.list_info_x_1= []
        self.list_info_y_1= []
        self.list_info_x_2= [] 
        self.list_info_y_2= []

        # set netx target position
        self.next_target_node_x_0=0.0
        self.next_target_node_y_0=0.0
        self.next_target_node_x_1=0.0
        self.next_target_node_y_1=0.0
        self.next_target_node_x_2=0.0
        self.next_target_node_y_2=0.0

        # whether complete
        self.done = False

        # Is there any information node point within the detection range
        self.detect_info_node = False
        
        # whether arrive information node 
        self.whe_arr_info_node_0 = False 
        self.whe_arr_info_node_1 = False
        self.whe_arr_info_node_2 = False


        # set turtlebot index in gazebo world
        self.model_index = 10 #25

        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)


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

        return (Point(*trans), rotation[2])






    def shutdown(self):
        self.cmd_vel.publish(Twist())
        rospy.sleep(1)


    def print_odom(self,robot_name):
        while True:
            (position, rotation) = self.get_odom(robot_name)
            print("position is %s, %s, %s, ", position.x, position.y, position.z)
            print("rotation is %s, ", rotation)


    def reset(self):
        index_list = [-1, 1]
        index_x = random.choice(index_list)
        index_y = random.choice(index_list)
        index_turtlebot_y = random.choice(index_list)
        # for maze
        # self.target_x = (np.random.random()-0.5)*5 + 12*index_x
        # self.target_y = (np.random.random()-0.5)*5 + 12*index_y
        # random_turtlebot_y = (np.random.random())*4 + index_turtlebot_y

        # for corridor
        # self.target_x = (np.random.random()-0.5)*5 + 12*index_x
        # self.target_y = (np.random.random()-0.5)*3
        random_turtlebot_y = (np.random.random())*5 #+ index_turtlebot_y


        self.crash_indicator = 0




        state_msg_0 = ModelState()    
        state_msg_0.model_name = 'tb3_0'
        state_msg_0.pose.position.x = -5.9
        state_msg_0.pose.position.y = -2.9 #random_turtlebot_y
        state_msg_0.pose.position.z = 0.0
        state_msg_0.pose.orientation.x = 0
        state_msg_0.pose.orientation.y = 0
        state_msg_0.pose.orientation.z = 0
        state_msg_0.pose.orientation.w = 0

        state_msg_1 = ModelState()    
        state_msg_1.model_name = 'tb3_1'
        state_msg_1.pose.position.x = -7.0
        state_msg_1.pose.position.y = -1.4 #random_turtlebot_y
        state_msg_1.pose.position.z = 0.0
        state_msg_1.pose.orientation.x = 0
        state_msg_1.pose.orientation.y = 0
        state_msg_1.pose.orientation.z = 0
        state_msg_1.pose.orientation.w = 0

        state_msg_2 = ModelState()    
        state_msg_2.model_name = 'tb3_2'
        state_msg_2.pose.position.x = -5.8
        state_msg_2.pose.position.y = -0.1 #random_turtlebot_y
        state_msg_2.pose.position.z = 0.0
        state_msg_2.pose.orientation.x = 0
        state_msg_2.pose.orientation.y = 0
        state_msg_2.pose.orientation.z = 0
        state_msg_2.pose.orientation.w = 0




        # state_target_msg = ModelState()    
        # state_target_msg.model_name = 'unit_sphere_0_0' #'unit_sphere_0_0' #'unit_box_1' #'cube_20k_0'
        # state_target_msg.pose.position.x = self.target_x
        # state_target_msg.pose.position.y = self.target_y
        # state_target_msg.pose.position.z = 0.0
        # state_target_msg.pose.orientation.x = 0
        # state_target_msg.pose.orientation.y = 0
        # state_target_msg.pose.orientation.z = -0.2
        # state_target_msg.pose.orientation.w = 0


        rospy.wait_for_service('gazebo/reset_simulation')
        # try:
        #     self.reset_proxy()
        # except (rospy.ServiceException) as e:
        #     print("gazebo/reset_simulation service call failed")

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            set_state(state_msg_0)
            set_state(state_msg_1)
            set_state(state_msg_2)
            
            #resp_target = set_state(state_target_msg)

        except rospy.ServiceException as e:
            print ("Service call failed: %s" % e)

        initial_state = np.ones(self.state_num)
        #initial_state[self.state_num-2] = 0
        initial_state[self.state_num-1] = 0
        initial_state[self.state_num-2] = 0
        initial_state[self.state_num-3] = 0
        initial_state[self.state_num-4] = 0


        # self.move_cmd_0.linear.x = 0    
        # self.move_cmd_0.angular.z = 0

        # self.move_cmd_1.linear.x = 0
        # self.move_cmd_1.angular.z = 0

        # self.move_cmd_2.linear.x = 0
        # self.move_cmd_2.angular.z = 0

        # time.sleep(1)
        # self.pub0.publish(self.move_cmd_0)
        # time.sleep(1)
        # self.pub1.publish(self.move_cmd_1)
        # time.sleep(1)
        # self.pub2.publish(self.move_cmd_2)
        # time.sleep(1)

        self.rate.sleep()


        return initial_state

    def line_distance(self,position0_x,position0_y,position1_x,position1_y):
        return math.sqrt((position0_x - position1_x)**2 + (position0_y - position1_y)**2)

    def turtlebot_is_crashed(self, laser_values, range_limit):
        self.laser_crashed_value = 0
        self.laser_crashed_reward = 0

        for i in range(len(laser_values)):
            if (laser_values[i] < 2*range_limit):
                self.laser_crashed_reward = -80
            if (laser_values[i] < range_limit):
                self.laser_crashed_value = 1
                self.laser_crashed_reward = -200
                self.reset()
                break
        return self.laser_crashed_reward

    def whether_arrive_info_node(self,robot_name):
        self.base_frame = robot_name + '/base_footprint'
        (position, rotation) = self.get_odom(robot_name)
        if robot_name == self.robot_0:
            if self.line_distance(self.next_target_node_x_0,self.next_target_node_y_0,position.x,position.y) < 0.2:
                self.whe_arr_info_node_0 = True
            else:
                self.whe_arr_info_node_0 = False
        elif robot_name == self.robot_1:
            if self.line_distance(self.next_target_node_x_1,self.next_target_node_y_1,position.x,position.y) < 0.2:
                self.whe_arr_info_node_0 = True
            else:
                self.whe_arr_info_node_0 = False
        else:
            if self.line_distance(self.next_target_node_x_2,self.next_target_node_y_2,position.x,position.y) < 0.2:
                self.whe_arr_info_node_0 = True
            else:
                self.whe_arr_info_node_0 = False
    
    def whether_detect_info_node(self):
        pass
    
    def get_min_Omega_distance_point(self,position_x,position_y,option_target_point):
        Omega = 0 # distance of d_ik and phi_ik
        min_Omega = np.inf
        target=[]
        for i in range(len(option_target_point)):
            Omega = self.line_distance(self.list_info_x_0[0],self.list_info_y_0[0],option_target_point[0][i][0],option_target_point[0][i][1]) \
                     + self.line_distance(position_x,position_y,option_target_point[0][i][0],option_target_point[0][i][1])
            if Omega < min_Omega:
                min_Omega = Omega
                target[0],target[1] = option_target_point[0][i][0],option_target_point[0][i][1]
                return target[0],target[1]

    def get_recoed_next_info_node(self,robot_name,option_target_point):
        self.base_frame = robot_name + '/base_footprint'
        target = []
        (position, rotation) = self.get_odom(robot_name)
        if robot_name == self.robot_0:
            if not self.list_info_x_0:
                target=random.choice(option_target_point)
                self.list_info_x_0.append(position.x) 
                self.list_info_y_0.append(position.y)
            else:
                if self.whether_arrive_info_node_0 == True:
                    self.next_target_node_x_0,self.next_target_node_y_0= self.get_min_Omega_distance_point(position.x,position.y,option_target_point)
            self.whether_arrive_info_node_0 = False
        elif robot_name == self.robot_1:
            if not self.list_info_x_1:
                target=random.choice(option_target_point)
                self.list_info_x_1.append(position.x) 
                self.list_info_y_1.append(position.y)
            else:
                if self.whether_arrive_info_node_1 == True:
                    self.next_target_node_x_1,self.next_target_node_y_1= self.get_min_Omega_distance_point(position.x,position.y,option_target_point)
            self.whether_arrive_info_node_1 = False
        else:
            if not self.list_info_x_2:
                target=random.choice(option_target_point)
                self.next_target_node_x_2,self.next_target_node_y_2 = target[0],target[1]
                self.list_info_x_2.append(position.x) 
                self.list_info_y_2.append(position.y)
                self.whether_arrive_info_node_2 = False
            else:
                if self.whether_arrive_info_node_2 == True:
                    self.next_target_node_x_2,self.next_target_node_y_2= self.get_min_Omega_distance_point(position.x,position.y,option_target_point)
            self.whether_arrive_info_node_2 = False
        


    def game_step(self, robot_name, time_step=0.1, linear_x=0.8, angular_z=0.3):
        
        pub = rospy.Publisher(robot_name+'/cmd_vel', Twist, queue_size=10)
        laser_ig = InfoGetter()
        rospy.Subscriber(robot_name+'/scan', LaserScan, laser_ig)

        start_time = time.time()
        record_time = start_time
        record_time_step = 0
        self.move_cmd.linear.x = linear_x*0.26
        self.move_cmd.angular.z = angular_z
        self.rate.sleep()

        self.base_frame = robot_name + '/base_footprint'
        (position, rotation) = self.get_odom(robot_name)
        turtlebot_x_previous = position.x
        turtlebot_y_previous = position.y

        whe_arr_info_node = False
        # get list of optional target point
        option_target_point = []
        theta= 0 
        if whe_arr_info_node == True:
            laser_msg = laser_ig.get_msg()
            laser_values = laser_msg.ranges
            for i in range(len(laser_values)):
                if laser_values[i] == np.inf:
                    theta = (i+1)*laser_msg.angle_increment
                    option_target_point_x = turtlebot_x_previous + (laser_msg.range_max * math.sin(theta) )
                    option_target_point_y = turtlebot_y_previous - (laser_msg.range_max * math.cos(theta) )
                    option_target_point.append([option_target_point_x,option_target_point_y])
            
        target_x = 0.0
        target_y = 0.0
        if robot_name == self.robot_0:
            self.get_recoed_next_info_node(robot_name,option_target_point)
            whe_arr_info_node = self.whe_arr_info_node_0
            target_x,target_y = self.next_target_node_x_0,self.next_target_node_y_0
        elif robot_name == self.robot_1:
            self.get_recoed_next_info_node(robot_name,option_target_point)
            whe_arr_info_node = self.whe_arr_info_node_1
            target_x,target_y = self.next_target_node_x_1,self.next_target_node_y_1
        else:
            self.get_recoed_next_info_node(robot_name,option_target_point)
            whe_arr_info_node = self.whe_arr_info_node_2
            target_x,target_y = self.next_target_node_x_2,self.next_target_node_y_2
        
            
        while (record_time_step < time_step) and (self.crash_indicator==0):
            pub.publish(self.move_cmd)
            self.rate.sleep()
            record_time = time.time()
            record_time_step = record_time - start_time

        (position, rotation) = self.get_odom(robot_name)
        turtlebot_x = position.x
        turtlebot_y = position.y
        print(robot_name,":",position.x,position.y)
        angle_turtlebot = rotation

        # make input, angle between the turtlebot and the target
        angle_turtlebot_target = atan2(target_y - turtlebot_y, target_x- turtlebot_x)

        if angle_turtlebot < 0:
            angle_turtlebot = angle_turtlebot + 2*math.pi

        if angle_turtlebot_target < 0:
            angle_turtlebot_target = angle_turtlebot_target + 2*math.pi


        angle_diff = angle_turtlebot_target - angle_turtlebot
        if angle_diff < -math.pi:
            angle_diff = angle_diff + 2*math.pi
        if angle_diff > math.pi:
            angle_diff = angle_diff - 2*math.pi



        # prepare the normalized laser value and check if it is crash
        
        laser_msg = laser_ig.get_msg()
        laser_values = laser_msg.ranges

        #print('turtlebot laser_msg.ranges is %s', laser_msg.ranges)
        #print('turtlebot laser data is %s', laser_values)

        normalized_laser = list(laser_values)
        for i in range(len(normalized_laser)):
            if normalized_laser[i] == np.inf:
                normalized_laser[i] = 1.0
            else:
                normalized_laser[i] = normalized_laser[i]/laser_msg.range_max
        # normalized_laser = [(x)/3.5 for x in (laser_msg.ranges)]
        # print('turtlebot normalized laser range is %s', normalized_laser)


        # prepare state
        #state = np.append(normalized_laser, angle_diff)
        #state = np.append(normalized_laser,self.target_x- turtlebot_x)
        #state = np.append(state, self.target_y - turtlebot_y)
        current_distance_turtlebot_target = math.sqrt((target_x - turtlebot_x)**2 + (target_y - turtlebot_y)**2)

        state = np.append(normalized_laser, current_distance_turtlebot_target)
        state = np.append(state, angle_diff)
        state = np.append(state, linear_x*0.26)
        state = np.append(state, angular_z)
        # print("angle_turtlebot and angle_diff are %s %s", angle_turtlebot*180/math.pi, angle_diff*180/math.pi)
        # print("position x is %s position y is %s", turtlebot_x, turtlebot_y)
        # print("target position x is %s target position y is %s", self.target_x, self.target_y)
        # print("command angular is %s", angular_z*1.82)
        # print("command linear is %s", linear_x*0.26)
        #print("state is %s", state)

        state = state.reshape(1, self.state_num)


        # make distance reward
        (position, rotation) = self.get_odom(robot_name)
        turtlebot_x = position.x
        turtlebot_y = position.y
        distance_turtlebot_target_previous = math.sqrt((target_x - turtlebot_x_previous)**2 + (target_y - turtlebot_y_previous)**2)
        distance_turtlebot_target = math.sqrt((target_x - turtlebot_x)**2 + (target_y - turtlebot_y)**2)
        distance_reward = distance_turtlebot_target_previous - distance_turtlebot_target


        self.laser_crashed_reward = self.turtlebot_is_crashed(laser_values, range_limit=0.25)
        self.laser_reward = sum(normalized_laser)-24
        self.collision_reward = self.laser_crashed_reward + self.laser_reward
        self.collision_reward =0.0

        self.angular_punish_reward = 0
        self.linear_punish_reward = 0

        if angular_z > 0.8:
            self.angular_punish_reward = -1
        if angular_z < -0.8:
            self.angular_punish_reward = -1

        if linear_x < 0.2:
            self.linear_punish_reward = -2


        self.arrive_reward = 0
        if distance_turtlebot_target<0.2:
            self.arrive_reward = 100
            
            self.reset()
            time.sleep(1)


 

        reward  = distance_reward*(5/time_step)*1.2*7 + self.arrive_reward + self.collision_reward + self.angular_punish_reward + self.linear_punish_reward
        # print("laser_reward is %s", self.laser_reward)
        # print("laser_crashed_reward is %s", self.laser_crashed_reward)
        # print("arrive_reward is %s", self.arrive_reward)
        # print("distance reward is : %s", distance_reward*(5/time_step)*1.2*7)
        if reward>1000:
            pass

        return reward, state, self.laser_crashed_value




if __name__ == '__main__':
    try:
        sess = tensorflow.Session()
        K.set_session(sess)

        game_state = GameState()
        game_state.reset()

        for i in range(10):

            game_state.game_step(game_state.robot_0,time_step=0.1,linear_x=0.0,angular_z=0)
            game_state.game_step(game_state.robot_1,time_step=0.1,linear_x=0.0,angular_z=0)
            game_state.game_step(game_state.robot_2,time_step=0.1,linear_x=0.0,angular_z=0)


    except rospy.ROSInterruptException:
        pass


