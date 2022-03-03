#! /usr/bin/env python

from nav_msgs import msg
import rospy
import rospkg
from rospy import names
from rospy.timer import sleep
import tf
from nav_msgs.msg import Path
from std_msgs.msg import String
from geometry_msgs.msg import Twist, Point, Quaternion,PoseStamped
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

from nav_msgs.msg import OccupancyGrid
from map_msgs.msg import OccupancyGridUpdate
from nav_msgs.srv import GetMap
from nav_msgs.msg import MapMetaData
from std_msgs.msg import Int8MultiArray



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
        self.set_disable_signals =False
        self.talker_node = rospy.init_node('talker', anonymous=True,disable_signals= self.set_disable_signals)
        self.pose_ig = InfoGetter()
        self.collision_ig = InfoGetter()
        
        self.move_cmd = Twist()

        # tf
        self.tf_listener = tf.TransformListener()
        self.odom_frame = '/odom'
        self.base_frame = '/base_footprint'

        self.robot_name = ['tb3_0']
        self.position = {'tb3_0':Point()}

        self.rotation = {'tb3_0':0.0}
        
        # record target position
        self.record_info_node = {'tb3_0':[]}

        # set netx target position
        self.next_target_node = {'tb3_0':[]}

        # whether arrive information node 
        self.arr_info_node = {'tb3_0':False}

        # whether complete
        self.done = False

        # each robot share its own position max distacen
        self.communication_max_range = 8

        # Is there any information node point within the detection range
        self.detect_info_node = False

        # 
        self.laser_crashed_value = {'tb3_0':False}

        self.rate = rospy.Rate(100) # 100hz
        
        # cmd_vel
        cmd_vel_0 = rospy.Publisher('tb3_0/cmd_vel', Twist, queue_size=10,latch=True)

        self.cmd_vel_pub = {'tb3_0':cmd_vel_0}

        # mapprob
        # mapprob_0 = rospy.Publisher('tb3_0/mapprob', Int8MultiArray, queue_size=10)
        # mapprob_1 = rospy.Publisher('tb3_1/mapprob', Int8MultiArray, queue_size=10)
        # mapprob_2 = rospy.Publisher('tb3_2/mapprob', Int8MultiArray, queue_size=10)
        # self.mapprob_pub =  {'tb3_0':mapprob_0,'tb3_1':mapprob_1,'tb3_2':mapprob_2}

        # map data
        # rospy.wait_for_service('/tb3_0/dynamic_map')
        # try:
        #     self.map_service = rospy.ServiceProxy('/tb3_0/dynamic_map', GetMap)
        # except rospy.ServiceException as e:
        #     print("Service call failed: %s"%e)
        # self.map_pub = rospy.Publisher('tb3_0/map_new', OccupancyGridUpdate,queue_size=10)
        self.map_merge_data = OccupancyGrid()

        self.map1_free_num = 6251.0
        
        self.target_explored_region_rate = 0.4

        self.done = False


        # path_pub_0 = rospy.Publisher('tb3_0/trajectory', Path, queue_size=50)
        # path_pub_1 = rospy.Publisher('tb3_0/trajectory', Path, queue_size=50)
        # path_pub_2 = rospy.Publisher('tb3_0/trajectory', Path, queue_size=50)
        # self.path_pub = {'tb3_0':path_pub_0,'tb3_1':path_pub_1,'tb3_2':path_pub_2}




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
            #self.next_target_node[name] = self.get_min_Omega_distance_point(name,option_target_point)
            self.next_target_node[name] = random.choice(option_target_point)
            self.record_info_node[name].append(self.next_target_node[name])


    def reset(self):
        # set tf & get position
        for name in self.robot_name:
            self.set_tf(name)
            self.position[name],self.rotation[name] = self.get_odom(name)
            time.sleep(1)

        self.laser_crashed_value = {'tb3_0':False,'tb3_1':False,'tb3_2':False}
        self.rate.sleep()
        self.crash_indicator = 0
        current_time = rospy.Time.now()

        rospy.wait_for_service('/gazebo/set_model_state')

        for name in self.robot_name:
            cmd_vel_pub = self.cmd_vel_pub[name]
            self.move_cmd.linear.x = 0
            self.move_cmd.angular.z = 0
            cmd_vel_pub.publish(self.move_cmd)
            time.sleep(0.1)
            cmd_vel_pub.publish(self.move_cmd)
            self.rate.sleep()

        # location initialization
        state_msg_0 = ModelState()    
        state_msg_0.model_name = 'tb3_0'
        state_msg_0.pose.position.x = -3
        state_msg_0.pose.position.y = 0.0 #random_turtlebot_y
        state_msg_0.pose.position.z = 0
        state_msg_0.pose.orientation.x = 0
        state_msg_0.pose.orientation.y = 0
        state_msg_0.pose.orientation.z = 0
        state_msg_0.pose.orientation.w = 0
        state_msg_0.twist.linear.x= 0.0
        state_msg_0.twist.linear.y= 0.0
        state_msg_0.twist.linear.z= 0.0
        state_msg_0.twist.angular.x = 0.0
        state_msg_0.twist.angular.y = 0.0
        state_msg_0.twist.angular.z = 0.0



        rospy.wait_for_service('/gazebo/reset_simulation')


        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            set_state(state_msg_0)


            #resp_target = set_state(state_target_msg)

        except rospy.ServiceException as e:
            print ("Service call failed: %s" % e)



        for name in self.robot_name:
            cmd_vel_pub =  self.cmd_vel_pub[name]
            self.move_cmd.linear.x = 0
            self.move_cmd.angular.z = 0
            cmd_vel_pub.publish(self.move_cmd)
            time.sleep(0.1)
            cmd_vel_pub.publish(self.move_cmd)
            self.rate.sleep()


       # get position of robot
        for name in self.robot_name:
            self.base_frame = name + '/base_footprint'
            self.position[name],self.rotation[name] = self.get_odom(name)
            time.sleep(0.1)
            self.record_info_node[name].append([self.position[name].x,self.position[name].y])

        # get target node
        self.get_init_info_node()
        for name in self.robot_name:
            print(name,":target position",self.next_target_node[name][0],self.next_target_node[name][1])

        
        # map data

        # map_0_data = OccupancyGrid()
        # map_0_data.header.frame_id = map_msg.header
        # map_0_data.header.stamp = rospy.Time.now()
        # map_0_data.info = map_msg.info
        # map_0_data.data = [-1]*len(map_msg.data)
        
        

        # rospy.loginfo("Subscribing to updates!")
        # self._updates_sub = rospy.Subscriber('tb3_0/map_updates',OccupancyGridUpdate,self._occ_grid_update_cb,queue_size=10)



        # map_0_data.data = (0,)*25600
        # times = 0
        # while times < 10:
        #     self.map_service().map.data = (-1,)*25600
        #     times += 1
        #     rospy.sleep(1)
        
        # free_num = 0
        # map_ig = InfoGetter()
        # rospy.Subscriber('tb3_0/map',OccupancyGrid,map_ig)
        # map_msg = map_ig.get_msg()
        # for i in map_msg.data:
        #     if i == 0:
        #         free_num+=1
        # print(free_num)
        



        initial_state = np.ones(self.state_num)
        #initial_state[self.state_num-2] = 0
        initial_state[self.state_num-1] = 0
        initial_state[self.state_num-2] = 0
        initial_state[self.state_num-3] = 0
        initial_state[self.state_num-4] = 0

        self.rate.sleep()
        initial_state = [initial_state]*3
        return initial_state

    def _occ_grid_update_cb(self, data):
        rospy.loginfo("Got a partial OccupancyGrid update")
        # x, y origin point of the update
        # width and height of the update
        # data, the update
        # data comes in row-major order http://docs.ros.org/en/melodic/api/nav_msgs/html/msg/OccupancyGrid.html
        # first index is the row, second index the column
        data_np = np.array(data.data,
                           dtype=np.int8).reshape(data.height, data.width)
        self._grid_data[data.y:data.y +
                        data.height, data.x:data.x + data.width] = data_np
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

    # the number of robot 
    def num_robot_site(self,robot_name):
        option_site = []
        for name in self.robot_name:
            if robot_name == name:
                pass
            else:
                if self.line_distance(self.position[robot_name].x,self.position[robot_name].y,self.position[name].x,self.position[name].y) <= self.communication_max_range:
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

    def avoid_repeat_select_point(self,robot_name,option_target_point):
        avoid_repeat_option_target_point = []
        for i in range(len(option_target_point)):
            for j in range(len(self.record_info_node[robot_name])):
                if self.line_distance(option_target_point[i][0],option_target_point[i][1],self.record_info_node[robot_name][j][0],self.record_info_node[robot_name][j][1])>0.1:
                    avoid_repeat_option_target_point.append(option_target_point[i])
        return avoid_repeat_option_target_point

    def distance_other_point(self,robot_name,option_target_point,i):
        distance = 0.0
        for name in self.robot_name:
            if name != robot_name:
                if self.line_distance(self.position[robot_name].x,self.position[robot_name].y,self.position[name].x,self.position[name].y) <=self.communication_max_range:
                    distance += self.line_distance(self.position[name].x,self.position[name].y,option_target_point[i][0],option_target_point[i][1])
        return distance
    def get_min_Omega_distance_point(self,robot_name,option_target_point):
        Omega = 0 # distance of d_ik and phi_ik
        min_Omega = np.inf
        index = 0
        for i in range(len(option_target_point)):
            Omega = 0.3*(self.line_distance(self.record_info_node[robot_name][0][0],self.record_info_node[robot_name][0][1],option_target_point[i][0],option_target_point[i][1])) \
                     + 0.6*(self.line_distance(self.record_info_node[robot_name][-1][0], self.record_info_node[robot_name][-1][1],option_target_point[i][0],option_target_point[i][1])) \
                        -0.2*(self.distance_other_point(robot_name,option_target_point,i))
            if Omega < min_Omega:
                min_Omega = Omega
                index =  i
        return option_target_point[index]
    
    def get_record_next_info_node(self,robot_name,option_target_point):
        if(self.arr_info_node[robot_name] == True):
            option_target_point = self.voronoi_select_point(robot_name,option_target_point) # Further select the next point through the Voronoi algorithm
            option_target_point = self.avoid_repeat_select_point(robot_name,option_target_point)
            if len(option_target_point) == 0:
                return False
            else:
                self.next_target_node[robot_name] = self.get_min_Omega_distance_point(robot_name,option_target_point)
                print(robot_name,":target position",self.next_target_node[robot_name][0],self.next_target_node[robot_name][1])
                self.record_info_node[robot_name].append(self.next_target_node[robot_name])
                self.arr_info_node[robot_name] = False
                return True

    def map_data_handle(self):
        free_num=0
        explored_region_rate =0.0
        for data in self.map_merge_data.data:
            if data == 0:
                free_num += 1
        explored_region_rate = free_num/self.map1_free_num
        print(explored_region_rate)
        if explored_region_rate >= self.target_explored_region_rate:
            self.done = True

    def game_step(self, robot_name, time_step=0.1, linear_x=0.8, angular_z=0.3):
        cmd_vel_pub =  self.cmd_vel_pub[robot_name]
        # path_pub =  self.path_pub[robot_name]

        map_ig = InfoGetter()
        rospy.Subscriber('/tb3_0/map',OccupancyGrid,map_ig)
        map_msg = map_ig.get_msg()
        self.map_merge_data.data = map_msg.data
        self.map_data_handle()

        laser_ig = InfoGetter()
        rospy.Subscriber(robot_name+'/scan', LaserScan, laser_ig)
        

        start_time = time.time()
        current_time = rospy.Time.now()
        record_time = start_time
        record_time_step = 0
        self.move_cmd.linear.x = linear_x*0.26
        self.move_cmd.angular.z = angular_z
        self.rate.sleep()

        self.base_frame = robot_name + '/base_footprint'

        self.position[robot_name], self.rotation[robot_name]= self.get_odom(robot_name)
        turtlebot_x_previous = self.position[robot_name].x
        turtlebot_y_previous = self.position[robot_name].y

        while (record_time_step < time_step) and (self.crash_indicator==0):
            cmd_vel_pub.publish(self.move_cmd)
            self.rate.sleep()
            record_time = time.time()
            record_time_step = record_time - start_time

        self.position[robot_name], self.rotation[robot_name]= self.get_odom(robot_name)
        turtlebot_x = self.position[robot_name].x
        turtlebot_y = self.position[robot_name].y
        turtlebot_z = self.rotation[robot_name]

        angle_turtlebot = self.rotation[robot_name]

        # pose = PoseStamped()
        # pose.header.stamp = current_time
        # pose.header.frame_id = robot_name+'/odom'
        # pose.pose.position.x = turtlebot_x
        # pose.pose.position.y = turtlebot_y

        # path_record = Path()
        # path_record.header.stamp = current_time
        # path_record.header.frame_id = robot_name+'/odom'
        # path_record.poses.append(pose)
        
        # path_pub.publish(path_record)
        # self.rate.sleep()


        target_x = self.next_target_node[robot_name][0]
        target_y = self.next_target_node[robot_name][1]
        if self.line_distance(turtlebot_x,turtlebot_y,target_x,target_y)<1:
            self.arr_info_node[robot_name] = True
        else:
            self.arr_info_node[robot_name] = False

        
        # get list of optional target point
        #self.arr_info_node[robot_name] = True
        laser_msg = laser_ig.get_msg()
        laser_values = laser_msg.ranges
        for i in range(len(laser_values)):
            if (laser_values[i] < 0.15):
                self.reset()
                break
        #self.arr_info_node[robot_name] = True
        if self.arr_info_node[robot_name] == True:
            option_target_point = []
            theta= 0
            for i in range(len(laser_values)):
                if laser_values[i] == np.inf:
                    
                    theta = i*laser_msg.angle_increment + turtlebot_z+ (math.pi/2-laser_msg.angle_max)
                    option_target_point_x = turtlebot_x + (self.laser_msg_range_max * math.sin(theta))
                    option_target_point_y = turtlebot_y - (self.laser_msg_range_max * math.cos(theta))
                    option_target_point.append([option_target_point_x,option_target_point_y])
            if len(option_target_point):
                self.get_record_next_info_node(robot_name,option_target_point)
            else:
                self.move_cmd.linear.x = 0
                self.move_cmd.angular.z = 0
                while (record_time_step < time_step):
                    cmd_vel_pub.publish(self.move_cmd)
                    self.rate.sleep()
                    record_time = time.time()
                    record_time_step = record_time - start_time
                self.laser_crashed_value[robot_name] =1 


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

        normalized_laser = list(laser_values)
        for i in range(len(normalized_laser)):
            if normalized_laser[i] == np.inf:
                normalized_laser[i] = 1.0
            else:
                normalized_laser[i] = normalized_laser[i]/self.laser_msg_range_max

        current_distance_turtlebot_target = math.sqrt((target_x - turtlebot_x)**2 + (target_y - turtlebot_y)**2)

        state = np.append(normalized_laser, current_distance_turtlebot_target)
        state = np.append(state, angle_diff)
        state = np.append(state, linear_x*0.26)
        state = np.append(state, angular_z)

        state = state.reshape(1, self.state_num)

        return state
    def shut_down(self):
        rospy.on_shutdown(self.myhook)
    def myhook(self):
        print("shutdown time!")
    
if __name__ == '__main__':
    try:
        sess = tensorflow.Session()
        K.set_session(sess)

        game_state = GameState()
        game_state.reset()
        #for i in range(10):
        #print(game_state.position[game_state.robot_name[0]])
        #game_state.game_step(game_state.robot_name[0],time_step=0.1,linear_x=0,angular_z=0)
        #print(game_state.position[game_state.robot_name[0]])
        #game_state.game_step(game_state.robot_name[0],time_step=0.01,linear_x=0.0,angular_z=0)
    except rospy.ROSInterruptException:
        pass



