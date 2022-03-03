#! /usr/bin/env python
import tensorflow as tf
import keras.backend as K
import rospy
import turtlebot3_three_simulator
import turtlebot3_one_simulator
import turtlebot3_four_simulator
import ddpg_network_per_replay_human
import time



sess = tf.Session()
K.set_session(sess)




num_trials = 1
trial_len  = 2000
train_indicator = 0

robot_name = ['tb3_0','tb3_1','tb3_2','tb3_3']



complete_num = 0
for i in range(num_trials):
    game_state= turtlebot3_three_simulator.GameState()   # game_state has frame_step(action) function
    actor_critic = ddpg_network_per_replay_human.ActorCritic(game_state, sess)
    actor_critic.actor_model.load_weights("run1-7/human_per_actormodel-360-400.h5")
    actor_critic.critic_model.load_weights("run1-7/human_per_criticmodel-360-400.h5")
    print("trial:" + str(i))
    start_time =time.time()
    current_state = game_state.reset()
    total_reward = 0
    for j in range(trial_len):
        if game_state.done == True:
            game_state.done = False
            game_state.shut_down()
            rospy.sleep(2)
            complete_num +=1
            break
        for k in range(3):
            current_state[k] = current_state[k].reshape((1, game_state.observation_space.shape[0]))
            action = actor_critic.play(current_state[k])  # need to change the network input output, do I need to change the output to be [0, 2*pi]
            action = action.reshape((1, game_state.action_space.shape[0]))
            if game_state.laser_crashed_value[robot_name[k]] == True:
                game_state.game_step(robot_name[k],0.1, 0, 0)
            else:
                new_state = game_state.game_step(robot_name[k],0.1, action[0][1], action[0][0]) # we get reward and state here, then we need to calculate if it is crashed! for 'dones' value

            new_state = new_state.reshape((1, game_state.observation_space.shape[0]))
            current_state[k] = new_state
    end_time = time.time()
    print("total:",end_time-start_time)
