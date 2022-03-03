#! /usr/bin/env python
import tensorflow as tf
import keras.backend as K
import turtlebot3_multi_simulator
import ddpg_network_per_replay_human
import time



sess = tf.Session()
K.set_session(sess)


game_state= turtlebot3_multi_simulator.GameState()   # game_state has frame_step(action) function
actor_critic = ddpg_network_per_replay_human.ActorCritic(game_state, sess)

num_trials = 10000
trial_len  = 200
train_indicator = 0

robot_name = ['tb3_0','tb3_1','tb3_2']

actor_critic.actor_model.load_weights("run1-6/human_per_actormodel-300-500.h5")
actor_critic.critic_model.load_weights("run1-6/human_per_criticmodel-300-500.h5")

for i in range(num_trials):
    print("trial:" + str(i))
    current_state = game_state.reset()

    total_reward = 0
    crashed_value_list = {'tb3_0':False,'tb3_1':False,'tb3_2':False}
    for j in range(trial_len):
        for k in range(len(robot_name)):
            current_state[k] = current_state[k].reshape((1, game_state.observation_space.shape[0]))
            start_time = time.time()
            action = actor_critic.play(current_state[k])  # need to change the network input output, do I need to change the output to be [0, 2*pi]
            action = action.reshape((1, game_state.action_space.shape[0]))
            end_time = time.time()
            if game_state.laser_crashed_value[robot_name[k]] == True:
                game_state.game_step(robot_name[k],0.1, 0, 0)
            else:
                new_state = game_state.game_step(robot_name[k],0.1, action[0][1], action[0][0]) # we get reward and state here, then we need to calculate if it is crashed! for 'dones' value
            new_state = new_state.reshape((1, game_state.observation_space.shape[0]))
            current_state[k] = new_state

