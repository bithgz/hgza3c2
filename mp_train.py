#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 16:02:03 2019

@author: hu
"""
import copy
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import logging
import numpy as np
import multiprocessing as mp
import tensorflow as tf
# from a3cnet import AC_Network
import a3cnet
import random
from Helper import *



from time import sleep

from GymPursuit import GymPursuit


number_of_agents = 4
amount_of_agents_to_send_message_to = number_of_agents - 1

env = GymPursuit(number_of_agents = number_of_agents)
state_size = env.agent_observation_space
s_size_central = env.central_observation_space
action_size = env.agent_action_space
env.close()

comm_size = 15


gamma = 0.95
learning_rate = 0.0001



NUM_AGENTS = 6


max_episode_length = 100  # take as a train batch##############################


TRAIN_EPOCH = 500000
MODEL_SAVE_INTERVAL = 100

RANDOM_SEED = 42
# RAND_RANGE = 1000


comm_gaussian_noise = 0
comm_delivery_failure_chance = 0
comm_jumble_chance = 0

spread_rewards = True
spread_messages = False

SUMMARY_DIR = './results'
MODEL_DIR = './models'
TRAIN_TRACES = './cooked_traces/'
# NN_MODEL = './results/nn_model_ep_10800.ckpt'
NN_MODEL = None


batch_size = 20
'''
episode_buffer[i].append([previous_screen[i],  0
                          previous_screen_central[i],  1
                          previous_sent_message[i],  2
                          previous_recv_comm[i],  3
                          value[i],  4
                          actions[i],  5
                          reward[i] if spread_rewards else reward,  6
                          current_screen[i],  7
                          current_screen_central[i],  8
                          curr_sent_message[i],  9
                          curr_recv_comm[i],  10
                          curr_value[i],  11
                          terminal]  12)
'''
def train_weights_and_get_comm_gradients(rollout, sess, gamma, ac_network, bootstrap_value=0):
    rollout = np.array(rollout)
    observations = np.stack(rollout[:, 0])
    screen_central = np.stack(rollout[:, 1])
    mess_sent = np.stack(rollout[:, 2])
    mess_received = np.stack(rollout[:, 3])  
    values = rollout[:, 4]
    actions = rollout[:, 5]
    rewards = rollout[:, 6]
    bootstrap_value = rollout[-1, -2]
    
    
    rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value]) #[-10,-5,1]
    discounted_rewards = discount(rewards_plus, gamma)[:-1] #[-13.69,-4.1,1]   [-13.69,-4.1]
    value_plus = np.asarray(values.tolist() + bootstrap_value.tolist())#[3,2,1]
    epsilon = 0

    if epsilon == 0:
        advantages = adv(discounted_rewards, value_plus)#[-16.69,-6]
    else:
        advantages = gae(gamma, epsilon, rewards, value_plus)
        advantages = discount(advantages, gamma)

 
    v_l, p_l, grads_m, e_l, g_n, v_n, _ = sess.run([ac_network.value_loss,
                                                    ac_network.policy_loss,
                                                    ac_network.gradients_q_message,                                                    
                                                    ac_network.entropy,
                                                    ac_network.grad_norms,
                                                    ac_network.var_norms,
                                                    ac_network.apply_grads],
                                    feed_dict = {ac_network.target_v: discounted_rewards,
                                                 ac_network.inputs_state: observations,
                                                 ac_network.inputs_central: screen_central,
                                                 ac_network.receiv_message: mess_received,
                                                 ac_network.actions: actions,
                                                 ac_network.advantages: advantages})  
    
    return observations, mess_received, mess_sent, grads_m[0]




def apply_comm_gradients(observations, mess_received, message_sent, message_loss, sess, ac_network):
  
    target_message = message_sent - message_loss

    feed_dict = {ac_network.target_message: target_message,
                 ac_network.inputs_state: observations,
                 ac_network.receiv_message: mess_received}
    v_l_m, _ = sess.run([ac_network.loss_m, ac_network.apply_grads_m], feed_dict=feed_dict)

    return v_l_m



def input_mloss_to_output_mloss(batch_size, mgrad_per_received, comm_map):


    mgrad_per_sent = [[[0 for _ in range(comm_size)] for _ in range(batch_size)] for _ in
                      range(number_of_agents)]
    mgrad_per_sent_mean_counter = [[0 for _ in range(batch_size)] for _ in
                                   range(number_of_agents)]
    for j in range(number_of_agents):
        for t in range(batch_size):
            for index, neighbor in enumerate(comm_map[j][t]):
                if neighbor != -1:
                    for m in range(comm_size):
                        mgrad_per_sent[neighbor][t][m] = (mgrad_per_sent_mean_counter[neighbor][t] *
                                                          mgrad_per_sent[neighbor][t][m] +
                                                          mgrad_per_received[j][t][
                                                              index * comm_size + m]) / (
                                                             mgrad_per_sent_mean_counter[neighbor][t] + 1)
                    mgrad_per_sent_mean_counter[neighbor][t] += 1

    return mgrad_per_sent




def output_mess_to_input_mess(message, comm_map):
    curr_comm = []
    no_mess = np.ones(comm_size) * 0

    for j, agent_state in enumerate(comm_map):
        curr_agent_comm = []
        for neighbor in agent_state:
            if neighbor != -1:
                curr_agent_comm.extend(message[neighbor])
            else:
                curr_agent_comm.extend(no_mess)
        curr_comm.append(curr_agent_comm)

  
    return curr_comm




def central_agent(net_params_queues, trained_params_queues):

    assert len(net_params_queues) == NUM_AGENTS
    assert len(trained_params_queues) == NUM_AGENTS
       

    
    with tf.Session() as sess, open(SUMMARY_DIR + '/log_central', 'w') as log_file:


        trainer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        
        
        a3c2net = a3cnet.AC_Network(sess,
                                 s_size = state_size, 
                                 s_size_central = s_size_central,
                                 a_size = action_size,
                                 comm_size_input = comm_size * (number_of_agents - 1),
                                 comm_size_output = comm_size, 
                                 trainer = trainer)
        
        summary_ops, summary_vars = a3cnet.build_summaries()

        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)  # training monitor
        
        trained_net_params = [None for _ in range(NUM_AGENTS)]
        
        actor_critic_net_params = a3c2net.get_network_params()
        
        for i in range(NUM_AGENTS):
            net_params_queues[i].put(actor_critic_net_params)

        
        for ep in range(TRAIN_EPOCH):
              
            count = 1
            temp_mean_net_params = trained_params_queues[0].get()
            mean_net_params = copy.deepcopy(temp_mean_net_params)
            for j in range(1, NUM_AGENTS):
                trained_net_params[j] = trained_params_queues[j].get()
                for k, element in enumerate(trained_net_params[j]):
                    temp_mean_net_params[k] = (count * mean_net_params[k] + element)/(count + 1)
                    mean_net_params[k] = copy.deepcopy(temp_mean_net_params[k])
                count += 1

            for i in range(NUM_AGENTS):
                net_params_queues[i].put(mean_net_params)         
            

            # log training information
            avg_reward = 0
            avg_td_loss = 0

            log_file.write('Epoch: ' + str(ep) +
                         ' TD_loss: ' + str(avg_td_loss) +
                         ' Avg_reward: ' + str(avg_reward) + '\n')
            log_file.flush()

            summary_str = sess.run(summary_ops, feed_dict={
                summary_vars[0]: avg_td_loss,
                summary_vars[1]: avg_reward
            })

            writer.add_summary(summary_str, ep)
            writer.flush()






def agent(agent_id, net_params_queue, trained_params_queue):

    env = GymPursuit(number_of_agents = number_of_agents)

    with tf.Session() as sess, open(SUMMARY_DIR + '/log_agent_' + str(agent_id), 'w') as log_file:
        
        trainer = tf.train.AdamOptimizer(learning_rate = learning_rate)

        
        a3c2net = a3cnet.AC_Network(sess,
                                 s_size = state_size, 
                                 s_size_central = s_size_central,
                                 a_size = action_size,
                                 comm_size_input = comm_size * (number_of_agents - 1),
                                 comm_size_output = comm_size, 
                                 trainer = trainer)
        
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        summary_ops, summary_vars = a3cnet.build_summaries()
        
        
        
        actor_critic_net_params = net_params_queue.get()
        a3c2net.set_network_params(actor_critic_net_params)


        writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)  # training monitor

        # time_stamp = 0
        for ep in range(TRAIN_EPOCH): 
            episode_buffer = [[] for _ in range(number_of_agents)]
            episode_comm_maps = [[] for _ in range(number_of_agents)]
            # episode_values = [[] for _ in range(number_of_agents)]
            episode_reward = 0
            episode_step_count = 0
            action_indexes = list(range(env.max_actions))

            v_l, p_l, e_l, g_n, v_n = get_empty_loss_arrays(number_of_agents)
            partial_obs = [None for _ in range(number_of_agents)]
            partial_mess_rec = [None for _ in range(number_of_agents)]
            partial_sent_message = [None for _ in range(number_of_agents)]
            mgrad_per_received = [None for _ in range(number_of_agents)]
            mgrad_per_sent = [None for _ in range(number_of_agents)]



            # start new epi
            current_screen, info = env.reset()
            current_screen_central = info["state_central"]
            
            
            '''
            for i in range(number_of_agents):
                comm_map = list(range(number_of_agents))
                comm_map.remove(i)
                episode_comm_maps[i].append(comm_map)
            '''


            curr_recv_comm = [[] for _ in range(number_of_agents)]
            for curr_agent in range(number_of_agents):
                for from_agent in range(amount_of_agents_to_send_message_to):
                    curr_recv_comm[curr_agent].extend([0] * comm_size)  


            for episode_step_count in range(max_episode_length):
                
                # feedforward pass
                curr_sent_message = sess.run(a3c2net.sent_message,
                                   feed_dict={a3c2net.inputs_state: current_screen})

                # message gauss noise
                if comm_gaussian_noise != 0:
                    for index in range(len(curr_sent_message)):
                        curr_sent_message[index] += np.random.normal(0, comm_gaussian_noise)

                
                this_turns_comm_map = []
                for i in range(number_of_agents):
                    surviving_comms = list(range(number_of_agents))
                    surviving_comms.remove(i)
                    for index in range(len(surviving_comms)):
                        if random.random() < comm_delivery_failure_chance:
                            surviving_comms[index] = -1
                    episode_comm_maps[i].append(surviving_comms)
                    this_turns_comm_map.append(surviving_comms)
                
                    
                curr_recv_comm = output_mess_to_input_mess(curr_sent_message, this_turns_comm_map)


                action_distribution, value = sess.run([a3c2net.policy, a3c2net.value],
                                     feed_dict={a3c2net.inputs_state: current_screen, 
                                                a3c2net.inputs_central: current_screen_central,
                                                a3c2net.receiv_message: curr_recv_comm})

                actions = [np.random.choice(action_indexes, p=act_distribution)
                           for act_distribution in action_distribution]

                previous_screen = current_screen
                previous_screen_central = current_screen_central
                previous_recv_comm = curr_recv_comm
                previous_sent_message = curr_sent_message

                current_screen, reward, terminal, info = env.step(actions)
                current_screen_central = info["state_central"]
                         
                episode_reward += sum(reward) if spread_rewards else reward

                curr_sent_message = sess.run(a3c2net.sent_message,
                                   feed_dict={a3c2net.inputs_state: current_screen})

                curr_recv_comm = output_mess_to_input_mess(curr_sent_message, this_turns_comm_map)
                curr_value = sess.run(a3c2net.value,
                                 feed_dict={a3c2net.inputs_state: current_screen, 
                                            a3c2net.inputs_central: current_screen_central,
                                            a3c2net.receiv_message: curr_recv_comm})

                
                for i in range(number_of_agents):
                    episode_buffer[i].append([previous_screen[i],
                                              previous_screen_central[i],
                                              previous_sent_message[i],
                                              previous_recv_comm[i],
                                              value[i],
                                              actions[i],
                                              reward[i] if spread_rewards else reward,
                                              current_screen[i],
                                              current_screen_central[i],
                                              curr_sent_message[i],
                                              curr_recv_comm[i],
                                              curr_value[i],
                                              terminal])
        
        
                if len(episode_buffer[0]) == batch_size and not terminal and \
                                episode_step_count < max_episode_length - 1:
                                    
                    for i in range(number_of_agents):
                        partial_obs[i], partial_mess_rec[i], partial_sent_message[i], mgrad_per_received[i] = \
                            train_weights_and_get_comm_gradients(episode_buffer[i], sess, gamma, a3c2net)
#def train_weights_and_get_comm_gradients(rollout, sess, gamma, ac_network, bootstrap_value=0):
#    return observations, mess_received, mess_sent, grads_m[0]
                    if comm_size != 0:
                        mgrad_per_sent = input_mloss_to_output_mloss(batch_size, 
                                                                     mgrad_per_received, 
                                                                     episode_comm_maps)

                        
                        for i in range(number_of_agents):
                            apply_comm_gradients(partial_obs[i], partial_mess_rec[i],
                                                      partial_sent_message[i], mgrad_per_sent[i], 
                                                      sess, a3c2net)

                        episode_comm_maps = [[] for _ in range(number_of_agents)]     
                        episode_buffer = [[] for _ in range(number_of_agents)]
                
                if terminal:
                    break
            


            if  episode_step_count != 0 :  
                if  agent_id == 0 and ep % MODEL_SAVE_INTERVAL == 0:
                    print('epoch' + str(ep) + '        reward' + 
                          str(episode_reward) + '        step' + 
                          str(episode_step_count))
        
                log_file.write('epoch' + str(ep) + 'reward' + 
                               str(episode_reward) + 'step' + 
                               str(episode_step_count))
           
                for i in range(number_of_agents):
                    partial_obs[i], partial_mess_rec[i], partial_sent_message[i], mgrad_per_received[i] = \
                        train_weights_and_get_comm_gradients(episode_buffer[i], sess, gamma, a3c2net)

                if comm_size != 0:
                    mgrad_per_sent = input_mloss_to_output_mloss(len(partial_sent_message[0]), 
                                                                 mgrad_per_received, 
                                                                 episode_comm_maps)

                    for i in range(number_of_agents):
                        apply_comm_gradients(partial_obs[i], partial_mess_rec[i],
                                             partial_sent_message[i], mgrad_per_sent[i], 
                                             sess, a3c2net)
                      
                episode_buffer = [[] for _ in range(number_of_agents)]
                
                trained_net_params = a3c2net.get_network_params()
                trained_params_queue.put(trained_net_params)
 
                # log training information
                avg_reward = 0
                avg_td_loss = 0
    
                log_file.write(' TD_loss: ' + str(avg_td_loss) +
                             ' Avg_reward: ' + str(avg_reward) + '\n')
                log_file.flush()
    
                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: avg_td_loss,
                    summary_vars[1]: avg_reward
                })
    
                writer.add_summary(summary_str, ep)
                writer.flush()  
                
                # restore neural net parameters
                nn_model = NN_MODEL
    
                if nn_model is not None:  # nn_model is the path to file
                    saver.restore(sess, nn_model)
                    print("Model restored.")
                   
                if  agent_id == 0 and ep % MODEL_SAVE_INTERVAL == 0:
                    
                    saver.save(sess, MODEL_DIR + "/nn_model_ep_" +
                                            str(ep) + ".ckpt")
    
                actor_critic_net_params = net_params_queue.get()
                a3c2net.set_network_params(actor_critic_net_params)
    env.close()
            
                   



def main():

    np.random.seed(RANDOM_SEED)

    # inter-process communication queues
    net_params_queues = []
    trained_params_queues = []
    
    for i in range(NUM_AGENTS):
        net_params_queues.append(mp.Queue(1))
        trained_params_queues.append(mp.Queue(1))

    # create a coordinator and multiple agent processes
    # (note: threading is not desirable due to python GIL)
    coordinator = mp.Process(target = central_agent,
                             args = (net_params_queues, trained_params_queues))
    coordinator.start()

    agents = []
    for i in range(NUM_AGENTS):
        agents.append(mp.Process(target = agent,
                                 args=(i,
                                       net_params_queues[i], trained_params_queues[i])))
    for i in range(NUM_AGENTS):
        agents[i].start()

    # wait unit training is done
    coordinator.join()


if __name__ == '__main__':
    main()
