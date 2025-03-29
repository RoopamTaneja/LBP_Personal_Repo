import argparse
import numpy as np
import os
import tensorflow as tf
import time

import maddpg.common.tf_util as U
import log0 as Log
from experiments.env0.data_collection0 import Env
from maddpg.common.summary import Summary
from maddpg.trainer.maddpg import MADDPGAgentTrainer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Hyperparameters
ARGUMENTS = [
    # Environment
    ["--scenario", str, "simple_adversary", "name of the scenario script"],
    ["--max-episode-len", int, 500, "maximum episode length"],
    ["--num-episodes", int, 500, "number of episodes"],
    ["--num-adversaries", int, 0, "number of adversaries(enemy)"],
    ["--good-policy", str, "maddpg", "policy for good agents"],
    ["--adv-policy", str, "maddpg", "policy of adversaries"],

    # Core training parameters
    ["--lr", float, 5e-4, "learning rate for Adam optimizer"],
    ["--decay_rate", float, 0.99995, "learning rate exponential decay"],
    ["--gamma", float, 0.95, "discount factor"],
    ["--batch-size", int, 32, "number of epochs to optimize at the same time"],
    ["--num-units", int, 600, "number of units in the mlp"],

    # Priority Replay Buffer
    ["--alpha", float, 0.5, "priority parameter"],
    ["--beta", float, 0.4, "IS parameter"],
    ["--epsilon", float, 0.5, "a small positive constant"],
    ["--buffer_size", int, 200000, "buffer size for each agent"],

    # N-steps
    ["--N", int, 5, "steps of N-step"],

    # RNN
    ["--rnn_length", int, 0, "time_step in rnn, if ==0, not use rnn; else, use rnn."],
    ["--rnn_cell_size", int, 64, "LSTM-cell output's size"],

    # Checkpointing
    ["--exp-name", str, None, "name of the experiment"],
    ["--save-dir", str, "/policy/", "directory in which training state and model should be saved"],
    ["--save-rate", int, 2, "save model once every time this many episodes are completed"],
    ["--model_to_keep", int, 100, "the number of saved models"],
    ["--load-dir", str, "/home/linc/Desktop/maddpg-final/saved_state.ckpt", "directory in which training state and model are loaded"],

    # Evaluation
    ["--benchmark-iters", int, 100000, "number of iterations run for benchmarking"],
    ["--benchmark-dir", str, "./benchm", "directory where benchmark data is saved"],
    ["--plots-dir", str, "./learning_curves/", "directory where plot data is saved"],

    # Training
    ["--random_seed", int, 0, "random seed"]
]

ACTIONS = [
    ["--restore", "store_true", False],
    ["--display", "store_true", False],
    ["--benchmark", "store_true", False]
]

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    for arg in ARGUMENTS:
        parser.add_argument(arg[0], type=arg[1], default=arg[2], help=arg[3])
    for action in ACTIONS:
        parser.add_argument(action[0], action=action[1], default=action[2])
    return parser.parse_args()

def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    trainer = MADDPGAgentTrainer

    # Adversaries (if any)
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy == 'ddpg')))

    # Regular agents
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy == 'ddpg')))

    return trainers

def train(arglist, log):
    with U.multi_threaded_session() as sess:
        # Create a single environment
        env = Env(log)
        log.log(ARGUMENTS)
        log.log(ACTIONS)

        # Create summary
        summary = Summary(sess, env.log_dir)
        for i in range(env.n):
            summary.add_variable(tf.Variable(0.), 'reward_%d' % i)
            summary.add_variable(tf.Variable(0.), 'loss_%d' % i)
            summary.add_variable(tf.Variable(0.), 'wall_%d' % i)
            summary.add_variable(tf.Variable(0.), 'energy_%d' % i)
            summary.add_variable(tf.Variable(0.), 'gained_info_%d' % i)
        summary.add_variable(tf.Variable(0.), 'buffer_size')
        summary.add_variable(tf.Variable(0.), 'acc_reward')
        summary.add_variable(tf.Variable(0.), 'leftrewards')
        summary.add_variable(tf.Variable(0.), 'efficiency')
        summary.build()

        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)  # 0
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)

        # Initialize all variables
        U.initialize()

        if arglist.restore:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        saver = tf.train.Saver(max_to_keep=arglist.model_to_keep)

        # Initialize tracking variables
        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[]]  # placeholder for benchmarking info
        
        # Initialize environment
        obs_n = env.reset()
        state_step_n = []
        if arglist.rnn_length > 0:
            state_step_i = []
            for _ in range(0, arglist.rnn_length - 1):
                state_step_i.append(obs_n)
            state_step_n = state_step_i
            
        episode_step = 0
        t_start = time.time()
        m_time = t_start

        print('Starting iterations...')
        print('Log_dir:', env.log_dir)
        iteration = 0
        global_total_step = 0
        loss = [0.] * env.n
        model_index = 0
        efficiency = 0
        indicator = [0] * env.n
        meaningful_fill = [0] * env.n
        meaningful_get = [0] * env.n

        # Training loop
        while iteration < arglist.num_episodes:
            global_total_step += 1
            
            # Get actions for each agent
            if arglist.rnn_length > 0:
                action_n = []
                state_step_n.append(obs_n)
                for i, agent, obs in zip(range(0, len(trainers)), trainers, obs_n):
                    obs_sequence = []
                    for j in range(-1 * arglist.rnn_length, 0, 1):
                        obs_sequence.append(state_step_n[j][i])
                    action_n.append(agent.action(np.array(obs_sequence)))
            else:
                action_n = [agent.action(obs[None]) for agent, obs in zip(trainers, obs_n)]

            # Environment step
            new_obs_n, rew_n, done_n, info_n, indicator = env.step(actions=action_n, indicator=indicator)
            log.step_information(action_n, env, episode_step, iteration, meaningful_fill, meaningful_get, indicator)
            indicator = [0] * env.n
            
            episode_step += 1
            done = done_n
            terminal = (episode_step >= arglist.max_episode_len)

            # Collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n, terminal, 1)
            
            obs_n = new_obs_n

            # Update rewards
            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            # End of episode handling
            if done or terminal:
                # Report episode results
                print('\n%d th episode:\n' % iteration)
                print('\t%d steps, %.2f seconds, wasted %.2f seconds.' % (
                    episode_step, time.time() - m_time, env.time_))
                print('\tobstacle collisions:', env.walls)
                print('\tdata collection:', env.collection / env.totaldata)
                print('\treminding energy:', env.energy)
                efficiency = env.efficiency
                log.draw_path(env, iteration, meaningful_fill, meaningful_get)
                iteration += 1

                # Reset for next episode
                meaningful_fill = [0] * env.n
                meaningful_get = [0] * env.n
                m_time = time.time()
                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([])

            # Display if needed
            if arglist.display:
                env.render()
                continue

            # Update all trainers
            _loss = []
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                _loss.append(agent.update(env, trainers, global_total_step)[0])
            if np.sum(_loss) != 0:
                loss = _loss

            # Update summary visualization
            feed_dict = {}
            for i_summary in range(env.n):
                feed_dict['reward_%d' % i_summary] = rew_n[i_summary]
                feed_dict['loss_%d' % i_summary] = loss[i_summary]
                feed_dict['wall_%d' % i_summary] = env.walls[i_summary] / (float(episode_step) + 1e-4)
                feed_dict['energy_%d' % i_summary] = env.energy[i_summary]
                feed_dict['gained_info_%d' % i_summary] = env.collection[i_summary]
            feed_dict['buffer_size'] = trainers[0].filled_size
            feed_dict['leftrewards'] = env.leftrewards
            feed_dict['acc_reward'] = episode_rewards[-1]
            feed_dict['efficiency'] = efficiency
            summary.run(feed_dict=feed_dict, step=global_total_step)

            # Save model periodically
            if (done or terminal) and (len(episode_rewards) % arglist.save_rate == 0):
                U.save_state(
                    env.log_dir + arglist.save_dir + "/" + str(model_index % arglist.model_to_keep) + ".ckpt",
                    saver=saver)
                model_index += 1
                
                print("------------------------------------------------------------------------------------------")
                print("Steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                    global_total_step, len(episode_rewards) - 1,
                    np.mean(episode_rewards[-arglist.save_rate:]),
                    round(time.time() - t_start, 3)))
                print("------------------------------------------------------------------------------------------")
                
                t_start = time.time()
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

if __name__ == '__main__':
    print('Let\'s train, go! go! go!')
    arglist = parse_args()
    log = Log.Log()
    train(arglist, log)
