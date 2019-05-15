import gym
import tensorflow as tf
import numpy as np
import random

# General Parameters
# -- DO NOT MODIFY --
ENV_NAME = 'CartPole-v0'
EPISODE = 200000  # Episode limitation
STEP = 200  # Step limitation in an episode
TEST = 10  # The number of tests to run every TEST_FREQUENCY episodes
TEST_FREQUENCY = 100  # Num episodes to run before visualizing test accuracy

# TODO: HyperParameters
GAMMA = 1 # discount factor
INITIAL_EPSILON =  0.1 # starting value of epsilon
FINAL_EPSILON = 0.1 # final value of epsilon
EPSILON_DECAY_STEPS = 100 # decay period
HIDDEN_NODES = [24,48]
global replay_buffer
replay_buffer = []
BATCH_SIZE = 256
REPLAY_SIZE = 100000
LR = 0.01

# Create environment
# -- DO NOT MODIFY --
env = gym.make(ENV_NAME)
epsilon = INITIAL_EPSILON
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n
hidden_nodes=HIDDEN_NODES

# Placeholders
# -- DO NOT MODIFY --
state_in = tf.placeholder("float", [None, STATE_DIM])
action_in = tf.placeholder("float", [None, ACTION_DIM])
target_in = tf.placeholder("float", [None])

# TODO: Define Network Graph

c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
w_initializer = tf.random_normal_initializer(0., 0.01)
b_initializer = tf.constant_initializer(0.01) 
H1, H2 = HIDDEN_NODES[0], HIDDEN_NODES[1]
l = tf.contrib.layers.fully_connected(state_in, H1, activation_fn=tf.nn.tanh)
l = tf.contrib.layers.fully_connected(l, H2, activation_fn=tf.nn.tanh)
l_final = tf.contrib.layers.fully_connected(l, ACTION_DIM, activation_fn=None)
q_values = l_final
q_action = tf.reduce_sum(tf.multiply(q_values, action_in), reduction_indices=1)
loss = tf.reduce_mean(tf.square(target_in - q_action))
optimizer = tf.train.AdamOptimizer(LR).minimize(loss)
train_loss_summary_op = tf.summary.scalar("TrainingLoss", loss)


# TODO: Network outputs
# got this q_values =
# got this q_action =

# TODO: Loss/Optimizer Definition
# got this loss =
# got this optimizer =

# Start session - Tensorflow housekeeping
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())


# -- DO NOT MODIFY ---
def explore(state, epsilon):
    """
    Exploration function: given a state and an epsilon value,
    and assuming the network has already been defined, decide which action to
    take using e-greedy exploration based on the current q-value estimates.
    """
    Q_estimates = q_values.eval(feed_dict={
        state_in: [state]
    })
    if random.random() <= epsilon:
        action = random.randint(0, ACTION_DIM - 1)
    else:
        action = np.argmax(Q_estimates)
    one_hot_action = np.zeros(ACTION_DIM)
    one_hot_action[action] = 1
    return one_hot_action

batch_presentation_count = total_steps = total_reward = 0
consecutive_rewards = []
best_record = 0
best_ep = 0
# Main learning loop
for episode in range(EPISODE):

    # initialize task
    state = env.reset()

    # Update epsilon once per episode
    epsilon -= (epsilon-FINAL_EPSILON) / EPSILON_DECAY_STEPS
    ep_reward = 0
    # Move through env according to e-greedy policy
    for step in range(STEP):
        total_steps += 1
        action = explore(state, epsilon)
        next_state, reward, done, _ = env.step(np.argmax(action))
        ep_reward += reward
##        nextstate_q_values = q_values.eval(feed_dict={
##            state_in: [next_state]
##        })

        replay_buffer.append([state, action, reward, next_state, done])
        if len(replay_buffer) > REPLAY_SIZE:
            replay_buffer.pop(0)
        state = next_state
        target_batch = []
        if (len(replay_buffer) > BATCH_SIZE):
            minibatch = random.sample(replay_buffer, BATCH_SIZE)
            state_batch = [data[0] for data in minibatch]
            action_batch = [data[1] for data in minibatch]
            reward_batch = [data[2] for data in minibatch]
            next_state_batch = [data[3] for data in minibatch]

            
            Q_value_batch = q_values.eval(feed_dict={ state_in: next_state_batch })
            for i in range(0, BATCH_SIZE):
                sample_is_done = minibatch[i][4]
                if sample_is_done:
                    target_batch.append(reward_batch[i])
                else:
                    target_val = reward_batch[i] + GAMMA * np.max(Q_value_batch[i])
                    target_batch.append(target_val)
            
        
        # TODO: Calculate the target q-value.
        # hint1: Bellman
        # hint2: consider if the episode has terminated
            target = target_batch

        # Do one training step
            session.run([train_loss_summary_op,optimizer], feed_dict={
                target_in: target,
                action_in: action_batch,
                state_in: state_batch
            })

        # Update
        #state = next_state
        if done:
            break
    total_reward += ep_reward


##    if len(consecutive_rewards) < 100:
##        consecutive_rewards.append(ep_reward)
##    else:
##
##        consecutive_rewards.pop(0)
##        consecutive_rewards.append(ep_reward)
##        curr = np.mean(consecutive_rewards)
##        if best_record < curr:
##            best_record = curr
##            best_ep = episode
##        if episode % 25 == 0:
##            print("Current ep : {}, last 100 avg ep_reward : {}".format(episode, curr))
##            print("Till now best record : {}, at episode {}".format(best_record, best_ep))
##            print("=====")
##        if curr > 195:
##            print("###   Solved at ep {}, congratulations!   ###".format(episode))
##            exit()





    

    # Test and view sample runs - can disable render to save time
    # -- DO NOT MODIFY --
    if (episode % TEST_FREQUENCY == 0 and episode != 0):
        total_reward = 0
        for i in range(TEST):
            state = env.reset()
            for j in range(STEP):
                env.render()
                action = np.argmax(q_values.eval(feed_dict={
                    state_in: [state]
                }))
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
        ave_reward = total_reward / TEST
        print('episode:', episode, 'epsilon:', epsilon, 'Evaluation '
                                                        'Average Reward:', ave_reward)

env.close()
