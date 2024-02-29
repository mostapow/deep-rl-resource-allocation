import random
import os
import numpy as np
import math
import time
import shutil
import tensorflow as tf
from tensorflow import keras
# import json
# import matplotlib.pyplot as plt
from datetime import datetime
from statistics import mean, median
from simprocess.simprocess_variant import Simulation
import mlflow.keras
from mlflow import log_metric, log_param, log_artifacts

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = np.empty((buffer_size,), dtype=np.object)
        self.buffer_size = buffer_size
        self.idx = 0
        self.size = 0

    def append(self, experience):
        self.buffer[self.idx] = experience
        self.idx = (self.idx + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, b_size):
        indices = random.choices(range(self.size), k=b_size)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards, dtype='float32'), np.array(next_states)

random.seed()
np.random.seed()


nmb_of_train_episodes = 100
nmb_of_test_episodes = 100
nmb_of_iterations_per_episode = 93000
nmb_of_episodes_before_training = 10
# memory_length = 50000
memory_length = int(nmb_of_train_episodes * nmb_of_iterations_per_episode)
#replay_memory = deque(maxlen=memory_length)


state_repr = "a0"
last_5000_samples_ns = np.empty((0, 26), float)
last_5000_samples_s = np.empty((0, 26), float)
normalization_step_counter = 0
mean_ns = 0
sd_ns = 0
mean_s = 0
sd_s = 0
initializer = tf.keras.initializers.GlorotUniform()
initializer_name = initializer._keras_api_names[0]
batch_size = 2048
discount_rate = 0.99
learning_rate = 0.001
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
loss_fn = keras.losses.mean_squared_error
layer_1_size = 256
layer_2_size = 256
epsilon_type = 'classic'  # classic or exp
A = 0.5
B = 0.3
C = 0.1



replay_memory = ReplayBuffer(memory_length)

env = Simulation([0, 1, 2], ["./receipt.xes"], 0, 100, state_repr)
eval_env = Simulation([0, 1, 2], ["./receipt.xes"], 0, 100, state_repr)


action_space = env.action_space

nmb_of_inputs = action_space[0] + action_space[1]
#nmb_of_outputs = action_space[0] * action_space[1] + 1
nmb_of_outputs = len(env.nn_eligibilites) + 1
useful_actions = [0, 0, 0]
first_visit_viable = [0, 0, 0]
eligibility_filename = "longer_process_log_sx.csv"

model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(nmb_of_inputs,)),
        keras.layers.Dense(layer_1_size, kernel_initializer=initializer),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.Dense(layer_2_size, kernel_initializer=initializer),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.Dense(nmb_of_outputs)
    ])

target_model = keras.models.clone_model(model)

q_values_mean = 0
q_values_counter = 0.001

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")


def epsilon_exp(step):
    standardized_time = ((step-25) - A * 50) / (B * 50)
    cosh = np.cosh(math.exp(-standardized_time))
    e = 1.1 - (1 / cosh + ((step-25) * C / 50))
    return e

def epsilon_new(step):
    return -4.75 * step + 337.5
    #return -0.011875 * step + 1
    # if step > 50:
    #     return 0
    # else:
    #return -0.0098 * step + 0.99

def standarize(samples, mean, sd):
    bc_arr, bc_row_means = np.broadcast_arrays(samples, mean)
    bc_arr, bc_row_sd = np.broadcast_arrays(samples, sd)
    bc_arr = samples - bc_row_means
    bc_arr = bc_arr / bc_row_sd
    bc_arr = np.nan_to_num(bc_arr, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    return bc_arr

def training_step(training_batch_size):
    global q_values_mean
    global q_values_counter
    experiences = sample_experiences(training_batch_size)
    states, actions, rewards, next_states = experiences
    next_Q_values = model(next_states)
    best_next_actions = np.argmax(next_Q_values, axis=1)
    next_mask = tf.one_hot(best_next_actions, nmb_of_outputs)
    next_best_Q_values = (target_model(next_states).numpy() * next_mask.numpy()).sum(axis=1)
    target_Q_values = (rewards + discount_rate * next_best_Q_values)
    target_Q_values = target_Q_values.reshape(-1, 1)
    q_values_mean += np.mean(target_Q_values)
    q_values_counter += 1
    mask = tf.one_hot(actions, nmb_of_outputs)
    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, np.mean(target_Q_values)


@tf.function
def training_step_tf(model, target_model, experiences):
    states, actions, rewards, next_states = experiences # , indices, weights
    next_Q_values = model(next_states)
    best_next_actions = tf.argmax(next_Q_values, axis=1)
    next_mask = tf.one_hot(best_next_actions, nmb_of_outputs)
    next_best_Q_values = tf.reduce_sum(target_model(next_states) * next_mask, axis=1)
    target_Q_values = (rewards + discount_rate * next_best_Q_values)
    target_Q_values = tf.reshape(target_Q_values, (-1, 1))
    mask = tf.one_hot(actions, nmb_of_outputs)
    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values)) # weights *
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    q_values_mean = tf.reduce_mean(target_Q_values)
    return loss, q_values_mean

@tf.function
def training_step_dqn(model, target_model, experiences):
    states, actions, rewards, next_states = experiences
    next_Q_values = model(next_states)
    max_next_Q_values = tf.argmax(next_Q_values, axis=1)
    target_Q_values = (rewards + discount_rate * max_next_Q_values)
    target_Q_values = tf.reshape(target_Q_values, (-1, 1))
    mask = tf.one_hot(actions, nmb_of_outputs)
    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    q_values_mean = tf.reduce_mean(target_Q_values)
    return loss, q_values_mean


def sample_experiences(batch_size):
    return replay_memory.sample(batch_size)


def epsilon_greedy_policy(prediction_model, environment, state, nmb_of_tasks, epsilon=0, type_of_epsilon_exp="random"):
    """

    :param prediction_model:
    :param environment:
    :param state:
    :param nmb_of_tasks:
    :param epsilon:
    :param type_of_epsilon_exp: IMPORTANT - using anything besides random here may result in learning policy that is
                dependent on particular (in this case fifo) strategy present with epsilon probability
    :return:
    """
    if np.random.rand() < epsilon:
        if type_of_epsilon_exp == "random":
            action = np.random.randint(nmb_of_outputs)
            sim_action = environment.get_action_from_int(action)
            next_state, reward = env_state_mask(environment.step(sim_action))
            return next_state, reward, action, "r"
        elif type_of_epsilon_exp == "fifo":
            next_state, reward, action = env_state_mask(environment.step_fifo())
            if action[1] != -1:
                action[1] = environment.id_for_task[action[1]]
            return next_state, reward, get_int_from_sim_action(action), "r"
    else:
        Q_values = prediction_model(state[np.newaxis], training=False)
        global q_values_mean
        global q_values_counter
        global useful_actions
        global first_visit_viable
        # q_values_mean += np.mean(Q_values[0])
        # q_values_counter += 1
        action = np.argmax(Q_values[0])
        sim_action = environment.get_action_from_int(action)
        next_state, reward = env_state_mask(environment.step(sim_action))
        return next_state, reward, action, "q"


def get_action_from_int(int_action_value, nmb_of_tasks):
    """
    :param nmb_of_tasks: number of tasks
    :param nmb_of_resources: number of resources
    :param int_action_value: action value from the range NxM where N - nmb_of_resources, M - nmb_of_tasks
    :return: [resource, task] Vector of length 2 of actions to be taken by the environment
    """
    if int_action_value == 0:
        return [-1, -1]
    else:
        action_coded_value = int_action_value - 1
        resource = math.floor(action_coded_value / nmb_of_tasks)
        task = action_coded_value % nmb_of_tasks
        return [resource, task]


def get_int_from_sim_action(sim_action):
    return sim_action[0]*sim_action[1] + sim_action[1] + 1


def play_one_step_and_collect_memory(env, state, epsilon, type_of_epsilon_exp="random"):

    next_state, reward, action, action_type = epsilon_greedy_policy(model, env, state, action_space[1],
                                                         epsilon, type_of_epsilon_exp)
    replay_memory.append((state, action, reward, next_state))
    return next_state, reward, action, action_type



def env_state_mask(env_step_return_tuple):
    # loc_list = list(env_step_return_tuple)
    # loc_list[0] = loc_list[0][0:action_space[0], ]
    # x = tuple(loc_list)
    return env_step_return_tuple


def main():
    global model
    global target_model
    train_rewards = []
    type_of_epsilon_exp = "random"
    # type_of_epsilon_exp = "fifo"
    # 1 - epsilon decreases to 0.1 over nmb_of_train_episodes
    # 0.5 - epsilon decreases to 0.1 over nmb_of_train_episodes * 0.5 etc.
    epsilon_decreasing_factor = 0.1
    eval_episodes = 2
    eval_steps = 500
    #nmb_of_episodes_for_e_to_anneal = int(nmb_of_train_episodes * epsilon_decreasing_factor)
    nmb_of_episodes_for_e_to_anneal = 100
    target_update_steps = 10000

    start_time_all = time.time()

    print("Action-space: {}".format(env.action_space))
    print("Number of inputs: {} and outputs: {}".format(nmb_of_inputs, nmb_of_outputs))

    model.summary()


    best_eval_weights = model.get_weights()
    best_eval_completed = 0
    best_episode_reward_sum = 0
    best_completed_counter = 0
    best_weights = model.get_weights()
    best_completed_weights = model.get_weights()
    mlflow.set_experiment("paper")
    with mlflow.start_run():

        log_param('state_repr', state_repr)
        log_param('C', env.C)
        log_param('nmb_of_train_episodes', nmb_of_train_episodes)
        log_param('nmb_of_iterations_per_episode', nmb_of_iterations_per_episode)
        log_param('nmb_of_episodes_before_training', nmb_of_episodes_before_training)
        log_param('nmb_of_episodes_for_e_to_anneal', nmb_of_episodes_for_e_to_anneal)
        log_param('target_update_steps',  target_update_steps)
        log_param('memory_length', memory_length)
        log_param('batch_size', batch_size)
        log_param('type_of_epsilon_exp', type_of_epsilon_exp)
        log_param('discount_rate', discount_rate)
        log_param('nmb_of_test_episodes', nmb_of_test_episodes)
        log_param('loss_function', loss_fn.__name__)
        log_param('action_space', env.action_space)
        log_param('process_case_probability', env.process_case_probability)
        log_param('optimizer', optimizer._name)
        log_param('learning_rate', learning_rate)
        log_param('layer_1', layer_1_size)
        log_param('layer_2', layer_2_size)
        log_param('initializer', initializer_name)
        log_param("number of actions", len(env.nn_eligibilites))
        log_param("epsilon_type", epsilon_type)
        for episode_nmb in range(nmb_of_train_episodes):
            # obs = env.reset().to_numpy()[0:action_space[0], ]
            action_exists_counter = 0
            #obs = env.reset().to_numpy()
            #obs = np.asarray(env.soft_reset())
            obs = np.asarray(env.reset())
            train_episode_reward_sum = 0
            train_episode_action_sum = 0
            global q_values_mean
            global q_values_counter
            global normalization_step_counter
            q_values_mean = 0
            q_values_counter = 0.0001
            train_episode_model_actions_count = {}
            train_episode_model_actions_count_random = {}
            start_time = time.time()
            normalization_step_counter = 0
            episode_loss = 0
            training_time_per_episode = 0
            sampling_time_per_episode = 0
            step_time_per_episode = 0
            for step in range(nmb_of_iterations_per_episode):
                if epsilon_type == 'classic':
                    epsilon = max(1 - episode_nmb / (nmb_of_episodes_for_e_to_anneal), 0.05)
                elif epsilon_type == 'exp':
                    epsilon = epsilon_exp(episode_nmb)


                if env.action_exists():
                    action_exists_counter += 1
                    start_time_step = time.time()
                    obs, reward, action, action_type = play_one_step_and_collect_memory(env, obs, epsilon, type_of_epsilon_exp)
                    step_time_per_episode += time.time() - start_time_step
                    if action_type == "q":
                        if action in train_episode_model_actions_count:
                            train_episode_model_actions_count[action] += 1
                        else:
                            train_episode_model_actions_count[action] = 1
                    if action_type == "r":
                        if action in train_episode_model_actions_count_random:
                            train_episode_model_actions_count_random[action] += 1
                        else:
                            train_episode_model_actions_count_random[action] = 1
                    train_episode_reward_sum += reward
                    train_episode_action_sum += action
                else:
                    obs = env.continue_without_action()
                obs = np.asarray(obs)

                #
                if episode_nmb > nmb_of_episodes_before_training and env.action_exists():
                    start_time_training = time.time()
                    experiences = sample_experiences(batch_size)
                    sampling_time_per_episode += time.time() - start_time_training

                    start_time_training = time.time()
                    loss, q_val = training_step_tf(model, target_model, experiences)
                    episode_loss += loss.numpy()
                    q_values_mean += q_val.numpy()
                    q_values_counter += 1
                    training_time_per_episode += time.time() - start_time_training


                if episode_nmb > nmb_of_episodes_before_training and (episode_nmb * nmb_of_iterations_per_episode + step) % target_update_steps == 0:
                    target_model.set_weights(model.get_weights())

                if (episode_nmb * nmb_of_iterations_per_episode + step) % eval_steps == 0 and episode_nmb > nmb_of_episodes_before_training:
                    state = np.asarray(eval_env.reset())
                    eval_episode_reward_sum = 0
                    for _ in range(93000):
                        state, reward, action, action_type = epsilon_greedy_policy(model, eval_env, state,
                                                                                   action_space[1], epsilon=0.05)
                        state = np.asarray(state)
                        eval_episode_reward_sum += reward

                    eval_completed_counter = eval_env.completed_counter[0]
                    eval_completed_counter_X = eval_env.completed_counter[1]
                    log_metric('eval_completed', eval_completed_counter,
                               step=(episode_nmb * nmb_of_iterations_per_episode + step))
                    log_metric('eval_completed_X', eval_completed_counter_X,
                               step=(episode_nmb * nmb_of_iterations_per_episode + step))
                    if best_eval_completed < eval_completed_counter + eval_completed_counter_X:
                        best_eval_completed = eval_completed_counter + eval_completed_counter_X
                        mlflow.keras.log_model(model, "best_eval_model")
                        log_metric('best_eval_completed', best_eval_completed,
                                   step=(episode_nmb * nmb_of_iterations_per_episode + step))

                # else:
                #     env.continue_without_action()

            log_metric("action_exists_counter", action_exists_counter, step=episode_nmb)
            log_metric("training_time", training_time_per_episode, step=episode_nmb)
            log_metric("sampling_time", sampling_time_per_episode, step=episode_nmb)
            log_metric("step_time", step_time_per_episode, step=episode_nmb)

            # if episode_nmb > nmb_of_episodes_before_training:
            #     episode_loss += training_step(batch_size)
            #
            # if episode_nmb > nmb_of_episodes_before_training and episode_nmb % 5 == 0:
            #     target_model.set_weights(model.get_weights())


            # statistics
            train_rewards.append(train_episode_reward_sum)

            log_metric("train_episode_reward_sum", train_episode_reward_sum, step=episode_nmb)
            log_metric("train_loss", episode_loss/nmb_of_iterations_per_episode, step=episode_nmb)

            train_episode_action_average = train_episode_action_sum / nmb_of_iterations_per_episode
            log_metric("train_episode_action_average", train_episode_action_average, step=episode_nmb)
            log_metric("train_completed_counter", env.completed_counter[0] + env.completed_counter[1], step=episode_nmb)
            log_metric('train_avg_rewards', sum(train_rewards) / len(train_rewards), step=episode_nmb)

            if best_completed_counter < env.completed_counter[0] + env.completed_counter[1] and episode_nmb > 99:
                best_completed_weights = model.get_weights()
                best_completed_counter = env.completed_counter[0] + env.completed_counter[1]
                mlflow.keras.log_model(model, "best_train_model")
                log_metric('best_train_episode_completed_counter', best_completed_counter, step=episode_nmb)

            # episode_nmb % eval_episodes == 0
            # if (episode_nmb * nmb_of_iterations_per_episode + step) % eval_steps == 0 and episode_nmb > nmb_of_episodes_before_training:
            #
            #     state = np.asarray(eval_env.reset())
            #     eval_episode_reward_sum = 0
            #     for step in range(nmb_of_iterations_per_episode):
            #         state, reward, action, action_type = epsilon_greedy_policy(model, eval_env, state,
            #                                                                    action_space[1], epsilon=0.05)
            #         #state = state.to_numpy()
            #         state = np.asarray(state)
            #         eval_episode_reward_sum += reward
            #
            #     eval_completed_counter = eval_env.completed_counter[0]
            #     log_metric('eval_completed',  eval_completed_counter, step=episode_nmb * nmb_of_iterations_per_episode + step)
            #     if best_eval_completed < eval_completed_counter:
            #         best_eval_completed = eval_completed_counter
            #         mlflow.keras.log_model(model, "best_eval_model")
            #         log_metric('best_eval_completed', best_eval_completed, step=episode_nmb * nmb_of_iterations_per_episode + step)

            #  and episode_nmb > nmb_of_episodes_for_e_to_anneal and episode_nmb > 99
            if best_episode_reward_sum < train_episode_reward_sum:
                best_weights = model.get_weights()
                best_episode_reward_sum = train_episode_reward_sum
                log_metric('best_train_episode_reward_sum', train_episode_reward_sum, step=episode_nmb)


            log_metric("Q_value_mean", q_values_mean/q_values_counter, step=episode_nmb)
            print("Episode: {}, loss: {}, rewards: {}, q_value_mean: {} actions: {} eps: {:.3f} time: {}".format(episode_nmb,
                                                                                                                 episode_loss/nmb_of_iterations_per_episode,
                                                                                                                 train_episode_reward_sum,
                                                                                                                 q_values_mean/q_values_counter,
                                                                                                                 train_episode_action_average ,
                                                                                                                 epsilon,
                                                                                                                 time.time() - start_time), end="\n")
            print(f"Completed counter: 0: {env.completed_counter[0]}, X: {env.completed_counter[1]}")
            print(f"Model actions: {train_episode_model_actions_count}")
            print(f"Random actions: {train_episode_model_actions_count_random}")

        best_score_model = keras.models.clone_model(model)
        best_score_model.set_weights(best_weights)
        mlflow.keras.log_model(best_score_model, "best_model")
        mlflow.keras.log_model(model, "last_model")

        best_completed_model = keras.models.clone_model(model)
        best_completed_model.set_weights(best_completed_weights)
        mlflow.keras.log_model(best_completed_model, "best_completed_model")

        print("Whole process took: {}", time.time() - start_time_all)


def save_model(best_score_model, last_model):
    path = "results_" + datetime.now().strftime("%Y%m%d_%H_%M_%S_%f")
    os.makedirs(path)
    os.makedirs(path + "/weights")

    best_score_model.save(path + "/best_model")
    last_model.save(path + "/last_model")

    best_score_model.save_weights(path + '/weights/best_weights.h5')
    last_model.save_weights(path + '/weights/last_weights.h5')


    shutil.copytree('./conf', path + "/conf", dirs_exist_ok=True)


def eval_and_create_graph(path, suffix=None):
    global useful_actions
    global first_visit_viable
    global nmb_of_iterations_per_episode
    env = Simulation([0, 1, 2], ["./receipt.xes"], 0, 100, state_repr, True)
    print("Start testow...")
    print("Ladowanie modelu...")
    best_score_model = mlflow.keras.load_model(path)
    steps = nmb_of_iterations_per_episode
    epos = 50


    dqn_metrics = {
                    "completed_process_cases": [],
                    "processing_time_completed": [],
                    "processing_time_current": [],
                    "processing_time_enabled": [],
                    "response_time_completed": [],
                    "response_time_current": [],
                    "response_time_enabled": [],
                    "waiting_time_completed": [],
                    "waiting_time_total": [],
                    "waiting_time_enabled": [],
                    "waiting_time_current": [],
                    "arrived_process_cases": [],
                    "lead_time_completed": [],
                    "lead_time_enabled": [],
                    "lead_time_current": [],
                    "duration_total": [],
                    "resource_utilization": [],
                    "resource_idle": [],
                    "idle_waiting_time": [],
                    "completed_percentage": [],
                    "duration_mean": [],
                    "tasks_lead_time": {},
                    "tasks_resource_utilization": {},
                   }

    for task in env.id_for_task.keys():
        dqn_metrics["tasks_lead_time"][task] = list()
        dqn_metrics["tasks_resource_utilization"][task] = list()

    test_rewards_dqn = []
    test_rewards_fifo = []
    test_rewards_spt = []
    test_rewards_lst_task = []
    test_rewards_lst_process = []
    test_rewards_edf_task = []
    test_rewards_edf_process = []
    results = {}
    completed_dqn = []
    completed_fifo = []
    completed_spt = []
    completed_lst_task = []
    completed_lst_process = []
    completed_edf_task = []
    completed_edf_process = []




    all_finished_length = []
    all_enabled_length = []
    print("[###] DDQN evaluation...")
    mlflow.set_experiment("paper_eval")
    with mlflow.start_run():
        mlflow.keras.log_model(best_score_model, "used_model")
        log_param('nmb_of_train_episodes', epos)
        log_param('nmb_of_iterations_per_episode', steps)

        for e in range(epos):
            print("Epizod " + str(e) + " started...")
            actions = []
            states_no_action = []
            states_action = []
            state = np.asarray(env.reset())
            #state = env.reset().to_numpy()
            useful_actions = [0, 0, 0]
            first_visit_viable = [0, 0, 0]
            for step in range(steps):
                state, reward, action, action_type = epsilon_greedy_policy(best_score_model, env, state,
                                                                           action_space[1], epsilon=0.05)
                state = np.asarray(state)
                # [state, reward, action] = env.step_lst_process_case()
                # state = np.asarray(state)

            # states_set_a = set(map(tuple, states_action))
            # states_set_n = set(map(tuple, states_no_action))
            # print(states_set_a)
            # print(states_set_n)
            print({action: actions.count(action) for action in actions})
            log_metric("dqn_completed_counter", env.completed_counter[0], step=e)
            completed_dqn.append(env.completed_counter[0])

            dqn_metrics["completed_process_cases"].append(env.completed_counter[0])
            dqn_metrics["idle_waiting_time"].append(env.idle_waiting_time)

            dqn_metrics["arrived_process_cases"].append(
                                    len(env.completed_process_cases) +
                                    len(env.enabled_process_cases) +
                                    len(env.current_process_cases))

            dqn_metrics["completed_percentage"].append(env.completed_counter[0] / (len(env.completed_process_cases) +
                                                                                    len(env.enabled_process_cases) +
                                                                                    len(env.current_process_cases)))


            dqn_metrics["waiting_time_completed"].append([case.process_waiting_time for case in env.completed_process_cases])
            dqn_metrics["waiting_time_enabled"].append([case.process_waiting_time for case in env.enabled_process_cases])
            dqn_metrics["waiting_time_current"].append([case.process_waiting_time for case in env.current_process_cases])
            #dqn_metrics["waiting_time_total"].append(waiting_time_current + waiting_time_completed + waiting_time_enabled)

            dqn_metrics["lead_time_completed"].append([case.time_length for case in env.completed_process_cases])
            dqn_metrics["lead_time_enabled"].append([case.time_length for case in env.enabled_process_cases])
            dqn_metrics["lead_time_current"].append([case.time_length for case in env.current_process_cases])

            dqn_metrics["processing_time_completed"].append([case.process_processing_time for case in env.completed_process_cases])
            dqn_metrics["processing_time_enabled"].append([case.process_processing_time for case in env.enabled_process_cases])
            dqn_metrics["processing_time_current"].append([case.process_processing_time for case in env.current_process_cases])

            dqn_metrics["response_time_completed"].append([case.process_response_time for case in env.completed_process_cases])
            dqn_metrics["response_time_enabled"].append([case.process_response_time for case in env.enabled_process_cases])
            dqn_metrics["response_time_current"].append([case.process_response_time for case in env.current_process_cases])
            #dqn_metrics["duration_total"].append(duration_completed + duration_enabled + duration_current)

            temp = []
            temp.extend([case.time_length for case in env.completed_process_cases])
            temp.extend([case.time_length for case in env.enabled_process_cases])
            temp.extend([case.time_length for case in env.current_process_cases])
            dqn_metrics["duration_mean"].append(mean(temp))
            log_metric("completed_avg_dqn", sum(completed_dqn) / len(completed_dqn))

            dqn_metrics["resource_utilization"].append({res: util/steps for res, util in env.resource_utilization.items()})
            dqn_metrics["resource_idle"].append({res: (1 - util/steps) for res, util in env.resource_utilization.items()})
            for task in env.task_duration:
                dqn_metrics["tasks_lead_time"][task].append(env.task_duration[task])
                dqn_metrics["tasks_resource_utilization"][task].append(env.task_resources[task])

            with open("dqn_metrics_" + suffix[0:6] + ".txt", 'w') as f:
                f.write(str(dqn_metrics))

            mlflow.log_artifact("dqn_metrics_" + suffix[0:6] + ".txt", artifact_path="metrics")

            # #test FIFO
            # env.reset()
            # reward_sum = 0
            # for j in range(steps):
            #     [state, reward, action] = env.step_fifo()
            #
            # log_metric("fifo_completed_counter", env.completed_counter[0], step=e)
            #
            # completed_fifo.append(env.completed_counter[0])
            # test_rewards_fifo.append(reward_sum)
            #
            # fifo_metrics["completed_process_cases"].append(env.completed_counter[0])
            # fifo_metrics["idle_waiting_time"].append(env.idle_waiting_time)
            #
            # fifo_metrics["arrived_process_cases"].append(
            #                         len(env.completed_process_cases) +
            #                         len(env.enabled_process_cases) +
            #                         len(env.current_process_cases))
            # fifo_metrics["completed_percentage"].append(env.completed_counter[0]/(len(env.completed_process_cases) +
            #                         len(env.enabled_process_cases) +
            #                         len(env.current_process_cases)))
            #
            # fifo_metrics["waiting_time_completed"].append([case.process_waiting_time for case in env.completed_process_cases])
            # fifo_metrics["waiting_time_enabled"].append([case.process_waiting_time for case in env.enabled_process_cases])
            # fifo_metrics["waiting_time_current"].append([case.process_waiting_time for case in env.current_process_cases])
            # #dqn_metrics["waiting_time_total"].append(waiting_time_current + waiting_time_completed + waiting_time_enabled)
            #
            # fifo_metrics["duration_completed"].append([case.time_length for case in env.completed_process_cases])
            # fifo_metrics["duration_enabled"].append([case.time_length for case in env.enabled_process_cases])
            # fifo_metrics["duration_current"].append([case.time_length for case in env.current_process_cases])
            # temp = []
            # temp.extend([case.time_length for case in env.completed_process_cases])
            # temp.extend([case.time_length for case in env.enabled_process_cases])
            # temp.extend([case.time_length for case in env.current_process_cases])
            # fifo_metrics["duration_mean"].append(mean(temp))
            #
            # #dqn_metrics["duration_total"].append(duration_completed + duration_enabled + duration_current)
            #
            # fifo_metrics["resource_utilization"].append({res: util/steps for res, util in env.resource_utilization.items()})
            # fifo_metrics["resource_idle"].append({res: (1 - util/steps) for res, util in env.resource_utilization.items()})
            # log_metric("completed_avg_fifo", sum(completed_fifo) / len(completed_fifo))

            # #test SPT
            # env.reset()
            # for j in range(steps):
            #     [state, reward, action] = env.step_spt()
            #
            # log_metric("spt_completed_counter", env.completed_counter[0], step=e)
            # completed_spt.append(env.completed_counter[0])
            # log_metric("completed_avg_spt", sum(completed_spt) / len(completed_spt))

            # #test EDF PROCESS
            # env.reset()
            # for j in range(steps):
            #     [state, reward, action] = env.step_edf_process_case()
            #
            # log_metric("edf_process_completed_counter", env.completed_counter[0], step=e)
            # completed_edf_process.append(env.completed_counter[0])
            # log_metric("completed_avg_edf_process", sum(completed_edf_process) / len(completed_edf_process))
            #
            # #test EDF TASK
            # env.reset()
            # for j in range(steps):
            #     [state, reward, action] = env.step_edf_task()
            #
            # log_metric("edf_task_completed_counter", env.completed_counter[0], step=e)
            # completed_edf_task.append(env.completed_counter[0])
            # log_metric("completed_avg_edf_task", sum(completed_edf_task) / len(completed_edf_task))
            #
            # #test LST_PROCESS
            # env.reset()
            # for j in range(steps):
            #     [state, reward, action] = env.step_lst_process_case()
            #
            # log_metric("lst_process_completed_counter", env.completed_counter[0], step=e)
            # completed_lst_process.append(env.completed_counter[0])
            # log_metric("completed_avg_lst_process", sum(completed_lst_process) / len(completed_lst_process))
            #
            #
            # #test LST_TASK
            # env.reset()
            # for j in range(steps):
            #     [state, reward, action] = env.step_lst_task()
            #
            # log_metric("lst_task_completed_counter", env.completed_counter[0], step=e)
            # completed_lst_task.append(env.completed_counter[0])
            # log_metric("completed_avg_lst_task", sum(completed_lst_task) / len(completed_lst_task))


        # with open("fifo_metrics_54b497e2.txt", 'w') as f:
        #     f.write(str(fifo_metrics))

        # mlflow.log_artifact("fifo_metrics_54b497e2.txt", artifact_path="metrics")
        #
        # plt.figure(figsize=(20, 10))
        # plt.plot(completed_dqn, label="DDQN rewards")
        # plt.plot(completed_fifo, label="FIFO rewards")
        # plt.plot(completed_spt, label="SPT rewards")
        # plt.plot(completed_lst_task, label="LST_TASK rewards")
        # plt.plot(completed_lst_process, label="LST_PROCESS rewards")
        # plt.plot(completed_edf_task, label="EDF_TASK rewards")
        # plt.plot(completed_edf_process, label="EDF_PROCESS rewards")
        # plt.ylim(ymin=0)
        # plt.xlabel("Episode", fontsize=14)
        # plt.ylabel("Number of completed process cases", fontsize=14)
        # # plt.yticks(np.arange(0, 300, step=25))
        # plt.grid(b=True, which='major', axis='both', color='black', linestyle='-', linewidth=0.1)
        # plt.legend()
        # plt.savefig("Comparison_with_" + str(steps) + "_steps.png", format="png", dpi=300)
        #
        # mlflow.log_artifact("Comparison_with_" + str(steps) + "_steps.png")
        #
        # # print("========= " + suffix + " ==========")
        # tmp = test_rewards_dqn
        # #print("DDQN avg: " + str(sum(tmp) / len(tmp)))
        # print(f"Completed avg {sum(completed_dqn)/len(completed_dqn)}")
        # print("=============================\n\n\n")
        # # return test_rewards
        # print(str(fifo_metrics["duration_mean"]))
        # print("#######################################")
        # print(str(fifo_metrics["completed_percentage"]))
        # print("#######################################")
        # print(f"Mean: {mean(fifo_metrics['completed_percentage'])}")

if __name__ == '__main__':
    main()
