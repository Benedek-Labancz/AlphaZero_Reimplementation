from argparse import ArgumentParser
import os
import json
import time

import random
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.optim import SGD
from collections import deque

from src.mcts.tree import Tree, Node
from src.training.self_play import play_episode
from src.environments.f4ce.two_dims import TwoDims
from src.environments.f4ce.three_dims import ThreeDims
from src.environments.f4ce.four_dims import FourDims
from src.policy.network import PolicyScoreNetwork
from src.training.train_policy import train_batch
from src.training.evaluate import evaluate_policy

def seed_everything(seed: int):
    # TODO
    pass

def save_checkpoint(policy, path: str, timestep: int):
    if not os.path.isdir(path):
        os.mkdir(path)
    filename = f'checkpoint_{timestep}.pth'
    torch.save(policy.state_dict(), os.path.join(path, filename))

def get_policy(state_dict, config):
    policy = PolicyScoreNetwork(**config["network_arguments"])
    policy.load_state_dict(state_dict)
    return policy

def self_play_worker(env, 
                     config, 
                     from_eval_conn, 
                     data_queue, 
                     lock, 
                     episode_counter, 
                     initial_model_weights):
    policy = get_policy(initial_model_weights, config)
    while episode_counter.value < config["global"]["num_games"]:
        new_best_weights_available = False
        while from_eval_conn.poll():
            model_weights = from_eval_conn.recv()
            new_best_weights_available = True
        if new_best_weights_available:
            policy.load_state_dict(model_weights)
        data = play_episode(
            tree=Tree(root=Node(env=env, state=env.get_board_state())),
            policy=policy,
            num_simulations=config["self_play"]["num_simulations"],
            c=config["self_play"]["c"],
            tau=config["self_play"]["tau"],
            early_selection_threshold=config["self_play"]["early_selection_threshold"]
        )
        data_queue.put(data)
        with lock:
            episode_counter.value += 1


def train_policy_worker(config,
                        data_queue,
                        checkpoint_queue,
                        episode_counter,
                        update_counter,
                        lock,
                        initial_model_weights):
    policy = get_policy(initial_model_weights, config)
    optimizer = SGD(
        params=policy.parameters(), 
        lr=config["training"]["lr"], 
        momentum=config["training"]["momentum"],
        weight_decay=config["training"]["weight_decay"]
        )
    data_buffer = deque(maxlen=config["training"]["buffer_size"])
    while episode_counter.value < config["global"]["num_games"]:
        while not data_queue.empty():
            data_buffer.append(data_queue.get())
        if len(data_buffer) < config["training"]["batch_size"]:
            continue
        b = time.time()
        # Sample and batch data
        samples = random.sample(data_buffer, k=config["training"]["batch_size"])
        states = []
        pi_values = []
        winners = []
        for data in samples:
            s, pi, z = data
            states.append(s)
            pi_values.append(pi)
            winners.append(z)
        batch_loss = train_batch(
            data=(states, pi_values, winners),
            policy=policy,
            optimizer=optimizer
        )
        e = time.time()
        # TODO: log loss to wandb
        print(f"Timestep: {update_counter.value}\t-\tLoss: {batch_loss}\t-\tTime taken: {round(e - b, 3)}s")
        with lock:
            update_counter.value += 1
        if update_counter.value % config["training"]["checkpoint_frequency"] == 0:
            checkpoint_queue.put(policy.state_dict())
            save_checkpoint(
                policy=policy,
                path=config["training"]["save_path"],
                timestep=update_counter.value
            )

def eval_worker(env,
                config,
                checkpoint_queue,
                to_self_play_conn,
                episode_counter,
                initial_model_weights):
    best_policy = get_policy(initial_model_weights, config)
    while episode_counter.value < config["global"]["num_games"]:
        if checkpoint_queue.empty():
            continue
        candidate_weights = checkpoint_queue.get()
        candidate_policy = get_policy(candidate_weights, config)
        win_rate = evaluate_policy(
            env=env,
            best_policy=best_policy,
            candidate_policy=candidate_policy,
            num_games=config["eval"]["num_games"],
            num_simulations=config["eval"]["num_simulations"],
            c=config["eval"]["c"]
        )
        if win_rate > config["eval"]["win_rate_margin"]:
            new_best_weights = candidate_policy.state_dict()
            to_self_play_conn.send(new_best_weights)
            best_policy.load_state_dict(new_best_weights)
            print(f'New Best Model!\t-\tWin Rate: {round(win_rate, 3)}')


if __name__ == '__main__':
    seed_everything(42) # TODO
    parser = ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        config = json.loads(f.read())

    env = TwoDims()
    initial_policy = PolicyScoreNetwork(**config["network_arguments"])
    initial_model_weights = initial_policy.state_dict()

    lock = mp.Lock()
    episode_counter = mp.Value('i', 0)
    update_counter = mp.Value('i', 0)
    
    data_queue = mp.Queue()
    checkpoint_queue = mp.Queue()

    from_eval_conn, to_self_play_conn = mp.Pipe(duplex=True)

    self_play_w = mp.Process(
        target=self_play_worker,
        args=(
            env,
            config,
            from_eval_conn,
            data_queue,
            lock,
            episode_counter,
            initial_model_weights
        )
    )

    train_policy_w = mp.Process(
        target=train_policy_worker,
        args=(
            config,
            data_queue,
            checkpoint_queue,
            episode_counter,
            update_counter,
            lock,
            initial_model_weights
        )
    )

    eval_w = mp.Process(
        target=eval_worker,
        args=(
            env,
            config,
            checkpoint_queue,
            to_self_play_conn,
            episode_counter,
            initial_model_weights
        )
    )

    self_play_w.start()
    # train_policy_w.start()
    # eval_w.start()