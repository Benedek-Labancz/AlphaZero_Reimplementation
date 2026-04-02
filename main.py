from argparse import ArgumentParser
import os
import json
import time
import psutil

import random
import numpy as np
from numpy.random import SeedSequence, default_rng
import torch
import torch.multiprocessing as mp
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from collections import deque
import wandb


from src.mcts.tree import Tree, Node
from src.training.self_play import self_play_episode
from src.environments.f4ce.two_dims import TwoDims
from src.environments.f4ce.three_dims import ThreeDims
from src.environments.f4ce.four_dims import FourDims
from src.policy.network import PolicyScoreNetwork
from src.training.train_policy import train_batch
from src.training.play import play_episodes, select_az_action
from src.training.utils import batch_entropy
from src.evaluation.minimax import select_minimax_action
from src.evaluation.random import select_random_action

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(child_seed: SeedSequence):
    """Call at the top of every worker function."""
    seeds = child_seed.generate_state(3, dtype=np.uint64)

    rng = default_rng(child_seed)          # numpy
    torch.manual_seed(int(seeds[0]))       # torch CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seeds[0]))  # torch CUDA
    random.seed(int(seeds[1]))             # Python random
    # seeds[2] spare — e.g. for an environment's own RNG
    return rng

def save_checkpoint(policy, path: str, timestep: int):
    if not os.path.isdir(path):
        os.mkdir(path)
    filename = f'checkpoint_{timestep}.pth'
    torch.save(policy.state_dict(), os.path.join(path, filename))

def get_policy(state_dict, config):
    policy = PolicyScoreNetwork(**config["network_arguments"])
    policy.load_state_dict(state_dict)
    return policy

def self_play_worker(child_seed,
                     env, 
                     wandb_run,
                     config, 
                     from_eval_conn, 
                     data_queue, 
                     lock,
                     episode_counter,
                     update_counter,
                     episode_time,
                     initial_model_weights):
    print(f"Self-play worker affinity: {psutil.Process().cpu_affinity()}")
    rng = seed_worker(child_seed)
    policy = get_policy(initial_model_weights, config)
    while update_counter.value < config["global"]["num_updates"]:
        new_best_weights_available = False
        while from_eval_conn.poll():
            model_weights = from_eval_conn.recv()
            new_best_weights_available = True
        if new_best_weights_available:
            policy.load_state_dict(model_weights)
            print("New best policy loaded.")
        b = time.time()
        env.reset()
        data = self_play_episode(
            rng=rng,
            tree=Tree(root=Node(env=env, state=env.get_board_state())),
            policy=policy,
            num_simulations=config["self_play"]["num_simulations"],
            c=config["self_play"]["c"],
            tau=config["self_play"]["tau"],
            early_selection_threshold=config["self_play"]["early_selection_threshold"]
        )
        data_queue.put(data)
        wandb_run.log({'pi_entropy': np.mean(batch_entropy(data[1]))})
        e = time.time()
        ep_time = e - b
        with lock:
            episode_counter.value += 1
            episode_time.value += ep_time
        with open(os.path.join(config["global"]["log_dir"], "self_play_worker.txt"), 'w') as f:
            print(f'Number of games generated: {episode_counter.value}', file=f)
            print(f'Average generation rate: {round(episode_time.value / episode_counter.value, 3)}s/game', file=f)
            print(f'Last game took {round(ep_time, 3)}s to generate.', file=f)
    data_queue.close()
    data_queue.cancel_join_thread()
    print('Data queue closed.')

def train_policy_worker(child_seed,
                        wandb_run,
                        config,
                        data_queue,
                        checkpoint_queue,
                        lock,
                        episode_counter,
                        update_counter,
                        eval_counter,
                        update_time,
                        initial_model_weights):
    
    print(f"Train-policy worker affinity: {psutil.Process().cpu_affinity()}")
    rng = seed_worker(child_seed)
    policy = get_policy(initial_model_weights, config)

    optimizer = SGD(
        params=policy.parameters(), 
        lr=config["training"]["lr"], 
        momentum=config["training"]["momentum"],
        weight_decay=config["training"]["weight_decay"]
        )
    scheduler = StepLR(
        optimizer=optimizer,
        step_size=config["training"]["lr_annealing_step_size"],
        gamma=config["training"]["lr_annealing"]
    )
    data_buffer = deque(maxlen=config["training"]["buffer_size"])

    while update_counter.value < config["global"]["num_updates"]:
        # Control the rate of policy updates on several levels
        # Wait for self-play episode / policy update ratio
        if update_counter.value > 0:
            while (episode_counter.value / update_counter.value) < config["training"]["episodes_per_update"]:
                continue
        # Wait for evaluation of the last checkpoint
        if update_counter.value >= config["training"]["checkpoint_frequency"]:
            while (eval_counter.value / (update_counter.value // config["training"]["checkpoint_frequency"])) != 1:
                continue
        try:
            while not data_queue.empty():
                data_buffer.append(data_queue.get())
        except ValueError:
            print("Data Queue has been closed. Aborting Training.")
            break
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
        states = np.vstack(np.array(states))
        pi_values = np.vstack(np.array(pi_values))
        winners = np.array(winners).reshape(-1)
        batch_loss = train_batch(
            data=(states, pi_values, winners),
            policy=policy,
            optimizer=optimizer,
            scheduler=scheduler
        )
        wandb_run.log({"loss": batch_loss})
        e = time.time()
        up_time = e - b
        with lock:
            update_counter.value += 1
            update_time.value += up_time
        with open(os.path.join(config["global"]["log_dir"], "train_policy_worker.txt"), 'w') as f:
            print(f'Number of policy updates: {update_counter.value}', file=f)
            print(f'Average update rate: {round(update_time.value / update_counter.value, 3)}s/batch', file=f)
            print(f'Last update took {round(up_time, 3)}s.', file=f)
            print(f'Average checkpoint rate: {round((update_time.value / update_counter.value) * config["training"]["checkpoint_frequency"], 3)}s/checkpoint', file=f)
            print(f'Batch size: {config["training"]["batch_size"]}', file=f)
        if update_counter.value % config["training"]["checkpoint_frequency"] == 0:
            checkpoint_queue.put(policy.state_dict())
            save_checkpoint(
                policy=policy,
                path=config["training"]["save_path"],
                timestep=update_counter.value
            )
    checkpoint_queue.close()
    checkpoint_queue.cancel_join_thread()
    print('Checkpoint queue closed.')

def eval_worker(child_seed,
                env,
                wandb_run,
                config,
                checkpoint_queue,
                to_self_play_conn,
                lock,
                eval_counter,
                update_counter,
                promotion_counter,
                eval_time,
                initial_model_weights):
    print(f"Eval worker affinity: {psutil.Process().cpu_affinity()}")
    rng = seed_worker(child_seed)
    best_policy = get_policy(initial_model_weights, config)
    while update_counter.value < config["global"]["num_updates"]:
        if checkpoint_queue.empty():
            continue
        try:
            candidate_weights = checkpoint_queue.get()
        except ValueError as e:
            print("Checkpoint Queue has been closed. Aborting Evaluation.")
            break
        b = time.time()
        candidate_policy = get_policy(candidate_weights, config)
        with lock:
            num_wins = play_episodes(
                rng=rng,
                env=env,
                best_policy=best_policy,
                candidate_policy=candidate_policy,
                best_select_action_fn=select_az_action,
                candidate_select_action_fn=select_az_action,
                num_games=config["eval"]["num_games"],
                num_simulations=config["eval"]["num_simulations"],
                c=config["eval"]["c"]
            )
        win_rate = num_wins[1] / config["eval"]["num_games"]
        wandb_run.log({'num_losses_vs_best': num_wins[0]})
        wandb_run.log({'num_wins_vs_best': num_wins[1]})
        if num_wins[0] < num_wins[1] or num_wins == [0, 0]:
            new_best_weights = candidate_policy.state_dict()
            to_self_play_conn.send(new_best_weights)
            best_policy.load_state_dict(new_best_weights)
            save_checkpoint(
                policy=best_policy,
                path=os.path.join(config["training"]["save_path"], 'best'),
                timestep=update_counter.value
            )
            print(f'New Best Model!\t-\tWin Rate: {round(win_rate, 3)}')
            with lock:
                promotion_counter.value += 1
            # Evaluate against minimax
            with lock:
                num_wins_m = play_episodes(
                    rng=rng,
                    env=env,
                    best_policy=best_policy,
                    candidate_policy=None,
                    best_select_action_fn=select_az_action,
                    candidate_select_action_fn=select_minimax_action,
                    num_games=config["eval"]["num_games"],
                    num_simulations=config["eval"]["num_simulations"],
                    c=config["eval"]["c"],
                    max_depth=config["eval"]["minimax_max_depth"],
                    epsilon=config["eval"]["minimax_epsilon"]
                )
                wandb_run.log({f'num_losses_vs_minimax-{config["eval"]["minimax_max_depth"]}': num_wins_m[0]})
                wandb_run.log({f'num_wins_vs_minimax-{config["eval"]["minimax_max_depth"]}': num_wins_m[1]})
                num_wins_r = play_episodes(
                    rng=rng,
                    env=env,
                    best_policy=best_policy,
                    candidate_policy=None,
                    best_select_action_fn=select_az_action,
                    candidate_select_action_fn=select_random_action,
                    num_games=config["eval"]["num_games"],
                    num_simulations=config["eval"]["num_simulations"],
                    c=config["eval"]["c"],
                )
                wandb_run.log({'num_losses_vs_random': num_wins_r[0]})
                wandb_run.log({'num_wins_vs_random': num_wins_r[1]})
        e = time.time()
        ev_time = e - b
        with lock:
            eval_counter.value += 1
            eval_time.value += ev_time
        with open(os.path.join(config["global"]["log_dir"], "eval_worker.txt"), 'w') as f:
            print(f'Number of evaluations: {eval_counter.value}', file=f)
            print(f'Average eval rate: {round(eval_time.value / eval_counter.value, 3)}s/eval ({config["eval"]["num_games"]} games each)', file=f)
            print(f'Last evaluation took {round(ev_time, 3)}s.', file=f)
            print(f'Win rate {round(win_rate, 3)}', file=f)
            print(f'Promotion rate: {round((promotion_counter.value / eval_counter.value), 3)}', file=f)
    checkpoint_queue.close()
    checkpoint_queue.cancel_join_thread()
    print('Checkpoint queue closed.')


if __name__ == '__main__':
    GLOBAL_SEED = 42
    NUM_WORKERS = 3
    seed_everything(GLOBAL_SEED)
    ss = SeedSequence(GLOBAL_SEED)
    child_seeds = ss.spawn(NUM_WORKERS)


    parser = ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        config = json.loads(f.read())


    wandb_run = wandb.init(
        entity=config["logging"]["entity_name"],
        project=config["logging"]["project_name"],
        config=config,
    )

    env = TwoDims()
    initial_policy = PolicyScoreNetwork(**config["network_arguments"])
    initial_model_weights = initial_policy.state_dict()

    lock = mp.Lock()
    episode_counter = mp.Value('i', 0)
    update_counter = mp.Value('i', 0)
    eval_counter = mp.Value('i', 0)
    promotion_counter = mp.Value('i', 0)
    episode_time = mp.Value('f', 0)
    update_time = mp.Value('f', 0)
    eval_time = mp.Value('f', 0)
    
    data_queue = mp.Queue()
    checkpoint_queue = mp.Queue()

    from_eval_conn, to_self_play_conn = mp.Pipe(duplex=True)

    self_play_w = mp.Process(
        target=self_play_worker,
        args=(
            child_seeds[0],
            env,
            wandb_run,
            config,
            from_eval_conn,
            data_queue,
            lock,
            episode_counter,
            update_counter,
            episode_time,
            initial_model_weights
        )
    )
    self_play_w.start()

    train_policy_w = mp.Process(
        target=train_policy_worker,
        args=(
            child_seeds[1],
            wandb_run,
            config,
            data_queue,
            checkpoint_queue,
            lock,
            episode_counter,
            update_counter,
            eval_counter,
            update_time,
            initial_model_weights
        )
    )
    train_policy_w.start()


    eval_w = mp.Process(
        target=eval_worker,
        args=(
            child_seeds[2],
            env,
            wandb_run,
            config,
            checkpoint_queue,
            to_self_play_conn,
            lock,
            eval_counter,
            update_counter,
            promotion_counter,
            eval_time,
            initial_model_weights
        )
    )
    eval_w.start()
    
    self_play_w.join()
    train_policy_w.join()
    eval_w.join()
    wandb_run.finish()