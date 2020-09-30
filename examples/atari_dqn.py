from time import time
import gym
# from gym.utils.play import play import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import hydra
import mlflow
import os
import subprocess
import requests
import pathlib
import getpass

from accel.utils.atari_wrappers import make_atari, make_atari_ram
from accel.explorers import epsilon_greedy
from accel.replay_buffers.replay_buffer import Transition, ReplayBuffer
from accel.replay_buffers.prioritized_replay_buffer import PrioritizedReplayBuffer
from accel.agents import dqn
from accel.utils.utils import set_seed


class Net(nn.Module):
    def __init__(self, input, output, dueling=False, high_reso=False):
        super().__init__()
        self.dueling = dueling
        self.conv1 = nn.Conv2d(input, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        linear_size = 7 * 7 * 64 if not high_reso else 12 * 12 * 64
        self.fc1 = nn.Linear(linear_size, 512)
        self.fc2 = nn.Linear(512, output)
        if self.dueling:
            self.v_fc1 = nn.Linear(linear_size, 512)
            self.v_fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = x / 255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)

        adv = F.relu(self.fc1(x))
        adv = self.fc2(adv)
        if not self.dueling:
            return adv

        v = F.relu(self.v_fc1(x))
        v = self.v_fc2(v)
        return v + adv - adv.mean(dim=1, keepdim=True)


class RamNet(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.fc1 = nn.Linear(input, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.relu(self.fc4(x))


def get_commitid():
    # return short commit id
    cmd = "git rev-parse --short HEAD"
    commitid = subprocess.check_output(cmd.split()).strip()
    return commitid

def slack_notify(msg = 'done'):
    # Notification to slack
    # Register id and url in environment variables
    slack_user_id = os.getenv('SLACK_USER_ID')
    slack_webhook_url = os.getenv('SLACK_WEBHOOK_URL')
    if slack_user_id is not None and slack_webhook_url is not None:
        requests.post(slack_webhook_url, json={"text":msg})

def get_model_path(model_evacuation):
    model_save_path = os.getcwd()
    if model_evacuation:
        # Save only the model under localhome
        hydra_abs_cwd = pathlib.Path(os.getcwd())
        org_abs_cwd = hydra.utils.get_original_cwd()
        hydra_rel_cwb  = str(hydra_abs_cwd.relative_to(org_abs_cwd))
        local_home = os.getenv('LOCALHOME')
        if local_home is not None:
            model_save_path = os.path.join(local_home, hydra_rel_cwb)
            os.makedirs(model_save_path, exist_ok=True)
    user_name = getpass.getuser()
    machine_name = os.uname()[1]
    model_save_upath = user_name + '@' + machine_name + ':' + model_save_path
    return model_save_path, model_save_upath

@hydra.main(config_name='config/atari_dqn_config.yaml')
def main(cfg):
    set_seed(cfg.seed)

    model_save_path, model_save_upath = get_model_path(cfg.model_evacuation)
    slack_notify("start {} on {}".format(cfg.name, os.uname()[1]))

    cwd = hydra.utils.get_original_cwd()
    mlflow.set_tracking_uri(os.path.join(cwd, 'mlruns'))
    mlflow.set_experiment('atari_dqn')

    with mlflow.start_run(run_name=cfg.name):
        mlflow.log_param('seed', cfg.seed)
        mlflow.log_param('gamma', cfg.gamma)
        mlflow.log_param('replay', cfg.replay_capacity)
        mlflow.log_param('dueling', cfg.dueling)
        mlflow.log_param('prioritized', cfg.prioritized)
        # mlflow.log_param('color', cfg.color)
        # mlflow.log_param('high', cfg.high_reso)
        mlflow.log_param('no_stack', cfg.no_stack)
        mlflow.log_param('nstep', cfg.nstep)
        mlflow.log_param('huber', cfg.huber)
        mlflow.set_tag('env', cfg.env)
        mlflow.set_tag('commitid', get_commitid())
        mlflow.set_tag('machine', os.uname()[1])

        if not cfg.device:
            cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        is_ram = '-ram' in cfg.env
        if is_ram:
            env = make_atari_ram(cfg.env)
            eval_env = make_atari_ram(cfg.env, clip_rewards=False)
        else:
            if cfg.high_reso:
                env = make_atari(cfg.env, color=cfg.color,
                                 image_size=128, frame_stack=not cfg.no_stack)
                eval_env = make_atari(
                    cfg.env, clip_rewards=False, color=cfg.color, image_size=128, frame_stack=not cfg.no_stack)
            else:
                env = make_atari(cfg.env, color=cfg.color,
                                 frame_stack=not cfg.no_stack)
                eval_env = make_atari(
                    cfg.env, clip_rewards=False, color=cfg.color, frame_stack=not cfg.no_stack)

        env.seed(cfg.seed)
        eval_env.seed(cfg.seed)

        dim_state = env.observation_space.shape[0]
        dim_action = env.action_space.n

        if is_ram:
            q_func = RamNet(dim_state, dim_action)
        else:
            q_func = Net(dim_state, dim_action,
                         dueling=cfg.dueling, high_reso=cfg.high_reso)

        if cfg.load:
            q_func.load_state_dict(torch.load(os.path.join(
                cwd, cfg.load), map_location=cfg.device))

        optimizer = optim.RMSprop(
            q_func.parameters(), lr=0.00025, alpha=0.95, eps=1e-2)

        if cfg.prioritized:
            memory = PrioritizedReplayBuffer(capacity=cfg.replay_capacity, beta_steps=cfg.steps - cfg.replay_start_step, nstep=cfg.nstep)
        else:
            memory = ReplayBuffer(capacity=cfg.replay_capacity, nstep=cfg.nstep)

        score_steps = []
        scores = []

        explorer = epsilon_greedy.LinearDecayEpsilonGreedy(
            start_eps=1.0, end_eps=0.1, decay_steps=1e6)

        agent = dqn.DoubleDQN(q_func, optimizer, memory, cfg.gamma,
                              explorer, cfg.device, batch_size=32,
                              target_update_interval=10000,
                              replay_start_step=cfg.replay_start_step,
                              huber=cfg.huber)

        if cfg.demo:
            for x in range(10):
                total_reward = 0

                while True:
                    obs = eval_env.reset()
                    eval_env.render()
                    done = False

                    while not done:
                        action = agent.act(obs, greedy=True)
                        obs, reward, done, _ = eval_env.step(action)
                        eval_env.render()

                        # print(reward)
                        total_reward += reward

                    if eval_env.was_real_done:
                        break

                print('Episode:', x, 'Score:', total_reward)

            exit(0)

        next_eval_cnt = 1
        episode_cnt = 0

        train_start_time = time()

        log_file_name = 'scores.txt'
        model_file_name = 'model_path.txt'
        best_score = -1e10

        while agent.total_steps < cfg.steps:
            episode_cnt += 1

            obs = env.reset()
            done = False
            total_reward = 0
            step = 0

            while not done:
                action = agent.act(obs)
                next_obs, reward, done, _ = env.step(action)
                total_reward += reward
                step += 1

                next_valid = 1 if step == env.spec.max_episode_steps else float(
                    not done)
                loss_ = agent.update(obs, action, next_obs, reward, next_valid)
                if loss_ is not None: loss = loss_

                obs = next_obs

            if agent.total_steps > next_eval_cnt * cfg.eval_interval:
                total_reward = 0

                while True:
                    obs = eval_env.reset()
                    done = False

                    while not done:
                        action = agent.act(obs, greedy=True)
                        obs, reward, done, _ = eval_env.step(action)

                        total_reward += reward

                    if eval_env.was_real_done:
                        break

                next_eval_cnt += 1

                score_steps.append(agent.total_steps)
                scores.append(total_reward)

                if total_reward > best_score:
                    model_name = os.path.join(model_save_path, f'{agent.total_steps}.model')
                    torch.save(q_func.state_dict(), model_name)
                    model_path = os.path.join(model_save_upath, f'{agent.total_steps}.model')
                    # mlflow.log_artifact(model_path)
                    with open(model_file_name, 'a') as f:
                        f.write(model_path)
                    best_score = total_reward

                now = time()
                elapsed = now - train_start_time

                log = f'{agent.total_steps} {total_reward} {elapsed:.1f}\n'
                print(log, end='')
                mlflow.log_metric('reward', total_reward,
                                  step=agent.total_steps)
                if loss is not None: mlflow.log_metric('loss', loss, step=agent.total_steps)

                with open(log_file_name, 'a') as f:
                    f.write(log)

        # final evaluation
        total_reward = 0

        while True:
            obs = eval_env.reset()
            done = False

            while not done:
                action = agent.act(obs, greedy=True)
                obs, reward, done, _ = eval_env.step(action)

                total_reward += reward

            if eval_env.was_real_done:
                break

        score_steps.append(agent.total_steps)
        scores.append(total_reward)

        model_name = os.path.join(model_save_path, f'final.model')
        torch.save(q_func.state_dict(), model_name)
        model_path = os.path.join(model_save_upath, f'final.model')
        # mlflow.log_artifact(model_path)
        with open(model_file_name, 'a') as f:
            f.write(model_path)
        try:
            mlflow.log_artifact(model_file_name)
            mlflow.log_artifact(model_path)
        except Exception as e:
            with open(log_file_name, 'a') as f:
                f.write(e)

        now = time()
        elapsed = now - train_start_time

        log = f'{agent.total_steps} {total_reward} {elapsed:.1f}\n'
        print(log, end='')
        mlflow.log_metric('reward', total_reward, step=agent.total_steps)

        with open(log_file_name, 'a') as f:
            f.write(log)

        duration = np.round(elapsed / 60 / 60, 2)
        mlflow.log_metric('duration', duration)
        print('Complete')
        slack_notify("Complete {} on {}, duration: {}h".format(cfg.name, os.uname()[1], duration))
        env.close()


if __name__ == '__main__':
    main()
