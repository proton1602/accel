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
import math
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

class Net_1(nn.Module):
    def __init__(self, input, output, dueling=False, high_reso=False):
        super().__init__()
        self.dueling = dueling
        self.conv1 = nn.Conv2d(input, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        linear_size = 7 * 7 * 64 if not high_reso else 12 * 12 * 64
        self.fc1 = nn.Linear(linear_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, output)
        if self.dueling:
            self.v_fc1 = nn.Linear(linear_size, 512)
            self.v_fc2 = nn.Linear(512, 128)
            self.v_fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = x / 255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)

        adv = F.relu(self.fc1(x))
        adv = F.relu(self.fc2(adv))
        adv = self.fc3(adv)
        if not self.dueling:
            return adv

        v = F.relu(self.v_fc1(x))
        v = F.relu(self.v_fc2(v))
        v = self.v_fc3(v)
        return v + adv - adv.mean(dim=1, keepdim=True)

class Net_2(nn.Module):
    def __init__(self, input, output, dueling=False, high_reso=False):
        super().__init__()
        self.dueling = dueling
        self.conv1 = nn.Conv2d(input, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        linear_size = 7 * 7 * 64 if not high_reso else 12 * 12 * 64
        self.fc1 = nn.Linear(linear_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output)
        if self.dueling:
            self.v_fc1 = nn.Linear(linear_size, 512)
            self.v_fc2 = nn.Linear(512, 256)
            self.v_fc3 = nn.Linear(256, 128)
            self.v_fc4 = nn.Linear(128, 1)

    def forward(self, x):
        x = x / 255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)

        adv = F.relu(self.fc1(x))
        adv = F.relu(self.fc2(adv))
        adv = F.relu(self.fc3(adv))
        adv = self.fc4(adv)
        if not self.dueling:
            return adv

        v = F.relu(self.v_fc1(x))
        v = F.relu(self.v_fc2(v))
        v = F.relu(self.v_fc3(v))
        v = self.v_fc4(v)
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
        return self.fc4(x)


class RamNet_1(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.fc1 = nn.Linear(input, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class RamNet_2(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.fc1 = nn.Linear(input, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class Linear_hold(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear_hold, self).__init__(in_features, out_features, bias)
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.out = F.linear(input, self.weight, self.bias)
        return self.out

class Conv2d_hold(nn.Conv2d):
    def __init__(self,in_channels: int, out_channels: int, kernel_size,
        stride = 1, padding = 0, dilation = 1, groups: int = 1,
        bias: bool = True, padding_mode: str = 'zeros'):
        super(Conv2d_hold, self).__init__(in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias, padding_mode)
        self.out = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.out = self._conv_forward(input, self.weight)
        return self.out

class AdaptC(nn.Module):
    def __init__(self, main_link, main_index, input_size, output_size, paramset):
        self.paramset = paramset
        super(AdaptC, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.main = main_link
        self.index = main_index
        self.sub = nn.Conv2d(output_size, output_size, kernel_size=1, stride=1)
        self.out = None

    def __call__(self, x): 
        if self.paramset.phase == 2 or self.paramset.phase == 3:
            y = self.main(x)
            self.out = y + self.sub(y) 
        else:
            self.out = self.main.out + self.sub(self.main.out) 
        return self.out

class AdaptL(nn.Module):
    def __init__(self, main_link, main_index, input_size, output_size, paramset, comp_rate=2**(-4)):
        self.paramset = paramset
        super(AdaptL, self).__init__()
        self.input_size = input_size
        self.comp_size = self.floor(input_size*comp_rate)
        self.output_size = output_size
        self.main = main_link
        self.index = main_index
        self.sub0 = nn.Linear(self.input_size, self.comp_size)
        self.sub1 = nn.Linear(self.comp_size, self.output_size)
        self.out = None

    def __call__(self, x): 
        comp_data = self.sub0(x)
        if self.paramset.phase == 2 or self.paramset.phase == 3:
            self.out = self.main(x) + self.sub1(comp_data) 
        else:
            self.out = self.main.out + self.sub1(comp_data) 
        return self.out

    def floor(self, x): 
        return int(-(-x//1)) # Round up

class GConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride = 1,
        module_list=[]):
        super(GConv2d, self).__init__()
        self.LinearDict = nn.ModuleDict()
        self.key_list = [] # Guarantee the order
        for i, module in enumerate(module_list):
            self.LinearDict[f'reuse{i}'] = module
            self.key_list.append(f'reuse{i}')
        for i, module in enumerate(module_list):
            self.LinearDict[f'adapt{i}'] = AdaptC(module, i, in_channels, out_channels, self)
            self.key_list.append(f'adapt{i}')
        self.LinearDict['new'] = Conv2d_hold(in_channels, out_channels, kernel_size, stride=stride)
        self.key_list.append('new')
        self.len = len(self.LinearDict)
        if self.len == 1:
            self.alpha = nn.Parameter(torch.ones(1), requires_grad=False)
        else:
            bound = 1 / math.sqrt(self.len)
            self.alpha = nn.Parameter(nn.init.uniform_(torch.empty(self.len), -bound, bound),
                requires_grad=True)

    def forward(self, x, phase):
        self.phase = phase
        self.out = torch.zeros(1)
        for i, key in enumerate(self.key_list):
            self.out += self.LinearDict[key](x) * self.alpha[i]
        return self.out
    
    def reset_alpha(self):
        if self.len == 1:
            self.alpha = nn.Parameter(torch.ones(1), requires_grad=False)
        else:
            bound = 1 / math.sqrt(self.len)
            self.alpha = nn.Parameter(nn.init.uniform_(torch.empty(self.len), -bound, bound),
                requires_grad=True)


class GLinear(nn.Module):
    def __init__(self, in_channels, out_channels, module_list=[], no_grow=False):
        super(GLinear, self).__init__()
        self.LinearDict = nn.ModuleDict()
        self.key_list = [] # Guarantee the order
        for i, module in enumerate(module_list):
            self.LinearDict[f'reuse{i}'] = module
            self.key_list.append(f'reuse{i}')
        if not no_grow:
            for i, module in enumerate(module_list):
                self.LinearDict[f'adapt{i}'] = AdaptL(module, i, in_channels, out_channels, self)
                self.key_list.append(f'adapt{i}')
            self.LinearDict['new'] = Linear_hold(in_channels, out_channels)
            self.key_list.append('new')
        self.len = len(self.LinearDict)
        if self.len == 1:
            self.alpha = nn.Parameter(torch.ones(1), requires_grad=False)
        else:
            bound = 1 / math.sqrt(self.len)
            self.alpha = nn.Parameter(nn.init.uniform_(torch.empty(self.len), -bound, bound),
                requires_grad=True)

    def forward(self, x, phase):
        self.phase = phase
        self.out = torch.zeros(1)
        for i, key in enumerate(self.key_list):
            self.out += self.LinearDict[key](x) * self.alpha[i]
        return self.out
    
    def reset_alpha(self):
        if self.len == 1:
            self.alpha = nn.Parameter(torch.ones(1), requires_grad=False)
        else:
            bound = 1 / math.sqrt(self.len)
            self.alpha = nn.Parameter(nn.init.uniform_(torch.empty(self.len), -bound, bound),
                requires_grad=True)

class GNet(nn.Module):
    def __init__(self, in_channels, out_channels, first_task, first_model, second_task, no_grow=False,
         high_reso=False, task_num = 1):
        super(GNet, self).__init__()
        self.linear_size = 7 * 7 * 64 if not high_reso else 12 * 12 * 64
        self.first_task = make_atari_ram(first_task) if is_ram(first_task) else make_atari(first_task)
        self.first_in_channels = self.first_task.observation_space.shape[0]
        self.first_out_channels = self.first_task.action_space.n
        self.f2s = ('ram' if is_ram(first_task) else 'img') + '2' + ('ram' if is_ram(second_task) else 'img')
        self.task_num = task_num # 0: first_task, 1: second_task, 2: third_task(same as the first)
        self.in_ram = is_ram(second_task) if self.task_num==1 else is_ram(first_task)
        self.fc1_1_exist = False
        self.fc4_1_exist = False
        if self.f2s == 'ram2ram':
            if self.first_in_channels == in_channels:
                self.fc1 = GLinear(self.first_in_channels, 512, module_list=[first_model.fc1])
            else:
                self.fc1 = GLinear(self.first_in_channels, 512, module_list=[first_model.fc1], no_grow=no_grow)
                self.fc1_1 = GLinear(in_channels, 512)
                self.fc1_1_exist = True
            self.fc2 = GLinear(512, 256, module_list=[first_model.fc2])
            self.fc3 = GLinear(256, 128, module_list=[first_model.fc3])
            if self.first_out_channels == out_channels:
                self.fc4 = GLinear(128, self.first_out_channels, module_list=[first_model.fc4])
            else: 
                self.fc4 = GLinear(128, self.first_out_channels, module_list=[first_model.fc4], no_grow=no_grow)
                self.fc4_1 = GLinear(128, out_channels)
                self.fc4_1_exist = True
        elif self.f2s == 'ram2img': 
            self.conv1 = Gconv2d(in_channels, 32, kernel_size=8, stride=4)
            self.conv2 = Gconv2d(32, 64, kernel_size=4, stride=2)
            self.conv3 = Gconv2d(64, 64, kernel_size=3, stride=1)
            if self.first_in_channels == self.linear_size:
                self.fc1 = GLinear(self.first_in_channels, 512, module_list=[first_model.fc1])
            else:
                self.fc1 = GLinear(self.first_in_channels, 512, module_list=[first_model.fc1], no_grow=no_grow)
                self.fc1_1 = GLinear(self.linear_size, 512)
                self.fc1_1_exist = True
            self.fc2 = GLinear(512, 256, module_list=[first_model.fc2])
            self.fc3 = GLinear(256, 128, module_list=[first_model.fc3])
            if self.first_out_channels == out_channels:
                self.fc4 = GLinear(128, self.first_out_channels, module_list=[first_model.fc4])
            else: 
                self.fc4 = GLinear(128, self.first_out_channels, module_list=[first_model.fc4], no_grow=no_grow)
                self.fc4_1 = GLinear(128, out_channels)
                self.fc4_1_exist = True
        elif self.f2s == 'img2ram':
            self.conv1 = Gconv2d(in_channels, 32, kernel_size=8, stride=4, 
                module_list=[first_model.conv1], no_grow=no_grow)
            self.conv2 = Gconv2d(32, 64, kernel_size=4, stride=2, 
                module_list=[first_model.conv2], no_grow=no_grow)
            self.conv3 = Gconv2d(64, 64, kernel_size=3, stride=1, 
                module_list=[first_model.conv3], no_grow=no_grow)
            if self.linear_size == in_channels:
                self.fc1 = GLinear(self.linear_size, 512, module_list=[first_model.fc1])
            else:
                self.fc1 = GLinear(self.linear_size, 512, module_list=[first_model.fc1], no_grow=no_grow)
                self.fc1_1 = GLinear(in_channels, 512)
                self.fc1_1_exist = True
            self.fc2 = GLinear(512, 256, module_list=[first_model.fc2])
            self.fc3 = GLinear(256, 128, module_list=[first_model.fc3])
            if self.first_out_channels == out_channels:
                self.fc4 = GLinear(128, self.first_out_channels, module_list=[first_model.fc4])
            else: 
                self.fc4 = GLinear(128, self.first_out_channels, module_list=[first_model.fc4], no_grow=no_grow)
                self.fc4_1 = GLinear(128, out_channels)
                self.fc4_1_exist = True
        elif self.f2s == 'img2img':
            self.conv1 = Gconv2d(in_channels, 32, kernel_size=8, stride=4, 
                module_list=[first_model.conv1])
            self.conv2 = Gconv2d(32, 64, kernel_size=4, stride=2, 
                module_list=[first_model.conv2])
            self.conv3 = Gconv2d(64, 64, kernel_size=3, stride=1, 
                module_list=[first_model.conv3])
            self.fc1 = GLinear(self.linear_size, 512, module_list=[first_model.fc1])
            self.fc2 = GLinear(512, 256, module_list=[first_model.fc2])
            self.fc3 = GLinear(256, 128, module_list=[first_model.fc3])
            if self.first_out_channels == out_channels:
                self.fc4 = GLinear(128, self.first_out_channels, module_list=[first_model.fc4])
            else: 
                self.fc4 = GLinear(128, self.first_out_channels, module_list=[first_model.fc4], no_grow=no_grow)
                self.fc4_1 = GLinear(128, out_channels)
                self.fc4_1_exist = True

    def forward(self, x):
        if not self.in_ram:
            x = x / 255.
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = x.reshape(x.size(0), -1)
        if self.fc1_1_exist:
            adv = F.relu(self.fc1_1(x))
        else:
            adv = F.relu(self.fc1(x))
        adv = F.relu(self.fc2(adv))
        adv = F.relu(self.fc3(adv))
        if self.fc4_1_exist:
            adv = self.fc4_1(adv)
        else:
            adv = self.fc4(adv)
        return adv

    def is_ram(self, task_name):
        return '-ram' in task_name

    def reset_alpha(self):
        if not in_ram:
            self.conv1.reset_alpha()
            self.conv2.reset_alpha()
            self.conv3.reset_alpha()
        if not self.fc1_1_exist: self.fc1.reset_alpha()
        self.fc2.reset_alpha()
        self.fc3.reset_alpha()
        if not self.fc4_1_exist: self.fc4.reset_alpha()


class Action_log():
    def __init__(self, action_space_n):
        self.n = action_space_n
        self.action_log = []
        self.action_count = [0] * self.n

    def reset(self):
        self.action_log = []
        self.action_count = [0] * self.n

    def __call__(self, action):
        self.action_log.append(action)
        self.action_count[action] += 1

    def log(self, raw_log=False):
        action_len = str(len(self.action_log))
        action_count = str(self.action_count)
        if raw_log:
            action_log = str(self.action_log)
            return action_len + '; ' + action_count + '; ' + action_log
        else:
            return action_len + '; ' + action_count


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
    return model_save_path

def check_and_get(model_upath):
    user_machine, model_path = model_upath.split(':')
    if not os.path.exists(model_path):
        subprocess.call(f'scp {model_upath} {model_path}', shell=True)

@hydra.main(config_name='config/atari_dqn_config.yaml')
def main(cfg):
    set_seed(cfg.seed)

    model_save_path = get_model_path(cfg.model_evacuation)
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
        mlflow.log_param('net_version', cfg.net_version)
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
            if cfg.net_version==0:
                q_func = RamNet(dim_state, dim_action)
            elif cfg.net_version==1:
                q_func = RamNet_1(dim_state, dim_action)
            elif cfg.net_version==2:
                q_func = RamNet_2(dim_state, dim_action)
            else: raise IndexError('net_version in [0,2]')
        else:
            if cfg.net_version==0:
                q_func = Net(dim_state, dim_action,
                         dueling=cfg.dueling, high_reso=cfg.high_reso)
            elif cfg.net_version==1:
                q_func = Net_1(dim_state, dim_action,
                         dueling=cfg.dueling, high_reso=cfg.high_reso)
            elif cfg.net_version==2:
                q_func = Net_2(dim_state, dim_action,
                         dueling=cfg.dueling, high_reso=cfg.high_reso)
            else: raise IndexError('net_version in [0,1]')

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
        action_log = Action_log(env.action_space.n)

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
                    # eval_env.render()
                    action_log.reset()
                    done = False

                    while not done:
                        action = agent.act(obs, greedy=True)
                        action_log(action)
                        obs, reward, done, _ = eval_env.step(action)
                        # eval_env.render()

                        # print(reward)
                        total_reward += reward

                    if eval_env.was_real_done:
                        break

                print('Episode:', x, 'Score:', total_reward)
                print(action_log.log())

            exit(0)

        next_eval_cnt = 1
        episode_cnt = 0

        train_start_time = time()

        log_file_name = 'scores.txt'
        model_file_name = 'model_path.txt'
        action_file_name = 'action.txt'
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
                    action_log.reset()
                    done = False

                    while not done:
                        action = agent.act(obs, greedy=True)
                        action_log(action)
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
                    user_name = getpass.getuser()
                    machine_name = os.uname()[1]
                    with open(model_file_name, 'a') as f:
                        f.write(user_name + '@' + machine_name + ':'+ model_name+'\n')
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
                with open(action_file_name, 'a') as f:
                    f.write(action_log.log() + '\n')

        # final evaluation
        total_reward = 0

        while True:
            obs = eval_env.reset()
            action_log.reset()
            done = False

            while not done:
                action = agent.act(obs, greedy=True)
                action_log(action)
                obs, reward, done, _ = eval_env.step(action)

                total_reward += reward

            if eval_env.was_real_done:
                break

        score_steps.append(agent.total_steps)
        scores.append(total_reward)

        model_name = os.path.join(model_save_path, f'final.model')
        torch.save(q_func.state_dict(), model_name)
        user_name = getpass.getuser()
        machine_name = os.uname()[1]
        with open(model_file_name, 'a') as f:
            f.write(user_name + '@' + machine_name + ':' + model_name+'\n')
        mlflow.log_artifact(model_file_name)

        now = time()
        elapsed = now - train_start_time

        log = f'{agent.total_steps} {total_reward} {elapsed:.1f}\n'
        print(log, end='')
        mlflow.log_metric('reward', total_reward, step=agent.total_steps)
        with open(log_file_name, 'a') as f:
            f.write(log)

        with open(action_file_name, 'a') as f:
            f.write(action_log.log() + '\n')
        mlflow.log_artifact(action_file_name)

        duration = np.round(elapsed / 60 / 60, 2)
        mlflow.log_metric('duration', duration)
        print('Complete')
        slack_notify("Complete {} on {}, duration: {}h".format(cfg.name, os.uname()[1], duration))
        env.close()


if __name__ == '__main__':
    main()
