import torch
from torch.distributions import Normal
import numpy as np
import torch.nn as nn
import copy
import torch.nn.functional as F
from torch.optim import Adam
from accel.replay_buffers.replay_buffer import Transition


class CriticNet(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim=256):
        super().__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        return self.linear3(x1)


class ActorNet(torch.nn.Module):
    def __init__(self, n_input, n_output, n_hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)

        self.fc3 = nn.Linear(n_hidden, n_output)
        self.fc4 = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc3(x)
        log_std = torch.clamp(self.fc4(x), -20, 2)
        return mean, log_std


class SAC:
    def __init__(self, device,
                 observation_space,
                 action_space,
                 gamma, replay_buffer, tau=0.005,
                 lr=3e-4,
                 batch_size=256,
                 update_interval=1,
                 target_update_interval=1,
                 load=None):
        self.device = device
        self.n_obs = observation_space.shape[0]
        self.n_actions = action_space.shape[0]
        self.critic1 = CriticNet(self.n_obs, self.n_actions).to(self.device)
        self.critic2 = CriticNet(self.n_obs, self.n_actions).to(self.device)

        self.gamma = gamma
        self.actor = ActorNet(self.n_obs, self.n_actions).to(device)
        if load is not None:
            self.critic1.load_state_dict(torch.load(
                f'{load}/q1.model', map_location=device))
            self.critic2.load_state_dict(torch.load(
                f'{load}/q2.model', map_location=device))
            self.actor.load_state_dict(torch.load(
                f'{load}/pi.model', map_location=device))

        self.q1_optim = Adam(self.critic1.parameters(), lr=lr)
        self.q2_optim = Adam(self.critic2.parameters(), lr=lr)
        self.actor_optim = Adam(self.actor.parameters(), lr=lr)

        self.target_critic1 = copy.deepcopy(self.critic1).to(device)
        self.target_critic2 = copy.deepcopy(self.critic2).to(device)

        self.target_update_interval = target_update_interval

        self.replay_buffer = replay_buffer

        self.total_steps = 0
        self.n_actions = len(action_space.low)
        self.action_scale = torch.tensor(
            (action_space.high - action_space.low) / 2).to(device)
        self.action_bias = torch.tensor(
            (action_space.low + action_space.high) / 2).to(device)

        self.train_cnt = 0

        # that is -|A|
        self.target_entropy = -self.n_actions

        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = 0.2
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=3e-4)

        self.tau = tau
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.prev_target_update_time = 0

    def act(self, obs, greedy=False):
        obs = torch.tensor(obs, device=self.device, dtype=torch.float32)[None]
        if greedy:
            _, _, action = self.try_act(obs)
        else:
            action, _, _ = self.try_act(obs)

        return action.detach().cpu().numpy()[0]

    def try_act(self, obs):
        mean, log_std = self.actor(obs)

        normal = Normal(mean, log_std.exp())
        x_t = normal.rsample()  # latent space
        y_t = torch.tanh(x_t)   # squash
        action = y_t * self.action_scale + self.action_bias

        # enforcing bound
        eps = 1e-6
        log_pi = (normal.log_prob(x_t) -
                  torch.log(self.action_scale * (1-y_t.pow(2)) + eps)).sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_pi, mean

    def calc_target_q(self, obs, action, reward, next_states, valid):
        with torch.no_grad():
            next_action, log_pi, _ = self.try_act(next_states)
            q1 = self.target_critic1(next_states, next_action)
            q2 = self.target_critic2(next_states, next_action)
            q = torch.min(q1, q2) - self.alpha * log_pi

            return reward + valid * self.gamma * q

    def update(self, obs, action, next_obs, reward, valid):
        self.replay_buffer.push(obs, action, next_obs,
                                np.float32(reward), valid)

        self.total_steps += 1

        if self.total_steps % self.update_interval == 0:
            self.train()

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        self.train_cnt += 1

        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.tensor(
            np.array(batch.state, dtype=np.float32), device=self.device)
        action_batch = torch.tensor(
            np.array(batch.action, dtype=np.float32), device=self.device)
        next_state_batch = torch.tensor(
            np.array(batch.next_state, dtype=np.float32), device=self.device)
        reward_batch = torch.tensor(
            np.array(batch.reward, dtype=np.float32), device=self.device).unsqueeze(1)
        valid_batch = torch.tensor(
            np.array(batch.valid, dtype=np.float32), device=self.device).unsqueeze(1)

        target_q = self.calc_target_q(
            state_batch, action_batch, reward_batch, next_state_batch, valid_batch)
        q1 = self.critic1(state_batch, action_batch)
        q2 = self.critic2(state_batch, action_batch)
        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)
        q_loss = q1_loss + q2_loss
        self.q1_optim.zero_grad()
        self.q2_optim.zero_grad()
        q_loss.backward()
        self.q1_optim.step()
        self.q2_optim.step()

        pi, log_pi, _ = self.try_act(state_batch)

        qf1_pi = self.critic1(state_batch, pi)
        qf2_pi = self.critic2(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        # adjust alpha
        alpha_loss = -(self.log_alpha *
                       (self.target_entropy + log_pi).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()

        if self.train_cnt % self.target_update_interval == 0:
            self.soft_update(self.target_critic1, self.critic1)
            self.soft_update(self.target_critic2, self.critic2)
            self.prev_target_update_time = self.total_steps

    def soft_update(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(t.data * (1.0 - self.tau) + s.data * self.tau)
