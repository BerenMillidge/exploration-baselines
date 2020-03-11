
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class ActorModel(nn.Module):
  def __init__(self, state_dim, action_dim, hidden_size,action_limit,act_fn,device="cpu"):
    super().__init__()
    self.state_dim = state_dim
    self.action_dim = action_dim
    self.hidden_size = hidden_size
    self.action_limit = action_limit
    self.act_fn = act_fn
    self.LOG_STD_MAX = 2
    self.LOG_STD_MIN = -20
    self.device=device

    self.fc1 = nn.Linear(state_dim, hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.fc3 = nn.Linear(hidden_size, hidden_size)
    self.mu_layer = nn.Linear(hidden_size, action_dim)
    self.log_std_layer = nn.Linear(hidden_size, action_dim)
    self.to(self.device)


  def forward(self, state, deterministic=False, with_logprob = True):
    out = self.act_fn(self.fc1(state))
    out = self.act_fn(self.fc2(out))
    out = self.fc3(out)
    mu = self.mu_layer(out)
    log_std = self.log_std_layer(out)
    log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
    std = torch.exp(log_std)
    action_distribution = Normal(mu, std)
    if deterministic:
      action = mu
    else:
      action = action_distribution.rsample()

    if with_logprob:
        # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
        # NOTE: The correction formula is a little bit magic. To get an understanding
        # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
        # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
        # Try deriving it yourself as a (very difficult) exercise. :)
        logp_a = action_distribution.log_prob(action).sum(axis=-1)
        logp_a -= (2*(np.log(2) - action - F.softplus(-2*action))).sum(axis=1)
    else:
      logp_a = None

    action = torch.tanh(action)
    action = action * self.action_limit
    return action, logp_a

class ValueModel(nn.Module):
  def __init__(self, state_dim, action_dim, hidden_size, act_fn,device="cpu"):
    super().__init__()
    self.state_dim=state_dim
    self.action_dim = action_dim
    self.hidden_size = hidden_size
    self.act_fn = act_fn
    self.device=  device

    self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.fc3 = nn.Linear(hidden_size, 1)
    self.to(self.device)

  def forward(self, state, action):
    out = self.act_fn(self.fc1(torch.cat([state,action],dim=-1)))
    out = self.act_fn(self.fc2(out))
    out = self.fc3(out)
    return torch.squeeze(out,-1)

class ActorCritic(nn.Module):
  def __init__(self, state_dim,action_dim,action_limit, hidden_size,gamma,alpha, act_fn = F.relu,device="cpu"):
    super().__init__()
    self.state_dim = state_dim
    self.action_dim = action_dim
    self.action_limit = action_limit
    self.hidden_size = hidden_size
    self.act_fn = act_fn
    self.gamma = gamma
    self.alpha = alpha
    self.device=device

    self.pi = ActorModel(self.state_dim, self.action_dim, self.hidden_size,self.action_limit, self.act_fn,device=self.device)
    self.q1 = ValueModel(self.state_dim, self.action_dim, self.hidden_size, self.act_fn,device=self.device)
    self.q2 = ValueModel(self.state_dim, self.action_dim, self.hidden_size, self.act_fn,device=self.device)

  def act(self, state, deterministic=False):
    with torch.no_grad():
      a,_ = self.pi(state, deterministic, False)
      return a.cpu().numpy()

  def compute_loss_q(self,data,ac_targ):
      o, a, r, odelta, d,o2 = data
      r = r.squeeze(-1)
      d = d.squeeze(-1)

      q1 = self.q1(o,a)
      q2 = self.q2(o,a)

      # Bellman backup for Q functions
      with torch.no_grad():
          # Target actions come from *current* policy
          a2, logp_a2 = self.pi(o2)

          # Target Q-values
          q1_pi_targ = ac_targ.q1(o2, a2)
          q2_pi_targ = ac_targ.q2(o2, a2)
          q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
          backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

      # MSE loss against Bellman backup
      loss_q1 = ((q1 - backup)**2).mean()
      loss_q2 = ((q2 - backup)**2).mean()
      loss_q = loss_q1 + loss_q2

      return loss_q

  def compute_loss_pi(self,data):
      #o = data['obs']
      o, a, r, odelta, d,o2 = data
      pi, logp_pi = self.pi(o)
      q1_pi = self.q1(o, pi)
      q2_pi = self.q2(o, pi)
      q_pi = torch.min(q1_pi, q2_pi)

      # Entropy-regularized policy loss
      loss_pi = (self.alpha * logp_pi - q_pi).mean()

      return loss_pi
