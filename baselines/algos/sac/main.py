from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import argparse
from buffer import Buffer
from models import ActorCritic
from baselines.envs import TorchEnv


def sac(args):
    #set seed if non default is entered
    if args.seed != -1:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    env, test_env = TorchEnv(args.env_name, args.max_ep_len), TorchEnv(args.env_name, args.max_ep_len)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_limit = env.action_space.high[0]
    # Create actor-critic module and target networks
    ac = ActorCritic(state_dim,action_dim,action_limit,args.hidden_size,args.gamma,args.alpha,device=args.device)
    ac_targ = deepcopy(ac)
    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False
    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())
    # Experience buffer
    buffer = Buffer(state_dim,action_dim, buffer_size=args.buffer_size,device=args.device)
    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=args.lr)
    q_optimizer = Adam(q_params, lr=args.lr)

    def update(data):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q = ac.compute_loss_q(data,ac_targ)
        loss_q.backward()
        q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort computing gradients for them during the policy learning step.
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi = ac.compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params:
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(args.polyak)
                p_targ.data.add_((1 - args.polyak) * p.data)

    def test_agent(deterministic=True):
        for j in range(args.num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == args.max_ep_len)):
                # Take deterministic actions at test time
                o, r, d = test_env.step(ac.act(torch.as_tensor(o, dtype=torch.float32).to(args.device), deterministic))
                ep_ret += r
                ep_len += 1

    # Prepare for interaction with environment
    total_steps = args.steps_per_epoch * args.epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        # Until start_steps have elapsed, randomly sample actions from a uniform distribution for better exploration. Afterwards, use the learned policy.
        if t > args.start_steps:
            a = ac.act(torch.as_tensor(o, dtype=torch.float32).to(args.device))
        else:
            a = env.action_space.sample()
        # Step the env
        o2, r, d = env.step(a)
        if args.render_env:
            env.render()
        ep_ret += r
        ep_len += 1
        # Ignore the "done" signal if it comes from hitting the time horizon (that is, when it's an artificial terminal signal that isn't based on the agent's state)
        d = False if ep_len==args.max_ep_len else d
        # Store experience to replay buffer
        buffer.add(o,a,r,o2,d)
        o = o2
        # End of trajectory handling
        if d or (ep_len == args.max_ep_len):
            print("EPISODE REWARD: ", ep_ret)
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= args.update_after and t % args.update_every == 0:
            batch_generator = buffer.get_train_batches(args.batch_size)
            for j in range(args.update_every):
              #my_batch = my_buffer.get_train_batches(args.batch_size).__next__()
              try:
                    batch = batch_generator.__next__()
              except:
                    batch_generator = buffer.get_train_batches(args.batch_size)
                    batch = batch_generator.__next__()
              update(batch)

        # End of epoch handling
        if (t+1) % args.steps_per_epoch == 0:
            epoch = (t+1) // args.steps_per_epoch
            # Test the performance of the deterministic version of the agent.
            test_agent()


def boolcheck(x):
        return str(x).lower() in ["true", "1", "yes"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    parser.add_argument("--env_name", type=str, default="Pendulum-v0")
    parser.add_argument("--max_ep_len", type=int, default=1000)
    parser.add_argument("--seed",type=int, default=-1)
    parser.add_argument("--hidden_size",type=int, default=256)
    parser.add_argument("--steps_per_epoch", type=int, default=4000)
    parser.add_argument("--epochs",type=int,default=100)
    parser.add_argument("--buffer_size",type=int,default=int(1e6))
    parser.add_argument("--gamma",type=float,default=0.99)
    parser.add_argument("--polyak",type=float,default=0.995)
    parser.add_argument("--lr",type=float,default=1e-3)
    parser.add_argument("--alpha",type=float,default=0.2)
    parser.add_argument("--args.batch_size",type=int,default=100)
    parser.add_argument("--start_steps",type=int,default=10000)
    parser.add_argument("--update_after",type=int,default=1000)
    parser.add_argument("--update_every",type=int,default=50)
    parser.add_argument("--num_test_episodes",type=int,default=10)
    parser.add_argument("--batch_size",type=int,default=100)
    parser.add_argument("--render_env",type=boolcheck,default=False)
    args =parser.parse_args()
    args.device = DEVICE
    sac(args)
