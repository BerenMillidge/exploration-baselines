# pylint: disable=not-callable
# pylint: disable=no-member

import numpy as np

import torch

import os
import sys
from copy import deepcopy

from baselines.algos.max import (
    Buffer,
    Model,
    JensenRenyiDivergenceUtilityMeasure,
    TransitionNormalizer,
    Imagination,
    SAC,
)
from baselines.envs import TorchEnv, const

max_episode_len = 500

n_eval_episodes = 3
env_noise_stdev = 0.02
n_warm_up_steps = 256
n_exploration_steps = 20000
eval_freq = 2000
data_buffer_size = n_exploration_steps + 1

env = TorchEnv(const.SPARSE_MOUNTAIN_CAR, max_episode_len)
d_state = env.state_dims[0]
d_action = env.action_dims[0]

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

ensemble_size = 32
n_hidden = 512
n_layers = 4
non_linearity = "swish"

exploring_model_epochs = 50
evaluation_model_epochs = 200
batch_size = 256
learning_rate = 1e-3
normalize_data = True
weight_decay = 0
training_noise_stdev = 0
grad_clip = 5

policy_actors = 128
policy_warm_up_episodes = 3
policy_replay_size = int(1e7)
policy_batch_size = 4096
policy_reactive_updates = 100
policy_active_updates = 1

policy_n_hidden = 256
policy_lr = 1e-3
policy_gamma = 0.99
policy_tau = 0.005

buffer_reuse = True
use_best_policy = False

policy_explore_horizon = 50
policy_explore_episodes = 50
policy_explore_alpha = 0.02

policy_exploit_horizon = 100
policy_exploit_episodes = 250
policy_exploit_alpha = 0.4

exploration_mode = "active"
model_train_freq = 25
utility_measure = "renyi_div"
renyi_decay = 0.1
utility_action_norm_penalty = 0
action_noise_stdev = 0

max_exploration = True
exploitation = True
render = True


def get_model():
    model = Model(
        d_action=d_action,
        d_state=d_state,
        ensemble_size=ensemble_size,
        n_hidden=n_hidden,
        n_layers=n_layers,
        non_linearity=non_linearity,
        device=device,
    )
    return model


def get_buffer():
    return Buffer(
        d_action=d_action,
        d_state=d_state,
        ensemble_size=ensemble_size,
        buffer_size=data_buffer_size,
    )


def get_optimizer_factory():
    return lambda params: torch.optim.Adam(
        params, lr=learning_rate, weight_decay=weight_decay
    )


def get_utility_measure():
    return JensenRenyiDivergenceUtilityMeasure(
        decay=renyi_decay, action_norm_penalty=utility_action_norm_penalty
    )


def train_epoch(model, buffer, optimizer):
    losses = []
    for tr_states, tr_actions, tr_state_deltas in buffer.train_batches(
        batch_size=batch_size
    ):
        optimizer.zero_grad()
        loss = model.loss(
            tr_states,
            tr_actions,
            tr_state_deltas,
            training_noise_stdev=training_noise_stdev,
        )
        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip)
        optimizer.step()

    return np.mean(losses)


def fit_model(buffer, n_epochs, step_num, mode):
    model = get_model()
    model.setup_normalizer(buffer.normalizer)
    optimizer = get_optimizer_factory()(model.parameters())

    print(f"step: {step_num}\t training")

    for epoch_i in range(1, n_epochs + 1):
        tr_loss = train_epoch(model=model, buffer=buffer, optimizer=optimizer)
        print(f"epoch: {epoch_i:3d} training_loss: {tr_loss:.2f}")

    print(
        f"step: {step_num}\t training done for {n_epochs} epochs, final loss: {np.round(tr_loss, 3)}"
    )

    if mode == "explore":
        print("explore_loss [{}] step [{}]".format(tr_loss, step_num))
    elif mode == "exploit":
        print("exploit_loss [{}] step [{}]".format(tr_loss, step_num))

    return model

def get_policy(
    buffer,
    model,
    measure,
    mode,
):

    print("... getting fresh agent")

    policy_alpha = policy_explore_alpha if mode == "explore" else policy_exploit_alpha

    agent = SAC(
        d_state=d_state,
        d_action=d_action,
        replay_size=policy_replay_size,
        batch_size=policy_batch_size,
        n_updates=policy_active_updates,
        n_hidden=policy_n_hidden,
        gamma=policy_gamma,
        alpha=policy_alpha,
        lr=policy_lr,
        tau=policy_tau,
    )

    agent = agent.to(device)
    agent.setup_normalizer(model.normalizer)

    if not buffer_reuse:
        return agent

    print("... transferring exploration buffer")

    size = len(buffer)
    for i in range(0, size, 1024):
        j = min(i + 1024, size)
        s, a = buffer.states[i:j], buffer.actions[i:j]
        ns = buffer.states[i:j] + buffer.state_deltas[i:j]
        r = buffer.rewards[i:j]
        s, a, ns = s.to(device), a.to(device), ns.to(device)
        with torch.no_grad():
            mu, var = model.forward_all(s, a)

        if measure is not None:
            r = measure(s, a, ns, mu, var, model)
            
        agent.replay.add(s, a, r, ns)

    print("... transferred exploration buffer")

    return agent


def get_action(mdp, agent):
    current_state = mdp.reset()
    actions = agent(current_state, eval=True)
    action = actions[0].detach().data.cpu().numpy()
    policy_value = torch.mean(agent.get_state_value(current_state)).item()
    return action, mdp, agent, policy_value


def act(
    state,
    agent,
    mdp,
    buffer,
    model,
    measure,
    mode,
):

    if mode == "explore":
        policy_horizon = policy_explore_horizon
        policy_episodes = policy_explore_episodes
    elif mode == "exploit":
        policy_horizon = policy_exploit_horizon
        policy_episodes = policy_exploit_episodes
    else:
        raise Exception("invalid acting mode")

    fresh_agent = True if agent is None else False

    if mdp is None:
        mdp = Imagination(
            horizon=policy_horizon, n_actors=policy_actors, model=model, measure=measure
        )

    if fresh_agent:
        agent = get_policy(buffer=buffer, model=model, measure=measure, mode=mode)

    # update state to current env state
    mdp.update_init_state(state)

    if not fresh_agent:
        # agent is not stale, use it to return action
        return get_action(mdp, agent)

    # reactive updates
    for _ in range(policy_reactive_updates):
        agent.update()

    # active updates
    perform_active_exploration = mode == "explore" and exploration_mode == "active"
    perform_exploitation = mode == "exploit"
    if perform_active_exploration or perform_exploitation:

        # to be fair to reactive methods, clear real env data in SAC buffer, to prevent further gradient updates from it.
        # for active exploration, only effect of on-policy training remains
        if perform_active_exploration:
            agent.reset_replay()

        ep_returns = []
        best_return, best_params = -np.inf, deepcopy(agent.state_dict())
        for ep_i in range(policy_episodes):
            warm_up = (
                True if ((ep_i < policy_warm_up_episodes) and fresh_agent) else False
            )
            ep_return = agent.episode(
                env=mdp, warm_up=warm_up
            )
            ep_returns.append(ep_return)

            if use_best_policy and ep_return > best_return:
                best_return, best_params = ep_return, deepcopy(agent.state_dict())

            step_return = ep_return / policy_horizon
            print(f"\tep: {ep_i}\taverage step return: {np.round(step_return, 3)}")

        if use_best_policy:
            agent.load_state_dict(best_params)

        if mode == "explore" and len(ep_returns) >= 3:
            first_return = ep_returns[0]
            last_return = max(ep_returns) if use_best_policy else ep_returns[-1]
            print(
                "policy_improvement_first_return {}".format(
                    first_return / policy_horizon
                )
            )
            print(
                "policy_improvement_second_return {}".format(
                    ep_returns[1] / policy_horizon
                )
            )
            print(
                "policy_improvement_last_return {}".format(last_return / policy_horizon)
            )

    return get_action(mdp, agent)

def evaluate_task(env, model, buffer):
    state = env.reset()
    ep_return = 0
    agent = None
    mdp = None
    done = False
    novelty = []
    while not done:
        action, mdp, agent, _ = act(state=state, agent=agent, mdp=mdp, buffer=buffer, measure=None, model=model, mode='exploit')
        next_state, reward, done = env.step(action)

        print(f'reward: {reward:5.2f} action: {action}')
        ep_return += reward

        if render:
            env.render()

        state = next_state

    env.close()

    return ep_return, np.mean(novelty)

def evaluate_tasks(buffer, step_num):
    model = fit_model(buffer=buffer, n_epochs=evaluation_model_epochs, step_num=step_num, mode='exploit')

    average_returns = []
    task_returns = []
    task_novelty = []
    for ep_idx in range(1, n_eval_episodes + 1):
        ep_return, ep_novelty = evaluate_task(env=env, model=model, buffer=buffer)

        print(f"task: \tepisode: {ep_idx}\treward: {np.round(ep_return, 4)}")
        task_returns.append(ep_return)
        task_novelty.append(ep_novelty)

        average_returns.append(task_returns)
        print(f"task: taverage return: {np.round(np.mean(task_returns), 4)}")

    average_return = np.mean(average_returns)
    print("average_return {} / {}".format(average_return, step_num))
    return average_return

def do_max_exploration():

    buffer = get_buffer()
    exploration_measure = get_utility_measure()

    if normalize_data:
        normalizer = TransitionNormalizer()
        buffer.setup_normalizer(normalizer)

    model = None
    mdp = None
    agent = None
    average_performances = []

    state = env.reset()

    for step_num in range(1, n_exploration_steps + 1):
        print("> step number {}".format(step_num))

        if step_num > n_warm_up_steps:
            action, mdp, agent, policy_value = act(
                state=state,
                agent=agent,
                mdp=mdp,
                buffer=buffer,
                model=model,
                measure=exploration_measure,
                mode="explore",
            )

            print("action_norm {} / {}".format(np.sum(np.square(action)), step_num))
            print("exploration_policy_value {}/ {}".format(policy_value, step_num))

            if action_noise_stdev:
                action = action + np.random.normal(
                    scale=action_noise_stdev, size=action.shape
                )
        else:
            action = env.sample_action()

        next_state, reward, done = env.step(action)
        buffer.add(state, action, next_state, reward)

        if render:
            env.render()

        if done:
            print(f"step: {step_num}\tepisode complete")
            agent = None
            mdp = None
            next_state = env.reset()

        state = next_state

        if step_num < n_warm_up_steps:
            continue

        episode_done = done
        train_at_end_of_episode = model_train_freq is np.inf
        time_to_update = (step_num % model_train_freq) == 0
        just_finished_warm_up = step_num == n_warm_up_steps
        if (
            (train_at_end_of_episode and episode_done)
            or time_to_update
            or just_finished_warm_up
        ):  
            model = fit_model(
                buffer=buffer,
                n_epochs=exploring_model_epochs,
                step_num=step_num,
                mode="explore",
            )

            mdp = None
            agent = None

        time_to_evaluate = (step_num % eval_freq) == 0
        if time_to_evaluate or just_finished_warm_up:
            print("evaluating performance")
            average_performance = evaluate_tasks(buffer=buffer, step_num=step_num)
            average_performances.append(average_performance)


    return max(average_performances)


def main(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    do_max_exploration()


if __name__ == "__main__":
    main(1)