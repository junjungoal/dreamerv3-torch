import argparse
import functools
import os
import pathlib
import sys

os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models
import tools
import envs.wrappers as wrappers
from envs.gym import DWMBufferToEnv
import dill as pickle
from parallel import Parallel, Damy
from dreamer import Dreamer, make_env, make_dataset, make_eval_dataset, count_steps
from dwm.a2c import ActorCritic
from os.path import join

import torch
from torch import nn
from torch import distributions as torchd


to_np = lambda x: x.detach().cpu().numpy()

def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    # step in logger is environmental step
    # logger = tools.Logger(logdir, config.action_repeat * step)
    logger = tools.WandBLogger(logdir, config.action_repeat * step, config.group, config)

    print("Create envs.")
    if config.offline_traindir:
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir
    train_eps = tools.load_episodes(directory, limit=config.dataset_size)
    if config.offline_evaldir:
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir
    eval_eps = tools.load_episodes(directory, limit=1)
    make = lambda mode: make_env(config, mode)
    eval_envs = [make("eval") for _ in range(config.envs)]
    if config.parallel:
        eval_envs = [Parallel(env, "process") for env in eval_envs]
    else:
        eval_envs = [Damy(env) for env in eval_envs]
    acts = eval_envs[0].action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    num_obs = eval_envs[0].observation_space["states"].shape[0]
    
    # Load actor critic from checkpoint
    print("Num states", num_obs, "Num Actions", config.num_actions)
    a2c = ActorCritic(in_dim=num_obs, out_actions=config.num_actions, normalizer=None)
    ac_path = join(config.load_path, f"step-{config.load_step}-ac.pt")
    a2c.load_state_dict(torch.load(ac_path))
    print(f"Loaded actor critic from {ac_path}")
    # print(f"Actor std dev: ", a2c.logstd(torch.Tensor([0.0])).exp().detach().cpu().mean().item() + a2c.min_std)

    # load dataset
    dataset_path = join(config.load_path, f"step-{config.load_step}-dataset.pkl")
    with open(dataset_path, 'rb') as f:
        dwm_dataset = pickle.load(f)
    print(f"Loaded dataset from {dataset_path}")    

    train_envs = [DWMBufferToEnv(dwm_dataset.data_buffer)]

    state = None
    if hasattr(acts, "discrete"):
        random_actor = tools.OneHotDist(
            torch.zeros(config.num_actions).repeat(config.envs, 1)
        )
    else:
        random_actor = torchd.independent.Independent(
            torchd.uniform.Uniform(
                torch.Tensor(acts.low).repeat(config.envs, 1),
                torch.Tensor(acts.high).repeat(config.envs, 1),
            ),
            1,
        )

    def random_agent(o, d, s):
        action = random_actor.sample()
        logprob = random_actor.log_prob(action)
        return {"action": action, "logprob": logprob}, None
    
    print("Adding dataset to replay buffer...")
    episodes = 10 # TODO: REMOVE
    state = tools.simulate(
        random_agent,
        train_envs,
        train_eps,
        config.traindir,
        logger,
        limit=config.dataset_size,
        # steps=config.load_step,
        # episodes=train_envs[0].data_buffer._count,
        episodes=episodes,
        env_is_dataset=True,
        state=state,
    )
    print("Done.")

    print("Simulate agent.")
    train_dataset = make_dataset(train_eps, config)
    eval_dataset = make_eval_dataset(eval_eps, config)
    agent = Dreamer(
        eval_envs[0].observation_space,
        eval_envs[0].action_space,
        config,
        logger,
        train_dataset,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)

    # def eval_policy(latent_state, decoder):
    #     with torch.no_grad():
    #         # decode latent state to observation
    #         obs = decoder(latent_state)

    #         obs = torch.Tensor(obs).to(config.device)
    #         post, prior = agent._world_model(obs)
    #         action, logprob = agent._policy(post)
    #         action = action.cpu().numpy()
    #         logprob = logprob.cpu().numpy()
    #         return {"action": action, "logprob": logprob}, post

    epochs = 200
    epoch_length = 50
    train_steps = 0
    for epoch in range(epochs):
        for _ in range(epoch_length):

            # train world model and world model policy
            agent._train(next(train_dataset))
            train_steps += 1
            

        # gather some real episodes under policy
        eval_policy = functools.partial(agent, training=False)
        tools.simulate(
            eval_policy,
            eval_envs,
            eval_eps,
            config.evaldir,
            logger,
            is_eval=True,
            episodes=config.eval_episode_num,
        )

        # eval prediction errors under policy
        data = next(eval_dataset)
        error_metrics, post = agent._wm.compute_traj_errors(eval_envs[0],data)
        agent_error_metrics = agent._task_behavior.compute_traj_errors(eval_envs[0], post, data, horizon=config.eval_batch_length)
        for key, val in error_metrics.items():
            logger.scalar(key, float(val))
        for key, val in agent_error_metrics.items():
            logger.scalar(key, float(val))

        print("Train_steps: ", train_steps)
        print("Agent error metrics: ")
        print(agent_error_metrics)
        print("\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    main(parser.parse_args(remaining))
