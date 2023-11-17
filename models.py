import copy
import torch
from torch import nn
import numpy as np
from PIL import ImageColor, Image, ImageDraw, ImageFont

import networks
import tools

to_np = lambda x: x.detach().cpu().numpy()

def map_dict(fn, d):
    """takes a dictionary and applies the function to every element"""
    return type(d)(map(lambda kv: (kv[0], fn(kv[1])), d.items()))

def make_recursive(fn, *argv, **kwargs):
    """ Takes a fn and returns a function that can apply fn on tensor structure
     which can be a single tensor, tuple or a list. """

    def recursive_map(tensors):
        if tensors is None:
            return tensors
        elif isinstance(tensors, list) or isinstance(tensors, tuple):
            return type(tensors)(map(recursive_map, tensors))
        elif isinstance(tensors, dict):
            return type(tensors)(map_dict(recursive_map, tensors))
        elif isinstance(tensors, torch.Tensor) or isinstance(tensors, np.ndarray):
            return fn(tensors, *argv, **kwargs)
        else:
            try:
                return fn(tensors, *argv, **kwargs)
            except Exception as e:
                print("The following error was raised when recursively applying a function:")
                print(e)
                raise ValueError("Type {} not supported for recursive map".format(type(tensors)))

    return recursive_map

def map_recursive(fn, tensors):
    return make_recursive(fn)(tensors)

def map2np(struct):
    """Recursively maps all elements in struct to numpy ndarrays."""
    return map_recursive(to_np, struct)

def listdict2dictlist(LD):
    """ Converts a list of dicts to a dict of lists """

    # Take intersection of keys
    keys = reduce(lambda x,y: x & y, (map(lambda d: d.keys(), LD)))
    return AttrDict({k: [dic[k] for dic in LD] for k in keys})


def dictlist2listdict(DL):
    " Converts a dict of lists to a list of dicts "
    return [dict(zip(DL,t)) for t in zip(*DL.values())]


class RewardEMA(object):
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.values = torch.zeros((2,)).to(device)
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95]).to(device)

    def __call__(self, x):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        self.values = self.alpha * x_quantile + (1 - self.alpha) * self.values
        scale = torch.clip(self.values[1] - self.values[0], min=1.0)
        offset = self.values[0]
        return offset.detach(), scale.detach()


class WorldModel(nn.Module):
    def __init__(self, obs_space, act_space, step, config):
        super(WorldModel, self).__init__()
        self._step = step
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        self.encoder = networks.MultiEncoder(shapes, **config.encoder)
        self.embed_size = self.encoder.outdim
        self.dynamics = networks.RSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_input_layers,
            config.dyn_output_layers,
            config.dyn_rec_depth,
            config.dyn_shared,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_temp_post,
            config.dyn_min_std,
            config.dyn_cell,
            config.unimix_ratio,
            config.initial,
            config.num_actions,
            self.embed_size,
            config.device,
        )
        self.heads = nn.ModuleDict()
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.heads["decoder"] = networks.MultiDecoder(
            feat_size, shapes, **config.decoder
        )
        if config.reward_head == "symlog_disc":
            self.heads["reward"] = networks.MLP(
                feat_size,  # pytorch version
                (255,),
                config.reward_layers,
                config.units,
                config.act,
                config.norm,
                dist=config.reward_head,
                outscale=0.0,
                device=config.device,
            )
        else:
            self.heads["reward"] = networks.MLP(
                feat_size,  # pytorch version
                [],
                config.reward_layers,
                config.units,
                config.act,
                config.norm,
                dist=config.reward_head,
                outscale=0.0,
                device=config.device,
            )
        self.heads["cont"] = networks.MLP(
            feat_size,  # pytorch version
            [],
            config.cont_layers,
            config.units,
            config.act,
            config.norm,
            dist="binary",
            device=config.device,
        )
        for name in config.grad_heads:
            assert name in self.heads, name
        self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )
        self._scales = dict(reward=config.reward_scale, cont=config.cont_scale)

    def _train(self, data):
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        data = self.preprocess(data)

        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                embed = self.encoder(data)
                post, prior = self.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
                kl_free = tools.schedule(self._config.kl_free, self._step)
                dyn_scale = tools.schedule(self._config.dyn_scale, self._step)
                rep_scale = tools.schedule(self._config.rep_scale, self._step)
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                preds = {}
                for name, head in self.heads.items():
                    grad_head = name in self._config.grad_heads
                    feat = self.dynamics.get_feat(post)
                    feat = feat if grad_head else feat.detach()
                    pred = head(feat)
                    if type(pred) is dict:
                        preds.update(pred)
                    else:
                        preds[name] = pred
                losses = {}
                for name, pred in preds.items():
                    like = pred.log_prob(data[name])
                    losses[name] = -torch.mean(like) * self._scales.get(name, 1.0)
                model_loss = sum(losses.values()) + kl_loss
            metrics = self._model_opt(model_loss, self.parameters())

        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl"] = to_np(torch.mean(kl_value))
        with torch.cuda.amp.autocast(self._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(post).entropy())
            )
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
            )
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics

    # this function is called during both rollout and training
    def preprocess(self, obs):
        obs = obs.copy()
        # obs["image"] = torch.Tensor(obs["image"]) / 255.0 - 0.5
        if "discount" in obs:
            obs["discount"] *= self._config.discount
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            obs["discount"] = torch.Tensor(obs["discount"]).unsqueeze(-1)
        # 'is_first' is necesarry to initialize hidden state at training
        assert "is_first" in obs
        # 'is_terminal' is necesarry to train cont_head
        assert "is_terminal" in obs
        obs["cont"] = torch.Tensor(1.0 - obs["is_terminal"]).unsqueeze(-1)
        obs = {k: torch.Tensor(v).to(self._config.device) for k, v in obs.items()}
        return obs

    def video_pred(self, data):
        data = self.preprocess(data)
        embed = self.encoder(data)

        states, _ = self.dynamics.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        recon = self.heads["decoder"](self.dynamics.get_feat(states))["image"].mode()[
            :6
        ]
        reward_post = self.heads["reward"](self.dynamics.get_feat(states)).mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine(data["action"][:6, 5:], init)
        openl = self.heads["decoder"](self.dynamics.get_feat(prior))["image"].mode()
        reward_prior = self.heads["reward"](self.dynamics.get_feat(prior)).mode()
        # observed image is given until 5 steps
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:6] + 0.5
        model = model + 0.5
        error = (model - truth + 1.0) / 2.0

        return torch.cat([truth, model, error], 2)

    def compute_traj_errors(self, env, data):
        env.reset()()
        error_lists  = dict()
        for init_steps in range(1, 50):
            data = self.preprocess(data)
            embed = self.encoder(data)
            states, post = self.dynamics.observe(
                embed[:, 0:init_steps], data["action"][:, 0:init_steps], data["is_first"][:, 0:init_steps]
            )
            recon = self.heads["decoder"](self.dynamics.get_feat(states))["states"].mode()
            init = {k: v[:, -1] for k, v in states.items()}
            prior = self.dynamics.imagine(data["action"][:, init_steps:], init)
            openl = self.heads["decoder"](self.dynamics.get_feat(prior))["states"].mode()
            state_predictions = torch.cat([recon, openl], 1)
            max_steps = data['action'].shape[1] - 1
            
            # for i in range(0, data['action'].shape[1] - num_step - 1, num_step):
            for obs, action, sim_state in zip(state_predictions, data['action'], data['sim_state']):
                init_t = 0
                env.reset()()
                env.set_state(obs[init_t], to_np(sim_state)[init_t])
                for i in range(max_steps):
                    num_step = i + 1
                    open_l_steps = num_step - init_steps + 2
                    act = to_np(action[init_t+i])
                    gt_obs, _, term, _ = env.step({'action': act})()
                    obs_error = np.square(gt_obs['states'] - obs[init_t+num_step].detach().cpu().numpy()).mean()
                    if open_l_steps > 0 and ((open_l_steps <= 20 or open_l_steps % 5 == 0) or num_step == max_steps):
                        if open_l_steps not in error_lists:
                            error_lists[open_l_steps] = [obs_error]
                        else:
                            error_lists[open_l_steps].append(obs_error)

        metrics = dict()
        for step in error_lists:
            metrics.update({
                f"rssm_errors/dynamics_mse_{step:04}_step": np.array(error_lists[step]).mean(),
                f"rssm_errors/dynamics_mse_std_{step:04}_step": np.array(error_lists[step]).std(),
            })
        return metrics, post

class ImagBehavior(nn.Module):
    def __init__(self, config, world_model, stop_grad_actor=True, reward=None):
        super(ImagBehavior, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self._world_model = world_model
        self._stop_grad_actor = stop_grad_actor
        self._reward = reward
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.actor = networks.ActionHead(
            feat_size,
            config.num_actions,
            config.actor_layers,
            config.units,
            config.act,
            config.norm,
            config.actor_dist,
            config.actor_init_std,
            config.actor_min_std,
            config.actor_max_std,
            config.actor_temp,
            outscale=1.0,
            unimix_ratio=config.action_unimix_ratio,
        )
        if config.value_head == "symlog_disc":
            self.value = networks.MLP(
                feat_size,
                (255,),
                config.value_layers,
                config.units,
                config.act,
                config.norm,
                config.value_head,
                outscale=0.0,
                device=config.device,
            )
        else:
            self.value = networks.MLP(
                feat_size,
                [],
                config.value_layers,
                config.units,
                config.act,
                config.norm,
                config.value_head,
                outscale=0.0,
                device=config.device,
            )
        if config.slow_value_target:
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._actor_opt = tools.Optimizer(
            "actor",
            self.actor.parameters(),
            config.actor_lr,
            config.ac_opt_eps,
            config.actor_grad_clip,
            **kw,
        )
        self._value_opt = tools.Optimizer(
            "value",
            self.value.parameters(),
            config.value_lr,
            config.ac_opt_eps,
            config.value_grad_clip,
            **kw,
        )
        if self._config.reward_EMA:
            self.reward_ema = RewardEMA(device=self._config.device)

    def _train(
        self,
        start,
        objective=None,
        action=None,
        reward=None,
        imagine=None,
        tape=None,
        repeats=None,
    ):
        objective = objective or self._reward
        self._update_slow_target()
        metrics = {}

        with tools.RequiresGrad(self.actor):
            with torch.cuda.amp.autocast(self._use_amp):
                imag_feat, imag_state, imag_action = self._imagine(
                    start, self.actor, self._config.imag_horizon, repeats
                )
                reward = objective(imag_feat, imag_state, imag_action)
                actor_ent = self.actor(imag_feat).entropy()
                state_ent = self._world_model.dynamics.get_dist(imag_state).entropy()
                # this target is not scaled
                # slow is flag to indicate whether slow_target is used for lambda-return
                target, weights, base = self._compute_target(
                    imag_feat, imag_state, imag_action, reward, actor_ent, state_ent
                )
                actor_loss, mets = self._compute_actor_loss(
                    imag_feat,
                    imag_state,
                    imag_action,
                    target,
                    actor_ent,
                    state_ent,
                    weights,
                    base,
                )
                metrics.update(mets)
                value_input = imag_feat

        with tools.RequiresGrad(self.value):
            with torch.cuda.amp.autocast(self._use_amp):
                value = self.value(value_input[:-1].detach())
                target = torch.stack(target, dim=1)
                # (time, batch, 1), (time, batch, 1) -> (time, batch)
                value_loss = -value.log_prob(target.detach())
                slow_target = self._slow_value(value_input[:-1].detach())
                if self._config.slow_value_target:
                    value_loss = value_loss - value.log_prob(
                        slow_target.mode().detach()
                    )
                if self._config.value_decay:
                    value_loss += self._config.value_decay * value.mode()
                # (time, batch, 1), (time, batch, 1) -> (1,)
                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])

        metrics.update(tools.tensorstats(value.mode(), "value"))
        metrics.update(tools.tensorstats(target, "target"))
        metrics.update(tools.tensorstats(reward, "imag_reward"))
        if self._config.actor_dist in ["onehot"]:
            metrics.update(
                tools.tensorstats(
                    torch.argmax(imag_action, dim=-1).float(), "imag_action"
                )
            )
        else:
            metrics.update(tools.tensorstats(imag_action, "imag_action"))
        metrics["actor_entropy"] = to_np(torch.mean(actor_ent))
        with tools.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            metrics.update(self._value_opt(value_loss, self.value.parameters()))
        return imag_feat, imag_state, imag_action, weights, metrics


    def compute_traj_errors(self, env, start, data, horizon, policy):
        env.reset()()
        init_steps = 10
        max_init_steps = 50
        max_eval = 128
        batch, _, _ = data["action"].shape
        data = self._world_model.preprocess(data)
        embed = self._world_model.encoder(data)
        states, post = self._world_model.dynamics.observe(
            embed[:, :max_init_steps], data["action"][:, :max_init_steps], data["is_first"][:, :max_init_steps]
        )
        init = {k: v[:, :] for k, v in states.items()}
        print("Init shape", init['stoch'].shape)
        
        with torch.cuda.amp.autocast(self._use_amp):
            imag_feat, imag_state, imag_action = self._imagine(
                init, policy, horizon, None, reload_policy=True
            )
            init_feat = self._world_model.dynamics.get_feat(init)
            imag_feat = imag_feat.permute((1, 0, 2))
            imag_recon = self._world_model.heads["decoder"](imag_feat)["states"].mode()
        imag_action = imag_action.permute((1, 0, 2))
        max_steps = data['action'].shape[1] - 1
        error_lists  = dict()

        max_steps = horizon - 1
        error_lists_per_cond  = dict()
        for _ in range(max_init_steps):
            error_lists_per_cond[init_steps] = dict()
        # print("imagine action shape", imag_action.shape)
        # print("Imag recon state shape", imag_recon.shape)
        imag_action = imag_action.reshape(batch, -1, imag_action.shape[-2], imag_action.shape[-1])
        imag_recon = imag_recon.reshape(batch, -1, imag_recon.shape[-2], imag_recon.shape[-1])
        for obs, action, sim_state, real_act in zip(imag_recon, imag_action, data['sim_state'], data["action"]):
            for init_steps in range(1, max_init_steps):
                init_t = 0
                env.reset()()
                if torch.is_tensor(sim_state[init_steps]):
                    init_sim_state = to_np(sim_state[init_steps])
                else:
                    init_sim_state = sim_state[init_t]
                env.set_state(map2np(obs[init_t]), init_sim_state)

                for openl_steps in range(horizon):
                    act = to_np(action[init_steps, openl_steps])
                    # print("Action", act)
                    gt_obs, _, term, _ = env.step({'action': act})()
                    obs_error = np.square(gt_obs['states'] - to_np(obs[init_steps, openl_steps])).mean()
                    if ((openl_steps + 1) <= 20 or (openl_steps + 1) % 5 == 0) or (openl_steps + 1) == max_steps:
                        if (openl_steps + 1) not in error_lists:
                            error_lists[(openl_steps + 1)] = [obs_error]
                        else:
                            error_lists[(openl_steps + 1)].append(obs_error)
                            
                        if (openl_steps + 1) not in error_lists_per_cond[init_steps]:
                            error_lists_per_cond[init_steps][(openl_steps + 1)] = [obs_error]
                        else:
                            error_lists_per_cond[init_steps][(openl_steps + 1)].append(obs_error)
        metrics = dict()
        for step in error_lists:
            metrics.update({
                f"agent_errors/dynamics_mse_{step:04}_step": np.array(error_lists[step]).mean(),
                f"agent_errors/dynamics_mse_std_{step:04}_step": np.array(error_lists[step]).std(),
            })

        detailed_metrics = dict()
        for cond_steps in error_lists_per_cond:
            for step in error_lists_per_cond[cond_steps]:
                detailed_metrics.update({
                    f"agent_errors/{cond_steps}_cond_steps/dynamics_mse_{step:04}_step": np.array(error_lists_per_cond[cond_steps][step]).mean(),
                    f"agent_errors/{cond_steps}_cond_steps/dynamics_mse_std_{step:04}_step": np.array(error_lists_per_cond[cond_steps][step]).std(),
                })

        return metrics, detailed_metrics

    def _imagine(self, start, policy, horizon, repeats=None, reload_policy=False):
        dynamics = self._world_model.dynamics
        if repeats:
            raise NotImplemented("repeats is not implemented in this version")
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            inp = feat.detach() if self._stop_grad_actor else feat
            if reload_policy:
                recon = self._world_model.heads["decoder"](inp)
                action = policy(recon['states'].mode(), normed_input=False).sample()
            else:
                action = policy(inp).sample()
            succ = dynamics.img_step(state, action, sample=self._config.imag_sample)
            return succ, feat, action

        succ, feats, actions = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None)
        )
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}
        if repeats:
            raise NotImplemented("repeats is not implemented in this version")

        return feats, states, actions

    def _compute_target(
        self, imag_feat, imag_state, imag_action, reward, actor_ent, state_ent
    ):
        if "cont" in self._world_model.heads:
            inp = self._world_model.dynamics.get_feat(imag_state)
            discount = self._config.discount * self._world_model.heads["cont"](inp).mean
        else:
            discount = self._config.discount * torch.ones_like(reward)
        if self._config.future_entropy and self._config.actor_entropy() > 0:
            reward += self._config.actor_entropy() * actor_ent
        if self._config.future_entropy and self._config.actor_state_entropy() > 0:
            reward += self._config.actor_state_entropy() * state_ent
        value = self.value(imag_feat).mode()
        target = tools.lambda_return(
            reward[1:],
            value[:-1],
            discount[1:],
            bootstrap=value[-1],
            lambda_=self._config.discount_lambda,
            axis=0,
        )
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        return target, weights, value[:-1]

    def _compute_actor_loss(
        self,
        imag_feat,
        imag_state,
        imag_action,
        target,
        actor_ent,
        state_ent,
        weights,
        base,
    ):
        metrics = {}
        inp = imag_feat.detach() if self._stop_grad_actor else imag_feat
        policy = self.actor(inp)
        actor_ent = policy.entropy()
        # Q-val for actor is not transformed using symlog
        target = torch.stack(target, dim=1)
        if self._config.reward_EMA:
            offset, scale = self.reward_ema(target)
            normed_target = (target - offset) / scale
            normed_base = (base - offset) / scale
            adv = normed_target - normed_base
            metrics.update(tools.tensorstats(normed_target, "normed_target"))
            values = self.reward_ema.values
            metrics["EMA_005"] = to_np(values[0])
            metrics["EMA_095"] = to_np(values[1])

        if self._config.imag_gradient == "dynamics":
            actor_target = adv
        elif self._config.imag_gradient == "reinforce":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
        elif self._config.imag_gradient == "both":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
            mix = self._config.imag_gradient_mix()
            actor_target = mix * target + (1 - mix) * actor_target
            metrics["imag_gradient_mix"] = mix
        else:
            raise NotImplementedError(self._config.imag_gradient)
        if not self._config.future_entropy and (self._config.actor_entropy() > 0):
            actor_entropy = self._config.actor_entropy() * actor_ent[:-1][:, :, None]
            actor_target += actor_entropy
        if not self._config.future_entropy and (self._config.actor_state_entropy() > 0):
            state_entropy = self._config.actor_state_entropy() * state_ent[:-1]
            actor_target += state_entropy
            metrics["actor_state_entropy"] = to_np(torch.mean(state_entropy))
        actor_loss = -torch.mean(weights[:-1] * actor_target)
        return actor_loss, metrics

    def _update_slow_target(self):
        if self._config.slow_value_target:
            if self._updates % self._config.slow_target_update == 0:
                mix = self._config.slow_target_fraction
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1
