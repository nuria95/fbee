# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=unused-import
import pdb
import copy
import math
import logging
import dataclasses
from collections import OrderedDict
import typing as tp
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
import omegaconf
from dm_env import specs

from url_benchmark import utils
from url_benchmark.in_memory_replay_buffer import ReplayBuffer
from url_benchmark.dmc import TimeStep
from url_benchmark import goals as _goals
from .fb_modules import IdentityMap
from .fb_modules import Actor, ForwardMap, BackwardMap, EnsembleMLP, HighLevelActor, Critic, RNDCuriosity

MetaDict = tp.Mapping[str, np.ndarray]
logger = logging.getLogger(__name__)

""" 
Code adapted from  https://github.com/facebookresearch/controllable_agent
"""

@dataclasses.dataclass
class FBDDPGAgentConfig:
    # @package agent
    _target_: str = "url_benchmark.agent.fb_ddpg.FBDDPGAgent"
    name: str = "fb_ddpg"
    obs_type: str = omegaconf.MISSING  # to be specified later
    obs_shape: tp.Tuple[int, ...] = omegaconf.MISSING  # to be specified later
    # to be specified later
    action_shape: tp.Tuple[int, ...] = omegaconf.MISSING
    device: str = omegaconf.II("device")  # ${device}
    lr: float = 1e-4
    lr_coef: float = 1
    fb_target_tau: float = 0.01  # 0.001-0.01
    update_every_steps: int = 2
    use_tb: bool = omegaconf.II("use_tb")  # ${use_tb}
    use_wandb: bool = omegaconf.II("use_wandb")  # ${use_wandb}
    num_expl_steps: int = omegaconf.MISSING  # ???  # to be specified later
    num_inference_steps: int = 5120
    hidden_dim: int = 1024   # 128, 2048
    backward_hidden_dim: int = 526   # 512
    feature_dim: int = 512   # 128, 1024
    z_dim: int = 50  # 100
    stddev_schedule: str = "0.2"  # "linear(1,0.2,200000)" #
    stddev_clip: float = 0.3  # 1
    update_z_every_step: int = 300
    update_z_proba: float = 1.0
    batch_size: int = 1024  # 512
    init_fb: bool = True
    goal_space: tp.Optional[str] = omegaconf.II("goal_space")
    ortho_coef: float = 1.0  # 0.01-10
    debug: bool = False
    future_ratio: float = 0.0
    mix_ratio: float = 0.5  # 0-1
    rand_weight: bool = False  # True, False
    preprocess: bool = True
    norm_z: bool = True
    add_trunk: bool = False
    uncertainty: bool = omegaconf.II("uncertainty")
    one_target: bool = False
    n_ensemble: int = 5
    sampling: bool = False  # if False: sample and pick the argmax, otw policy
    myopic: bool = True  #  kept for consistency with configs, not used anymore
    num_z_samples: int = 100
    # rnd params
    rnd_coeff: float = 0.5
    rnd: bool = False
    rnd_embed_dim: int = 100
    f_uncertainty: bool = False


cs = ConfigStore.instance()
cs.store(group="agent", name="fb_ddpg", node=FBDDPGAgentConfig)


class FBDDPGAgent:

    # pylint: disable=unused-argument
    def __init__(self,
                 **kwargs: tp.Any
                 ):
        cfg = FBDDPGAgentConfig(**kwargs)
        self.cfg = cfg
        assert len(cfg.action_shape) == 1
        self.action_dim = cfg.action_shape[0]

        self.aug = nn.Identity()
        self.encoder = nn.Identity()
        self.obs_dim = cfg.obs_shape[0]
        if cfg.feature_dim < self.obs_dim:
            logger.warning(
                f"feature_dim {cfg.feature_dim} should not be smaller that obs_dim {self.obs_dim}")
        assert not (
            cfg.rnd and cfg.uncertainty), 'Cannot use both RND and uncertainty'

        goal_dim = self.obs_dim
        if cfg.goal_space is not None:
            goal_dim = _goals.get_goal_space_dim(cfg.goal_space)
        if cfg.z_dim < goal_dim:
            logger.warning(
                f"z_dim {cfg.z_dim} should not be smaller that goal_dim {goal_dim}")
        # create the network
        self.actor = Actor(self.obs_dim, cfg.z_dim, self.action_dim,
                           cfg.feature_dim, cfg.hidden_dim,
                           preprocess=cfg.preprocess, add_trunk=self.cfg.add_trunk).to(cfg.device)
        if self.cfg.uncertainty and not self.cfg.sampling:
            self.high_expl_actor = HighLevelActor(
                self.obs_dim, cfg.z_dim, cfg.hidden_dim).to(cfg.device)

        f_dict = {'obs_dim': self.obs_dim, 'z_dim': cfg.z_dim, 'action_dim': self.action_dim,
                  'feature_dim': cfg.feature_dim, 'hidden_dim': cfg.hidden_dim,
                  'preprocess': cfg.preprocess, 'add_trunk': self.cfg.add_trunk}
        if not cfg.uncertainty:
            self.forward_net = ForwardMap(**f_dict).to(cfg.device)
        else:
            self.forward_net = EnsembleMLP(
                f_dict, n_ensemble=self.cfg.n_ensemble, device=cfg.device)
        if cfg.debug:
            self.backward_net: nn.Module = IdentityMap().to(cfg.device)
            self.backward_target_net: nn.Module = IdentityMap().to(cfg.device)
        else:
            self.backward_net = BackwardMap(
                goal_dim, cfg.z_dim, cfg.backward_hidden_dim, norm_z=cfg.norm_z).to(cfg.device)
            self.backward_target_net = BackwardMap(goal_dim,
                                                   cfg.z_dim, cfg.backward_hidden_dim, norm_z=cfg.norm_z).to(cfg.device)
        # build up the target network
        if not cfg.uncertainty:
            self.forward_target_net = ForwardMap(**f_dict).to(cfg.device)
        else:
            self.forward_target_net = EnsembleMLP(
                f_dict, n_ensemble=self.cfg.n_ensemble, device=cfg.device)
        # load the weights into the target networks
        self.forward_target_net.load_state_dict(self.forward_net.state_dict())
        if cfg.one_target and cfg.uncertainty:
            # Initialise target params for each ensemble member to the mean of the forward_net
            utils.soft_update_params_mean(
                self.forward_net, self.forward_target_net, tau=1.0)
        self.backward_target_net.load_state_dict(
            self.backward_net.state_dict())
        # optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        if self.cfg.uncertainty and not self.cfg.sampling:
            self.high_expl_actor_opt = torch.optim.Adam(
                self.high_expl_actor.parameters(), lr=cfg.lr)
        self.fb_opt = torch.optim.Adam([{'params': self.forward_net.parameters()},  # type: ignore
                                        {'params': self.backward_net.parameters(), 'lr': cfg.lr_coef * cfg.lr}],
                                       lr=cfg.lr)

        if self.cfg.rnd:
            self.Q_rnd = Critic(self.obs_dim, self.action_dim,
                                cfg.z_dim, self.cfg.hidden_dim).to(cfg.device)
            self.target_Q_rnd = Critic(
                self.obs_dim, self.action_dim, cfg.z_dim, self.cfg.hidden_dim).to(cfg.device)
            self.target_Q_rnd.load_state_dict(self.Q_rnd.state_dict())
            self.Q_rnd_opt = torch.optim.Adam(
                self.Q_rnd.parameters(), lr=cfg.lr)
            self.rnd_module = RNDCuriosity(
                self.obs_dim, self.cfg.hidden_dim, self.cfg.rnd_embed_dim, self.cfg.lr, cfg.device)

        self.train()
        self.forward_target_net.train()
        self.backward_target_net.train()

    def train(self, training: bool = True) -> None:
        self.training = training
        for net in [self.encoder, self.actor, self.forward_net, self.backward_net]:
            net.train(training)

    def init_from(self, other) -> None:
        # copy parameters over
        names = ["encoder", "actor"]
        if self.cfg.init_fb:
            names += ["forward_net", "backward_net",
                      "backward_target_net", "forward_target_net"]
        for name in names:
            utils.hard_update_params(getattr(other, name), getattr(self, name))
        for key, val in self.__dict__.items():
            if isinstance(val, torch.optim.Optimizer):
                val.load_state_dict(copy.deepcopy(
                    getattr(other, key).state_dict()))

    def get_goal_meta(self, goal_array: np.ndarray) -> MetaDict:
        desired_goal = torch.tensor(
            goal_array).unsqueeze(0).to(self.cfg.device)
        with torch.no_grad():
            z = self.backward_net(desired_goal)
        z = z.squeeze(0).cpu().numpy()
        meta = OrderedDict()
        meta['z'] = z
        return meta

    def infer_meta(self, replay_loader: ReplayBuffer) -> MetaDict:
        obs_list, reward_list = [], []
        batch_size = 0
        while batch_size < self.cfg.num_inference_steps:
            batch = replay_loader.sample(self.cfg.batch_size)
            batch = batch.to(self.cfg.device)
            obs_list.append(
                batch.next_goal if self.cfg.goal_space is not None else batch.next_obs)
            reward_list.append(batch.reward)
            batch_size += batch.next_obs.size(0)
        obs, reward = torch.cat(obs_list, 0), torch.cat(
            reward_list, 0)  # type: ignore
        obs, reward = obs[:self.cfg.num_inference_steps], reward[:self.cfg.num_inference_steps]
        return self.infer_meta_from_obs_and_rewards(obs, reward)

    def infer_meta_from_obs_and_rewards(self, obs: torch.Tensor, reward: torch.Tensor) -> MetaDict:
        with torch.no_grad():
            B = self.backward_net(obs)
        z = torch.matmul(reward.T, B) / reward.shape[0]
        if self.cfg.norm_z:
            z = math.sqrt(self.cfg.z_dim) * F.normalize(z, dim=1)
        meta = OrderedDict()
        meta['z'] = z.squeeze().cpu().numpy()
        return meta

    def sample_z(self, size, device: str = "cpu"):
        gaussian_rdv = torch.randn(
            (size, self.cfg.z_dim), dtype=torch.float32, device=device)
        gaussian_rdv = F.normalize(gaussian_rdv, dim=1)
        if self.cfg.norm_z:
            z = math.sqrt(self.cfg.z_dim) * gaussian_rdv
        else:
            uniform_rdv = torch.rand(
                (size, self.cfg.z_dim), dtype=torch.float32, device=device)
            z = np.sqrt(self.cfg.z_dim) * uniform_rdv * gaussian_rdv
        return z

    def init_meta(self, obs: np.ndarray = None, replay_loader: tp.Optional[ReplayBuffer] = None) -> MetaDict:
        if self.cfg.uncertainty:
            return self.init_curious_meta(obs,)
        else:
            z = self.sample_z(1)
            z = z.squeeze().numpy()
            meta = OrderedDict()
            meta['z'] = z
        meta['updated'] = True
        return meta

    def init_curious_meta(self, obs: np.ndarray) -> MetaDict:
        meta = OrderedDict()
        if self.cfg.sampling:
            with torch.no_grad():
                num_zs = self.cfg.num_z_samples
                # num_zs x z_dim
                z = self.sample_z(size=num_zs, device=self.cfg.device)
                obs = torch.as_tensor(obs, device=self.cfg.device, dtype=torch.float32).expand(
                    num_zs, -1)  # num_zs x obs_dim
                h = self.encoder(obs)
                # num_zs x act_dim take the mean, although querying with std 0 anyways
                acts = self.actor(h, z, std=1.).mean
                # ensemble_size x num_zs x z_dim
                F1, F2 = self.forward_net((obs, z, acts))
                Q1, Q2 = [torch.einsum('esd, ...sd -> es', Fi, z)
                          for Fi in [F1, F2]]  # ensemble_size x num_zs

            if self.cfg.f_uncertainty:
                eF1 = []
                for i in range(num_zs):
                    F11 = F1[:, i, :].squeeze().to(torch.float64)
                    F11 = F11.transpose(1, 0)  # d x ensemble_size
                    cov_F1 = torch.cov(F11)
                    epistemic_F1 = torch.trace(cov_F1)
                    eF1.append(epistemic_F1)
                # num_zs, naming it for consistency (but its trace of F1)
                epistemic_std1 = torch.tensor(eF1)

            else:
                epistemic_std1, epistemic_std2 = Q1.std(
                    dim=0), Q2.std(dim=0)  # num_zs

            idxs = torch.argmax(epistemic_std1, dim=0)
            # take the z with the highest epistemic uncertainty
            uncertain_z = z[idxs].cpu().numpy()
            meta['z'] = uncertain_z
            meta['disagr'] = epistemic_std1.std().item()
            meta['updated'] = True
        else:
            with torch.no_grad():
                obs = torch.as_tensor(obs, device=self.cfg.device)
                z = self.high_expl_actor(obs, std=1.).sample()
                if self.cfg.norm_z:
                    z = math.sqrt(self.cfg.z_dim) * F.normalize(z, dim=0)
                meta['z'] = z.cpu().numpy()
                meta['updated'] = True
        return meta

    # pylint: disable=unused-argument
    def update_meta(
        self,
        meta: MetaDict,
        global_step: int,
        obs: np.ndarray = None,
    ) -> MetaDict:
        if global_step % self.cfg.update_z_every_step == 0 and np.random.rand() < self.cfg.update_z_proba:
            return self.init_meta() if not self.cfg.uncertainty else self.init_curious_meta(obs)
        meta['updated'] = False
        return meta

    def act(self, obs, meta, step, eval_mode) -> tp.Any:
        obs = torch.as_tensor(obs, device=self.cfg.device,
                              dtype=torch.float32).unsqueeze(0)  # type: ignore
        h = self.encoder(obs)
        z = torch.as_tensor(meta['z'], device=self.cfg.device).unsqueeze(
            0)  # type: ignore
        stddev = utils.schedule(self.cfg.stddev_schedule, step)
        dist = self.actor(h, z, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample()
            if step < self.cfg.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def compute_z_correl(self, time_step: TimeStep, meta: MetaDict) -> float:
        goal = time_step.goal if self.cfg.goal_space is not None else time_step.observation  # type: ignore
        with torch.no_grad():
            zs = [torch.Tensor(x).unsqueeze(0).float().to(
                self.cfg.device) for x in [goal, meta["z"]]]
            zs[0] = self.backward_net(zs[0])
            zs = [F.normalize(z, 1) for z in zs]
            return torch.matmul(zs[0], zs[1].T).item()

    def update_fb(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor,
        next_goal: torch.Tensor,
        z: torch.Tensor,
        step: int,
        goal: torch.Tensor,
    ) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}
        # compute target successor measure
        with torch.no_grad():
            stddev = utils.schedule(self.cfg.stddev_schedule, step)
            dist = self.actor(next_obs, z, stddev)
            next_action = dist.sample(clip=self.cfg.stddev_clip)
            target_F1, target_F2 = self.forward_target_net(
                (next_obs, z, next_action))  # e? x batch x z_dim
            target_B = self.backward_target_net(goal)  # batch x z_dim
            if not self.cfg.uncertainty:
                target_M1 = torch.einsum(
                    'sd, td -> st', target_F1, target_B)  # batch x batch
                target_M2 = torch.einsum(
                    'sd, td -> st', target_F2, target_B)  # batch x batch
            else:
                # e x batch x batch torch.ones((self.n_ensemble, 1024, 1024), device = target_F1.device) #
                target_M1 = torch.einsum(
                    'esd, ...td -> est', target_F1, target_B)
                # e x batch x batch torch.ones((self.n_ensemble, 1024, 1024), device = target_F1.device) #
                target_M2 = torch.einsum(
                    'esd, ...td -> est', target_F2, target_B)
            target_M = torch.min(target_M1, target_M2)
        # compute FB loss
        F1, F2 = self.forward_net((obs, z, action))  # batch x z_dim
        B = self.backward_net(goal)  # batch x z_dim
        if not self.cfg.uncertainty:
            M1 = torch.einsum('sd, td -> st', F1, B)  # batch x batch
            M2 = torch.einsum('sd, td -> st', F2, B)  # batch x batch
            I = torch.eye(*M1.size(), device=M1.device)
            off_diag = ~I.bool()
            fb_offdiag: tp.Any = 0.5 * \
                sum((M - discount * target_M)
                    [off_diag].pow(2).mean() for M in [M1, M2])
            fb_diag: tp.Any = -sum(M.diag().mean() for M in [M1, M2])
        else:
            M1 = torch.einsum('esd, ...td -> est', F1, B)  # e x batch x batch
            M2 = torch.einsum('esd, ...td -> est', F2, B)  # e x batch x batch
            I = torch.eye(*M1.size()[1:], device=M1.device)
            off_diag = ~I.bool()
            # # get indices for the first dimension of the M ensemble matrix
            E_indices = torch.arange(
                M1.shape[0]).unsqueeze(-1).unsqueeze(-1)  # (e, 1, 1)
            # compute the offidagonal term for each member averaging over batch dim, and summing over E and over M1 and M2
            # this one seems to be quite costly
            scaled_T = discount * target_M
            fb_offdiag: tp.Any = 0.5 * \
                (sum((M - scaled_T)[E_indices, off_diag].pow(2).mean()
                 for M in [M1, M2]))

            # M.diagonal(dim1=-2, dim2=-1) returns diagonals over every ensemble so size is: E x batch
            # then we average over B and sum over E and over M1 and M2
            fb_diag: tp.Any = - \
                sum(M.diagonal(dim1=-2, dim2=-1).mean() for M in [M1, M2])
        fb_loss = fb_offdiag + fb_diag

        # ORTHONORMALITY LOSS FOR BACKWARD EMBEDDING

        Cov = torch.matmul(B, B.T)
        orth_loss_diag = - 2 * Cov.diag().mean()
        orth_loss_offdiag = Cov[off_diag].pow(2).mean()
        orth_loss = orth_loss_offdiag + orth_loss_diag
        fb_loss += self.cfg.ortho_coef * orth_loss

        if self.cfg.use_tb or self.cfg.use_wandb:
            metrics['target_M'] = target_M.mean().item()
            metrics['M1'] = M1.mean().item()
            metrics['F1'] = F1.mean().item()
            metrics['B'] = B.mean().item()
            metrics['B_norm'] = torch.norm(B, dim=-1).mean().item()
            metrics['z_norm'] = torch.norm(z, dim=-1).mean().item()
            metrics['fb_loss'] = fb_loss.item()
            metrics['fb_diag'] = fb_diag.item()
            metrics['fb_offdiag'] = fb_offdiag.item()
            metrics['orth_loss'] = orth_loss.item()
            metrics['orth_loss_diag'] = orth_loss_diag.item()
            metrics['orth_loss_offdiag'] = orth_loss_offdiag.item()
            eye_diff = torch.matmul(
                B.T, B) / B.shape[0] - torch.eye(B.shape[1], device=B.device)
            metrics['orth_linf'] = torch.max(torch.abs(eye_diff)).item()
            metrics['orth_l2'] = eye_diff.norm().item() / math.sqrt(B.shape[1])
            if isinstance(self.fb_opt, torch.optim.Adam):
                metrics["fb_opt_lr"] = self.fb_opt.param_groups[0]["lr"]

        # optimize FB
        self.fb_opt.zero_grad(set_to_none=True)
        fb_loss.backward()
        self.fb_opt.step()
        return metrics

    def update_actor(self, obs: torch.Tensor, z: torch.Tensor, step: int) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}
        stddev = utils.schedule(self.cfg.stddev_schedule, step)
        dist = self.actor(obs, z, stddev)
        # non differentiable / differentiable?
        action = dist.sample(clip=self.cfg.stddev_clip)

        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        F1, F2 = self.forward_net((obs, z, action))

        if self.cfg.uncertainty:
            Q1 = torch.einsum('esd, ...sd -> es', F1, z).mean(0)
            Q2 = torch.einsum('esd, ...sd -> es', F2, z).mean(0)
        else:
            Q1 = torch.einsum('sd, sd -> s', F1, z)
            Q2 = torch.einsum('sd, sd -> s', F2, z)
        Q = torch.min(Q1, Q2)
        actor_loss = -Q.mean()
        if self.cfg.rnd:
            rnd_loss = -self.Q_rnd(obs, action, z).mean()
            actor_loss += self.cfg.rnd_coeff * rnd_loss
            metrics['actor_loss_rnd'] = rnd_loss.item() if self.cfg.rnd else 0
            metrics['actor_loss_exploit'] = -Q.mean().item()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.cfg.use_tb or self.cfg.use_wandb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['q'] = Q.mean().item()
            metrics['actor_logprob'] = log_prob.mean().item()
            # metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update_high_expl_actor(self, obs: torch.Tensor, step: int) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}

        stddev = utils.schedule(self.cfg.stddev_schedule, step)
        dist = self.high_expl_actor(obs, stddev)
        z = dist.sample()
        if self.cfg.norm_z:
            z = math.sqrt(self.cfg.z_dim) * F.normalize(z, dim=1)
        with torch.no_grad():
            a = self.actor(obs, z, std=1.).mean
        F1, F2 = self.forward_net((obs, z, a))

        Q1, Q2 = [torch.einsum('esd, ...sd -> es', Fi, z) for Fi in [F1, F2]]
        epistemic_std1, epistemic_std2 = Q1.std(dim=0), Q2.std(
            dim=0)
        epistemic_std = epistemic_std1.mean()
        actor_loss = -epistemic_std
        # optimize actor
        self.high_expl_actor_opt.zero_grad(set_to_none=True)
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.high_expl_actor_opt.step()

        if self.cfg.use_tb or self.cfg.use_wandb:
            metrics['high_actor_loss'] = actor_loss.item()
        return metrics

    def update_Qrnd(self, obs: torch.Tensor, action: torch.Tensor, discount: torch.Tensor, next_obs: torch.Tensor, z: torch.Tensor, step: int) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}
        values_rnd = self.Q_rnd(obs, action, z)

        with torch.no_grad():
            next_action = self.actor(next_obs, z, std=1.).sample()
            next_values_rnd = self.target_Q_rnd(next_obs, next_action, z)
        reward, rnd_loss = self.rnd_module.update_curiosity(next_obs)
        value_loss = F.mse_loss(values_rnd, reward +
                                discount * next_values_rnd)
        self.Q_rnd_opt.zero_grad()
        value_loss.backward()
        self.Q_rnd_opt.step()

        metrics['Qrnd_loss'] = value_loss.item()
        metrics['RND_net_loss'] = rnd_loss.item()
        return metrics

    def aug_and_encode(self, obs: torch.Tensor) -> torch.Tensor:
        obs = self.aug(obs)
        return self.encoder(obs)

    def update(self, replay_loader: ReplayBuffer, step: int) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}

        if step % self.cfg.update_every_steps != 0:
            return metrics

        batch = replay_loader.sample(self.cfg.batch_size)
        batch = batch.to(self.cfg.device)

        obs = batch.obs
        goal = batch.obs
        action = batch.action
        discount = batch.discount
        next_obs = next_goal = batch.next_obs
        if self.cfg.goal_space is not None:
            # in case goal_space is defined and next_goal is not in batch (case of prefill)
            if batch.next_goal is None and self.cfg.goal_space == 'simplified_point_mass_maze':
                batch.next_goal = batch.next_obs[:, :2]
            assert batch.next_goal is not None
            next_goal = batch.next_goal

        z = self.sample_z(self.cfg.batch_size, device=self.cfg.device)
        if not z.shape[-1] == self.cfg.z_dim:
            raise RuntimeError("There's something wrong with the logic here")
        backward_input = batch.obs
        future_goal = batch.future_obs
        if self.cfg.goal_space is not None:
            # in case goal_space is defined and next_goal is not in batch (case of prefill)
            if batch.goal is None and self.cfg.goal_space == 'simplified_point_mass_maze':
                batch.goal = batch.obs[:, :2]
            assert batch.goal is not None
            backward_input = batch.goal
            goal = batch.goal
            future_goal = batch.future_goal

        perm = torch.randperm(self.cfg.batch_size)
        backward_input = backward_input[perm]

        if self.cfg.mix_ratio > 0:
            mix_idxs: tp.Any = np.where(np.random.uniform(
                size=self.cfg.batch_size) < self.cfg.mix_ratio)[0]
            if not self.cfg.rand_weight:
                with torch.no_grad():
                    mix_z = self.backward_net(
                        backward_input[mix_idxs]).detach()
            else:
                # generate random weight
                weight = torch.rand(
                    size=(mix_idxs.shape[0], self.cfg.batch_size)).to(self.cfg.device)
                weight = F.normalize(weight, dim=1)
                uniform_rdv = torch.rand(
                    mix_idxs.shape[0], 1).to(self.cfg.device)
                weight = uniform_rdv * weight
                with torch.no_grad():
                    mix_z = torch.matmul(
                        weight, self.backward_net(backward_input).detach())
            if self.cfg.norm_z:
                mix_z = math.sqrt(self.cfg.z_dim) * F.normalize(mix_z, dim=1)
            z[mix_idxs] = mix_z

        # hindsight replay
        if self.cfg.future_ratio > 0:
            assert future_goal is not None
            future_idxs = np.where(np.random.uniform(
                size=self.cfg.batch_size) < self.cfg.future_ratio)
            z[future_idxs] = self.backward_net(
                future_goal[future_idxs]).detach()

        metrics.update(self.update_fb(obs=obs, action=action, discount=discount,
                                      next_obs=next_obs, next_goal=next_goal, z=z, step=step,
                                      goal=goal))

        # update high expl actor
        if self.cfg.uncertainty and not self.cfg.sampling:
            metrics.update(self.update_high_expl_actor(obs, step))

        if self.cfg.rnd:
            metrics.update(self.update_Qrnd(obs=obs, action=action,
                           discount=discount, next_obs=next_obs, z=z, step=step))

        # update actor
        metrics.update(self.update_actor(obs, z, step))

        # update critic target
        if self.cfg.one_target and self.cfg.uncertainty:
            utils.soft_update_params_mean(self.forward_net, self.forward_target_net,
                                          self.cfg.fb_target_tau)
        else:
            utils.soft_update_params(self.forward_net, self.forward_target_net,
                                     self.cfg.fb_target_tau)
        utils.soft_update_params(self.backward_net, self.backward_target_net,
                                 self.cfg.fb_target_tau)

        if self.cfg.rnd:
            utils.soft_update_params(self.Q_rnd, self.target_Q_rnd,
                                     self.cfg.fb_target_tau)
        return metrics
