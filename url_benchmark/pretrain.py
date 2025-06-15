# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Alert: do not change import order to avoid 
import os
import json
import pdb  # pylint: disable=unused-import
import logging
import dataclasses
import typing as tp
import warnings
from pathlib import Path
import sys

warnings.filterwarnings('ignore', category=DeprecationWarning)

""" 
Code adapted from  https://github.com/facebookresearch/controllable_agent
"""

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
# if the default egl does not work, you may want to try:
# export MUJOCO_GL=glfw
os.environ['MUJOCO_GL'] = os.environ.get('MUJOCO_GL', 'egl')
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import hydra
from hydra.core.config_store import ConfigStore
import numpy as np
import torch
import wandb
import omegaconf as omgcf
# from dm_env import specs

from url_benchmark import dmc
from dm_env import specs
from url_benchmark import utils
from url_benchmark import goals as _goals
from url_benchmark.logger import Logger
from url_benchmark.in_memory_replay_buffer import ReplayBuffer
from url_benchmark.video import TrainVideoRecorder, VideoRecorder
from url_benchmark import agent as agents
logger = logging.getLogger(__name__)
torch.backends.cudnn.benchmark = True
# os.environ['WANDB_MODE']='offline'

# # # Config # # #


@dataclasses.dataclass
class Config:
    agent: tp.Any
    # misc
    seed: int = 1
    device: str = "cuda"
    save_video: bool = False
    use_tb: bool = False
    use_wandb: bool = False
    # experiment
    experiment: str = "online"
    # task settings
    task: str = "walker_stand"
    obs_type: str = "states"  # [states, pixels]
    frame_stack: int = 3  # only works if obs_type=pixels
    action_repeat: int = 1  # set to 2 for pixels
    discount: float = 0.99
    future: float = 0.99  # discount of future sampling, future=1 means no future sampling
    goal_space: tp.Optional[str] = None
    append_goal_to_observation: bool = False
    # eval
    num_eval_episodes: int = 10
    custom_reward: tp.Optional[str] = None  # activates custom eval if not None
    final_tests: int = 10
    # checkpoint # num episode * length of episode
    snapshot_at: tp.Tuple[int, ...] = (0, 250, 500, 1000, 1500, 2000)
    checkpoint_every: int = 100000
    load_model: tp.Optional[str] = None
    # training
    num_seed_frames: int = 4000
    replay_buffer_episodes: int = 5000
    update_encoder: bool = True
    batch_size: int = omgcf.II("agent.batch_size")
    uncertainty: bool = False
    update_every_steps: int = 1
    num_agent_updates: int = 1
    pretrain_update_steps: int = 1000
    # to avoid hydra issues
    project_dir: str = ""
    results_dir: str = ""
    id: int = 0
    working_dir: str = ""
    debug: bool = False
    eval: bool = False
    # mode
    reward_free: bool = True
    # train settings
    num_train_frames: int = 2000010
    # snapshot
    eval_every_frames: int = 10000
    load_replay_buffer: tp.Optional[str] = None
    save_train_video: bool = False


#  Name the Config as "workspace_config".
#  When we load workspace_config in the main config, we are telling it to load: Config.
ConfigStore.instance().store(name="workspace_config", node=Config)


# # # Implem # # #


def make_agent(
    obs_type: str, obs_spec, action_spec, num_expl_steps: int, cfg: omgcf.DictConfig
) -> tp.Union[agents.FBDDPGAgent]:
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = (action_spec.num_values, ) if isinstance(action_spec, specs.DiscreteArray) \
        else action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    return hydra.utils.instantiate(cfg)


C = tp.TypeVar("C", bound=Config)


def _init_eval_meta(workspace: "BaseWorkspace", custom_reward: tp.Optional[_goals.BaseReward] = None) -> agents.MetaDict:
    if custom_reward is not None:
        obs_list, reward_list = [], []
        batch_size = 0
        num_steps = workspace.agent.cfg.num_inference_steps  # type: ignore
        while batch_size < num_steps:
            batch = workspace.replay_loader.sample(
                workspace.cfg.batch_size, custom_reward=custom_reward)
            batch = batch.to(workspace.cfg.device)
            obs_list.append(
                batch.next_goal if workspace.cfg.goal_space is not None else batch.next_obs)
            reward_list.append(batch.reward)
            batch_size += batch.next_obs.size(0)
        obs, reward = torch.cat(obs_list, 0), torch.cat(
            reward_list, 0)  # type: ignore
        obs_t, reward_t = obs[:num_steps], reward[:num_steps]
        return workspace.agent.infer_meta_from_obs_and_rewards(obs_t, reward_t)

    if workspace.cfg.goal_space is not None:  # goal reaching task defined from "task" variable, no custom reward
        funcs = _goals.goals.funcs.get(workspace.cfg.goal_space, {})
        if workspace.cfg.task in funcs:
            g = funcs[workspace.cfg.task]()
            return workspace.agent.get_goal_meta(g)
    print('\n***\n inferring eval meta from replay buffer :s\n***\n')
    return workspace.agent.infer_meta(workspace.replay_loader)


class BaseWorkspace(tp.Generic[C]):
    def __init__(self, cfg: C) -> None:
        self.work_dir = Path.cwd() if len(cfg.working_dir) == 0 else Path(cfg.working_dir)
        self.model_dir = self.work_dir if 'cluster' not in str(
            self.work_dir) else Path(str(self.work_dir).replace('home', 'scratch'))
        print(f'Workspace: {self.work_dir}')
        print(
            f'Running code in : {Path(__file__).parent.resolve().absolute()}')
        logger.info(f'Workspace: {self.work_dir}')
        logger.info(
            f'Running code in : {Path(__file__).parent.resolve().absolute()}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        if not torch.cuda.is_available():
            if cfg.device != "cpu":
                logger.warning(
                    f"Falling back to cpu as {cfg.device} is not available")
                cfg.device = "cpu"
                cfg.agent.device = "cpu"
        self.device = torch.device(cfg.device)
        task = cfg.task
        if task.startswith('point_mass_maze'):
            self.domain = 'point_mass_maze'
        else:
            self.domain = task.split('_', maxsplit=1)[0]
        if cfg.goal_space is not None:
            if cfg.goal_space not in _goals.goal_spaces.funcs[self.domain]:
                raise ValueError(
                    f"Unregistered goal space {cfg.goal_space} for domain {self.domain}")
        self.train_env = self._make_env()
        self.eval_env = self._make_env()
        # create agent
        self.agent = make_agent(cfg.obs_type,
                                self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                cfg.num_seed_frames // cfg.action_repeat,
                                cfg.agent)

        # create logger
        self.logger = Logger(self.work_dir,
                             use_tb=cfg.use_tb,
                             use_wandb=cfg.use_wandb)

        if cfg.use_wandb:
            exp_name = '_'.join([
                cfg.experiment, cfg.agent.name, self.domain, str(cfg.id)
            ])
            wandb.init(project="controllable_agent", group=cfg.experiment, name=exp_name,  # mode="disabled",
                       config=omgcf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True), dir=self.work_dir)  # type: ignore

        self.replay_loader = ReplayBuffer(
            max_episodes=cfg.replay_buffer_episodes, discount=cfg.discount, future=cfg.future)

        cam_id = 0 if 'quadruped' not in self.domain else 2

        self.video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None,
                                            camera_id=cam_id, use_wandb=self.cfg.use_wandb)

        self.timer = utils.Timer()
        self.global_step = 0
        self.global_episode = 0
        self.eval_rewards_history: tp.List[float] = []
        self.eval_dist_history: tp.List[float] = []
        self._checkpoint_filepath = self.model_dir / "models" / "latest.pt"
        # This is for continuing training in case workdir is the same
        if self._checkpoint_filepath.exists():
            self.load_checkpoint(self._checkpoint_filepath)
        # This is for loading an existing model
        elif cfg.load_model is not None:
            # , exclude=["replay_loader"])
            self.load_checkpoint(cfg.load_model)

        self.domain_tasks = {
            "cheetah": ['walk', 'walk_backward', 'run', 'run_backward', 'flip', 'flip_backward'],
            "quadruped": ['stand', 'walk', 'run', 'jump'],
            "walker": ['stand', 'walk', 'run', 'flip'],
            "hopper": ['hop', 'stand', 'hop_backward', 'flip', 'flip_backward'],
        }

    def _make_env(self) -> dmc.EnvWrapper:
        cfg = self.cfg
        return dmc.make(cfg.task, cfg.obs_type, cfg.frame_stack, cfg.action_repeat, cfg.seed,
                        goal_space=cfg.goal_space, append_goal_to_observation=cfg.append_goal_to_observation)

    @property
    def global_frame(self) -> int:
        return self.global_step * self.cfg.action_repeat

    def _make_custom_reward(self, seed: int) -> tp.Optional[_goals.BaseReward]:
        """Creates a custom reward function if provided in configuration
        else returns None
        """
        if self.cfg.custom_reward is None:
            return None
        return _goals.get_reward_function(self.cfg.custom_reward, seed)

    def eval_maze_goals(self) -> None:
        reward_cls = _goals.MazeMultiGoal()
        rewards = list()
        dists = list()
        successes = list()
        for g in reward_cls.goals:
            goal_rewards = list()
            goal_distances = list()
            goal_successes = list()
            meta = self.agent.get_goal_meta(g)
            for episode in range(self.cfg.num_eval_episodes):
                self.video_recorder.init(self.eval_env, enabled=(episode == 0))
                time_step = self.eval_env.reset()
                episode_reward = 0.0
                while not time_step.last():
                    with torch.no_grad(), utils.eval_mode(self.agent):
                        action = self.agent.act(time_step.observation,
                                                meta,
                                                0,
                                                eval_mode=True)
                    time_step = self.eval_env.step(action)
                    self.video_recorder.record(self.eval_env)
                    assert isinstance(time_step, dmc.ExtendedGoalTimeStep)
                    step_reward, distance, success = reward_cls.from_goal(
                        time_step.goal, g)
                    episode_reward += step_reward
                goal_rewards.append(episode_reward)
                goal_distances.append(float(distance))
                goal_successes.append(success)
                self.video_recorder.save(f'{g}_{self.global_frame}.mp4')
            # print(f"goal: {g}, avg_reward: {round(float(np.mean(goal_rewards)), 2)}, "
            #       f"avg_distance: {round(float(np.mean(goal_distances)), 5)}, "
            #       f"avg_success: {round(float(np.mean(goal_successes)), 5)}")
            rewards.append(float(np.mean(goal_rewards)))
            dists.append(float(np.mean(goal_distances)))
            successes.append(float(np.mean(goal_successes)))  # num goals x 1
        total_avg_reward = float(np.mean(rewards))
        total_avg_dist = float(np.mean(dists))
        total_avg_success = float(np.mean(successes))
        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_avg_reward)
            log('episode_distance', total_avg_dist)
            log('step', self.global_step)
            log('episode', self.global_episode)
            log('success_rate', total_avg_success)
            log('buffer_size', len(self.replay_loader))
            for i, room in zip(range(0, len(successes), reward_cls.goals_per_room), range(1, 5)):
                log(f'success_room{room}', float(
                    np.mean(successes[i:i+reward_cls.goals_per_room])))
                log(f'reward_room{room}', float(
                    np.mean(rewards[i:i+reward_cls.goals_per_room])))
                log(f'dist_{room}', float(
                    np.mean(dists[i:i+reward_cls.goals_per_room])))

    def eval(self, task=None) -> None:
        if task is not None:
            self.domain_tasks = {self.domain: ['_'.join(task.split('_')[1:])]}
        # Test if enough data to compute meta from samples, otw quit already!
        # TODO Assuming fix episode length for now.
        num_steps = self.cfg.agent.num_inference_steps  # type: ignore
        if len(self.replay_loader) * self.replay_loader._episodes_length[0] < num_steps:
            # print("Not enough data for inference, skipping eval")
            return None

        total_tasks_rewards = []
        # add log_and_dump here to ensure csv_file has all the fields (columns)!
        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            for name in self.domain_tasks[self.domain]:
                task = "_".join([self.domain, name])
                self.cfg.task = task
                self.cfg.custom_reward = task  # set task to custom reward to compute z
                self.cfg.seed += 1  # for the sake of avoiding similar seeds
                self.eval_env = self._make_env()

                step, episode = 0, 0
                eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
                physics_agg = dmc.PhysicsAggregator()
                rewards: tp.List[float] = []
                z_correl = 0.0
                actor_success: tp.List[float] = []
                while eval_until_episode(episode):
                    time_step = self.eval_env.reset()
                    seed = 12 * self.cfg.num_eval_episodes + len(rewards)
                    custom_reward = self._make_custom_reward(seed=seed)
                    meta = _init_eval_meta(self, custom_reward)
                    total_reward = 0.0
                    self.video_recorder.init(
                        self.eval_env, enabled=(episode == 0))
                    while not time_step.last():
                        with torch.no_grad(), utils.eval_mode(self.agent):
                            action = self.agent.act(time_step.observation,
                                                    meta,
                                                    self.global_step,
                                                    eval_mode=True)
                        time_step = self.eval_env.step(action)
                        physics_agg.add(self.eval_env)
                        self.video_recorder.record(self.eval_env)
                        if custom_reward is not None:
                            time_step.reward = custom_reward.from_env(
                                self.eval_env)
                        total_reward += time_step.reward
                        step += 1
                    rewards.append(total_reward)
                    episode += 1
                    self.video_recorder.save(f'{task}_{self.global_frame}.mp4')

                total_avg_reward = float(np.mean(rewards))
                total_tasks_rewards.append(total_avg_reward)
                log(f'episode_reward_{task}', total_avg_reward)
                if len(rewards) > 1:
                    log(f'episode_reward#std_{task}', float(np.std(rewards)))
                log(f'episode_length_{task}', step *
                    self.cfg.action_repeat / episode)
                log(f'episode_{task}', self.global_episode)
                log(f'z_correl_{task}', z_correl / episode)
                log(f'step_{task}', self.global_step)
                log(f'z_norm_{task}', np.linalg.norm(meta['z']).item())
                for key, val in physics_agg.dump():
                    log(key+f'_{task}', val)
            log('episode_reward', np.mean(total_tasks_rewards))
            log('buffer_size', len(self.replay_loader))

    _CHECKPOINTED_KEYS = ('agent', 'global_step',
                          'global_episode', "replay_loader")

    def save_checkpoint(self, fp: tp.Union[Path, str], exclude: tp.Sequence[str] = ()) -> None:
        logger.info(f"Saving checkpoint to {fp}")
        exclude = list(exclude)
        assert all(x in self._CHECKPOINTED_KEYS for x in exclude)
        fp = Path(fp)
        fp.parent.mkdir(exist_ok=True, parents=True)
        assert isinstance(
            self.replay_loader, ReplayBuffer), "Is this buffer designed for checkpointing?"
        # this is just a dumb security check to not forget about it
        payload = {k: self.__dict__[k]
                   for k in self._CHECKPOINTED_KEYS if k not in exclude}
        with fp.open('wb') as f:
            torch.save(payload, f, pickle_protocol=4)

    def load_checkpoint(self, fp: tp.Union[Path, str], only: tp.Optional[tp.Sequence[str]] = None, exclude: tp.Sequence[str] = ()) -> None:
        """Reloads a checkpoint or part of it

        Parameters
        ----------
        only: None or sequence of str
            reloads only a specific subset (defaults to all)
        exclude: sequence of str
            does not reload the provided keys
        """
        print(f"loading checkpoint from {fp}")
        fp = Path(fp)
        with fp.open('rb') as f:
            payload = torch.load(f)
        if isinstance(payload, ReplayBuffer):  # compatibility with pure buffers pickles
            payload = {"replay_loader": payload}
        if only is not None:
            only = list(only)
            assert all(x in self._CHECKPOINTED_KEYS for x in only)
            payload = {x: payload[x] for x in only}
        exclude = list(exclude)
        assert all(x in self._CHECKPOINTED_KEYS for x in exclude)
        for x in exclude:
            payload.pop(x, None)
        for name, val in payload.items():
            logger.info("Reloading %s from %s", name, fp)
            if name == "agent":
                self.agent.init_from(val)
            elif name == "replay_loader":
                assert isinstance(val, ReplayBuffer)
                # pylint: disable=protected-access
                # drop unecessary meta which could make a mess
                val._current_episode.clear()  # make sure we can start over
                val._future = self.cfg.future
                val._discount = self.cfg.discount
                if not self.cfg.eval:  # Otherwise keep original buffer size
                    val._max_episodes = self.cfg.replay_buffer_episodes
                val._episodes_length = np.array(
                    [len(array) - 1 for array in val._storage["discount"]], dtype=np.int32)
                self.replay_loader = val
                if len(self.replay_loader._storage['discount']) < self.replay_loader._max_episodes:
                    self.replay_loader.resize()
            else:
                assert hasattr(self, name)
                setattr(self, name, val)
                if name == "global_episode":
                    logger.warning(
                        f"Reloaded agent at global episode {self.global_episode}")


class Workspace(BaseWorkspace[Config]):
    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        self.train_video_recorder = TrainVideoRecorder(self.work_dir if cfg.save_train_video else None,
                                                       camera_id=self.video_recorder.camera_id, use_wandb=self.cfg.use_wandb)
        if not self._checkpoint_filepath.exists():  # don't relay if there is a checkpoint
            if cfg.load_replay_buffer is not None:

                self.load_checkpoint(cfg.load_replay_buffer,
                                     only=["replay_loader"])

    def _init_meta(self, obs: np.ndarray, replay_loader: tp.Optional[ReplayBuffer]):
        meta = self.agent.init_meta(obs, replay_loader)
        return meta

    def train(self) -> None:
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)
        update_every_step = utils.Every(self.agent.cfg.update_every_steps,
                                        self.cfg.action_repeat)
        episode_step, episode_reward, z_correl = 0, 0.0, 0.0
        time_step = self.train_env.reset()
        meta = self._init_meta(time_step.observation, self.replay_loader)
        self.replay_loader.add(time_step, meta)
        self.train_video_recorder.init(time_step.observation)
        metrics = None
        meta_disagr = []
        physics_agg = dmc.PhysicsAggregator()

        while train_until_step(self.global_step):
            # try to update the agent: only if we collected enough random data (seed until step steps)
            if not seed_until_step(self.global_step) and update_every_step(self.global_step):
                for _ in range(self.cfg.num_agent_updates):
                    metrics = self.agent.update(
                        self.replay_loader, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            if not self.replay_loader._full and time_step.last():
                self.global_episode += 1
                self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_loader))
                        log('step', self.global_step)
                        log('z_correl', z_correl)
                        if self.cfg.uncertainty and len(meta_disagr) > 0:
                            log('z_disagr', np.mean(meta_disagr))

                        for key, val in physics_agg.dump():
                            log(key, val)
                # reset env
                time_step = self.train_env.reset()
                meta = self._init_meta(
                    time_step.observation, self.replay_loader)
                self.replay_loader.add(time_step, meta)
                self.train_video_recorder.init(time_step.observation)

                episode_step = 0
                episode_reward = 0.0
                z_correl = 0.0
                meta_disagr = []
                # save checkpoint to reload
                if self.global_episode in self.cfg.snapshot_at:
                    #  notice the replay loader will get very big, so can be excluded, but its needed at eval time!
                    # , exclude=["replay_loader"])
                    self.save_checkpoint(self._checkpoint_filepath)

            # Very ugly way of keeping the xaxis of evals updated when not collecting more data (and hence not calling log above). TODO pass step to log !
            if self.replay_loader._full and self.global_frame % self.replay_loader._episodes_length[0] == 0:
                wandb.log({})

            # try to evaluate
            if eval_every_step(self.global_step) and not self.cfg.debug:
                if self.cfg.custom_reward == "maze_multi_goal":
                    self.eval_maze_goals()
                else:
                    self.eval()
            # Collect more data if buffer is not full:
            if not self.replay_loader._full:
                meta = self.agent.update_meta(meta, self.global_step, obs=time_step.observation)
                if self.cfg.uncertainty and 'disagr' in meta and meta['updated']:
                    meta_disagr.append(meta['disagr'])
                # sample action
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            meta,
                                            self.global_step,
                                            eval_mode=False)

                # take env step
                time_step = self.train_env.step(action)
                physics_agg.add(self.train_env)
                episode_reward += time_step.reward
                self.replay_loader.add(time_step, meta)
                self.train_video_recorder.record(time_step.observation)
                z_correl += self.agent.compute_z_correl(time_step, meta)
                episode_step += 1

            self.global_step += 1
        # , exclude=["replay_loader"])  # make sure we save the final checkpoint
        self.save_checkpoint(self._checkpoint_filepath)

    def eval_model(self) -> None:
        if self.cfg.custom_reward == "maze_multi_goal":
            self.eval_maze_goals()
        # self.eval(task=self.cfg.task) # if only one task to evaluate
        self.eval()


@hydra.main(config_path='configs', config_name='base_config', version_base="1.1")
def main(cfg: omgcf.DictConfig) -> None:
    # we assume cfg is a PretrainConfig (but actually not really)
    # calls Config and PretrainConfig
    workspace = Workspace(cfg)  # type: ignore
    if not cfg.eval:
        workspace.train()
    else:
        workspace.eval_model()


if __name__ == '__main__':
    main()
