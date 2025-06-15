# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pdb  # pylint: disable=unused-import
import token
import tokenize
import functools
import typing as tp
from io import BytesIO
from collections import OrderedDict
import numpy as np
from url_benchmark import dmc
from dm_control.utils import rewards
import torch

""" 
Code adapted from  https://github.com/facebookresearch/controllable_agent
"""

from url_benchmark.custom_dmc_tasks.point_mass_maze import TASKS as point_mass_maze_tasks_list

point_mass_maze_tasks = dict(point_mass_maze_tasks_list)

F = tp.TypeVar("F", bound=tp.Callable[..., np.ndarray])


class Register(tp.Generic[F]):

    def __init__(self) -> None:
        self.funcs: tp.Dict[str, tp.Dict[str, F]] = {}

    def __call__(self, name: str) -> tp.Callable[[F], F]:
        return functools.partial(self._register, name=name)

    def _register(self, func: F, name: str) -> F:
        fname = func.__name__
        subdict = self.funcs.setdefault(name, {})
        if fname in subdict:
            raise ValueError(
                f"Already registered a function {fname} for {name}")
        subdict[fname] = func
        return func


goal_spaces: Register[tp.Callable[[dmc.EnvWrapper], np.ndarray]] = Register()
goals: Register[tp.Callable[[], np.ndarray]] = Register()


# # # # #
# goal spaces, defined on one environment to specify:
# # # # #

# pylint: disable=function-redefined


@goal_spaces("point_mass_maze")
def simplified_point_mass_maze(env: dmc.EnvWrapper) -> np.ndarray:
    return np.array(env.physics.named.data.geom_xpos['pointmass'][:2],
                    dtype=np.float32)


@goal_spaces("walker")
def simplified_walker(env: dmc.EnvWrapper) -> np.ndarray:
    # check the physics here:
    # https://github.com/deepmind/dm_control/blob/d72c22f3bb89178bff38728957daf62965632c2f/dm_control/suite/walker.py
    return np.array([env.physics.torso_height(),
                     env.physics.torso_upright(),
                     env.physics.horizontal_velocity()],
                    dtype=np.float32)


@goal_spaces("walker")
def walker_pos_speed(env: dmc.EnvWrapper) -> np.ndarray:
    """simplifed walker, with x position as additional variable"""
    # check the physics here:
    # https://github.com/deepmind/dm_control/blob/d72c22f3bb89178bff38728957daf62965632c2f/dm_control/suite/walker.py
    x = env.physics.named.data.xpos['torso', 'x']
    # type: ignore
    return np.concatenate([simplified_walker(env), [x]], axis=0, dtype=np.float32)


@goal_spaces("walker")
def walker_pos_speed_z(env: dmc.EnvWrapper) -> np.ndarray:
    """simplifed walker, with x position as additional variable"""
    # check the physics here:
    # https://github.com/deepmind/dm_control/blob/d72c22f3bb89178bff38728957daf62965632c2f/dm_control/suite/walker.py
    # vz = env.physics.named.data.sensordata["torso_subtreelinvel"][-1]
    # om_y = env.physics.named.data.subtree_angmom['torso'][1]
    vz = env.physics.named.data.subtree_linvel['torso', 'z']
    om_y = env.physics.named.data.subtree_angmom['torso', 'y']
    # type: ignore
    return np.concatenate([walker_pos_speed(env), [vz, om_y]], axis=0, dtype=np.float32)


@goal_spaces("quadruped")
def simplified_quadruped(env: dmc.EnvWrapper) -> np.ndarray:
    # check the physics here:
    # https://github.com/deepmind/dm_control/blob/d72c22f3bb89178bff38728957daf62965632c2f/dm_control/suite/quadruped.py#L145
    return np.array([env.physics.torso_upright(),
                     np.linalg.norm(env.physics.torso_velocity())],
                    dtype=np.float32)


@goal_spaces("quadruped")
def simplified_quadruped_vel(env: dmc.EnvWrapper) -> np.ndarray:
    # check the physics here:
    # https://github.com/deepmind/dm_control/blob/d72c22f3bb89178bff38728957daf62965632c2f/dm_control/suite/quadruped.py#L145
    return np.concatenate([[env.physics.torso_upright()],
                           env.physics.torso_velocity()],
                          dtype=np.float32)


@goal_spaces("quadruped")
def simplified_quadruped_velx(env: dmc.EnvWrapper) -> np.ndarray:
    # check the physics here:
    # https://github.com/deepmind/dm_control/blob/d72c22f3bb89178bff38728957daf62965632c2f/dm_control/suite/quadruped.py#L145
    return np.array([env.physics.torso_upright(),
                     env.physics.torso_velocity()[0]],
                    dtype=np.float32)


@goal_spaces("quadruped")
def quad_pos_speed(env: dmc.EnvWrapper) -> np.ndarray:
    # check the physics here:
    # https://github.com/deepmind/dm_control/blob/d72c22f3bb89178bff38728957daf62965632c2f/dm_control/suite/quadruped.py#L145
    x = np.array(env.physics.named.data.site_xpos['workspace'])
    states = [[env.physics.torso_upright()], x, env.physics.torso_velocity()]
    return np.concatenate(states, dtype=np.float32)


@goal_spaces("hopper")
def simplified_hopper(env: dmc.EnvWrapper) -> np.ndarray:
    # check the physics here:
    # https://github.com/deepmind/dm_control/blob/d72c22f3bb89178bff38728957daf62965632c2f/dm_control/suite/hopper.py
    return np.array([env.physics.height(),
                     env.physics.speed()],
                    dtype=np.float32)


@goal_spaces("cheetah")
def simplified_cheetah(env: dmc.EnvWrapper) -> np.ndarray:
    # check the physics here:
    # https://github.com/deepmind/dm_control/blob/d72c22f3bb89178bff38728957daf62965632c2f/dm_control/suite/hopper.py
    return np.array([env.physics.speed()],
                    dtype=np.float32)


@goal_spaces("cheetah")
def simplified_cheetah_flip(env: dmc.EnvWrapper) -> np.ndarray:
    # check the physics here:
    # https://github.com/deepmind/dm_control/blob/d72c22f3bb89178bff38728957daf62965632c2f/dm_control/suite/hopper.py
    return np.array([env.physics.speed(),
                     env.physics.angmomentum()],
                    dtype=np.float32)

# # # # #
# goals, defined on one goal_space to specify:
# # # # #


@goals("simplified_walker")
def walker_stand() -> np.ndarray:
    return np.array([1.2, 1.0, 0], dtype=np.float32)


@goals("simplified_walker")
def walker_walk() -> np.ndarray:
    return np.array([1.2, 1.0, 2], dtype=np.float32)


@goals("simplified_walker")
def walker_run() -> np.ndarray:
    return np.array([1.2, 1.0, 4], dtype=np.float32)


@goals("simplified_walker")
def walker_upside() -> np.ndarray:
    return np.array([1.2, -1.0, 0], dtype=np.float32)


@goals("simplified_quadruped")
def quadruped_stand() -> np.ndarray:
    return np.array([1.0, 0], dtype=np.float32)


@goals("simplified_quadruped")
def quadruped_walk() -> np.ndarray:
    return np.array([1.0, 0.6], dtype=np.float32)


@goals("simplified_quadruped")
def quadruped_run() -> np.ndarray:
    return np.array([1.0, 6], dtype=np.float32)


@goals("simplified_quadruped_velx")
def quadruped_walk() -> np.ndarray:
    return np.array([1.0, 0.6], dtype=np.float32)


@goals("simplified_point_mass_maze")
def point_mass_maze_reach_top_left() -> np.ndarray:
    return np.array(point_mass_maze_tasks['reach_top_left'],
                    dtype=np.float32)


@goals("simplified_point_mass_maze")
def point_mass_maze_reach_top_right() -> np.ndarray:
    return np.array(point_mass_maze_tasks['reach_top_right'],
                    dtype=np.float32)


@goals("simplified_point_mass_maze")
def point_mass_maze_reach_bottom_left() -> np.ndarray:
    return np.array(point_mass_maze_tasks['reach_bottom_left'],
                    dtype=np.float32)


@goals("simplified_point_mass_maze")
def point_mass_maze_reach_bottom_right() -> np.ndarray:
    return np.array(point_mass_maze_tasks['reach_bottom_right'],
                    dtype=np.float32)


# # # Custom Reward # # #


def _make_env(domain: str) -> dmc.EnvWrapper:
    task = {"quadruped": "stand",
            "walker": "walk",
            "point_mass_maze": "reach_bottom_right",
            "hopper": "hop",
            "cheetah": "walk"}[domain]
    return dmc.make(f"{domain}_{task}", obs_type="states", frame_stack=1, action_repeat=1, seed=12)


def get_goal_space_dim(name: str) -> int:
    domain = {space: domain for domain, spaces in goal_spaces.funcs.items()
              for space in spaces}[name]
    env = _make_env(domain)
    return goal_spaces.funcs[domain][name](env).size


class BaseReward:

    def __init__(self, seed: tp.Optional[int] = None) -> None:
        self._env: dmc.EnvWrapper  # to be instantiated in subclasses
        self._rng = np.random.RandomState(seed)

    def get_goal(self, goal_space: str) -> np.ndarray:
        raise NotImplementedError

    def from_physics(self, physics: np.ndarray) -> float:
        "careful this is not threadsafe"
        with self._env.physics.reset_context():
            self._env.physics.set_state(physics)
        return self.from_env(self._env)

    def from_env(self, env: dmc.EnvWrapper) -> float:
        raise NotImplementedError


def get_reward_function(name: str, seed: tp.Optional[int] = None) -> BaseReward:
    if name == "maze_multi_goal":
        return MazeMultiGoal(seed)
    return DmcReward(name)


def _inv(distance: float) -> float:
    # print("dist", distance)
    return 1 / (1 + abs(distance))


class DmcReward(BaseReward):

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

        env_name, task_name = name.split("_", maxsplit=1)
        try:
            from dm_control import suite  # import
            from url_benchmark import custom_dmc_tasks as cdmc
        except ImportError as e:
            raise dmc.UnsupportedPlatform("DMC does not run on Mac") from e
        make = suite.load if (
            env_name, task_name) in suite.ALL_TASKS else cdmc.make
        self._env = make(env_name, task_name)

    def from_env(self, env: dmc.EnvWrapper) -> float:
        return float(self._env.task.get_reward(env.physics))


class MazeMultiGoal(BaseReward):
    def __init__(self, seed: tp.Optional[int] = None) -> None:
        super().__init__(seed)
        self.goals_per_room = 5
        self.goals = np.array([
            [-0.15, 0.15],  # room 1: top left
            [-0.22, 0.22],  # room 1
            [-0.08, 0.08],  # room 1
            [-0.22, 0.08],  # room 1
            [-0.08, 0.22],  # room 1
            [0.15, 0.15],  # room 2: top right
            [0.22, 0.22],  # room 2
            [0.08, 0.08],  # room 2
            [0.22, 0.08],  # room 2
            [0.08, 0.22],  # room 2
            [-0.15, -0.15],  # room 3: bottom left
            [-0.22, -0.22],  # room 3
            [-0.08, -0.08],  # room 3
            [-0.22, -0.08],  # room 3
            [-0.08, -0.22],  # room 3
            [0.15, -0.15],  # room 4: bottom right
            [0.22, -0.22],  # room 4
            [0.08, -0.08],  # room 4
            [0.22, -0.08],  # room 4
            [0.08, -0.22],  # room 4
        ], dtype=np.float32)
        assert len(self.goals) == self.goals_per_room * 4

    def from_goal(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> tp.Tuple[float, float]:
        """returns reward and distance"""
        assert achieved_goal.shape == desired_goal.shape
        target_size = .03
        d: np.ndarray = achieved_goal - desired_goal
        distance = np.linalg.norm(
            d, axis=-1) if len(d.shape) > 0 else np.linalg.norm(d)
        reward = rewards.tolerance(distance,
                                   bounds=(0, target_size), margin=target_size)
        success = float(distance < target_size)
        return reward, distance, success

    def get_eval_states(self, num_states: int) -> torch.Tensor:
        # Samples states unioformly from the maze (avoiding region inside walls)
        state_max, state_min = [0.29, 0.29, 0.09,
                                0.09], [-0.29, -0.29, -0.09, -0.09]
        dis = torch.distributions.Uniform(low=torch.tensor(
            state_min), high=torch.tensor(state_max))
        samples = dis.sample((num_states,))
        # avoid sampling inside walls
        condition = (samples[:, 0] > 0.2) | (samples[:, 0] < -0.2) | (samples[:, 1] > 0.2) | (samples[:, 1] < -0.2) | \
            (((samples[:, 0] > 0.04) | (samples[:, 0] < -0.04)) &
             ((samples[:, 1] < -0.04) | (samples[:, 1] > 0.04)))
        samples = samples[condition]
        while len(samples) < 1000:
            new_samples = dis.sample((20,))
            samples = torch.concatenate((samples, new_samples), axis=0)
            condition = (samples[:, 0] > 0.2) | (samples[:, 0] < -0.2) | (samples[:, 1] > 0.2) | (samples[:, 1] < -0.2) | \
                (((samples[:, 0] > 0.04) | (samples[:, 0] < -0.04)) &
                 ((samples[:, 1] < -0.04) | (samples[:, 1] > 0.04)))
            samples = samples[condition]
        samples = samples[:num_states, :]
        return samples

    def get_eval_midroom_states(self,) -> torch.Tensor:
        states = torch.tensor([
            [-0.15, 0.15, 0., 0.],  # room 1: top left
            [0.15, 0.15, 0., 0.],  # room 2: top right
            [-0.15, -0.15, 0., 0.],  # room 3: bottom left
            [0.15, -0.15, 0., 0.],  # room 4: bottom right
        ])
        return states
