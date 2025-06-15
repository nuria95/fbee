# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pdb  # pylint: disable=unused-import
import sys
import unittest
import dataclasses
import typing as tp
from typing import Any

from dm_env import Environment
from dm_env import StepType, specs
import numpy as np

""" 
Code from  https://github.com/facebookresearch/controllable_agent
"""


class UnsupportedPlatform(unittest.SkipTest, RuntimeError):
    """The platform is not supported for running"""


try:
    from dm_control import suite  # , manipulation
    from dm_control.suite.wrappers import action_scale, pixels
    from url_benchmark import custom_dmc_tasks as cdmc
except ImportError as e:
    raise UnsupportedPlatform(
        f"Import error (Note: DMC does not run on Mac):\n{e}") from e


S = tp.TypeVar("S", bound="TimeStep")
Env = tp.Union["EnvWrapper", Environment]


@dataclasses.dataclass
class TimeStep:
    step_type: StepType
    reward: float
    discount: float
    observation: np.ndarray
    physics: np.ndarray = dataclasses.field(default=np.ndarray([]), init=False)

    def first(self) -> bool:
        return self.step_type == StepType.FIRST  # type: ignore

    def mid(self) -> bool:
        return self.step_type == StepType.MID  # type: ignore

    def last(self) -> bool:
        return self.step_type == StepType.LAST  # type: ignore

    def __getitem__(self, attr: str) -> tp.Any:
        return getattr(self, attr)

    def _replace(self: S, **kwargs: tp.Any) -> S:
        for name, val in kwargs.items():
            setattr(self, name, val)
        return self


@dataclasses.dataclass
class GoalTimeStep(TimeStep):
    goal: np.ndarray


@dataclasses.dataclass
class ExtendedGoalTimeStep(GoalTimeStep):
    action: tp.Any


@dataclasses.dataclass
class ExtendedTimeStep(TimeStep):
    action: tp.Any


class EnvWrapper:
    def __init__(self, env: Env) -> None:
        self._env = env

    def _augment_time_step(self, time_step: TimeStep, action: tp.Optional[np.ndarray] = None) -> TimeStep:
        if not isinstance(time_step, TimeStep):
            # dm_env time step is a named tuple
            time_step = TimeStep(**time_step._asdict())
        if self.physics is not None:
            return time_step._replace(physics=self.physics.get_state())
        else:
            return time_step

    def reset(self) -> TimeStep:
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action: np.ndarray) -> TimeStep:
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def observation_spec(self) -> tp.Any:
        assert isinstance(self, EnvWrapper)
        return self._env.observation_spec()

    def action_spec(self) -> specs.Array:
        return self._env.action_spec()

    def render(self, *args: tp.Any, **kwargs: tp.Any) -> np.ndarray:
        return self._env.render(*args, **kwargs)  # type: ignore

    @property
    def base_env(self) -> tp.Any:
        env = self._env
        if isinstance(env, EnvWrapper):
            return self.base_env
        return env

    @property
    def physics(self) -> tp.Any:
        if hasattr(self._env, "physics"):
            return self._env.physics

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionRepeatWrapper(EnvWrapper):
    def __init__(self, env: tp.Any, num_repeats: int) -> None:
        super().__init__(env)
        self._num_repeats = num_repeats

    def step(self, action: np.ndarray) -> TimeStep:
        reward = 0.0
        discount = 1.0
        for _ in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)


class GoalWrapper(EnvWrapper):
    def __init__(self, env: Env, goal_func: tp.Callable[[Env], np.ndarray], append_goal_to_observation: bool = False) -> None:
        """Adds a goal space with a predefined function.
        This can also append the observation with the goal to make sure the goal is achievable
        """
        super().__init__(env)
        self.append_goal_to_observation = append_goal_to_observation
        self.goal_func = goal_func

    def _augment_time_step(self, time_step: TimeStep, action: tp.Optional[np.ndarray] = None) -> TimeStep:
        goal = self.goal_func(self)
        obs = time_step.observation.copy()
        if self.append_goal_to_observation:
            k = "observations"
            obs[k] = np.concatenate([obs[k], goal], axis=0)
            # obs[k] = np.concatenate([obs[k], np.random.normal(size=goal.shape)], axis=0)
        ts = GoalTimeStep(
            step_type=time_step.step_type,
            reward=time_step.reward,
            discount=time_step.discount,
            observation=obs,
            goal=goal,
        )
        return super()._augment_time_step(time_step=ts, action=action)

    def observation_spec(self) -> specs.Array:
        spec = super().observation_spec().copy()
        k = "observations"
        if not self.append_goal_to_observation:
            return spec
        goal = self.goal_func(self)
        spec[k] = specs.Array(
            (spec[k].shape[0] + goal.shape[0],), dtype=np.float32, name=k)
        return spec


class ActionDTypeWrapper(EnvWrapper):
    def __init__(self, env: Env, dtype) -> None:
        super().__init__(env)
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                               dtype,
                                               wrapped_action_spec.minimum,
                                               wrapped_action_spec.maximum,
                                               'action')

    def action_spec(self) -> specs.BoundedArray:
        return self._action_spec

    def step(self, action) -> Any:
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)


class ObservationDTypeWrapper(EnvWrapper):
    def __init__(self, env: Env, dtype) -> None:
        super().__init__(env)
        self._dtype = dtype
        wrapped_obs_spec = env.observation_spec()['observations']
        self._obs_spec = specs.Array(wrapped_obs_spec.shape, dtype,
                                     'observation')

    def _augment_time_step(self, time_step: TimeStep, action: tp.Optional[np.ndarray] = None) -> TimeStep:
        obs = time_step.observation['observations'].astype(self._dtype)
        return time_step._replace(observation=obs)

    def observation_spec(self) -> Any:
        return self._obs_spec


class ExtendedGoalTimeStepWrapper(EnvWrapper):

    def _augment_time_step(self, time_step: TimeStep, action: tp.Optional[np.ndarray] = None) -> TimeStep:
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        assert isinstance(time_step, GoalTimeStep)
        ts = ExtendedGoalTimeStep(observation=time_step.observation,
                                  step_type=time_step.step_type,
                                  action=action,
                                  reward=time_step.reward or 0.0,
                                  discount=time_step.discount or 1.0,
                                  goal=time_step.goal)
        return super()._augment_time_step(time_step=ts, action=action)


class ExtendedTimeStepWrapper(EnvWrapper):

    def _augment_time_step(self, time_step: TimeStep, action: tp.Optional[np.ndarray] = None) -> TimeStep:
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        ts = ExtendedTimeStep(observation=time_step.observation,
                              step_type=time_step.step_type,
                              action=action,
                              reward=time_step.reward or 0.0,
                              discount=time_step.discount or 1.0)
        return super()._augment_time_step(time_step=ts, action=action)


def _make_dmc(obs_type, domain, task, frame_stack, action_repeat, seed,
              goal_space: tp.Optional[str] = None, append_goal_to_observation: bool = False):
    visualize_reward = False
    if (domain, task) in suite.ALL_TASKS:
        env = suite.load(domain,
                         task,
                         task_kwargs=dict(random=seed),
                         environment_kwargs=dict(flat_observation=True),
                         visualize_reward=visualize_reward)
    else:
        #  only use customized dmc in case task is not in default dmc! (jump, roll, stand...)
        env = cdmc.make(domain,
                        task,
                        task_kwargs=dict(random=seed),
                        environment_kwargs=dict(flat_observation=True),
                        visualize_reward=visualize_reward)
    if goal_space is not None:
        # inline because circular import
        # pytlint: disable=import-outside-toplevel
        from url_benchmark import goals as _goals
        funcs = _goals.goal_spaces.funcs[domain]
        if goal_space not in funcs:
            raise ValueError(
                f"No goal space {goal_space} for {domain}, avail: {list(funcs)}")
        goal_func = funcs[goal_space]
        env = GoalWrapper(
            env, goal_func, append_goal_to_observation=append_goal_to_observation)
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    return env


def make(
    name: str, obs_type='states', frame_stack=1, action_repeat=1,
    seed=1, goal_space: tp.Optional[str] = None, append_goal_to_observation: bool = False
) -> EnvWrapper:
    if append_goal_to_observation and goal_space is None:
        raise ValueError("Cannot append goal space since none is defined")
    assert obs_type in ['states', 'pixels']
    if name.startswith('point_mass_maze'):
        domain = 'point_mass_maze'
        _, _, _, task = name.split('_', 3)
    elif name.startswith('ball_in_cup'):
        domain = 'ball_in_cup'
        _, _, _, task = name.split('_', 3)
    else:
        domain, task = name.split('_', 1)
    if sys.platform == "darwin":
        raise UnsupportedPlatform("Mac platform is not supported")

    make_fn = _make_dmc
    env = make_fn(obs_type, domain, task, frame_stack, action_repeat, seed,
                  goal_space=goal_space, append_goal_to_observation=append_goal_to_observation)  # type: ignore

    env = ObservationDTypeWrapper(env, np.float32)

    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    if goal_space is not None:
        env = ExtendedGoalTimeStepWrapper(env)
    else:
        env = ExtendedTimeStepWrapper(env)
    return env


def extract_physics(env: Env) -> tp.Dict[str, float]:
    """Extract some physics available in the env"""
    output = {}
    names = ["torso_height", "torso_upright",
             "horizontal_velocity", "torso_velocity"]
    for name in names:
        if not hasattr(env.physics, name):
            continue
        val: tp.Union[float, np.ndarray] = getattr(env.physics, name)()
        if isinstance(val, (int, float)) or not val.ndim:
            output[name] = float(val)
        else:
            for k, v in enumerate(val):
                output[f"{name}#{k}"] = float(v)
    return output


class FloatStats:
    """Handle for keeping track of the statistics of a float variable"""

    def __init__(self) -> None:
        self.min = np.inf
        self.max = -np.inf
        self.mean = 0.0
        self._count = 0

    def add(self, value: float) -> "FloatStats":
        self.min = min(value, self.min)
        self.max = max(value, self.max)
        self._count += 1
        self.mean = (self._count - 1) / self._count * \
            self.mean + 1 / self._count * value
        return self

    def items(self) -> tp.Iterator[tp.Tuple[str, float]]:
        for name, val in self.__dict__.items():
            if not name.startswith("_"):
                yield name, val


class PhysicsAggregator:
    """Aggregate stats on the physics of an environment"""

    def __init__(self) -> None:
        self.stats: tp.Dict[str, FloatStats] = {}

    def add(self, env: Env) -> "PhysicsAggregator":
        phy = extract_physics(env)
        for key, val in phy.items():
            self.stats.setdefault(key, FloatStats()).add(val)
        return self

    def dump(self) -> tp.Iterator[tp.Tuple[str, float]]:
        """Exports all statistics and reset the statistics"""
        for key, stats in self.stats.items():
            for stat, val in stats.items():
                yield (f'{key}/{stat}', val)
        self.stats.clear()
