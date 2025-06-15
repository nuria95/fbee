# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" 
Code from  https://github.com/facebookresearch/controllable_agent
"""

from typing import List

DOMAINS = [
    'walker',
    'quadruped',
    'point_mass_maze'
    'cheetah'
]

CHEETAH_TASKS = [
    'cheetah_walk',
    'cheetah_walk_backward',
    'cheetah_run',
    'cheetah_run_backward'
]

WALKER_TASKS = [
    'walker_stand',
    'walker_walk',
    'walker_run',
    'walker_flip',
    'walker_upside'
]

QUADRUPED_TASKS = [
    'quadruped_walk',
    'quadruped_run',
    'quadruped_stand',
    'quadruped_jump',
]


POINT_MASS_MAZE_TASKS = [
    'point_mass_maze_reach_top_left',
    'point_mass_maze_reach_top_right',
    'point_mass_maze_reach_bottom_left',
    'point_mass_maze_reach_bottom_right',
]



TASKS: List[str] = WALKER_TASKS + QUADRUPED_TASKS + POINT_MASS_MAZE_TASKS

PRIMAL_TASKS = {
    'walker': 'walker_stand',
    'quadruped': 'quadruped_walk'
}
