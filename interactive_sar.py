#!/usr/bin/env python3
"""Test script for the SAR Environment."""

import os
import sys

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv

from SAREnv import SAREnv


def test_manual_control():
    env = SAREnv(room_size=10, num_rows=4, num_cols=5, num_people=3, num_exits=4, render_mode="human")
    manual_control = ManualControl(env)
    manual_control.start()

if __name__ == "__main__":
    # press tab to pick up a person
    # press shift to drop the person
    # arrow keys to rotate/move
    test_manual_control()
