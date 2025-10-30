from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv


class SAREnv(MiniGridEnv):
    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        agent_start_pos: tuple[int, int] = (1, 1),
        agent_start_dir: int = 0,
        max_steps: int | None = None,
        agent_view_size: int = 5,
        **kwargs,
    ):
        self.agent_pos = agent_start_pos
        self.agent_dir = agent_start_dir
        self.agent_view_size = agent_view_size

        mission_space = MissionSpace(mission_func=lambda: "reach the goal")

        super().__init__(
            mission_space=mission_space,
            width=width,  # width of the grid
            height=height,  # height of the grid
            see_through_walls=False,  # Agent cannot see through walls
            max_steps=max_steps,  # Maximum number of steps per episode
            agent_pov=True,  # Limit agent's field of view
            agent_view_size=self.agent_view_size,  # Size of the agent's view
        )

    def _gen_mission():
        return "Save the people"
