from __future__ import annotations

import math

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Lava, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv

from sar_objects import Exit, Person


class SAREnv(MiniGridEnv):
    def __init__(
        self,
        width: int = 15,
        height: int = 15,
        num_people: int = 1,
        num_exits: int = 2,
        num_collapsed_floors: int = 2,
        agent_start_pos: tuple[int, int] | None = None,
        agent_start_dir: int = 0,
        max_steps: int | None = None,
        agent_view_size: int = 5,
        **kwargs,
    ):
        self.num_people = num_people
        self.num_exits = num_exits
        self.num_collapsed_floors = num_collapsed_floors
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        self.people_rescued = 0  # Track number of people rescued

        # Set default max_steps if not provided
        if max_steps is None:
            max_steps = 4 * width * height

        mission_space = MissionSpace(mission_func=lambda: "Save the people")

        super().__init__(
            mission_space=mission_space,
            width=width,
            height=height,
            see_through_walls=False,  # Agent cannot see through walls
            max_steps=max_steps,
            agent_pov=False,
            agent_view_size=agent_view_size,  # Partial observability
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "Save the people"

    # Override the _gen_grid method to create a custom grid
    def _gen_grid(self, width: int, height: int) -> None:
        # Create an empty grid
        self.grid = Grid(width, height)

        # Initialize exit positions list
        self.exit_positions = []

        # Generate the building perimeter (walls around the entire grid)
        self._create_building_perimeter(width, height)

        # Place exits around the perimeter
        self._place_exits(width, height)

        # Place people in random free spots inside the building
        self._place_people()

        # Place collapsed floors in random free spots inside the building
        self._place_collapsed_floors()

        # Set agent starting position
        self._set_agent_start()

        # Set mission
        self.mission = "Save the people"

    def _create_building_perimeter(self, width: int, height: int) -> None:
        """Create walls around the perimeter of the building, inset by 1 to avoid edge
        issues."""
        # Create walls 1 space in from the edge to prevent out-of-bounds errors
        self.grid.wall_rect(1, 1, width - 2, height - 2)
        self.grid.wall_rect(0, 0, width, height)  # Outer walls

    def _place_exits(self, width: int, height: int) -> None:
        """Place exits randomly distributed around the building walls (inset by 1)"""

        # Get all available wall positions on the inset perimeter
        available_positions = []

        # Top wall (inset, excluding corners)
        for x in range(2, width - 2):
            if self.grid.get(x, 1) is not None and self.grid.get(x, 1).type == "wall":
                available_positions.append((x, 1))

        # Bottom wall (inset, excluding corners)
        for x in range(2, width - 2):
            if (
                self.grid.get(x, height - 2) is not None
                and self.grid.get(x, height - 2).type == "wall"
            ):
                available_positions.append((x, height - 2))

        # Left wall (inset, excluding corners)
        for y in range(2, height - 2):
            if self.grid.get(1, y) is not None and self.grid.get(1, y).type == "wall":
                available_positions.append((1, y))

        # Right wall (inset, excluding corners)
        for y in range(2, height - 2):
            if (
                self.grid.get(width - 2, y) is not None
                and self.grid.get(width - 2, y).type == "wall"
            ):
                available_positions.append((width - 2, y))

        for _ in range(self.num_exits):
            # Randomly select one position and place exit
            pos_idx = self._rand_int(0, len(available_positions))
            x, y = available_positions[pos_idx]
            self.put_obj(Exit(), x, y)
            # Store exit position for rescue detection
            self.exit_positions.append((x, y))
            # Remove used position
            available_positions.pop(pos_idx)

    def _place_people(self) -> None:
        """Place people in random free spots inside the building."""
        for i in range(self.num_people):
            person = Person(color="purple")

            # place_obj will find a random empty position inside the building interior
            self.place_obj(
                person,
                top=(2, 2),  # top: top left position of rectangle to place within
                size=(self.width - 4, self.height - 4),  # Avoid the perimeter and walls
            )

    def _place_collapsed_floors(self) -> None:
        for i in range(self.num_collapsed_floors):
            collapsed_floor = Lava()

            # place_obj will find a random empty position inside the building interior
            self.place_obj(
                collapsed_floor,
                top=(2, 2),  # top: top left position of rectangle to place within
                size=(self.width - 4, self.height - 4),  # Avoid the perimeter and walls
            )

    def _set_agent_start(self) -> None:
        """Set agent starting position."""
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            # Place agent randomly inside the building interior, avoiding walls
            self.place_agent(
                top=(2, 2), size=(self.width - 4, self.height - 4), rand_dir=True
            )

    def step(self, action):
        """Override step to handle rescue mechanics and rewards."""

        # Check if agent is carrying a person before taking action
        carrying_person_before = (
            self.carrying is not None and self.carrying.type == "ball"
        )

        fwd_cell = self.grid.get(*self.front_pos)

        # Take the action
        obs, reward, terminated, truncated, info = super().step(action)

        # Control termination ourselves based on rescue completion
        if self.people_rescued < self.num_people:
            terminated = False  # Override parent class termination

        # Penalty fo steppin on collapsed floor
        if fwd_cell is not None and fwd_cell.type == "lava":
            reward -= 100
            terminated = True  # should we terminate here? or just penalize?

        agent_pos = tuple(self.agent_pos)

        # Check if agent is on an exit position while carrying a person
        rescue_successful = False
        if (
            carrying_person_before
            and self.carrying
            and self.carrying.type == "ball"
            and agent_pos in self.exit_positions
        ):
            rescue_successful = True

        if rescue_successful:
            # Successful rescue
            reward += 100  # Large reward for rescue
            self.people_rescued += 1

            # Remove the rescued person from inventory
            self.carrying = None

            # Check if all people are rescued
            if self.people_rescued >= self.num_people:
                terminated = True  # All people rescued, can safely terminate
                reward += (
                    50  # Bonus for completing mission # TODO: should we keep this?
                )
                info["success"] = True
                info["message"] = f"All {self.num_people} people rescued!"
                print("Mission Complete: All people rescued!")
            else:
                # Continue mission - more people to rescue
                terminated = False
                info["rescue"] = True
                info["message"] = (
                    f"Person rescued! {self.people_rescued}/{self.num_people} complete"
                )
                print(
                    "Person rescued! {}/{}".format(self.people_rescued, self.num_people)
                )

        # Small penalty for each step
        reward -= 0.1

        # Add info about current state
        info["people_rescued"] = getattr(self, "people_rescued", 0)
        info["carrying_person"] = (
            self.carrying is not None and self.carrying.type == "ball"
        )

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset environment and rescue tracking."""
        self.people_rescued = 0
        # Don't reset exit_positions here
        obs, info = super().reset(**kwargs)
        return obs, info
