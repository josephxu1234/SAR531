from __future__ import annotations

import math

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.roomgrid import Room, RoomGrid
from minigrid.core.world_object import Door, Goal, Key, Lava, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv

from sar_objects import Exit, Person


class SAREnv(RoomGrid):
    def __init__(
        self,
        room_size: int = 7,
        num_rows: int = 3,
        num_cols: int = 3,
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
            max_steps = 4 * (room_size * num_rows) * (room_size * num_cols)

        mission_space = MissionSpace(mission_func=lambda: "Save the people")

        # Calculate size with padding of 2 (1 on each side)
        height = (room_size - 1) * num_rows + 1 + 2
        width = (room_size - 1) * num_cols + 1 + 2

        self.room_size = room_size
        self.num_rows = num_rows
        self.num_cols = num_cols

        # Initialize MiniGridEnv directly to bypass RoomGrid's size calculation
        MiniGridEnv.__init__(
            self,
            mission_space=mission_space,
            width=width,
            height=height,
            max_steps=max_steps,
            see_through_walls=False,
            agent_view_size=agent_view_size,  # Partial observability
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "Save the people"

    def room_from_pos(self, x: int, y: int) -> Room:
        """Get the room a given position maps to."""

        assert x >= 0
        assert y >= 0

        # Adjust for padding
        i = (x - 1) // (self.room_size - 1)
        j = (y - 1) // (self.room_size - 1)

        assert i < self.num_cols
        assert j < self.num_rows

        return self.room_grid[j][i]

    # Override the _gen_grid method to create a custom grid
    def _gen_grid(self, width: int, height: int) -> None:
        # Create the grid
        self.grid = Grid(width, height)

        # Create outer walls
        self.grid.wall_rect(0, 0, width, height)

        self.room_grid = []

        # For each row of rooms
        for j in range(0, self.num_rows):
            row = []

            # For each column of rooms
            for i in range(0, self.num_cols):
                # Add offset of (1, 1) to room position
                room = Room(
                    (
                        i * (self.room_size - 1) + 1,
                        j * (self.room_size - 1) + 1,
                    ),
                    (self.room_size, self.room_size),
                )
                row.append(room)

                # Generate the walls for this room
                self.grid.wall_rect(*room.top, *room.size)

            self.room_grid.append(row)

        # For each row of rooms
        for j in range(0, self.num_rows):
            # For each column of rooms
            for i in range(0, self.num_cols):
                room = self.room_grid[j][i]

                x_l, y_l = (room.top[0] + 1, room.top[1] + 1)
                x_m, y_m = (
                    room.top[0] + room.size[0] - 1,
                    room.top[1] + room.size[1] - 1,
                )

                # Door positions, order is right, down, left, up
                if i < self.num_cols - 1:
                    room.neighbors[0] = self.room_grid[j][i + 1]
                    room.door_pos[0] = (x_m, self._rand_int(y_l, y_m))
                if j < self.num_rows - 1:
                    room.neighbors[1] = self.room_grid[j + 1][i]
                    room.door_pos[1] = (self._rand_int(x_l, x_m), y_m)
                if i > 0:
                    room.neighbors[2] = self.room_grid[j][i - 1]
                    room.door_pos[2] = room.neighbors[2].door_pos[0]
                if j > 0:
                    room.neighbors[3] = self.room_grid[j - 1][i]
                    room.door_pos[3] = room.neighbors[3].door_pos[1]

        # The agent starts in the middle, facing right
        self.agent_pos = (
            (self.num_cols // 2) * (self.room_size - 1) + (self.room_size // 2) + 1,
            (self.num_rows // 2) * (self.room_size - 1) + (self.room_size // 2) + 1,
        )
        self.agent_dir = 0

        # Initialize exit positions list
        self.exit_positions = []

        # Connect all rooms with doors
        added_doors = self.connect_all()

        for door in added_doors:
            door.is_locked = False
            door.is_open = True

        # Place exits around the perimeter
        self._place_exits(width, height)

        # Place people in random free spots inside the building
        self._place_people()

        # Place collapsed floors in random free spots inside the building
        self._place_collapsed_floors()

        # Set agent starting position (if custom position is provided)
        self._set_agent_start()

        # Set mission
        self.mission = "Save the people"

    def _place_exits(self, width: int, height: int) -> None:
        """Place exits randomly distributed around the building perimeter."""

        # Get all available wall positions on the perimeter
        available_positions = []

        # Top wall (inset by 1 due to outer wall)
        # The building starts at y=1. The top wall of the building is at y=1.
        for x in range(2, width - 2):
            if self.grid.get(x, 1) is not None and self.grid.get(x, 1).type == "wall":
                available_positions.append((x, 1))

        # Bottom wall
        # The building ends at height-2. The bottom wall is at height-2.
        for x in range(2, width - 2):
            if (
                self.grid.get(x, height - 2) is not None
                and self.grid.get(x, height - 2).type == "wall"
            ):
                available_positions.append((x, height - 2))

        # Left wall
        # The building starts at x=1. The left wall is at x=1.
        for y in range(2, height - 2):
            if self.grid.get(1, y) is not None and self.grid.get(1, y).type == "wall":
                available_positions.append((1, y))

        # Right wall
        # The building ends at width-2. The right wall is at width-2.
        for y in range(2, height - 2):
            if (
                self.grid.get(width - 2, y) is not None
                and self.grid.get(width - 2, y).type == "wall"
            ):
                available_positions.append((width - 2, y))

        for _ in range(self.num_exits):
            if not available_positions:
                break
            # Randomly select one position and place exit
            pos_idx = self._rand_int(0, len(available_positions))
            x, y = available_positions[pos_idx]
            self.grid.set(x, y, Exit())
            # Store exit position for rescue detection
            self.exit_positions.append((x, y))
            # Remove used position
            available_positions.pop(pos_idx)

    def _place_people(self) -> None:
        """Place people in random free spots inside the building."""
        for i in range(self.num_people):
            person = Person(color="purple")
            # place_obj will find a random empty position
            self.place_obj(person)

    def _place_collapsed_floors(self) -> None:
        for i in range(self.num_collapsed_floors):
            collapsed_floor = Lava()
            # place_obj will find a random empty position
            self.place_obj(collapsed_floor)

    def _set_agent_start(self) -> None:
        """Set agent starting position."""
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
            # Ensure the start position is valid (cleared)
            self.grid.set(*self.agent_pos, None)
        # If agent_start_pos is None, RoomGrid._gen_grid has already placed the agent randomly

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
