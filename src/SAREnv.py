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
        agent_view_size: int = 3,
        **kwargs,
    ):
        self.num_people = num_people
        self.num_exits = num_exits
        self.num_collapsed_floors = num_collapsed_floors
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        self.people_rescued = 0  # Track number of people rescued
        self.exit_positions = []  # Initialize here to avoid reset issues

        # Set default max_steps if not provided
        if max_steps is None:
            max_steps = 4 * (room_size * num_rows) * (room_size * num_cols)

        mission_space = MissionSpace(mission_func=self._gen_mission)

        self.room_size = room_size
        self.num_rows = num_rows
        self.num_cols = num_cols

        # Calculate size with padding of 2 (1 on each side)
        height = (room_size - 1) * num_rows + 1 + 2
        width = (room_size - 1) * num_cols + 1 + 2

        # Initialize MiniGridEnv directly with specific parameters
        MiniGridEnv.__init__(
            self,
            mission_space=mission_space,
            width=width,
            height=height,
            max_steps=max_steps,
            see_through_walls=False,  # Cannot see through walls
            agent_view_size=agent_view_size,  # Limited field of view
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

        # Top wall
        for x in range(2, width - 2):
            if self.grid.get(x, 1) is not None and self.grid.get(x, 1).type == "wall":
                available_positions.append((x, 1))

        # Bottom wall
        for x in range(2, width - 2):
            if (
                self.grid.get(x, height - 2) is not None
                and self.grid.get(x, height - 2).type == "wall"
            ):
                available_positions.append((x, height - 2))

        # Left wall
        for y in range(2, height - 2):
            if self.grid.get(1, y) is not None and self.grid.get(1, y).type == "wall":
                available_positions.append((1, y))

        # Right wall
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
            # Make sure we don't place on walls or perimeter
            placed = False
            attempts = 0
            while not placed and attempts < 100:
                try:
                    self.place_obj(person, max_tries=100)
                    # Verify placement is not on perimeter
                    if person.cur_pos is not None:
                        x, y = person.cur_pos
                        # Check if on perimeter (should be at least 2 cells from edge)
                        if x >= 2 and x < self.width - 2 and y >= 2 and y < self.height - 2:
                            placed = True
                        else:
                            # Remove and try again
                            self.grid.set(x, y, None)
                            attempts += 1
                    else:
                        placed = True
                except:
                    attempts += 1
            
            if not placed:
                print(f"Warning: Could not place person {i} away from perimeter")

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
        elif self.exit_positions:
            # Start at a random exit
            self.agent_pos = self.exit_positions[
                self._rand_int(0, len(self.exit_positions))
            ]

            # Face towards the center of the building
            x, y = self.agent_pos
            if x == 1:  # Left wall
                self.agent_dir = 0  # Right
            elif x == self.width - 2:  # Right wall
                self.agent_dir = 2  # Left
            elif y == 1:  # Top wall
                self.agent_dir = 1  # Down
            elif y == self.height - 2:  # Bottom wall
                self.agent_dir = 3  # Up

    def gen_obs_grid(self, agent_view_size=None):
        """
        Generate the sub-grid observed by the agent.
        Overridden to ensure the agent sees what is under it (e.g. Exit) 
        when not carrying anything.
        """
        grid, vis_mask = super().gen_obs_grid(agent_view_size)

        # If not carrying anything, ensure the object under the agent is visible
        if self.carrying is None:
            agent_view_size = agent_view_size or self.agent_view_size
            agent_pos_in_view = (agent_view_size // 2, agent_view_size - 1)
            
            # Get what is actually at the agent's position in the world
            world_cell = self.grid.get(*self.agent_pos)
            
            # If there is something there (like an Exit), put it in the view
            if world_cell is not None:
                grid.set(*agent_pos_in_view, world_cell)
                
        return grid, vis_mask

    def step(self, action):
        """Override step to handle rescue mechanics and rewards."""

        # Store state before action
        agent_pos_before = tuple(self.agent_pos)
        carrying_person_before = (
            self.carrying is not None and self.carrying.type == "ball"
        )
        
        # Get forward cell before action
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)

        # Take the action
        obs, reward, terminated, truncated, info = super().step(action)

        # Control termination ourselves based on rescue completion
        if self.people_rescued < self.num_people:
            terminated = False  # Override parent class termination

        # Penalty for stepping on collapsed floor (only on forward action)
        if action == self.actions.forward and fwd_cell is not None and fwd_cell.type == "lava":
            reward -= 100
            print('Stepped on collapsed floor! -100 penalty.')
            terminated = True

        # Get current position after action
        agent_pos = tuple(self.agent_pos)
        
        # No auto-pickup - agent must use pickup action when adjacent to person

        # Check if agent is on an exit position while carrying a person
        rescue_successful = False
        if (
            self.carrying is not None
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
                terminated = True
                reward += 50  # Bonus for completing mission
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
        info["people_rescued"] = self.people_rescued
        info["carrying_person"] = (
            self.carrying is not None and self.carrying.type == "ball"
        )

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset environment and rescue tracking."""
        self.people_rescued = 0
        obs, info = super().reset(**kwargs)
        return obs, info