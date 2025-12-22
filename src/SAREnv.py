from __future__ import annotations

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.roomgrid import RoomGrid, Room
from minigrid.core.world_object import Lava
from minigrid.minigrid_env import MiniGridEnv

from sar_objects import Exit, Person, Floor


class SAREnv(RoomGrid):
    """Search and Rescue environment with zone-based person/lava placement."""
    
    # medical rooms likely to contain people, not lava
    # industral zones likely to contain lava, not people
    # common rooms have equal likelihood of people and lava
    # storage rooms likely to be empty
    ZONE_TYPES = {
        'medical': {'color': 'green', 'person_weight': 3.0, 'lava_weight': 0.2},
        'industrial': {'color': 'red', 'person_weight': 0.2, 'lava_weight': 3.0},
        'common': {'color': 'blue', 'person_weight': 1.0, 'lava_weight': 1.0},
        'storage': {'color': 'grey', 'person_weight': 0.5, 'lava_weight': 0.5}
    }
    
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
        self.people_rescued = 0
        self.exit_positions = []
        
        if max_steps is None:
            max_steps = 4 * (room_size * num_rows) * (room_size * num_cols)

        self.room_size = room_size
        self.num_rows = num_rows
        self.num_cols = num_cols

        # Determine total height and width of building
        height = (room_size - 1) * num_rows + 1 + 2
        width = (room_size - 1) * num_cols + 1 + 2

        MiniGridEnv.__init__(
            self,
            mission_space=MissionSpace(mission_func=lambda: "Save the people"),
            width=width,
            height=height,
            max_steps=max_steps,
            see_through_walls=False,
            agent_view_size=agent_view_size,
            **kwargs,
        )

    def room_from_pos(self, x: int, y: int) -> Room:
        """Get the room a given position maps to."""
        i = (x - 1) // (self.room_size - 1)
        j = (y - 1) // (self.room_size - 1)
        return self.room_grid[j][i]

    def _gen_grid(self, width: int, height: int) -> None:
        """Generate the grid including positions of all objects"""
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self.room_grid = []

        # Create rooms
        for j in range(self.num_rows):
            row = []
            for i in range(self.num_cols):
                room = Room(
                    (i * (self.room_size - 1) + 1, j * (self.room_size - 1) + 1),
                    (self.room_size, self.room_size),
                )
                row.append(room)
                self.grid.wall_rect(*room.top, *room.size)
            self.room_grid.append(row)

        # Set door positions
        for j in range(self.num_rows):
            for i in range(self.num_cols):
                room = self.room_grid[j][i]
                x_l, y_l = room.top[0] + 1, room.top[1] + 1
                x_m, y_m = room.top[0] + room.size[0] - 1, room.top[1] + room.size[1] - 1

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

        # Set agent starting position
        self.agent_pos = (
            (self.num_cols // 2) * (self.room_size - 1) + (self.room_size // 2) + 1,
            (self.num_rows // 2) * (self.room_size - 1) + (self.room_size // 2) + 1,
        )
        self.agent_dir = 0
        self.exit_positions = []

        # Assign zone types and place colored floor tiles
        for j in range(self.num_rows):
            for i in range(self.num_cols):
                room = self.room_grid[j][i]
                zone_type = self._rand_elem(list(self.ZONE_TYPES.keys()))
                room.zone_type = zone_type
                room.zone_color = self.ZONE_TYPES[zone_type]['color']
                
                for x in range(room.top[0] + 1, room.top[0] + room.size[0] - 1):
                    for y in range(room.top[1] + 1, room.top[1] + room.size[1] - 1):
                        self.grid.set(x, y, Floor(room.zone_color))

        # Connect rooms with open doors
        added_doors = self.connect_all()
        for door in added_doors:
            door.is_locked = False
            door.is_open = True

        self._place_exits(width, height)
        self._place_people()
        self._place_collapsed_floors()
        self._set_agent_start()

    def _place_exits(self, width: int, height: int) -> None:
        """Place exits randomly on building perimeter"""
        available_positions = []

        for x in range(2, width - 2):
            if self.grid.get(x, 1) is not None and self.grid.get(x, 1).type == "wall":
                available_positions.append((x, 1))
            if self.grid.get(x, height - 2) is not None and self.grid.get(x, height - 2).type == "wall":
                available_positions.append((x, height - 2))

        for y in range(2, height - 2):
            if self.grid.get(1, y) is not None and self.grid.get(1, y).type == "wall":
                available_positions.append((1, y))
            if self.grid.get(width - 2, y) is not None and self.grid.get(width - 2, y).type == "wall":
                available_positions.append((width - 2, y))

        for _ in range(self.num_exits):
            if not available_positions:
                break
            pos_idx = self._rand_int(0, len(available_positions))
            x, y = available_positions.pop(pos_idx)
            self.grid.set(x, y, Exit())
            self.exit_positions.append((x, y))

    def _place_people(self) -> None:
        """Place people weighted by zone type"""
        for i in range(self.num_people):
            rooms = []
            weights = []
            for row in self.room_grid:
                for room in row:
                    rooms.append(room)
                    weights.append(self.ZONE_TYPES[room.zone_type]['person_weight'])
            
            total_weight = sum(weights)
            probabilities = [w / total_weight for w in weights]
            selected_idx = self.np_random.choice(len(rooms), p=probabilities)
            selected_room = rooms[selected_idx]
            
            top_x, top_y = selected_room.top[0] + 1, selected_room.top[1] + 1
            size_x, size_y = selected_room.size[0] - 2, selected_room.size[1] - 2
            
            available_positions = []
            for x in range(top_x, top_x + size_x):
                for y in range(top_y, top_y + size_y):
                    cell = self.grid.get(x, y)
                    if cell is None or (hasattr(cell, 'type') and cell.type == 'floor'):
                        available_positions.append((x, y))
            
            if available_positions:
                pos_idx = self._rand_int(0, len(available_positions))
                x, y = available_positions[pos_idx]
                self.grid.set(x, y, Person(color="purple"))

    def _place_collapsed_floors(self) -> None:
        """Place lava weighted by zone type"""
        for i in range(self.num_collapsed_floors):
            rooms = []
            weights = []
            for row in self.room_grid:
                for room in row:
                    rooms.append(room)
                    weights.append(self.ZONE_TYPES[room.zone_type]['lava_weight'])
            
            total_weight = sum(weights)
            probabilities = [w / total_weight for w in weights]
            selected_idx = self.np_random.choice(len(rooms), p=probabilities)
            selected_room = rooms[selected_idx]
            
            top_x, top_y = selected_room.top[0] + 1, selected_room.top[1] + 1
            size_x, size_y = selected_room.size[0] - 2, selected_room.size[1] - 2
            
            available_positions = []
            for x in range(top_x, top_x + size_x):
                for y in range(top_y, top_y + size_y):
                    cell = self.grid.get(x, y)
                    if cell is None or (hasattr(cell, 'type') and cell.type == 'floor'):
                        available_positions.append((x, y))
            
            if available_positions:
                pos_idx = self._rand_int(0, len(available_positions))
                x, y = available_positions[pos_idx]
                self.grid.set(x, y, Lava())

    def _set_agent_start(self) -> None:
        """Set agent starting position"""
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
            self.grid.set(*self.agent_pos, None)
        elif self.exit_positions:
            self.agent_pos = self.exit_positions[self._rand_int(0, len(self.exit_positions))]
            x, y = self.agent_pos
            if x == 1:
                self.agent_dir = 0
            elif x == self.width - 2:
                self.agent_dir = 2
            elif y == 1:
                self.agent_dir = 1
            elif y == self.height - 2:
                self.agent_dir = 3

    def gen_obs_grid(self, agent_view_size=None):
        """Generate observation grid, ensuring agent sees what's under them"""
        grid, vis_mask = super().gen_obs_grid(agent_view_size)

        if self.carrying is None:
            agent_view_size = agent_view_size or self.agent_view_size
            agent_pos_in_view = (agent_view_size // 2, agent_view_size - 1)
            world_cell = self.grid.get(*self.agent_pos)
            if world_cell is not None:
                grid.set(*agent_pos_in_view, world_cell)
                
        return grid, vis_mask

    def step(self, action):
        """Execute action with rescue mechanics and rewards"""
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)

        obs, reward, terminated, truncated, info = super().step(action)

        if self.people_rescued < self.num_people:
            terminated = False

        # Penalty for stepping on lava
        if action == self.actions.forward and fwd_cell is not None and fwd_cell.type == "lava":
            reward -= 100
            terminated = True

        # Check for successful rescue
        agent_pos = tuple(self.agent_pos)
        if self.carrying is not None and self.carrying.type == "ball" and agent_pos in self.exit_positions:
            reward += 100
            self.people_rescued += 1
            self.carrying = None

            if self.people_rescued >= self.num_people:
                terminated = True
                reward += 50
                info["success"] = True
            else:
                terminated = False
                info["rescue"] = True

        reward -= 0.1
        info["people_rescued"] = self.people_rescued
        info["carrying_person"] = self.carrying is not None and self.carrying.type == "ball"

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset environment"""
        self.people_rescued = 0
        obs, info = super().reset(**kwargs)
        return obs, info