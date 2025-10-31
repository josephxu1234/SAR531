from __future__ import annotations

import numpy as np
from minigrid.core.constants import COLOR_TO_IDX, COLORS, OBJECT_TO_IDX
from minigrid.core.world_object import WorldObj
from minigrid.utils.rendering import fill_coords, point_in_circle, point_in_rect


# TODO: update OBJECT_TO_IDX to include "person" and "exit" types
class Person(WorldObj):
    """A person that needs to be rescued."""

    def __init__(self, color: str = "purple"):
        # Use "ball" as the base type since it's in OBJECT_TO_IDX and can be picked up
        super().__init__("ball", color)
        self.rescued = False  # Track if person has been rescued

    def can_pickup(self):
        return True

    def can_overlap(self):
        return False

    def render(self, img):
        c = COLORS[self.color]

        # Draw a simple humanoid figure
        # Head (circle at top)
        fill_coords(img, point_in_circle(cx=0.5, cy=0.2, r=0.12), c)

        # Body (rectangle in middle)
        fill_coords(img, point_in_rect(0.4, 0.6, 0.3, 0.7), c)

        # Arms (horizontal rectangles)
        fill_coords(img, point_in_rect(0.2, 0.8, 0.4, 0.5), c)

        # Legs (vertical rectangles)
        fill_coords(img, point_in_rect(0.35, 0.45, 0.7, 0.9), c)
        fill_coords(img, point_in_rect(0.55, 0.65, 0.7, 0.9), c)


class Exit(WorldObj):
    """An exit where people can be evacuated."""

    def __init__(self, color: str = "green"):
        # Use "goal" as the base type since it's in OBJECT_TO_IDX and can be overlapped
        super().__init__("goal", color)

    def can_overlap(self):
        return True

    def can_pickup(self):
        return False

    def render(self, img):
        c = COLORS[self.color]

        # Draw exit as a door with "EXIT" indication
        # Door frame
        fill_coords(img, point_in_rect(0.1, 0.9, 0.1, 0.9), c)
        fill_coords(img, point_in_rect(0.2, 0.8, 0.2, 0.8), (0, 0, 0))

        # Exit sign (bright rectangle at top)
        fill_coords(img, point_in_rect(0.25, 0.75, 0.15, 0.35), c)