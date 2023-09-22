# -----------------------------------------------------------
# Copyright (c) YPSOMED AG, Burgdorf / Switzerland
# YDS INNOVATION - Digital Innovation
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# email diginno@ypsomed.com
# author: Tim Leuenberger (Tim.leuenberger@ypsomed.com)
# -----------------------------------------------------------
import random

import pygame.draw

from ..config import *


class Bird:
    def __init__(self):
        self.cords = pygame.Vector2(200, 400)
        self.acceleration = pygame.Vector2(0, 0)

    def update(self):
        self.acceleration += GRAVITY
        self.cords += self.acceleration

    def jump(self):
        self.acceleration.y = -5

    def draw(self, screen: pygame.Surface):
        pygame.draw.rect(
            screen,
            (255, 255, 255),
            (self.cords.x, self.cords.y, 20, 20)
        )


class Pipe:
    def __init__(self, x: int, y: int):
        self.cords = pygame.Vector2(x, y)
        self.space_between = 200
        self.height = 500

    def update(self):
        if self.cords.x <= -51:
            self.cords = pygame.Vector2(850, random.randrange(300, 600))
        self.cords -= pygame.Vector2(1, 0)

    def draw(self, screen: pygame.Surface):
        pygame.draw.rect(
            screen,
            (255, 255, 255),
            (*self.cords, 50, self.height)
        )

        pygame.draw.rect(
            screen,
            (255, 255, 255),
            (self.cords.x, self.cords.y - (self.height + self.space_between), 50, self.height)
        )
