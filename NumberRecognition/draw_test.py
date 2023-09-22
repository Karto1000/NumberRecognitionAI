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
import numpy as np
import pygame

from NumberRecognition.NeuralNetwork import NeuralNetwork

pygame.init()
SCREEN = pygame.display.set_mode((1084, 784))
FONT = pygame.font.SysFont("Arial", 28)

network = NeuralNetwork([])
network.load_json("network.json")

GRID_SIZE = 28, 28
AMOUNT_OF_CELLS = SCREEN.get_width() // GRID_SIZE[0], SCREEN.get_height() // GRID_SIZE[1]

drawing_grid = np.zeros((28, 28))
result = None

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                drawing_grid = np.zeros((28, 28))

    SCREEN.fill((255, 255, 255))

    mouse_pos = pygame.mouse.get_pos()
    mouse_pressed = pygame.mouse.get_pressed()

    if mouse_pressed[0]:
        grid_x, grid_y = mouse_pos[0] // GRID_SIZE[0], mouse_pos[1] // GRID_SIZE[1]

        if grid_x >= GRID_SIZE[0] or grid_y >= GRID_SIZE[1] or grid_x < 0 or grid_y < 0:
            continue

        drawing_grid[grid_y][grid_x] = 1
        concatenated = np.concatenate(drawing_grid)
        concatenated.shape += (1, )
        result = np.argmax(network.predict(concatenated))

    for y, y_pair in enumerate(drawing_grid):
        for x, cell in enumerate(y_pair):
            if cell == 0.0:
                continue

            pygame.draw.rect(
                SCREEN,
                (0, 0, 0),
                (x * GRID_SIZE[0], y * GRID_SIZE[1], GRID_SIZE[0], GRID_SIZE[1])
            )

    pygame.draw.rect(
        SCREEN,
        (155, 155, 155),
        (784, 0, 300, 784)
    )

    if result:
        text = FONT.render(f"Network Predicts {result}", False, True)
        SCREEN.blit(
            text,
            (804, 20)
        )


    pygame.display.flip()
