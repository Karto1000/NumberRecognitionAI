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
import pygame

pygame.init()

from NN1.Flappy import Bird, Pipe
from NN1.NeuralNetwork import NeuralNetwork, NeuralNetworkLayerBuilder, sigmoid

SCREEN = pygame.display.set_mode((800, 800))
CLOCK = pygame.time.Clock()
running = True

layers = NeuralNetworkLayerBuilder() \
    .add_layer(5) \
    .add_layer(10) \
    .add_layer(2) \
    .add_result_dict(["STAY", "JUMP"]) \
    .build()

neural_network = NeuralNetwork(layers)
to_predict = None

bird = Bird()
pipe = Pipe(700, 400)


def run_events():
    global to_predict
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit(0)
        if event.type == pygame.MOUSEBUTTONDOWN:
            bird.jump()


while running:
    SCREEN.fill((0, 0, 0))
    CLOCK.tick(60)
    mouse_pos = pygame.mouse.get_pos()

    run_events()

    bird.update()
    bird.draw(SCREEN)

    pipe.update()
    pipe.draw(SCREEN)

    distance_to_top = pygame.Vector2(
        abs(pipe.cords.x - bird.cords.x),
        abs((pipe.cords.y - pipe.space_between) - bird.cords.y)
    ).normalize()

    distance_to_bottom = pygame.Vector2(
        abs(pipe.cords.x - bird.cords.x),
        abs(pipe.cords.y - bird.cords.y)
    ).normalize()

    to_predict = [*distance_to_top, *distance_to_bottom, sigmoid(bird.acceleration.y / 10)]
    prediction = neural_network.predict(to_predict)

    if prediction["JUMP"] > prediction["STAY"]:
        bird.jump()

    neural_network.draw(SCREEN)

    pygame.display.flip()
