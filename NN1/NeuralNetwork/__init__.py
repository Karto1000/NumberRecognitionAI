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
import dataclasses
import math
from dataclasses import dataclass

import numpy as np

from ..config import *


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


sigmoid = np.vectorize(sigmoid)


class LayerType:
    ENTRY = 0
    HIDDEN = 1
    EXIT = 2


@dataclass
class Layer:
    inputs: int
    type: int
    outputs: int = dataclasses.field(default=None)
    output_descriptor: list = dataclasses.field(default=None)


class NeuralNetworkLayerBuilder:
    def __init__(self):
        self.layers: list[Layer] = []

    def add_layer(self, nodes: int):
        if len(self.layers) > 0:
            self.layers[-1].outputs = nodes
        self.layers.append(Layer(inputs=nodes, type=LayerType.HIDDEN))
        return self

    def add_result_dict(self, result: list):
        if self.layers[-1].inputs < len(result):
            raise Exception(f"{len(result)} keys not enough for {self.layers[-1].outputs} outputs")

        self.layers[-1].output_descriptor = result
        return self

    def build(self):
        self.layers[-1].type = LayerType.EXIT
        self.layers[-1].outputs = self.layers[-1].inputs
        self.layers[0].type = LayerType.ENTRY
        return self.layers


@dataclass
class NetworkLayer:
    weights: np.ndarray
    bias: np.ndarray


class NeuralNetwork:
    def __init__(self, layers: list[Layer]):
        self.layers: list[Layer] = layers
        self.network: list[NetworkLayer] = []
        self.last_layer_calc = []
        self.__init_layers()

    def __init_layers(self):
        for layer in self.layers:
            self.network.append(
                NetworkLayer(
                    weights=np.random.randn(layer.inputs, layer.outputs) * 0.2,
                    bias=np.random.randn(layer.inputs, 1) * 0.2
                )
            )

    def predict(self, inputs: np.array) -> dict:
        self.last_layer_calc.clear()
        if len(inputs) != self.layers[0].inputs:
            raise Exception(
                f"Inputs in predict function do not have the correct amount of values give: "
                f"{len(inputs)} expected: "
                f"{self.layers[0].inputs}")

        self.last_layer_calc.append(inputs)
        dot_product = inputs
        for i, network_layer in enumerate(self.network):
            if i == len(self.network) - 1:
                result = {}

                for i_, val in enumerate(dot_product):
                    result[self.layers[-1].output_descriptor[i_]] = abs(val)

                return result

            dot_product = sigmoid(np.dot(dot_product, network_layer.weights))
            self.last_layer_calc.append(dot_product)

    def draw(self, screen: pygame.Surface):
        for x, calc in enumerate(self.last_layer_calc):
            amount_of_nodes = len(calc)

            for i in range(0, amount_of_nodes):
                val = calc[i]
                pygame.draw.circle(
                    screen,
                    (255, 255, 255),
                    (
                        x * PADDING + RADIUS + WINDOW_PADDING,
                        i * PADDING + RADIUS + WINDOW_PADDING
                    ),
                    RADIUS
                )

                text = FONT.render(f"{val:0.2f}", False, (0, 0, 0))
                screen.blit(
                    text,
                    (
                        x * PADDING + WINDOW_PADDING + RADIUS - text.get_width() / 2,
                        i * PADDING + WINDOW_PADDING + RADIUS - text.get_height() / 2
                    )
                )
