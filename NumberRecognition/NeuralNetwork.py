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
import json
import math
from dataclasses import dataclass

import numpy as np


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


class NeuralNetworkLayerBuilder:
    def __init__(self):
        self.layers: list[Layer] = []

    def add_layer(self, nodes: int):
        if len(self.layers) > 0:
            self.layers[-1].outputs = nodes
        self.layers.append(Layer(inputs=nodes, type=LayerType.HIDDEN))
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
        self.cached_values = []
        self.learn_rate = 0.01
        self.hidden_values = None
        self.__init_layers__()

    def __init_layers__(self):
        for layer in self.layers:
            self.network.append(NetworkLayer(
                weights=np.random.uniform(-0.5, 0.5, (layer.outputs, layer.inputs)),
                bias=np.zeros((layer.outputs, 1))
            ))

    def predict(self, data: np.array):
        hidden_pre = self.network[0].bias + np.dot(self.network[0].weights, data)
        result = 1 / (1 + np.exp(-hidden_pre))

        self.hidden_values = result

        output_pre = self.network[1].bias + np.dot(self.network[1].weights, result)
        result = 1 / (1 + np.exp(-output_pre))

        return np.reshape(result, (-1, 1))

    def backpropagation(self, result: np.array, correct_result: np.array, *, image):
        delta = result - correct_result

        w_h_o = -self.learn_rate * delta @ np.transpose(self.hidden_values)
        b_h_o = -self.learn_rate * delta

        delta_hidden = np.transpose(self.network[1].weights) @ delta * (self.hidden_values * (1 - self.hidden_values))

        w_i_h = -self.learn_rate * delta_hidden @ np.transpose(image)
        b_i_h = -self.learn_rate * delta_hidden

        self.network[1].weights += w_h_o
        self.network[1].bias += b_h_o
        self.network[0].weights += w_i_h
        self.network[0].bias += b_i_h

    def to_json(self):
        result = {"layers": []}

        for layer, network_layer in zip(self.layers, self.network):
            result["layers"].append({
                "inputs": layer.inputs,
                "outputs": layer.outputs,
                "weights": network_layer.weights.tolist(),
                "bias": network_layer.bias.tolist(),
                "type": layer.type
            })

        return json.dumps(result)

    def load_json(self, path: str):
        new_network = []
        new_layers = []

        with open(path, "r") as file:
            data = json.loads(file.read())

            for layer in data["layers"]:
                new_layers.append(Layer(
                    inputs=layer['inputs'],
                    outputs=layer["outputs"],
                    type=layer["type"]
                ))

                new_network.append(NetworkLayer(
                    bias=np.array(layer['bias']),
                    weights=np.array(layer['weights'])
                ))

        self.layers = new_layers
        self.network = new_network
