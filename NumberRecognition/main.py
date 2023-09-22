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
from mnist import MNIST

from NumberRecognition.NeuralNetwork import NeuralNetworkLayerBuilder, NeuralNetwork

# Test to see if collaborator stuff finally works

mdata = MNIST("./MNIST/")
images, labels = mdata.load_training()
# Convert arrays into numpy arrays with numbers ranging from 0 to 1
images, labels = np.array(images) / 255, np.array(labels)

layers = NeuralNetworkLayerBuilder() \
    .add_layer(784) \
    .add_layer(320) \
    .add_layer(10) \
    .build()

neural_network = NeuralNetwork(layers)

correct_count = 0
for epoch in range(5):
    for image, label in zip(images, labels):
        image.shape += (1,)

        correct_output = np.zeros((10, 1))
        correct_output[label] = [1]

        result = neural_network.predict(image)

        if np.argmax(result) == np.argmax(correct_output):
            correct_count += 1

        neural_network.backpropagation(result, correct_output, image=image)

    print(f"Epoch {epoch + 1}: Correct Count: {correct_count} / 60000")
    correct_count = 0

# Write to file in json format
with open("network.json", "w") as file:
    file.write(neural_network.to_json())
