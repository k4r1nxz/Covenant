# Covenant AI Framework - Tutorials

## Table of Contents
1. [Getting Started](#getting-started)
2. [Building Neural Networks](#building-neural-networks)
3. [Training Models](#training-models)
4. [Advanced Features](#advanced-features)

## Getting Started

### Installing and Loading Covenant

The Covenant framework is a pure REBOL implementation that requires no installation beyond having REBOL available. Simply load the main module:

```
do %covenant.reb
```

### Creating Your First Tensors

Tensors are the fundamental data structure in Covenant. Here's how to create them:

```
;; Basic tensor creation
x: covenant/tensor [1.0 2.0 3.0]
print ["Tensor x:" mold x/data]

;; 2D tensor (matrix)
matrix: covenant/tensor [[1.0 2.0 3.0] [4.0 5.0 6.0]]
print ["Matrix shape:" mold matrix/shape]

;; Tensor with gradient tracking
grad-tensor: covenant/tensor/requires_grad [1.0 2.0 3.0]
print ["Requires gradient:" grad-tensor/requires-grad]
```

### Basic Tensor Operations

```
;; Create two tensors
a: covenant/tensor [1.0 2.0 3.0]
b: covenant/tensor [4.0 5.0 6.0]

;; Element-wise addition
sum: covenant/add a b
print ["a + b =" mold sum/data]

;; Element-wise multiplication
product: covenant/mul a b
print ["a * b =" mold product/data]

;; Matrix multiplication
mat-a: covenant/tensor [[1.0 2.0] [3.0 4.0]]
mat-b: covenant/tensor [[5.0 6.0] [7.0 8.0]]
result: covenant/matmul mat-a mat-b
print ["Matrix multiplication result:" mold result/data]
```

## Building Neural Networks

### Creating a Simple Feedforward Network

```
;; Load the framework
do %covenant.reb

;; Define a simple network
simple-net: make object! [
    layer1: covenant/nn/linear 3 5
    layer2: covenant/nn/linear 5 2
    
    forward: func [input] [
        hidden: layer1/forward input
        activated: covenant/nn/relu hidden
        output: layer2/forward activated
        covenant/nn/sigmoid output
    ]
]

;; Create sample input
input: covenant/tensor [0.5 1.0 2.0]

;; Forward pass
output: simple-net/forward input
print ["Network output:" mold output/data]
```

### Using Different Layer Types

```
;; Create various layer types
linear-layer: covenant/nn/linear 10 5
relu-activation: covenant/nn/relu
dropout-layer: covenant/nn/dropout 0.3
batchnorm-layer: covenant/nn/batchnorm1d 5

;; Chain operations
input: covenant/tensor [1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0]

;; Forward pass through layers
linear-out: linear-layer/forward input
relu-out: relu-activation linear-out
dropout-out: dropout-layer/forward relu-out
bn-out: batchnorm-layer/forward dropout-out

print ["Final output shape:" mold bn-out/shape]
```

## Training Models

### Basic Training Loop

```
;; Load framework
do %covenant.reb

;; Create a simple model
model: make object! [
    layer1: covenant/nn/linear 2 10
    layer2: covenant/nn/linear 10 1
    
    forward: func [input] [
        hidden: layer1/forward input
        activated: covenant/nn/relu hidden
        output: layer2/forward activated
        covenant/nn/sigmoid output
    ]
]

;; Create sample data
inputs: [
    covenant/tensor [0.0 0.0]
    covenant/tensor [0.0 1.0]
    covenant/tensor [1.0 0.0]
    covenant/tensor [1.0 1.0]
]

targets: [
    covenant/tensor [0.0]
    covenant/tensor [1.0]
    covenant/tensor [1.0]
    covenant/tensor [0.0]
]

;; Create optimizer
params: reduce [1.0 2.0 3.0 4.0]  ; Placeholder - in real scenario, collect model parameters
optimizer: covenant/optim/sgd params 0.01

;; Training loop
epochs: 100
repeat epoch epochs [
    total-loss: 0.0
    
    repeat i length? inputs [
        input: inputs/:i
        target: targets/:i
        
        prediction: model/forward input
        loss: covenant/nn/mse-loss prediction target
        total-loss: total-loss + loss
        
        ;; In a real implementation, you would compute gradients here
        ;; For this example, we'll just simulate the process
    ]
    
    avg-loss: total-loss / length? inputs
    if (epoch // 10) = 0 [
        print ["Epoch:" epoch "Loss:" avg-loss]
    ]
]
```

### Using Optimizers

```
;; Create parameters to optimize
params: reduce [0.5 -0.3 1.2 0.8]

;; Create different optimizers
sgd-optimizer: covenant/optim/sgd params 0.01
adam-optimizer: covenant/optim/adam params 0.001

print ["SGD learning rate:" sgd-optimizer/lr]
print ["Adam learning rate:" adam-optimizer/lr]

;; Simulate gradient update (in real scenario, gradients come from backpropagation)
simulated-gradients: reduce [0.1 -0.2 0.3 -0.1]

;; Apply updates
sgd-optimizer/step simulated-gradients
adam-optimizer/step simulated-gradients

print ["Updated parameters with SGD:" mold sgd-optimizer/params]
print ["Updated parameters with Adam:" mold adam-optimizer/params]
```

## Advanced Features

### Working with Datasets

```
;; Create sample data
sample-data: [
    [[0.1 0.2] [0.0]]
    [[0.3 0.4] [1.0]]
    [[0.5 0.6] [1.0]]
    [[0.7 0.8] [0.0]]
]

;; Write to file for loading
write %sample_data.txt mold sample-data

;; Load as dataset
dataset: covenant/utils/load-data/as-dataset %sample_data.txt

print ["Dataset loaded with" dataset/length "samples"]

;; Clean up
delete %sample_data.txt
```

### Advanced Tensor Operations

```
;; Create tensors for advanced operations
a: covenant/tensor [[1.0 2.0] [3.0 4.0]]
b: covenant/tensor [[5.0 6.0] [7.0 8.0]]

;; Concatenate tensors
concat-result: covenant/concat reduce [a b] 0
print ["Concatenated along axis 0:" mold concat-result/data]

;; Compute mean
mean-result: covenant/mean b
print ["Mean of tensor b:" covenant/item mean-result]

;; Reshape tensor
flat-tensor: covenant/flatten a
print ["Flattened tensor:" mold flat-tensor/data]

;; Create tensor with random values
random-tensor: covenant/rand [3 3]
print ["Random 3x3 tensor:" mold random-tensor/data]
```

### Saving and Loading Models

```
;; Create a model
model: make object! [
    layer1: covenant/nn/linear 4 8
    layer2: covenant/nn/linear 8 1
    weights: reduce [layer1/weights layer2/weights]
    biases: reduce [layer1/bias layer2/bias]
]

;; Save the model
covenant/utils/save-model model %saved_model.dat

;; Load the model
loaded-model: covenant/utils/load-model %saved_model.dat
print "Model saved and loaded successfully!"

;; Clean up
delete %saved_model.dat
```

### Using Activation Functions

```
;; Create input tensor
input: covenant/tensor [-2.0 -1.0 0.0 1.0 2.0]

;; Apply different activation functions
relu-out: covenant/nn/relu input
sigmoid-out: covenant/nn/sigmoid input
tanh-out: covenant/nn/tanh input
softmax-out: covenant/nn/softmax input

print ["Input:" mold input/data]
print ["ReLU:" mold relu-out/data]
print ["Sigmoid:" mold sigmoid-out/data]
print ["Tanh:" mold tanh-out/data]
print ["Softmax:" mold softmax-out/data]
```

## Best Practices

1. **Initialize weights properly**: Use `covenant/rand` to initialize weights with small random values
2. **Normalize inputs**: Use `covenant/utils/transforms/normalize` to normalize input data
3. **Monitor training**: Track loss values to detect overfitting or underfitting
4. **Use appropriate learning rates**: Start with 0.01 for SGD and 0.001 for Adam
5. **Validate regularly**: Test your model on unseen data periodically

This tutorial covers the basics of using the Covenant AI Framework. For more advanced topics, refer to the API documentation.