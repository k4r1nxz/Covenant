# Covenant AI Framework Documentation

## Getting Started

### What is Covenant?

Covenant is a REBOL-based artificial intelligence framework designed for machine learning and neural network development. Inspired by PyTorch, it provides tensor operations, neural network layers, and optimization algorithms in the lightweight REBOL language.

### Key Features

- **Tensor Operations**: Multi-dimensional array operations with GPU-like semantics
- **Dynamic Neural Networks**: Define-by-run approach for building neural networks
- **Automatic Differentiation**: Automatic gradient computation for backpropagation
- **Optimization Algorithms**: SGD, Adam, and other optimization methods
- **Modular Design**: Clean separation of concerns with extensible components

### Installation

Covenant is a pure REBOL implementation that requires no installation beyond having REBOL available. Simply download the framework and load the main module:

```
do %covenant.reb
```

## Core Concepts

### Tensors

Tensors are the fundamental data structure in Covenant. They are multi-dimensional arrays that support various mathematical operations.

#### Creating Tensors

```
;; Create a 1D tensor
x: covenant/tensor [1.0 2.0 3.0]

;; Create a 2D tensor (matrix)
matrix: covenant/tensor [[1.0 2.0 3.0] [4.0 5.0 6.0]]

;; Create tensor with gradient tracking
grad-tensor: covenant/tensor/requires_grad [1.0 2.0 3.0]

;; Create tensor filled with zeros
zeros: covenant/zeros [3 4]

;; Create tensor filled with ones
ones: covenant/ones [2 2]

;; Create tensor with random values
random-tensor: covenant/rand [3 3]
```

#### Tensor Properties

Each tensor has the following properties:
- `data`: The underlying data as a flat block
- `shape`: The dimensions of the tensor as a block
- `dtype`: The data type (currently 'float32)

#### Tensor Operations

```
;; Element-wise operations
a: covenant/tensor [1.0 2.0 3.0]
b: covenant/tensor [4.0 5.0 6.0]

sum: covenant/add a b         ; [5.0 7.0 9.0]
product: covenant/mul a b     ; [4.0 10.0 18.0]

;; Matrix operations
mat-a: covenant/tensor [[1.0 2.0] [3.0 4.0]]
mat-b: covenant/tensor [[5.0 6.0] [7.0 8.0]]
result: covenant/matmul mat-a mat-b  ; [[19.0 22.0] [43.0 50.0]]

;; Reshaping
flat: covenant/flatten result
reshaped: covenant/reshape flat [4 1]

;; Reduction operations
total: covenant/sum a  ; Sum all elements
mean-val: covenant/mean a  ; Mean of all elements
```

### Neural Network Layers

Covenant provides various neural network layers that can be combined to build complex architectures.

#### Linear Layer

The linear (fully connected) layer applies a linear transformation to the incoming data.

```
;; Create a linear layer: 10 inputs -> 5 outputs
linear-layer: covenant/nn/linear 10 5

;; Forward pass
input: covenant/tensor [1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0]
output: linear-layer/forward input
```

#### Activation Functions

Activation functions introduce non-linearity to the network.

```
;; ReLU activation
input: covenant/tensor [-2.0 -1.0 0.0 1.0 2.0]
relu-output: covenant/nn/relu input  ; [0.0 0.0 0.0 1.0 2.0]

;; Sigmoid activation
sigmoid-output: covenant/nn/sigmoid input

;; Tanh activation
tanh-output: covenant/nn/tanh input

;; Softmax activation
softmax-output: covenant/nn/softmax input
```

#### Other Layers

```
;; Dropout layer
dropout-layer: covenant/nn/dropout 0.5  ; 50% dropout rate

;; Batch normalization
batchnorm-layer: covenant/nn/batchnorm1d 10  ; For 10 features

;; Convolutional layer (1D)
conv-layer: covenant/nn/conv1d 1 16 3  ; 1 input channel, 16 output, kernel size 3

;; Max pooling layer (1D)
pool-layer: covenant/nn/maxpool1d 2 2  ; Kernel size 2, stride 2
```

### Automatic Differentiation (Autograd)

Covenant supports automatic differentiation for computing gradients needed in training neural networks.

```
;; Create tensors with gradient tracking
x: covenant/tensor/requires_grad [2.0 3.0 4.0]
y: covenant/tensor/requires_grad [1.0 2.0 3.0]

;; Perform operations
z: covenant/mul x y  ; [2.0 6.0 12.0]
w: covenant/add z x  ; [4.0 9.0 16.0]

;; In a full implementation, you would call backward() to compute gradients
;; w/backward
;; print ["x grad:" mold x/grad]
```

## Deep Learning with Covenant

### Building a Neural Network

Here's how to build a simple feedforward neural network:

```
;; Load the framework
do %covenant.reb

;; Define a neural network
network: make object! [
    ;; Define layers
    layer1: covenant/nn/linear 784 128  ; Input: 784, Hidden: 128
    layer2: covenant/nn/linear 128 64   ; Hidden: 128, Hidden: 64
    layer3: covenant/nn/linear 64 10    ; Hidden: 64, Output: 10
    
    ;; Forward pass function
    forward: func [input] [
        ;; First hidden layer
        hidden1: layer1/forward input
        activated1: covenant/nn/relu hidden1
        
        ;; Second hidden layer
        hidden2: layer2/forward activated1
        activated2: covenant/nn/relu hidden2
        
        ;; Output layer
        output: layer3/forward activated2
        covenant/nn/softmax output
    ]
]

;; Create sample input (e.g., flattened 28x28 image)
input: covenant/rand [784]

;; Forward pass
output: network/forward input
print ["Network output shape:" mold output/shape]
```

### Training a Model

Training involves forward passes, loss computation, and parameter updates:

```
;; Define a simple training loop
train-model: func [
    model [object!]
    inputs [block!]
    targets [block!]
    epochs [integer!]
    learning-rate [number!]
] [
    ;; Create optimizer
    optimizer: covenant/optim/adam reduce [1.0 2.0 3.0] learning-rate  ; Placeholder parameters
    
    repeat epoch epochs [
        total-loss: 0.0
        
        ;; Iterate through data
        repeat i length? inputs [
            input: inputs/:i
            target: targets/:i
            
            ;; Forward pass
            prediction: model/forward input
            
            ;; Compute loss
            loss: covenant/nn/mse-loss prediction target
            total-loss: total-loss + loss
            
            ;; In a full implementation, compute gradients and update parameters
            ;; For this example, we'll just simulate
        ]
        
        avg-loss: total-loss / length? inputs
        if (epoch // 10) = 0 [
            print ["Epoch:" epoch "Average Loss:" avg-loss]
        ]
    ]
]
```

### Data Loading and Preprocessing

Covenant provides utilities for loading and preprocessing data:

```
;; Create sample dataset
sample-data: [
    [[0.1 0.2] [0.0]]
    [[0.3 0.4] [1.0]]
    [[0.5 0.6] [1.0]]
    [[0.7 0.8] [0.0]]
]

;; Write to file
write %sample_dataset.txt mold sample-data

;; Load as dataset
dataset: covenant/utils/load-data/as-dataset %sample_dataset.txt

;; Transformations
transformed-input: covenant/utils/transforms/normalize 
    covenant/tensor [0.5 0.8] 0.5 0.2  ; normalize with mean=0.5, std=0.2

;; Clean up
delete %sample_dataset.txt
```

## Optimization Algorithms

Covenant includes several optimization algorithms for training neural networks.

### Stochastic Gradient Descent (SGD)

```
;; Create parameters to optimize
params: reduce [0.1 -0.2 0.3 0.4]

;; Create SGD optimizer
sgd-optimizer: covenant/optim/sgd params 0.01  ; LR = 0.01

;; In training loop, after computing gradients:
;; sgd-optimizer/step gradients
```

### Adam Optimizer

```
;; Create Adam optimizer
adam-optimizer: covenant/optim/adam params 0.001  ; LR = 0.001

;; In training loop, after computing gradients:
;; adam-optimizer/step gradients
```

## Utilities and Helpers

### Model Saving and Loading

```
;; Save a model
model: make object! [
    layer1: covenant/nn/linear 5 3
    layer2: covenant/nn/linear 3 1
]
covenant/utils/save-model model %my_model.dat

;; Load a model
loaded-model: covenant/utils/load-model %my_model.dat

;; Clean up
delete %my_model.dat
```

### Visualization

```
;; Visualize a tensor
tensor: covenant/tensor [[1.0 2.0] [3.0 4.0]]
covenant/utils/visualize tensor
```

## Advanced Topics

### Custom Layers

You can create custom layers by defining objects with a `forward` method:

```
custom-layer: make object! [
    ;; Store layer parameters
    weight: covenant/rand [5 5]
    bias: covenant/zeros [5]
    
    ;; Forward pass function
    forward: func [input] [
        ;; Custom computation
        weighted: covenant/matmul input weight
        output: covenant/add weighted bias
        covenant/nn/relu output
    ]
]
```

### Functional Programming Style

Covenant supports a functional programming style for building networks:

```
;; Define a network as a composition of functions
network-fn: func [input] [
    input
    | (covenant/nn/linear 10 5)
    | covenant/nn/relu
    | (covenant/nn/linear 5 1)
    | covenant/nn/sigmoid
]

;; Apply to input
result: network-fn input
```

## Best Practices

### Weight Initialization

Proper weight initialization is crucial for training:

```
;; Good practice: Initialize weights with small random values
init-weights: func [shape] [
    ;; Xavier/Glorot initialization
    fan-in: shape/2  ; Assuming 2D tensor
    limit: sqrt (6.0 / (1.0 + fan-in))
    covenant/rand shape * (2 * limit) - limit
]
```

### Normalization

Normalize inputs to improve training stability:

```
;; Normalize input data
normalized-input: covenant/utils/transforms/normalize input 0.0 1.0
```

### Regularization

Use dropout and other regularization techniques:

```
;; Add dropout to prevent overfitting
dropout-layer: covenant/nn/dropout 0.3  ; 30% dropout
```

## Troubleshooting

### Common Issues

1. **Shape Mismatch Errors**: Ensure tensor dimensions are compatible for operations
2. **Gradient Vanishing/Exploding**: Use proper weight initialization and normalization
3. **Memory Issues**: REBOL has limited memory management; keep tensor sizes reasonable

### Debugging Tips

- Use `covenant/utils/visualize` to inspect tensor values
- Monitor loss values during training
- Check tensor shapes with `tensor/shape`
- Validate gradients if implementing backpropagation

## Contributing

Covenant is an open framework. Contributions are welcome for:
- New layer types
- Optimization algorithms
- Utility functions
- Documentation improvements

## API Reference

For detailed API documentation, see the [API Reference](api-reference.md).

## Tutorials

For hands-on examples, see the [Tutorials](tutorials.md).