# Covenant AI Framework - Examples and Tutorials

## Table of Contents
1. [Basic Tensor Operations](#basic-tensor-operations)
2. [Advanced Tensor Operations](#advanced-tensor-operations)
3. [Neural Network Examples](#neural-network-examples)
4. [Optimization Examples](#optimization-examples)
5. [Autograd Examples](#autograd-examples)

## Basic Tensor Operations

### Creating Tensors

```rebol
;; Load the framework
do %covenant.reb

;; Create a simple tensor
x: covenant/tensor [1.0 2.0 3.0]
print ["Simple tensor:" mold x/data]

;; Create a 2D tensor
matrix: covenant/tensor [[1.0 2.0] [3.0 4.0]]
print ["2D tensor shape:" mold matrix/shape]
print ["2D tensor data:" mold matrix/data]

;; Create tensor with specific data type
int-tensor: covenant/tensor/dtype [1 2 3] 'int32
print ["Integer tensor:" mold int-tensor/data "dtype:" int-tensor/dtype]

;; Create tensor with gradient tracking
x-grad: covenant/tensor/requires_grad [2.0 3.0 4.0]
print ["Tensor with grad tracking:" mold x-grad/data]
```

### Basic Operations

```rebol
;; Load the framework
do %covenant.reb

;; Create tensors
a: covenant/tensor [1.0 2.0 3.0]
b: covenant/tensor [4.0 5.0 6.0]

;; Addition
sum: covenant/add a b
print ["Addition result:" mold sum/data]

;; Multiplication
product: covenant/mul a b
print ["Multiplication result:" mold product/data]

;; Matrix multiplication
mat-a: covenant/tensor [[1.0 2.0] [3.0 4.0]]
mat-b: covenant/tensor [[5.0 6.0] [7.0 8.0]]
mat-product: covenant/matmul mat-a mat-b
print ["Matrix multiplication result:" mold mat-product/data]

;; Reshape
reshaped: covenant/reshape a [1 3]
print ["Reshaped tensor:" mold reshaped/data "shape:" mold reshaped/shape]
```

## Advanced Tensor Operations

### New Tensor Functions

```rebol
;; Load the framework
do %covenant.reb

;; Create tensor with arange
range-tensor: covenant/arange 0 10
print ["Arange tensor:" mold range-tensor/data]

;; Create tensor with linspace
linspace-tensor: covenant/linspace 0 1 5
print ["Linspace tensor:" mold linspace-tensor/data]

;; Power operation
base: covenant/tensor [1.0 2.0 3.0]
powered: covenant/pow base 2
print ["Powered tensor:" mold powered/data]

;; Square root
squared: covenant/tensor [4.0 9.0 16.0]
sqrt-result: covenant/sqrt squared
print ["Square root result:" mold sqrt-result/data]

;; Exponential
exp-input: covenant/tensor [0.0 1.0 2.0]
exp-result: covenant/exp exp-input
print ["Exponential result:" mold exp-result/data]

;; Natural logarithm
log-input: covenant/tensor [1.0 2.718 7.389]
log-result: covenant/log log-input
print ["Logarithm result:" mold log-result/data]

;; Statistical operations
data: covenant/tensor [[1.0 2.0 3.0] [4.0 5.0 6.0]]

;; Max along axis 0 (columns)
max-cols: covenant/max/axis data 0
print ["Max along columns:" mold max-cols/data]

;; Max along axis 1 (rows)
max-rows: covenant/max/axis data 1
print ["Max along rows:" mold max-rows/data]

;; Argmax
argmax-result: covenant/argmax/axis data 1
print ["Argmax along rows:" mold argmax-result/data]

;; Transpose
transposed: covenant/transpose data
print ["Transposed shape:" mold transposed/shape]
print ["Transposed data:" mold transposed/data]
```

## Neural Network Examples

### Simple Neural Network

```rebol
;; Load the framework
do %covenant.reb

;; Create a simple neural network with multiple layers
input-layer: covenant/nn/linear 3 5
hidden-layer: covenant/nn/linear 5 2
activation: covenant/nn/relu

;; Create input data
input-data: covenant/tensor/requires_grad [1.0 2.0 3.0]

;; Forward pass
hidden-output: input-layer/forward input-data
activated: activation hidden-output
final-output: hidden-layer/forward activated

print ["Network output:" mold final-output/data]
print ["Output shape:" mold final-output/shape]
```

### Sequential Model

```rebol
;; Load the framework
do %covenant.reb

;; Create layers
layer1: covenant/nn/linear 4 8
relu1: covenant/nn/relu
layer2: covenant/nn/linear 8 4
relu2: covenant/nn/relu
output-layer: covenant/nn/linear 4 1

;; Create sequential model
model: covenant/nn/sequential reduce [layer1 relu1 layer2 relu2 output-layer]

;; Create input
input: covenant/tensor/requires_grad [1.0 2.0 3.0 4.0]

;; Forward pass
output: model/forward input
print ["Sequential model output:" mold output/data]
```

### Convolutional Layer

```rebol
;; Load the framework
do %covenant.reb

;; Create a 1D convolutional layer
conv-layer: covenant/nn/conv1d 1 2 3  ; 1 input channel, 2 output channels, kernel size 3

;; Create input data (sequence of 5 values, 1 channel)
input-data: covenant/tensor [[1.0] [2.0] [3.0] [4.0] [5.0]]

;; Forward pass
output: conv-layer/forward input-data
print ["Conv1D output shape:" mold output/shape]
print ["Conv1D output data:" mold output/data]
```

### Dropout and Batch Normalization

```rebol
;; Load the framework
do %covenant.reb

;; Create batch normalization layer
bn-layer: covenant/nn/batchnorm1d 3  ; 3 features

;; Create input
input-data: covenant/tensor/requires_grad [[1.0 2.0 3.0] [4.0 5.0 6.0]]

;; Forward pass
bn-output: bn-layer/forward input-data
print ["Batch norm output:" mold bn-output/data]

;; Create dropout layer
dropout-layer: covenant/nn/dropout 0.5

;; Forward pass (training mode)
dropout-output: dropout-layer/forward input-data true
print ["Dropout output:" mold dropout-output/data]
```

## Optimization Examples

### Using Different Optimizers

```rebol
;; Load the framework
do %covenant.reb

;; Create some parameters to optimize
param1: covenant/tensor/requires_grad [0.5 -0.3 1.2]
param2: covenant/tensor/requires_grad [-0.1 0.8 0.4]

;; Create parameter list
params: reduce [param1 param2]

;; Create different optimizers
sgd-optimizer: covenant/optim/sgd/momentum params 0.01 0.9
adam-optimizer: covenant/optim/adam params 0.001
rmsprop-optimizer: covenant/optim/rmsprop params 0.01

;; Create some dummy gradients
gradients: reduce [
    covenant/tensor [0.1 -0.2 0.3]/data
    covenant/tensor [-0.05 0.15 0.25]/data
]

;; Update parameters with SGD
print "Before SGD update:"
print ["Param1:" mold param1/data]
sgd-optimizer/step gradients
print "After SGD update:"
print ["Param1:" mold param1/data]

;; Reset parameters and try Adam
param1/data: [0.5 -0.3 1.2]
param2/data: [-0.1 0.8 0.4]

print "^nBefore Adam update:"
print ["Param1:" mold param1/data]
adam-optimizer/step gradients
print "After Adam update:"
print ["Param1:" mold param1/data]
```

### Learning Rate Schedulers

```rebol
;; Load the framework
do %covenant.reb

;; Create parameters and optimizer
params: reduce [covenant/tensor/requires_grad [0.5 -0.3]]
optimizer: covenant/optim/sgd params 0.01

;; Create learning rate scheduler
scheduler: covenant/lr_scheduler/step_lr optimizer 10 0.5  ; Reduce LR by half every 10 epochs

print ["Initial LR:" optimizer/lr]

;; Simulate training for several epochs
repeat epoch 25 [
    if (epoch // 10) = 1 [  ; Apply scheduler at epoch 1, 11, 21, etc.
        scheduler/step epoch
        print ["Epoch" epoch "LR changed to:" optimizer/lr]
    ]
]
```

## Autograd Examples

### Automatic Differentiation

```rebol
;; Load the framework
do %covenant.reb

;; Create variables that require gradients
x: covenant/tensor/requires_grad [2.0]
y: covenant/tensor/requires_grad [3.0]

;; Perform operations
z: covenant/add x y
w: covenant/mul z x

print ["z = x + y =" mold z/data]
print ["w = z * x =" mold w/data]

;; Perform more complex operations
squared: covenant/pow w 2
result: covenant/sum squared

print ["result = sum((x + y) * x)^2 =" mold result/data]

;; Compute gradients using backward pass
; Note: In the current implementation, we need to manually set up the backward pass
; This would be more automated in a full implementation
```

### Training Loop Example

```rebol
;; Load the framework
do %covenant.reb

;; Create a simple model: y = w * x + b
w: covenant/tensor/requires_grad [0.5]
b: covenant/tensor/requires_grad [0.1]

;; Create optimizer
params: reduce [w b]
optimizer: covenant/optim/sgd/momentum params 0.01 0.9

;; Create some training data
x-train: covenant/tensor [1.0 2.0 3.0 4.0]
y-train: covenant/tensor [2.1 4.1 6.1 8.1]  ; y = 2*x + 0.1 (approximately)

;; Training loop
repeat epoch 100 [
    ;; Forward pass
    predictions: covenant/add (covenant/mul w x-train) b
    
    ;; Compute loss (MSE)
    loss: covenant/nn/mse_loss predictions y-train
    
    ;; In a real implementation, we would call loss/backward() here
    ;; For this example, we'll just print the loss
    if (epoch // 20) = 1 [
        print ["Epoch" epoch "Loss:" first loss/data]
    ]
    
    ;; Update parameters (this would use computed gradients in a real implementation)
    ; optimizer/step gradients
]

print ["Final w:" mold w/data]
print ["Final b:" mold b/data]
```
