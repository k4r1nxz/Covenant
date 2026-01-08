# Covenant AI Framework - Code Examples and Best Practices

## Table of Contents
1. [Basic Examples](#basic-examples)
2. [Intermediate Examples](#intermediate-examples)
3. [Advanced Examples](#advanced-examples)
4. [Best Practices](#best-practices)
5. [Performance Tips](#performance-tips)

## Basic Examples

### Example 1: Simple Tensor Operations

```
;; Load the framework
do %covenant.reb

;; Create tensors
a: covenant/tensor [1.0 2.0 3.0]
b: covenant/tensor [4.0 5.0 6.0]

;; Perform operations
sum: covenant/add a b
product: covenant/mul a b

print ["a:" mold a/data]
print ["b:" mold b/data]
print ["a + b:" mold sum/data]
print ["a * b:" mold product/data]
```

### Example 2: Basic Neural Network Layer

```
;; Load the framework
do %covenant.reb

;; Create a linear layer: 3 inputs -> 2 outputs
linear-layer: covenant/nn/linear 3 2

;; Create input tensor
input: covenant/tensor [1.0 2.0 3.0]

;; Forward pass
output: linear-layer/forward input

print ["Input:" mold input/data]
print ["Output:" mold output/data]
print ["Weights shape:" mold linear-layer/weights/shape]
print ["Bias shape:" mold linear-layer/bias/shape]
```

### Example 3: Activation Functions

```
;; Load the framework
do %covenant.reb

;; Create input tensor
input: covenant/tensor [-2.0 -1.0 0.0 1.0 2.0]

;; Apply different activation functions
relu-out: covenant/nn/relu input
sigmoid-out: covenant/nn/sigmoid input
tanh-out: covenant/nn/tanh input

print ["Input:" mold input/data]
print ["ReLU:" mold relu-out/data]
print ["Sigmoid:" mold sigmoid-out/data]
print ["Tanh:" mold tanh-out/data]
```

## Intermediate Examples

### Example 4: Multi-Layer Perceptron (MLP)

```
;; Load the framework
do %covenant.reb

;; Define a multi-layer perceptron
mlp: make object! [
    ;; Define layers
    hidden1: covenant/nn/linear 4 8
    hidden2: covenant/nn/linear 8 4
    output: covenant/nn/linear 4 1
    
    ;; Forward pass function
    forward: func [input] [
        ;; First hidden layer
        h1: hidden1/forward input
        a1: covenant/nn/relu h1
        
        ;; Second hidden layer
        h2: hidden2/forward a1
        a2: covenant/nn/relu h2
        
        ;; Output layer
        out: output/forward a2
        covenant/nn/sigmoid out
    ]
]

;; Create sample input
input: covenant/tensor [0.5 1.0 1.5 2.0]

;; Forward pass
output: mlp/forward input

print ["MLP Input:" mold input/data]
print ["MLP Output:" mold output/data]
```

### Example 5: Training Loop Template

```
;; Load the framework
do %covenant.reb

;; Define a simple model
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

;; Create sample training data
train-data: [
    [covenant/tensor [0.0 0.0] covenant/tensor [0.0]]
    [covenant/tensor [0.0 1.0] covenant/tensor [1.0]]
    [covenant/tensor [1.0 0.0] covenant/tensor [1.0]]
    [covenant/tensor [1.0 1.0] covenant/tensor [0.0]]
]

;; Training parameters
learning-rate: 0.01
epochs: 100

;; Create optimizer (using placeholder parameters)
params: reduce [1.0 2.0 3.0 4.0]  ; In real scenario, collect actual model parameters
optimizer: covenant/optim/sgd params learning-rate

;; Training loop
repeat epoch epochs [
    total-loss: 0.0
    
    foreach data-point train-data [
        input: first data-point
        target: second data-point
        
        ;; Forward pass
        prediction: model/forward input
        
        ;; Compute loss
        loss: covenant/nn/mse-loss prediction target
        total-loss: total-loss + loss
    ]
    
    avg-loss: total-loss / length? train-data
    
    ;; Print progress every 10 epochs
    if (epoch // 10) = 0 [
        print ["Epoch:" epoch "Loss:" avg-loss]
    ]
]

print "Training completed!"
```

### Example 6: Using Different Optimizers

```
;; Load the framework
do %covenant.reb

;; Create parameters to optimize
params: reduce [0.5 -0.3 1.2 0.8 -0.1]

;; Create different optimizers
sgd-optimizer: covenant/optim/sgd params 0.01
adam-optimizer: covenant/optim/adam params 0.001

print ["Original parameters:" mold params]

;; Simulate gradient values
gradients: reduce [0.1 -0.2 0.3 -0.1 0.05]

;; Apply one step of SGD
sgd-optimizer/step gradients
print ["After SGD step:" mold sgd-optimizer/params]

;; Reset parameters and apply Adam
adam-optimizer/params: copy params
adam-optimizer/step gradients
print ["After Adam step:" mold adam-optimizer/params]
```

## Advanced Examples

### Example 7: Convolutional Neural Network (Conceptual)

```
;; Load the framework
do %covenant.reb

;; Define a simple CNN-like structure
cnn: make object! [
    conv1: covenant/nn/conv1d 1 16 3  ; 1 input channel, 16 output, kernel size 3
    pool1: covenant/nn/maxpool1d 2 2    ; Pool size 2, stride 2
    conv2: covenant/nn/conv1d 16 32 3  ; 16 input, 32 output, kernel size 3
    pool2: covenant/nn/maxpool1d 2 2    ; Pool size 2, stride 2
    flatten-layer: func [input] [covenant/flatten input]
    fc1: covenant/nn/linear 32 64       ; Fully connected
    output: covenant/nn/linear 64 10    ; 10-class output
    
    forward: func [input] [
        ;; Convolutional layers
        c1: conv1/forward input
        p1: pool1/forward c1
        c2: conv2/forward p1
        p2: pool2/forward c2
        
        ;; Flatten and fully connected
        flat: flatten-layer p2
        f1: fc1/forward flat
        activated: covenant/nn/relu f1
        output/forward activated
    ]
]

;; Note: Actual implementation would depend on how conv1d and maxpool1d are implemented
print "CNN structure defined"
```

### Example 8: Model Saving and Loading

```
;; Load the framework
do %covenant.reb

;; Create a model
model: make object! [
    layer1: covenant/nn/linear 5 3
    layer2: covenant/nn/linear 3 1
    
    ;; Serialize model parameters
    serialize: func [] [
        make object! [
            layer1-weights: layer1/weights/data
            layer1-bias: layer1/bias/data
            layer2-weights: layer2/weights/data
            layer2-bias: layer2/bias/data
            layer1-shapes: reduce [layer1/weights/shape layer1/bias/shape]
            layer2-shapes: reduce [layer2/weights/shape layer2/bias/shape]
        ]
    ]
    
    ;; Deserialize model parameters
    deserialize: func [data] [
        ;; In a real implementation, reconstruct layers from saved data
        print "Model deserialized"
    ]
]

;; Save the model
serialized-model: model/serialize
covenant/utils/save-model serialized-model %advanced-model.dat

;; Load the model
loaded-data: covenant/utils/load-model %advanced-model.dat
print ["Model loaded with layer1 weights shape:" mold loaded-data/layer1-shapes/1]

;; Clean up
delete %advanced-model.dat

print "Model saving/loading example completed"
```

### Example 9: Custom Loss Function

```
;; Load the framework
do %covenant.reb

;; Define a custom loss function (Mean Absolute Error)
mae-loss: func [
    "Mean Absolute Error loss function"
    predictions [object!]
    targets [object!]
] [
    if predictions/shape <> targets/shape [
        throw "Prediction and target shapes must match"
    ]
    
    ;; Calculate absolute differences
    diff: covenant/add predictions (covenant/mul targets -1)
    abs-diff-data: copy diff/data
    repeat i length? abs-diff-data [
        abs-diff-data/:i: abs(abs-diff-data/:i)
    ]
    
    ;; Create tensor from absolute differences
    abs-diff: make object! [
        data: abs-diff-data
        shape: diff/shape
        dtype: diff/dtype
    ]
    
    ;; Calculate mean
    sum-abs: covenant/sum abs-diff 0  ; Sum all elements
    mae: sum-abs/data/1 / length? abs-diff-data
    mae
]

;; Test the custom loss function
pred: covenant/tensor [1.0 2.0 3.0]
target: covenant/tensor [1.1 1.9 3.2]
loss: mae-loss pred target

print ["Custom MAE Loss:" loss]

;; Compare with built-in MSE
mse: covenant/nn/mse-loss pred target
print ["Built-in MSE Loss:" mse]
```

### Example 10: Data Pipeline

```
;; Load the framework
do %covenant.reb

;; Create sample dataset
sample-data: [
    [[0.1 0.2 0.3] [0]]
    [[0.4 0.5 0.6] [1]]
    [[0.7 0.8 0.9] [1]]
    [[0.2 0.3 0.4] [0]]
    [[0.5 0.6 0.7] [1]]
]

;; Write to file
write %pipeline-data.txt mold sample-data

;; Load and process data
raw-data: covenant/utils/load-data %pipeline-data.txt

;; Extract inputs and targets
inputs: make block! length? raw-data
targets: make block! length? raw-data

foreach item raw-data [
    append inputs covenant/tensor first item
    append targets covenant/tensor second item
]

print ["Loaded" length? inputs "samples"]

;; Normalize inputs
normalized-inputs: make block! length? inputs
mean-tensor: covenant/mean covenant/stack inputs 0
; Note: This is a simplified normalization - in practice, you'd compute mean/std per feature

;; Clean up
delete %pipeline-data.txt

print "Data pipeline example completed"
```

## Best Practices

### 1. Proper Weight Initialization

```
;; Good practice: Initialize weights with small random values
initialize-weights: func [input-size output-size] [
    ;; Xavier/Glorot initialization
    limit: sqrt (6.0 / (input-size + output-size))
    covenant/rand reduce [output-size input-size] * (2 * limit) - limit
]

;; Example usage
layer: make object! [
    weights: initialize-weights 10 5
    bias: covenant/zeros [5]
]
```

### 2. Input Normalization

```
;; Normalize input data to have mean 0 and std 1
normalize-input: func [tensor] [
    mean-val: covenant/mean tensor
    centered: covenant/add tensor (covenant/mul mean-val -1)
    
    ;; Calculate standard deviation
    squared-diff: covenant/mul centered centered
    var: covenant/mean squared-diff
    std: sqrt covenant/item var
    
    ; Avoid division by zero
    if std < 1e-8 [std: 1e-8]
    
    normalized: covenant/mul centered (1.0 / std)
    normalized
]
```

### 3. Gradient Clipping

```
;; Clip gradients to prevent exploding gradients
clip-gradients: func [gradients max-norm] [
    ;; Calculate gradient norm
    squared: covenant/mul gradients gradients
    sum-squared: covenant/sum squared 0
    norm: sqrt covenant/item sum-squared
    
    ;; Clip if norm exceeds threshold
    if norm > max-norm [
        scale-factor: max-norm / norm
        gradients: covenant/mul gradients scale-factor
    ]
    gradients
]
```

### 4. Model Validation

```
;; Validate model on separate dataset
validate-model: func [model validation-data] [
    total-loss: 0.0
    count: 0
    
    foreach item validation-data [
        input: first item
        target: second item
        prediction: model/forward input
        loss: covenant/nn/mse-loss prediction target
        total-loss: total-loss + loss
        count: count + 1
    ]
    
    total-loss / count
]
```

### 5. Learning Rate Scheduling

```
;; Simple learning rate decay
update-learning-rate: func [initial-lr epoch decay-rate] [
    initial-lr / (1.0 + decay-rate * epoch)
]

;; Example usage in training loop
initial-lr: 0.01
decay-rate: 0.01

repeat epoch 100 [
    current-lr: update-learning-rate initial-lr epoch decay-rate
    print ["Epoch:" epoch "Learning rate:" current-lr]
]
```

## Performance Tips

### 1. Efficient Tensor Operations

```
;; Prefer vectorized operations over loops when possible
;; Good: Using built-in tensor operations
a: covenant/tensor [1.0 2.0 3.0]
b: covenant/tensor [4.0 5.0 6.0]
result: covenant/mul a b  ; Vectorized operation

;; Less efficient: Manual element-wise operations
manual-result: make block! 3
repeat i 3 [
    append manual-result a/data/:i * b/data/:i
]
```

### 2. Memory Management

```
;; Reuse tensors when possible to minimize allocations
reuse-tensor: func [existing-tensor new-data] [
    ;; Update tensor data in-place if shapes match
    if length? existing-tensor/data = length? new-data [
        repeat i length? new-data [
            existing-tensor/data/:i: new-data/:i
        ]
    ]
    existing-tensor
]
```

### 3. Batch Processing

```
;; Process data in batches for efficiency
process-batch: func [model inputs targets batch-size] [
    results: make block! batch-size
    
    repeat i batch-size [
        input: inputs/:i
        target: targets/:i
        prediction: model/forward input
        append results prediction
    ]
    
    results
]
```

## Common Pitfalls and Solutions

### 1. Shape Mismatches

```
;; Always verify tensor shapes before operations
verify-shapes: func [tensor1 tensor2 operation] [
    if tensor1/shape <> tensor2/shape [
        print ["Shape mismatch in" operation ": " mold tensor1/shape "<>" mold tensor2/shape]
        throw "Shape mismatch error"
    ]
]

;; Usage
a: covenant/tensor [1.0 2.0 3.0]
b: covenant/tensor [4.0 5.0]  ; Different shape
; verify-shapes a b "addition"  ; Would throw error
```

### 2. Numerical Stability

```
;; Add small epsilon to prevent division by zero
safe-divide: func [numerator denominator] [
    epsilon: 1e-8
    numerator / (denominator + epsilon)
]

;; Use numerically stable versions of functions
stable-softmax: func [x] [
    ; Subtract max for numerical stability
    max-val: first x/data
    foreach val x/data [if val > max-val [max-val: val]]
    
    shifted-data: copy x/data
    repeat i length? shifted-data [
        shifted-data/:i: shifted-data/:i - max-val
    ]
    
    exp-data: copy shifted-data
    repeat i length? exp-data [
        exp-data/:i: exp exp-data/:i  ; Assuming exp function exists
    ]
    
    sum-exp: 0.0
    foreach val exp-data [sum-exp: sum-exp + val]
    
    result-data: copy exp-data
    repeat i length? result-data [
        result-data/:i: result-data/:i / sum-exp
    ]
    
    make object! [
        data: result-data
        shape: x/shape
        dtype: x/dtype
    ]
]
```

This comprehensive guide provides practical examples and best practices for using the Covenant AI Framework effectively.