# Covenant AI Framework - API Reference

## Table of Contents
1. [Core Tensor Operations](#core-tensor-operations)
2. [Neural Network Layers](#neural-network-layers)
3. [Optimization Algorithms](#optimization-algorithms)
4. [Utilities](#utilities)
5. [Autograd System](#autograd-system)

## Core Tensor Operations

### `covenant/tensor`
Creates a tensor from input data with optional gradient tracking.

**Syntax:**
```
covenant/tensor data
covenant/tensor/requires_grad data
```

**Parameters:**
- `data`: Block of numbers or nested blocks representing tensor data
- `/requires_grad`: Optional refinement to enable gradient tracking

**Returns:**
An object with `data`, `shape`, and `dtype` fields. If gradient tracking is enabled, also includes `grad`, `requires_grad`, `grad_fn`, and `parents` fields.

**Example:**
```
x: covenant/tensor [1.0 2.0 3.0]
y: covenant/tensor/requires_grad [4.0 5.0 6.0]
```

### `covenant/zeros`
Creates a tensor filled with zeros.

**Syntax:**
```
covenant/zeros shape
```

**Parameters:**
- `shape`: Block specifying the tensor dimensions

**Returns:**
A tensor object filled with zeros.

**Example:**
```
zero-tensor: covenant/zeros [2 3]  ; Creates 2x3 matrix of zeros
```

### `covenant/ones`
Creates a tensor filled with ones.

**Syntax:**
```
covenant/ones shape
```

**Parameters:**
- `shape`: Block specifying the tensor dimensions

**Returns:**
A tensor object filled with ones.

**Example:**
```
ones-tensor: covenant/ones [3 2]  ; Creates 3x2 matrix of ones
```

### `covenant/rand`
Creates a tensor filled with random values.

**Syntax:**
```
covenant/rand shape
```

**Parameters:**
- `shape`: Block specifying the tensor dimensions

**Returns:**
A tensor object filled with random values between 0 and 1.

**Example:**
```
random-tensor: covenant/rand [2 2]  ; Creates 2x2 matrix of random values
```

### `covenant/add`
Element-wise addition of two tensors.

**Syntax:**
```
covenant/add tensor1 tensor2
```

**Parameters:**
- `tensor1`, `tensor2`: Tensor objects with the same shape

**Returns:**
A new tensor with element-wise sum.

**Example:**
```
a: covenant/tensor [1.0 2.0]
b: covenant/tensor [3.0 4.0]
c: covenant/add a b  ; Results in [4.0 6.0]
```

### `covenant/mul`
Element-wise multiplication of two tensors.

**Syntax:**
```
covenant/mul tensor1 tensor2
```

**Parameters:**
- `tensor1`, `tensor2`: Tensor objects with the same shape

**Returns:**
A new tensor with element-wise product.

**Example:**
```
a: covenant/tensor [1.0 2.0]
b: covenant/tensor [3.0 4.0]
c: covenant/mul a b  ; Results in [3.0 8.0]
```

### `covenant/matmul`
Matrix multiplication of two tensors.

**Syntax:**
```
covenant/matmul tensor1 tensor2
```

**Parameters:**
- `tensor1`, `tensor2`: 2D tensor objects with compatible dimensions

**Returns:**
A new tensor with the matrix product.

**Example:**
```
a: covenant/tensor [[1.0 2.0] [3.0 4.0]]
b: covenant/tensor [[5.0 6.0] [7.0 8.0]]
c: covenant/matmul a b  ; Results in [[19.0 22.0] [43.0 50.0]]
```

### `covenant/reshape`
Reshapes a tensor to a new shape.

**Syntax:**
```
covenant/reshape tensor new-shape
```

**Parameters:**
- `tensor`: Input tensor object
- `new-shape`: Block specifying the new dimensions

**Returns:**
A new tensor with the specified shape.

**Example:**
```
a: covenant/tensor [1.0 2.0 3.0 4.0]
b: covenant/reshape a [2 2]  ; Reshapes to 2x2 matrix
```

### `covenant/sum`
Computes the sum of tensor elements along a specified axis.

**Syntax:**
```
covenant/sum tensor axis
```

**Parameters:**
- `tensor`: Input tensor object
- `axis`: Integer specifying the axis along which to sum

**Returns:**
A new tensor with the sum computed along the specified axis.

**Example:**
```
a: covenant/tensor [[1.0 2.0] [3.0 4.0]]
row-sums: covenant/sum a 1  ; Sum along rows: [3.0 7.0]
col-sums: covenant/sum a 0  ; Sum along columns: [4.0 6.0]
```

### `covenant/concat`
Concatenates tensors along a specified axis.

**Syntax:**
```
covenant/concat tensors axis
covenant/concat/axis tensors axis-number
```

**Parameters:**
- `tensors`: Block of tensor objects with compatible shapes
- `axis`: Integer specifying the axis along which to concatenate

**Returns:**
A new tensor with concatenated data.

**Example:**
```
a: covenant/tensor [1.0 2.0]
b: covenant/tensor [3.0 4.0]
c: covenant/concat reduce [a b] 0  ; Concatenates along axis 0: [1.0 2.0 3.0 4.0]
```

### `covenant/stack`
Stacks tensors along a new axis.

**Syntax:**
```
covenant/stack tensors dim
```

**Parameters:**
- `tensors`: Block of tensor objects with the same shape
- `dim`: Integer specifying the dimension along which to stack

**Returns:**
A new tensor with an additional dimension.

**Example:**
```
a: covenant/tensor [1.0 2.0]
b: covenant/tensor [3.0 4.0]
c: covenant/stack reduce [a b] 0  ; Stacks along new axis 0
```

### `covenant/mean`
Computes the mean of tensor elements.

**Syntax:**
```
covenant/mean tensor
covenant/mean/axis tensor axis-number
```

**Parameters:**
- `tensor`: Input tensor object
- `axis`: Optional axis along which to compute the mean

**Returns:**
A new tensor with the computed mean(s).

**Example:**
```
a: covenant/tensor [1.0 2.0 3.0 4.0]
avg: covenant/mean a  ; Computes global mean: [2.5]
```

### `covenant/flatten`
Flattens a tensor to 1D.

**Syntax:**
```
covenant/flatten tensor
```

**Parameters:**
- `tensor`: Input tensor object

**Returns:**
A new 1D tensor with all elements.

**Example:**
```
a: covenant/tensor [[1.0 2.0] [3.0 4.0]]
flat: covenant/flatten a  ; Results in [1.0 2.0 3.0 4.0]
```

### `covenant/view`
Creates a view of a tensor with a different shape.

**Syntax:**
```
covenant/view tensor new-shape
```

**Parameters:**
- `tensor`: Input tensor object
- `new-shape`: Block specifying the new dimensions

**Returns:**
A new tensor with the specified shape.

**Example:**
```
a: covenant/tensor [1.0 2.0 3.0 4.0]
b: covenant/view a [2 2]  ; Creates 2x2 view
```

### `covenant/clone`
Creates a copy of a tensor.

**Syntax:**
```
covenant/clone tensor
```

**Parameters:**
- `tensor`: Input tensor object

**Returns:**
A new tensor with copied data.

**Example:**
```
a: covenant/tensor [1.0 2.0 3.0]
b: covenant/clone a  ; Creates a copy of a
```

## Neural Network Layers

### `covenant/nn/linear`
Creates a linear (fully connected) layer.

**Syntax:**
```
covenant/nn/linear input-size output-size
```

**Parameters:**
- `input-size`: Integer specifying the size of input features
- `output-size`: Integer specifying the size of output features

**Returns:**
A linear layer object with `forward` method.

**Example:**
```
layer: covenant/nn/linear 10 5  ; 10 inputs -> 5 outputs
input: covenant/tensor [1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0]
output: layer/forward input
```

### `covenant/nn/relu`
Applies ReLU activation function.

**Syntax:**
```
covenant/nn/relu tensor
```

**Parameters:**
- `tensor`: Input tensor object

**Returns:**
A new tensor with ReLU applied element-wise.

**Example:**
```
a: covenant/tensor [-1.0 0.0 1.0 2.0]
b: covenant/nn/relu a  ; Results in [0.0 0.0 1.0 2.0]
```

### `covenant/nn/sigmoid`
Applies Sigmoid activation function.

**Syntax:**
```
covenant/nn/sigmoid tensor
```

**Parameters:**
- `tensor`: Input tensor object

**Returns:**
A new tensor with Sigmoid applied element-wise.

**Example:**
```
a: covenant/tensor [0.0 1.0 2.0]
b: covenant/nn/sigmoid a
```

### `covenant/nn/tanh`
Applies Tanh activation function.

**Syntax:**
```
covenant/nn/tanh tensor
```

**Parameters:**
- `tensor`: Input tensor object

**Returns:**
A new tensor with Tanh applied element-wise.

**Example:**
```
a: covenant/tensor [0.0 1.0 2.0]
b: covenant/nn/tanh a
```

### `covenant/nn/softmax`
Applies Softmax activation function.

**Syntax:**
```
covenant/nn/softmax tensor
```

**Parameters:**
- `tensor`: Input tensor object

**Returns:**
A new tensor with Softmax applied.

**Example:**
```
a: covenant/tensor [1.0 2.0 3.0]
b: covenant/nn/softmax a
```

### `covenant/nn/conv1d`
Creates a 1D convolutional layer.

**Syntax:**
```
covenant/nn/conv1d in-channels out-channels kernel-size
```

**Parameters:**
- `in-channels`: Number of input channels
- `out-channels`: Number of output channels
- `kernel-size`: Size of the convolution kernel

**Returns:**
A 1D convolutional layer object with `forward` method.

**Example:**
```
conv-layer: covenant/nn/conv1d 1 16 3  ; 1 input channel, 16 output channels, kernel size 3
```

### `covenant/nn/maxpool1d`
Creates a 1D max pooling layer.

**Syntax:**
```
covenant/nn/maxpool1d kernel-size stride
```

**Parameters:**
- `kernel-size`: Size of the pooling kernel
- `stride`: Stride for pooling

**Returns:**
A 1D max pooling layer object with `forward` method.

**Example:**
```
pool-layer: covenant/nn/maxpool1d 2 2  ; Kernel size 2, stride 2
```

### `covenant/nn/dropout`
Creates a dropout layer.

**Syntax:**
```
covenant/nn/dropout probability
```

**Parameters:**
- `probability`: Dropout probability (0.0 to 1.0)

**Returns:**
A dropout layer object with `forward`, `train`, and `eval` methods.

**Example:**
```
dropout-layer: covenant/nn/dropout 0.5  ; 50% dropout
```

### `covenant/nn/batchnorm1d`
Creates a 1D batch normalization layer.

**Syntax:**
```
covenant/nn/batchnorm1d num-features
```

**Parameters:**
- `num-features`: Number of features

**Returns:**
A 1D batch normalization layer object with `forward`, `train`, and `eval` methods.

**Example:**
```
bn-layer: covenant/nn/batchnorm1d 10  ; Batch norm for 10 features
```

## Optimization Algorithms

### `covenant/optim/sgd`
Creates a Stochastic Gradient Descent optimizer.

**Syntax:**
```
covenant/optim/sgd parameters learning-rate
```

**Parameters:**
- `parameters`: Block of parameters to optimize
- `learning-rate`: Learning rate for updates

**Returns:**
An SGD optimizer object with `step` method.

**Example:**
```
params: reduce [1.0 2.0 3.0]
optimizer: covenant/optim/sgd params 0.01
; Later: optimizer/step gradients
```

### `covenant/optim/adam`
Creates an Adam optimizer.

**Syntax:**
```
covenant/optim/adam parameters learning-rate
```

**Parameters:**
- `parameters`: Block of parameters to optimize
- `learning-rate`: Learning rate for updates

**Returns:**
An Adam optimizer object with `step` method.

**Example:**
```
params: reduce [1.0 2.0 3.0]
optimizer: covenant/optim/adam params 0.001
; Later: optimizer/step gradients
```

## Utilities

### `covenant/utils/save-model`
Saves a model to a file.

**Syntax:**
```
covenant/utils/save-model model file-path
```

**Parameters:**
- `model`: Model object to save
- `file-path`: Path to save the model

**Example:**
```
covenant/utils/save-model my-model %my-model.dat
```

### `covenant/utils/load-model`
Loads a model from a file.

**Syntax:**
```
covenant/utils/load-model file-path
```

**Parameters:**
- `file-path`: Path to load the model from

**Returns:**
Loaded model data.

**Example:**
```
loaded-model: covenant/utils/load-model %my-model.dat
```

### `covenant/utils/visualize`
Visualizes a tensor.

**Syntax:**
```
covenant/utils/visualize tensor
```

**Parameters:**
- `tensor`: Tensor object to visualize

**Example:**
```
tensor: covenant/tensor [[1.0 2.0] [3.0 4.0]]
covenant/utils/visualize tensor
```

### `covenant/utils/load-data`
Loads data from a file.

**Syntax:**
```
covenant/utils/load-data file-path
covenant/utils/load-data/as-dataset file-path
```

**Parameters:**
- `file-path`: Path to load data from
- `/as-dataset`: Optional refinement to return as dataset object

**Returns:**
Loaded data or dataset object.

**Example:**
```
data: covenant/utils/load-data %data.txt
```

### `covenant/utils/evaluate`
Evaluates model performance.

**Syntax:**
```
covenant/utils/evaluate model test-data loss-function
```

**Parameters:**
- `model`: Model to evaluate
- `test-data`: Test data for evaluation
- `loss-function`: Loss function to use

**Returns:**
Average loss across test data.

**Example:**
```
avg-loss: covenant/utils/evaluate my-model test-set 'mse-loss
```

## Autograd System

### Gradient Tracking
Tensors can be created with gradient tracking enabled using the `/requires_grad` refinement:

```
x: covenant/tensor/requires_grad [1.0 2.0 3.0]
```

This enables automatic differentiation for backpropagation through the computational graph.