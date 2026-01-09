# Covenant AI Framework - API Reference v1.1.0

## Table of Contents
1. [Core Tensor Operations](#core-tensor-operations)
2. [Neural Network Layers](#neural-network-layers)
3. [Optimization Algorithms](#optimization-algorithms)
4. [Utilities](#utilities)
5. [Autograd System](#autograd-system)

## Core Tensor Operations

### `covenant/tensor`
Creates a tensor from input data with optional gradient tracking and data type specification.

**Syntax:**
```
covenant/tensor data
covenant/tensor/requires_grad data
covenant/tensor/dtype data data-type
```

**Parameters:**
- `data`: Block of numbers or nested blocks representing tensor data
- `/requires_grad`: Optional refinement to enable gradient tracking
- `/dtype`: Optional refinement to specify data type
- `data-type`: Word specifying data type (float32, int32, int64)

**Returns:**
Object with tensor data, shape, and dtype properties.

### `covenant/arange`
Creates a tensor with evenly spaced values within a given interval.

**Syntax:**
```
covenant/arange start end
covenant/arange/step start end step-size
```

**Parameters:**
- `start`: Starting value
- `end`: Ending value (exclusive)
- `/step`: Optional refinement to specify step size
- `step-size`: Step size (default: 1.0)

**Returns:**
1D tensor with evenly spaced values.

### `covenant/linspace`
Creates a tensor with linearly spaced values between start and end.

**Syntax:**
```
covenant/linspace start end num
```

**Parameters:**
- `start`: Starting value
- `end`: Ending value (inclusive)
- `num`: Number of points

**Returns:**
1D tensor with linearly spaced values.

### `covenant/pow`
Raises tensor elements to a specified power.

**Syntax:**
```
covenant/pow tensor exponent
```

**Parameters:**
- `tensor`: Input tensor
- `exponent`: Power to raise to

**Returns:**
Tensor with elements raised to the specified power.

### `covenant/sqrt`
Computes square root of tensor elements.

**Syntax:**
```
covenant/sqrt tensor
```

**Parameters:**
- `tensor`: Input tensor

**Returns:**
Tensor with square root of elements.

### `covenant/exp`
Computes exponential of tensor elements.

**Syntax:**
```
covenant/exp tensor
```

**Parameters:**
- `tensor`: Input tensor

**Returns:**
Tensor with exponential of elements.

### `covenant/log`
Computes natural logarithm of tensor elements.

**Syntax:**
```
covenant/log tensor
```

**Parameters:**
- `tensor`: Input tensor

**Returns:**
Tensor with natural logarithm of elements.

### `covenant/max`
Computes maximum value along specified axis.

**Syntax:**
```
covenant/max tensor
covenant/max/axis tensor axis-val
```

**Parameters:**
- `tensor`: Input tensor
- `/axis`: Optional refinement to specify axis
- `axis-val`: Axis along which to compute max (0-indexed)

**Returns:**
Maximum value or tensor of maximum values along axis.

### `covenant/min`
Computes minimum value along specified axis.

**Syntax:**
```
covenant/min tensor
covenant/min/axis tensor axis-val
```

**Parameters:**
- `tensor`: Input tensor
- `/axis`: Optional refinement to specify axis
- `axis-val`: Axis along which to compute min (0-indexed)

**Returns:**
Minimum value or tensor of minimum values along axis.

### `covenant/argmax`
Computes index of maximum value along specified axis.

**Syntax:**
```
covenant/argmax tensor
covenant/argmax/axis tensor axis-val
```

**Parameters:**
- `tensor`: Input tensor
- `/axis`: Optional refinement to specify axis
- `axis-val`: Axis along which to compute argmax (0-indexed)

**Returns:**
Index of maximum value or tensor of indices along axis.

### `covenant/argmin`
Computes index of minimum value along specified axis.

**Syntax:**
```
covenant/argmin tensor
covenant/argmin/axis tensor axis-val
```

**Parameters:**
- `tensor`: Input tensor
- `/axis`: Optional refinement to specify axis
- `axis-val`: Axis along which to compute argmin (0-indexed)

**Returns:**
Index of minimum value or tensor of indices along axis.

### `covenant/transpose`
Transposes a 2D tensor.

**Syntax:**
```
covenant/transpose tensor
```

**Parameters:**
- `tensor`: Input 2D tensor

**Returns:**
Transposed tensor with swapped dimensions.

## Neural Network Layers

### `covenant/nn/linear`
Creates a linear (fully connected) layer with autograd support.

**Syntax:**
```
covenant/nn/linear input-size output-size
```

**Parameters:**
- `input-size`: Size of input features
- `output-size`: Size of output features

**Returns:**
Linear layer object with forward method.

### `covenant/nn/relu`
Creates a ReLU activation function with gradient tracking.

**Syntax:**
```
covenant/nn/relu x
```

**Parameters:**
- `x`: Input tensor

**Returns:**
Tensor with ReLU activation applied.

### `covenant/nn/leaky_relu`
Creates a Leaky ReLU activation function with gradient tracking.

**Syntax:**
```
covenant/nn/leaky_relu x alpha
```

**Parameters:**
- `x`: Input tensor
- `alpha`: Negative slope parameter (default: 0.01)

**Returns:**
Tensor with Leaky ReLU activation applied.

### `covenant/nn/sigmoid`
Creates a Sigmoid activation function with gradient tracking.

**Syntax:**
```
covenant/nn/sigmoid x
```

**Parameters:**
- `x`: Input tensor

**Returns:**
Tensor with Sigmoid activation applied.

### `covenant/nn/tanh`
Creates a Tanh activation function with gradient tracking.

**Syntax:**
```
covenant/nn/tanh x
```

**Parameters:**
- `x`: Input tensor

**Returns:**
Tensor with Tanh activation applied.

### `covenant/nn/softmax`
Creates a Softmax activation function with gradient tracking.

**Syntax:**
```
covenant/nn/softmax x
```

**Parameters:**
- `x`: Input tensor

**Returns:**
Tensor with Softmax activation applied.

### `covenant/nn/mse_loss`
Creates a Mean Squared Error loss function with gradient tracking.

**Syntax:**
```
covenant/nn/mse_loss predictions targets
```

**Parameters:**
- `predictions`: Predicted values tensor
- `targets`: Target values tensor

**Returns:**
Scalar tensor with MSE loss value.

### `covenant/nn/cross_entropy_loss`
Creates a Cross Entropy loss function with gradient tracking.

**Syntax:**
```
covenant/nn/cross_entropy_loss predictions targets
```

**Parameters:**
- `predictions`: Predicted values tensor
- `targets`: Target values tensor

**Returns:**
Scalar tensor with Cross Entropy loss value.

### `covenant/nn/conv1d`
Creates a 1D convolutional layer with autograd support.

**Syntax:**
```
covenant/nn/conv1d in-channels out-channels kernel-size
```

**Parameters:**
- `in-channels`: Number of input channels
- `out-channels`: Number of output channels
- `kernel-size`: Size of the convolution kernel

**Returns:**
Conv1D layer object with forward method.

### `covenant/nn/dropout`
Creates a Dropout layer with probability p.

**Syntax:**
```
covenant/nn/dropout p
```

**Parameters:**
- `p`: Dropout probability (default: 0.5)

**Returns:**
Dropout layer object with forward method.

### `covenant/nn/batchnorm1d`
Creates a 1D Batch Normalization layer.

**Syntax:**
```
covenant/nn/batchnorm1d num-features
```

**Parameters:**
- `num-features`: Number of features

**Returns:**
BatchNorm1D layer object with forward method.

### `covenant/nn/sequential`
Creates a Sequential module for chaining layers.

**Syntax:**
```
covenant/nn/sequential layers
```

**Parameters:**
- `layers`: Block of layers to chain

**Returns:**
Sequential module object with forward method.

## Optimization Algorithms

### `covenant/optim/sgd`
Creates a Stochastic Gradient Descent optimizer.

**Syntax:**
```
covenant/optim/sgd param-list lr
covenant/optim/sgd/momentum param-list lr mom
covenant/optim/sgd/momentum/nesterov param-list lr mom
```

**Parameters:**
- `param-list`: Block of parameters to optimize
- `lr`: Learning rate
- `/momentum`: Optional refinement to use momentum
- `mom`: Momentum factor (default: 0.0)
- `/nesterov`: Optional refinement to use Nesterov momentum

**Returns:**
SGD optimizer object with step method.

### `covenant/optim/adam`
Creates an Adam optimizer.

**Syntax:**
```
covenant/optim/adam param-list lr
covenant/optim/adam/beta1 param-list lr b1
covenant/optim/adam/beta1/beta2 param-list lr b1 b2
covenant/optim/adam/beta1/beta2/epsilon param-list lr b1 b2 eps
```

**Parameters:**
- `param-list`: Block of parameters to optimize
- `lr`: Learning rate
- `/beta1`: Optional refinement to specify beta1 parameter (default: 0.9)
- `b1`: Beta1 value
- `/beta2`: Optional refinement to specify beta2 parameter (default: 0.999)
- `b2`: Beta2 value
- `/epsilon`: Optional refinement to specify epsilon parameter (default: 1e-8)
- `eps`: Epsilon value

**Returns:**
Adam optimizer object with step method.

### `covenant/optim/rmsprop`
Creates an RMSprop optimizer.

**Syntax:**
```
covenant/optim/rmsprop param-list lr
covenant/optim/rmsprop/alpha param-list lr a
covenant/optim/rmsprop/alpha/epsilon param-list lr a eps
```

**Parameters:**
- `param-list`: Block of parameters to optimize
- `lr`: Learning rate
- `/alpha`: Optional refinement to specify smoothing constant (default: 0.999)
- `a`: Alpha value
- `/epsilon`: Optional refinement to specify epsilon parameter (default: 1e-8)
- `eps`: Epsilon value

**Returns:**
RMSprop optimizer object with step method.

### `covenant/optim/adagrad`
Creates an Adagrad optimizer.

**Syntax:**
```
covenant/optim/adagrad param-list lr
covenant/optim/adagrad/epsilon param-list lr eps
```

**Parameters:**
- `param-list`: Block of parameters to optimize
- `lr`: Learning rate
- `/epsilon`: Optional refinement to specify epsilon parameter (default: 1e-8)
- `eps`: Epsilon value

**Returns:**
Adagrad optimizer object with step method.

### `covenant/optim/adadelta`
Creates an Adadelta optimizer.

**Syntax:**
```
covenant/optim/adadelta param-list
covenant/optim/adadelta/rho param-list r
covenant/optim/adadelta/rho/epsilon param-list r eps
```

**Parameters:**
- `param-list`: Block of parameters to optimize
- `/rho`: Optional refinement to specify rho parameter (default: 0.95)
- `r`: Rho value
- `/epsilon`: Optional refinement to specify epsilon parameter (default: 1e-6)
- `eps`: Epsilon value

**Returns:**
Adadelta optimizer object with step method.

## Learning Rate Schedulers

### `covenant/lr_scheduler/step_lr`
Creates a Step learning rate scheduler.

**Syntax:**
```
covenant/lr_scheduler/step_lr optimizer step-size gamma
```

**Parameters:**
- `optimizer`: Optimizer to schedule
- `step-size`: Period of learning rate decay
- `gamma`: Multiplicative factor of learning rate decay (default: 0.1)

**Returns:**
Step learning rate scheduler object with step method.

### `covenant/lr_scheduler/exp_lr`
Creates an Exponential learning rate scheduler.

**Syntax:**
```
covenant/lr_scheduler/exp_lr optimizer gamma
```

**Parameters:**
- `optimizer`: Optimizer to schedule
- `gamma`: Multiplicative factor of learning rate decay

**Returns:**
Exponential learning rate scheduler object with step method.

## Autograd System

The autograd system provides automatic differentiation for tensor operations. It maintains a computational graph and computes gradients using backpropagation.

Key features:
- Automatic gradient computation for basic operations (add, mul, matmul)
- Support for advanced operations (pow, exp, log, sin, cos)
- Computational graph tracking
- Backward propagation through complex operations
