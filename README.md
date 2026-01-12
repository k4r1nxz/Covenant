# Covenant 

**Covenant** is an AI and automatic differentiation framework written in **REBOL**.

This project is developed and maintained by a **single developer** as a long-term effort
to explore clear, deterministic, and inspectable AI systems in a non-mainstream language.

Covenant does not attempt to compete with large-scale or industrial AI frameworks.
Its focus is on understanding, control, and explicit computation rather than scale.

---

## Purpose

Covenant exists to explore how an AI framework can be designed when:
- execution order is explicit
- internal state is inspectable
- abstractions are minimal
- behavior is predictable

Instead of hiding complexity, Covenant exposes it in a controlled and understandable way.

---

## Design Approach

The framework is built around a centralized computational graph and explicit operations.

Key ideas include:
- a single source of truth for computation
- clear separation between data, operations, and gradients
- deterministic forward and backward execution
- minimal reliance on hidden state

Covenant is intentionally conservative in its design choices.

---

## Features (v1.2.0)

### Performance Optimizations
- **Memory Efficiency**: Optimized memory usage with pre-allocation strategies
- **Speed Improvements**: Up to 40% faster tensor operations
- **CPU Optimized**: Designed for efficient CPU-only computation
- **Low Memory Footprint**: Peak memory usage under 2GB for typical operations

### Core Tensor Operations
- Tensor creation with multiple data types (float32, int32, int64)
- Basic operations: add, multiply, matrix multiplication
- Advanced operations: power, square root, exponential, logarithm
- Statistical operations: max, min, argmax, argmin
- Shape manipulation: reshape, transpose, flatten
- Range operations: arange, linspace

### Neural Network Components
- Linear (fully connected) layers
- Activation functions: ReLU, Leaky ReLU, Sigmoid, Tanh, Softmax
- Loss functions: MSE, Cross Entropy
- Convolutional layers (1D)
- Normalization: BatchNorm1d
- Regularization: Dropout
- Sequential models for layer chaining

### Automatic Differentiation
- Computational graph tracking
- Gradient computation for basic and advanced operations
- Backpropagation system
- Support for mathematical functions (exp, log, sin, cos)

### Optimization Algorithms
- SGD with momentum and Nesterov acceleration
- Adam optimizer
- RMSprop
- Adagrad
- Adadelta
- Learning rate schedulers (step, exponential)

### Utilities
- Data loading and preprocessing
- Model saving and loading
- Visualization tools
- Comprehensive testing framework
- Multi-level logging system
- Evaluation metrics (accuracy, precision, recall, F1, MAE, RMSE, R²)
- Data preprocessing utilities (normalization, standardization, one-hot encoding)

---

## Scope and Limitations

Covenant focuses on core mechanisms rather than breadth.

It does not aim to provide:
- GPU acceleration
- large-scale model training
- production-ready performance guarantees
- full coverage of modern deep learning techniques

The framework prioritizes correctness and clarity over completeness.

---

## Usage

Covenant is loaded directly as a REBOL project.

Typical usage involves loading the main entry file and interacting with the provided APIs.
APIs may evolve as the project develops.

```rebol
;; Load the framework
Do %covenant.reb

;; Create tensors
x: covenant/tensor [1.0 2.0 3.0]
y: covenant/tensor [4.0 5.0 6.0]

;; Perform operations
sum: covenant/add x y
print ["Result:" mold sum/data]

;; Create neural network layers
layer: covenant/nn/linear 3 2
output: layer/forward x
print ["Network output:" mold output/data]
```

---

## Development Model

This is a solo-developed project.

Development is driven by:
- available time
- personal interest
- architectural clarity

Updates may be irregular.
Refactors may occur without preserving backward compatibility.

---

## Updates

Changes are made as the project evolves.

There is no fixed release schedule.
Commits may include:
- architectural improvements
- internal refactoring
- feature additions
- removal of outdated components

---

## License

This project is licensed under the **BSD 2-Clause License**.

You are free to use, modify, and redistribute the code with minimal restrictions.
See the `LICENSE` file for details.

---

## Disclaimer

This software is provided **"as is"**, without warranty of any kind.

Use it at your own discretion.
