# Covenant AI Framework - Documentation Index

Welcome to the Covenant AI Framework documentation! Covenant is a REBOL-based artificial intelligence framework designed for machine learning and neural network development, inspired by PyTorch.

## Table of Contents

### [Main Documentation](main-docs.md)
Complete overview of the Covenant framework, including:
- Getting Started
- Core Concepts (Tensors, Neural Networks, Autograd)
- Deep Learning with Covenant
- Optimization Algorithms
- Utilities and Helpers
- Advanced Topics
- Best Practices

### [API Reference](api-reference.md)
Detailed API documentation for all Covenant functions:
- Core Tensor Operations
- Neural Network Layers
- Optimization Algorithms
- Utilities
- Autograd System

### [Tutorials](tutorials.md)
Step-by-step guides for using Covenant:
- Getting Started
- Building Neural Networks
- Training Models
- Advanced Features

### [Code Examples and Best Practices](examples-best-practices.md)
Practical examples and recommended practices:
- Basic Examples
- Intermediate Examples
- Advanced Examples
- Best Practices
- Performance Tips
- Common Pitfalls and Solutions

## Quick Start

To begin using Covenant, simply load the main module:

```
do %covenant.reb
```

Then create your first tensor:

```
x: covenant/tensor [1.0 2.0 3.0]
print ["Tensor created:" mold x/data]
```

## About Covenant

Covenant provides:
- **Tensor Operations**: Multi-dimensional array operations with GPU-like semantics
- **Dynamic Neural Networks**: Define-by-run approach for building neural networks
- **Automatic Differentiation**: Automatic gradient computation for backpropagation
- **Optimization Algorithms**: SGD, Adam, and other optimization methods
- **Modular Design**: Clean separation of concerns with extensible components

## Author
**Karina Mikhailovna Chernykh**

## License
Covenant is released under an open license for research and educational purposes.

---
*For support and questions, please refer to the respective documentation sections.*