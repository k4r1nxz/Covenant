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

## Features

- Tensor abstraction
- Basic tensor operations
- Automatic differentiation
- Centralized computational graph
- Gradient propagation and accumulation
- Neural network components
- Optimizers for parameter updates

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
