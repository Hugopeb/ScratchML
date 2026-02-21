# ScratchML

![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A lightweight neural network framework that bridges the gap between fully abstracted deep learning libraries and from-scratch implementations.

This project uses PyTorch selectively for **optimized tensor operations and system-level utilities**, while implementing **model architecture, parameter management, and training logic explicitly**. The goal is to retain performance and practicality without relying on high-level abstractions such as `torch.nn.Module` or `torch.optim` for a better understanding of ML workflows.

---

## Motivation

Modern deep learning frameworks provide powerful abstractions that enable rapid experimentation, but they often obscure the underlying mechanics of training neural networks. Conversely, implementing neural networks entirely from scratch offers transparency but quickly becomes inefficient and impractical for real-world tasks.

This framework is designed to operate **between these two extremes**:

- Preserve transparency and control over model internals
- Avoid deeply nested Python loops and inefficient numerical code
- Leverage PyTorch only where it provides concrete advantages

The result is a framework suitable for learning, experimentation, and architectural exploration, without sacrificing computational efficiency.

---

## Design Principles

### Selective Use of PyTorch

PyTorch is used strictly as a **low-level numerical backend**, providing:

- Optimized tensor operations
- Convolution primitives (`torch.conv2d`)
- Serialization utilities (`torch.save`, `torch.load`)

The framework intentionally avoids:

- `torch.nn.Module`
- Built-in optimizers (`torch.optim`)
- Automatic layer composition
- End-to-end training abstractions

---

## Architecture Overview

- Neural networks are defined through a custom network class
- Forward passes are manually defined
- Convolutional layers rely on `torch.conv2d` for performance
- Activation functions are applied explicitly
- Training logic and parameter updates are controlled directly

This structure provides fine-grained control over every stage of computation while maintaining efficient execution.

---

## Convolutional Focus and CIFAR-10

The framework is currently oriented toward convolutional neural networks, with CIFAR-10 as a target dataset.

CIFAR-10 provides a practical benchmark that:
- Requires non-trivial spatial feature extraction
- Benefits from optimized convolution operations
- Enables meaningful evaluation of architectural choices and update rules

---

## Scope and Limitations

This project is **not** intended to:
- Replace PyTorch or existing deep learning frameworks
- Serve as a production-ready training system
- Maximize performance relative to highly optimized libraries

It **is** intended to:
- Provide insight into the mechanics of neural network training
- Support experimentation with custom architectures and update rules
- Serve as a pedagogical and research-oriented tool

---

## Intended Audience

- Developers seeking deeper understanding beyond high-level APIs
- Students studying neural network internals
- Researchers experimenting with alternative training logic
- Engineers interested in framework design tradeoffs

---

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/Hugopeb/torchLab.git
cd torchLab
```

2. **Create a virtual environment**

```bash
python3 -m venv torchLab_VENV
source torchLab_VENV/bin/activate
```

3. **Install required packages**

```bash
pip install -r requirements.txt
pip install -e .
```


---

