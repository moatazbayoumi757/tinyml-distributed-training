# Distributed Neural Network Training (PyTorch + Docker)

## Overview
This project implements a distributed model-parallel neural network using PyTorch Distributed and Docker. The model is split across multiple processes (containers), where each part performs forward and backward propagation while communicating intermediate data.

## Key Features
- Model parallelism across multiple processes using PyTorch Distributed (Gloo backend)
- Inter-process communication using `dist.send` and `dist.recv`
- Distributed forward and backward propagation
- Training on MNIST dataset with batch size 32 and 1800 steps
- Performance benchmarking for runtime and system bottlenecks

## Architecture
The neural network is split into two parts:

### Part 1 (Process 0)
- Input: MNIST images (28x28)
- Layers:
  - Flatten
  - Linear (784 → 500)
  - ReLU
- Sends activations and labels to Part 2
- Receives gradients and performs backpropagation

### Part 2 (Process 1)
- Input: Activations from Part 1
- Layers:
  - Linear (500 → 100)
  - ReLU
  - Linear (100 → 10)
- Computes loss and gradients
- Sends gradients back to Part 1

## Workflow
1. Part 1 processes input data and sends activations to Part 2
2. Part 2 computes loss and gradients
3. Gradients are sent back to Part 1
4. Both models update parameters synchronously

## Results
- Achieved ~93% accuracy on MNIST
- Achieved F1-score of ~0.91
- Identified CPU and memory trade-offs across configurations
- Measured runtime performance across distributed setups

## Tech Stack
- Python
- PyTorch (Distributed)
- Docker
- MNIST Dataset

## How to Run
```bash
docker compose up
