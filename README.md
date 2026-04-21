# Distributed Neural Network Training (PyTorch + Docker)

🚀 Built a distributed neural network training system using PyTorch and Docker, achieving 93% accuracy while analyzing CPU and memory trade-offs.

---

## Overview

This project implements a distributed model-parallel neural network using PyTorch Distributed and Docker. The model is split across multiple processes (containers), where each part performs forward and backward propagation while communicating intermediate data.

---

## Key Features

- Model parallelism across multiple processes using PyTorch Distributed (Gloo backend)
- Inter-process communication using `dist.send` and `dist.recv`
- Distributed forward and backward propagation
- Training on MNIST dataset with batch size 32 and 1800 steps
- Performance benchmarking for runtime and system bottlenecks

---

## Architecture

The neural network is split into two parts:

### Part 1 – Feature Extraction (Process 0)
- Input: MNIST images (28x28)
- Layers:
  - Flatten
  - Linear (784 → 500)
  - ReLU
- Sends activations and labels to Part 2
- Receives gradients and performs backpropagation

### Part 2 – Classification Head (Process 1)
- Input: Activations from Part 1
- Layers:
  - Linear (500 → 100)
  - ReLU
  - Linear (100 → 10)
- Computes loss and gradients
- Sends gradients back to Part 1

---

## Workflow

1. Part 1 processes input data and sends activations to Part 2  
2. Part 2 computes loss and gradients  
3. Gradients are sent back to Part 1  
4. Both models update parameters synchronously  

---

## Training vs Evaluation

This project separates training and evaluation into different components:

- Training is performed across distributed processes where model parameters are updated using backpropagation  
- Evaluation is performed separately using saved model weights without updating parameters  

This separation ensures:
- Unbiased performance measurement (no learning during evaluation)  
- Reproducibility of results  
- A clean machine learning pipeline (train → save → evaluate)  

This mirrors real-world ML systems where models are trained and then evaluated independently.

---

## Results

- Achieved ~93% accuracy on MNIST  
- Achieved F1-score of ~0.91  
- Identified CPU and memory trade-offs across configurations  
- Measured runtime performance across distributed setups  

---

## System Design Insights

- Used model parallelism instead of data parallelism to split computation across processes  
- Reduced memory load per process by partitioning model layers  
- Communication overhead introduced latency via send/recv operations  
- Trade-off: improved scalability vs increased inter-process communication cost  

---

## Why This Project Matters

This project demonstrates how large-scale neural networks can be efficiently scaled across multiple compute units, which is critical for modern AI systems and distributed machine learning infrastructure.

---

## Tech Stack

- Python  
- PyTorch (Distributed)  
- Docker  
- MNIST Dataset  

---

## How to Run

```bash
# Build and start containers
docker compose up --build

# Run training
docker compose run --rm part1 python part1_container.py
docker compose run --rm part2 python part2_container.py

# Test model
docker compose run --rm part1 python test_container.py
