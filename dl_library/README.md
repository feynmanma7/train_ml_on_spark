# torch_lite — A PyTorch-like Deep Learning Library with C++ Backend

A minimal deep learning framework with:
- **C++ backend** for fast tensor operations (matmul, activations, loss)
- **Python frontend** with PyTorch-like syntax and autograd
- **pybind11** bindings connecting C++ to Python

## Architecture

```
┌─────────────────────────────────────┐
│  Python API (torch_lite)           │
│  ├── tensor.py   → autograd engine │
│  ├── nn.py       → Module, Linear  │
│  ├── optim.py    → SGD optimizer   │
│  └── data.py     → DataLoader      │
├─────────────────────────────────────┤
│  pybind11 bindings (bindings.cpp)  │
├─────────────────────────────────────┤
│  C++ compute engine (tensor.cpp)   │
│  ├── matmul, element-wise ops      │
│  ├── relu, sigmoid activations     │
│  └── BCE loss                      │
└─────────────────────────────────────┘
```

## Quick Start

### Install

```bash
cd dl_library

# Install pybind11
pip install pybind11

# Build C++ extension
mkdir -p build && cd build
cmake .. && make -j4
cd ..

# Verify
python -c "import torch_lite; print(torch_lite.randn(3,3))"
```

### Train a Model

```bash
python examples/train_bank_marketing.py
```

## API (PyTorch-like)

```python
import torch_lite as tl

# Tensors
x = tl.randn(32, 10)
w = tl.randn(10, 1, requires_grad=True)

# Autograd
y = x @ w          # matmul
loss = ((y - target) ** 2).mean()
loss.backward()
print(w.grad)

# Neural Networks
model = tl.nn.Sequential(
    tl.nn.Linear(10, 64),
    tl.nn.ReLU(),
    tl.nn.Linear(64, 1),
)
criterion = tl.nn.BCELoss()
optimizer = tl.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

for epoch in range(100):
    for batch in dataloader:
        pred = model(batch.x)
        loss = criterion(pred, batch.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```
