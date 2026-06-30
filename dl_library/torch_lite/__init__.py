"""
torch_lite - A PyTorch-like deep learning library with C++ backend

Usage:
    import torch_lite as tl

    # Create tensors
    x = tl.randn(32, 10)
    w = tl.randn(10, 1)

    # Build model
    model = tl.nn.Sequential(
        tl.nn.Linear(10, 64),
        tl.nn.ReLU(),
        tl.nn.Linear(64, 1),
        tl.nn.Sigmoid(),
    )

    # Train
    opt = tl.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    for epoch in range(epochs):
        for batch in dataloader:
            pred = model(batch.x)
            loss = tl.nn.BCELoss()(pred, batch.y)
            opt.zero_grad()
            loss.backward()
            opt.step()
"""

# Attempt to import the C++ backend
try:
    from . import _torch_lite_core as _C
except ImportError:
    import os
    import sysconfig
    # Try to find the .so in the build directory
    _build_dir = os.path.join(os.path.dirname(__file__), '..', 'build')
    if os.path.exists(_build_dir):
        for root, dirs, files in os.walk(_build_dir):
            for f in files:
                if f.endswith('.so') and '_torch_lite_core' in f:
                    sys.path.insert(0, root)
                    try:
                        from . import _torch_lite_core as _C
                        break
                    except ImportError:
                        pass
    if '_C' not in dir():
        raise ImportError(
            "Cannot import _torch_lite_core C++ extension.\n"
            "Build it first:\n"
            "  cd dl_library && pip install -e .\n"
            "Or manually:\n"
            "  cd dl_library && mkdir -p build && cd build\n"
            "  cmake .. && make -j4\n"
        )

from .tensor import Tensor
from . import nn
from . import optim
from . import data

# Convenience functions (PyTorch-like)
def randn(*shape, mean=0.0, stddev=1.0, seed=None, requires_grad=False):
    """Create tensor with random normal values"""
    import time
    if seed is None:
        seed = int(time.time() * 1000) % (2**31)
    seed = seed % (2**31)
    return Tensor(_C.Tensor.randn(list(shape), float(mean), float(stddev), seed),
                  requires_grad=requires_grad)

def zeros(*shape, requires_grad=False):
    """Create tensor filled with zeros"""
    return Tensor(_C.Tensor.zeros(list(shape)), requires_grad=requires_grad)

def ones(*shape, requires_grad=False):
    """Create tensor filled with ones"""
    return Tensor(_C.Tensor.ones(list(shape)), requires_grad=requires_grad)

def tensor(data, shape=None):
    """Create tensor from list/nested-list"""
    if shape is None:
        # Infer shape from nested list structure
        def infer_shape(lst):
            if not isinstance(lst, (list, tuple)):
                return []
            if len(lst) == 0:
                return [0]
            inner = infer_shape(lst[0])
            return [len(lst)] + inner
        shape = infer_shape(data)
    return Tensor(_C.Tensor.from_list(data, list(shape)))

def from_numpy(arr):
    """Create tensor from numpy array"""
    import numpy as np
    arr = np.asarray(arr, dtype=np.float32)
    return Tensor(_C.Tensor.from_numpy(arr))

__all__ = ['Tensor', 'nn', 'optim', 'data', 'randn', 'zeros', 'ones', 'tensor', 'from_numpy']
