"""
torch_lite.data — Simple DataLoader for batching
"""
import numpy as np
from .tensor import Tensor


class DataLoader:
    """Simple DataLoader that yields (features, targets) batches as Tensors.

    Args:
        X: numpy array of shape [N, D] or [N]
        y: numpy array of shape [N, 1] or [N]
        batch_size: number of samples per batch
        shuffle: randomly shuffle each epoch
    """

    def __init__(self, X, y, batch_size=32, shuffle=True):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = len(self.X)

    def __iter__(self):
        indices = np.arange(self.n)
        if self.shuffle:
            np.random.shuffle(indices)

        for start in range(0, self.n, self.batch_size):
            batch_idx = indices[start:start + self.batch_size]
            yield Batch(
                Tensor(self.X[batch_idx].copy()),
                Tensor(self.y[batch_idx].copy())
            )

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size


class Batch:
    """A batch of (x, y) tensors"""
    __slots__ = ('x', 'y')

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Batch(x={self.x.shape}, y={self.y.shape})"


def train_test_split(X, y, test_size=0.2, random_state=42):
    """Split data into train/test sets."""
    np.random.seed(random_state)
    n = len(X)
    indices = np.random.permutation(n)
    split = int(n * (1 - test_size))
    train_idx, test_idx = indices[:split], indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
