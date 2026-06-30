"""
torch_lite.nn — Neural network building blocks (PyTorch-like API)
"""
import math
from . import _torch_lite_core as _C
from .tensor import Tensor, _bias_add, _relu, _sigmoid, _bce_loss


class Module:
    """Base class for all neural network modules."""

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def parameters(self):
        """Return list of all trainable parameters."""
        params = []
        for attr in dir(self):
            val = getattr(self, attr)
            if isinstance(val, Tensor) and val.requires_grad:
                params.append(val)
            elif isinstance(val, Module):
                params.extend(val.parameters())
        return params

    def zero_grad(self):
        for p in self.parameters():
            if p.grad is not None:
                p.grad.zero_()

    def train(self):
        self._training = True
        return self

    def eval(self):
        self._training = False
        return self

    def _reset_parameters(self):
        pass


class Linear(Module):
    """Fully-connected layer: y = x @ W.T + b

    Args:
        in_features: size of input features
        out_features: size of output features
        bias: if True, adds a learnable bias
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Kaiming uniform initialization
        bound = math.sqrt(6.0 / in_features)
        w_data = _C.Tensor.zeros([out_features, in_features])
        import numpy as np
        w_arr = np.random.uniform(-bound, bound, (out_features, in_features)).astype(np.float32)
        w_data = _C.Tensor.from_numpy(w_arr)
        self.weight = Tensor(w_data, requires_grad=True)

        if bias:
            self.bias = Tensor(_C.Tensor.zeros([1, out_features]), requires_grad=True)
        else:
            self.bias = None

    def forward(self, x):
        # x: [batch, in_features], weight: [out_features, in_features]
        # out: x @ W.T = [batch, out_features]
        out = x @ self.weight.t()
        if self.bias is not None:
            out = _bias_add(out, self.bias)
        return out

    def __repr__(self):
        return f"Linear(in={self.in_features}, out={self.out_features}, bias={self.bias is not None})"


class ReLU(Module):
    """ReLU activation: max(0, x)"""

    def forward(self, x):
        return _relu(x)

    def __repr__(self):
        return "ReLU()"


class Sigmoid(Module):
    """Sigmoid activation: 1/(1+exp(-x))"""

    def forward(self, x):
        return _sigmoid(x)

    def __repr__(self):
        return "Sigmoid()"


class Sequential(Module):
    """Sequential container: modules applied in order."""

    def __init__(self, *modules):
        super().__init__()
        self._modules = list(modules)
        for i, m in enumerate(self._modules):
            setattr(self, f"_m{i}", m)

    def forward(self, x):
        for m in self._modules:
            x = m(x)
        return x

    def parameters(self):
        params = []
        for m in self._modules:
            params.extend(m.parameters())
        return params

    def __repr__(self):
        lines = ["Sequential("]
        for m in self._modules:
            lines.append(f"  {m},")
        lines.append(")")
        return "\n".join(lines)

    def __getitem__(self, idx):
        return self._modules[idx]

    def __len__(self):
        return len(self._modules)


class BCELoss(Module):
    """Binary Cross Entropy Loss with logits.

    Computes: -mean(y*log(sigmoid(x)) + (1-y)*log(1-sigmoid(x)))
    where x is the logits (raw model output).

    Uses numerically stable C++ implementation.
    """

    def __init__(self):
        super().__init__()

    def forward(self, logits, target):
        return _bce_loss(logits, target)

    def __repr__(self):
        return "BCELoss()"


class Dropout(Module):
    """Dropout layer (only active in training mode)."""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self._training = True

    def forward(self, x):
        if not self._training or self.p == 0:
            return x
        import numpy as np
        mask = (np.random.rand(*x.shape) > self.p).astype(np.float32)
        scale = 1.0 / (1.0 - self.p)
        mask_t = Tensor(_C.Tensor.from_numpy(mask))
        scaled = x * mask_t
        return Tensor(_C.scale(scaled._c, scale))

    def __repr__(self):
        return f"Dropout(p={self.p})"
