"""
torch_lite.tensor — Tensor with autograd support, backed by C++ engine
"""
import numpy as np
from . import _torch_lite_core as _C


class Tensor:
    """Multi-dimensional array with automatic gradient tracking."""

    def __init__(self, data, requires_grad=False):
        if isinstance(data, _C.Tensor):
            self._c = data
        elif isinstance(data, np.ndarray):
            data = np.asarray(data, dtype=np.float32)
            self._c = _C.Tensor.from_numpy(data)
        elif isinstance(data, (list, tuple)):
            self._c = _C.Tensor.from_list(data, [len(data), 1])
        elif isinstance(data, Tensor):
            # Clone
            self._c = _C.Tensor.zeros(data._c.shape)
            self._c.copy_(data._c)
            if data.grad is not None:
                self.grad = Tensor(data.grad._c)
            requires_grad = data.requires_grad
        else:
            raise TypeError(f"Cannot create Tensor from {type(data)}")

        self.requires_grad = requires_grad
        self.grad = None
        self._backward_fn = None  # callable to compute gradients for inputs
        self._inputs = ()         # tensors used as input to create this one

    # --- Properties ---
    @property
    def shape(self): return tuple(self._c.shape)

    @property
    def ndim(self): return self._c.ndim

    @property
    def numel(self): return self._c.numel

    def numpy(self):
        return np.array(self._c.numpy())

    def __repr__(self):
        g = ", grad=True" if self.requires_grad else ""
        if self.numel <= 25:
            vals = self.numpy().flatten()
            s = ", ".join(f"{v:.4f}" for v in vals)
            return f"tensor([{s}], shape={list(self.shape)}{g})"
        return f"tensor(shape={list(self.shape)}{g})"

    # --- Operations ---

    def __matmul__(self, other):
        """C = A @ B.  Backward: dA = dC @ B.T, dB = A.T @ dC"""
        c = _C.matmul(self._c, other._c)
        out = Tensor(c, requires_grad=self.requires_grad or other.requires_grad)
        out._inputs = (self, other)

        def _backward():
            if out.grad is None:
                return
            dC = out.grad._c
            if self.requires_grad:
                g = _C.matmul(dC, other._c.t())
                _accum(self, g)
            if other.requires_grad:
                g = _C.matmul(self._c.t(), dC)
                _accum(other, g)
        out._backward_fn = _backward
        return out

    def __add__(self, other):
        if isinstance(other, (int, float)):
            c = _C.add(self._c, _C.Tensor.ones(self._c.shape))
            out = Tensor(c, requires_grad=self.requires_grad)
            out._inputs = (self,)
            def _backward():
                if out.grad is None: return
                _accum(self, out.grad._c)
            out._backward_fn = _backward
            return out
        c = _C.add(self._c, other._c)
        out = Tensor(c, requires_grad=self.requires_grad or other.requires_grad)
        out._inputs = (self, other)
        def _backward():
            if out.grad is None: return
            if self.requires_grad: _accum(self, out.grad._c)
            if other.requires_grad: _accum(other, out.grad._c)
        out._backward_fn = _backward
        return out

    def __sub__(self, other):
        c = _C.sub(self._c, other._c)
        out = Tensor(c, requires_grad=self.requires_grad or other.requires_grad)
        out._inputs = (self, other)
        def _backward():
            if out.grad is None: return
            if self.requires_grad: _accum(self, out.grad._c)
            if other.requires_grad: _accum(other, _C.scale(out.grad._c, -1.0))
        out._backward_fn = _backward
        return out

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            c = _C.scale(self._c, float(other))
            out = Tensor(c, requires_grad=self.requires_grad)
            out._inputs = (self,)
            def _backward():
                if out.grad is None: return
                _accum(self, _C.scale(out.grad._c, float(other)))
            out._backward_fn = _backward
            return out
        c = _C.mul(self._c, other._c)
        out = Tensor(c, requires_grad=self.requires_grad or other.requires_grad)
        out._inputs = (self, other)
        def _backward():
            if out.grad is None: return
            if self.requires_grad: _accum(self, _C.mul(out.grad._c, other._c))
            if other.requires_grad: _accum(other, _C.mul(out.grad._c, self._c))
        out._backward_fn = _backward
        return out

    def __radd__(self, other): return self.__add__(other)
    def __rmul__(self, other): return self.__mul__(other)
    def __neg__(self):
        c = _C.scale(self._c, -1.0)
        out = Tensor(c, requires_grad=self.requires_grad)
        out._inputs = (self,)
        def _backward():
            if out.grad is None: return
            _accum(self, _C.scale(out.grad._c, -1.0))
        out._backward_fn = _backward
        return out

    def t(self):
        """Transpose: [M,N] → [N,M]"""
        c = self._c.t()
        out = Tensor(c, requires_grad=self.requires_grad)
        out._inputs = (self,)
        def _backward():
            if out.grad is None: return
            _accum(self, out.grad._c.t())
        out._backward_fn = _backward
        return out

    # --- In-place ---
    def zero_(self):
        self._c.zero_()
        return self

    def fill_(self, val):
        self._c.fill_(float(val))
        return self

    # --- Autograd ---
    def backward(self, grad=None):
        """Run reverse-mode autodiff starting from this tensor."""
        if grad is None:
            grad = Tensor(_C.Tensor.ones(self._c.shape))
        elif isinstance(grad, _C.Tensor):
            grad = Tensor(grad)
        if self.grad is None:
            self.grad = grad
        else:
            self.grad = self.grad + grad

        # Topological sort of the computation graph
        visited = set()
        topo = []

        def visit(t):
            if id(t) not in visited:
                visited.add(id(t))
                for inp in t._inputs:
                    if isinstance(inp, Tensor):
                        visit(inp)
                topo.append(t)

        visit(self)

        # Execute backward in reverse order
        for node in reversed(topo):
            if node._backward_fn is not None:
                node._backward_fn()

    def __hash__(self):
        return id(self)


def _accum(tensor, c_grad):
    """Accumulate gradient into tensor.grad"""
    if tensor.grad is None:
        tensor.grad = Tensor(c_grad, requires_grad=False)
    else:
        tensor.grad = Tensor(_C.add(tensor.grad._c, c_grad), requires_grad=False)


# --- High-level ops registered on Tensor ---
def _bias_add(tensor_input, bias):
    """Broadcast add: [M,N] + [1,N] → [M,N]"""
    c = _C.broadcast_add(tensor_input._c, bias._c)
    out = Tensor(c, requires_grad=tensor_input.requires_grad or bias.requires_grad)
    out._inputs = (tensor_input, bias)

    def _backward():
        if out.grad is None: return
        if tensor_input.requires_grad:
            _accum(tensor_input, out.grad._c)
        if bias.requires_grad:
            g = _C.sum_dim0(out.grad._c)
            _accum(bias, g)
    out._backward_fn = _backward
    return out


def _relu(tensor_input):
    """ReLU activation: max(0, x)"""
    c = _C.relu(tensor_input._c)
    out = Tensor(c, requires_grad=tensor_input.requires_grad)
    out._inputs = (tensor_input,)

    def _backward():
        if out.grad is None: return
        g = _C.relu_backward(out.grad._c, tensor_input._c)
        _accum(tensor_input, g)
    out._backward_fn = _backward
    return out


def _sigmoid(tensor_input):
    """Sigmoid activation: 1/(1+exp(-x))"""
    c = _C.sigmoid(tensor_input._c)
    out = Tensor(c, requires_grad=tensor_input.requires_grad)
    out._inputs = (tensor_input,)

    def _backward():
        if out.grad is None: return
        s = _C.sigmoid(tensor_input._c)
        g = _C.mul(out.grad._c, _C.mul(s, _C.sub(_C.Tensor.ones(s.shape), s)))
        _accum(tensor_input, g)
    out._backward_fn = _backward
    return out


def _bce_loss(logits, target):
    """Binary Cross Entropy with logits input. Returns loss scalar tensor."""
    loss_val = _C.binary_cross_entropy(_C.sigmoid(logits._c), target._c)
    loss = Tensor(_C.Tensor.from_list([loss_val], [1, 1]),
                  requires_grad=logits.requires_grad)
    loss._inputs = (logits,)

    def _backward():
        if loss.grad is None: return
        # dBCE/dlogits = (sigmoid(logits) - target) / N, then scale by loss.grad
        d = _C.bce_sigmoid_backward(logits._c, target._c)
        if abs(float(loss.grad.numpy().flatten()[0]) - 1.0) > 1e-6:
            d = _C.scale(d, float(loss.grad.numpy().flatten()[0]))
        _accum(logits, d)
    loss._backward_fn = _backward
    return loss
