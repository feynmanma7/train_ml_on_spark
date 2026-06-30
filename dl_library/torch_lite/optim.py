"""
torch_lite.optim — Optimization algorithms (SGD)
"""
from . import _torch_lite_core as _C
from .tensor import Tensor


class SGD:
    """Stochastic Gradient Descent with optional momentum.

    Args:
        params: list of Tensors with requires_grad=True
        lr: learning rate
        momentum: momentum factor (0 = no momentum)
        weight_decay: L2 regularization coefficient
    """

    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._velocities = None

        if momentum > 0:
            self._velocities = [
                Tensor(_C.Tensor.zeros(p._c.shape), requires_grad=False)
                for p in self.params
            ]

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        """Update parameters using gradients."""
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            grad = p.grad._c

            # Weight decay: grad += wd * param
            if self.weight_decay > 0:
                grad = _C.add(grad, _C.scale(p._c, self.weight_decay))

            if self.momentum > 0:
                # v = momentum * v + grad
                v = self._velocities[i]
                v_new = _C.add(
                    _C.scale(v._c, self.momentum),
                    grad
                )
                v._c.copy_(v_new)
                update = _C.scale(v._c, -self.lr)
            else:
                update = _C.scale(grad, -self.lr)

            # param += update
            p._c.copy_(_C.add(p._c, update))

    def __repr__(self):
        return f"SGD(lr={self.lr}, momentum={self.momentum}, wd={self.weight_decay})"
