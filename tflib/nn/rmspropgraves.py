from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import tensorflow as tf


class RmsPropGraves(optimizer.Optimizer):

    def __init__(self, learning_rate=0.001, decay=0.9, momentum=0.0, epsilon=1e-10, use_locking=False, name="RmsPropGraves"):
        super(RmsPropGraves, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._decay = decay
        self._momentum = momentum
        self._epsilon = epsilon

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._decay_t = None
        self._momentum_t = None
        self._epsilon_t = None
        self._delta_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._decay_t = ops.convert_to_tensor(self._decay, name="decay_t")
        self._momentum_t = ops.convert_to_tensor(
            self._momentum, name="momentum_t")
        self._epsilon_t = ops.convert_to_tensor(
            self._epsilon, name="epsilon_t")

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
        for v in var_list:
            self._zeros_slot(v, "v", self._name)
        for v in var_list:
            self._zeros_slot(v, "delta", self._name)

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        decay_t = math_ops.cast(self._decay_t, var.dtype.base_dtype)
        momentum_t = math_ops.cast(self._momentum_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

        m = self.get_slot(var, "m")
        m_t = m.assign(decay_t * m + (1 - decay_t)*grad)
        v = self.get_slot(var, "v")
        v_t = v.assign(decay_t * v + (1 - decay_t)*grad**2)
        delta = self.get_slot(var, "delta")

        # Update 'ref' by subtracting 'value
        var_update = state_ops.assign_sub(
            var, lr_t * (1.0 / tf.sqrt(v_t - m_t**2 + epsilon_t)) * grad - momentum_t*delta)
        delta_t = delta.assign(momentum_t*delta - lr_t *
                               (1.0 / tf.sqrt(v_t - m_t**2 + epsilon_t)) * grad)
        # Create an op that groups multiple operations.
        # When this op finishes, all ops in input have finished
        return control_flow_ops.group(*[var_update, m_t, v_t, delta_t])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")
