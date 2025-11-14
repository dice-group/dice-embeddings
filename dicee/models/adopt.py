"""ADOPT Optimizer Implementation.

This module implements the ADOPT (Adaptive Optimization with Precise Tracking) algorithm,
an advanced optimization method for training neural networks.

ADOPT Overview:
--------------
ADOPT is an adaptive learning rate optimization algorithm that combines the benefits of
momentum-based methods with per-parameter learning rate adaptation. Unlike Adam, which
applies momentum to raw gradients, ADOPT normalizes gradients first and then applies
momentum, leading to more stable training dynamics.

Key Features:
- Gradient normalization before momentum application
- Adaptive per-parameter learning rates
- Optional gradient clipping that grows with training steps
- Support for decoupled weight decay (AdamW-style)
- Multiple execution modes: single-tensor, multi-tensor (foreach), and fused (planned)

Algorithm Comparison:
--------------------
Adam:    m = β₁*m + (1-β₁)*g,  θ = θ - α*m/√v
ADOPT:   m = β₁*m + (1-β₁)*g/√v,  θ = θ - α*m

The key difference is that ADOPT normalizes gradients before momentum, which provides
better stability and can lead to improved convergence.

Classes:
--------
- ADOPT: Main optimizer class (extends torch.optim.Optimizer)

Functions:
----------
- adopt: Functional API for ADOPT algorithm computation
- _single_tensor_adopt: Single-tensor implementation (TorchScript compatible)
- _multi_tensor_adopt: Multi-tensor implementation using foreach operations

Performance:
-----------
- Single-tensor: Default, compatible with torch.jit.script
- Multi-tensor (foreach): 2-3x faster on GPU through vectorization
- Fused (planned): Would provide maximum performance via specialized kernels

Example:
--------
>>> import torch
>>> from dicee.models.adopt import ADOPT
>>>
>>> model = torch.nn.Linear(10, 1)
>>> optimizer = ADOPT(model.parameters(), lr=0.001, weight_decay=0.01, decouple=True)
>>>
>>> # Training loop
>>> for epoch in range(num_epochs):
...     optimizer.zero_grad()
...     output = model(input)
...     loss = criterion(output, target)
...     loss.backward()
...     optimizer.step()

References:
----------
Original implementation: https://github.com/iShohei220/adopt

Notes:
-----
This implementation is based on the original ADOPT implementation and adapted to work
with the PyTorch optimizer interface and the dice-embeddings framework.
"""
# CD: copy pasted from https://raw.githubusercontent.com/iShohei220/adopt/refs/heads/main/adopt.py
# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
from typing import cast, Callable, List, Optional, Tuple, Union

import torch
from torch import Tensor

from torch.optim.optimizer import (
    _capturable_doc, # noqa: F401
    _default_to_fused_or_foreach,
    _device_dtype_check_for_fused,
    _differentiable_doc, # noqa: F401
    _disable_dynamo_if_unsupported,
    _foreach_doc, # noqa: F401
    _fused_doc, # noqa: F401
    _get_capturable_supported_devices,
    _get_scalar_dtype,
    _get_value, # noqa: F401
    _maximize_doc, # noqa: F401
    _stack_if_compiling, # noqa: F401
    _use_grad_for_differentiable,
    _view_as_real,
    DeviceDict, # noqa: F401
    Optimizer,
    ParamsT,
)


__all__ = ["ADOPT", "adopt"]


class ADOPT(Optimizer):
    """ADOPT  Optimizer.
    
    ADOPT is an adaptive learning rate optimization algorithm that combines momentum-based
    updates with adaptive per-parameter learning rates. It uses exponential moving averages
    of gradients and squared gradients, with gradient clipping for stability.
    
    The algorithm performs the following key operations:
    1. Normalizes gradients by the square root of the second moment estimate
    2. Applies optional gradient clipping based on the training step
    3. Updates parameters using momentum-smoothed normalized gradients
    4. Supports decoupled weight decay (AdamW-style) or L2 regularization
    
    Mathematical formulation:
        m_t = β₁ * m_{t-1} + (1 - β₁) * clip(g_t / √(v_t))
        v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
        θ_t = θ_{t-1} - α * m_t
    
    where:
        - θ_t: parameter at step t
        - g_t: gradient at step t
        - m_t: first moment estimate (momentum)
        - v_t: second moment estimate (variance)
        - α: learning rate
        - β₁, β₂: exponential decay rates
        - clip(): optional gradient clipping function
    
    Reference:
        Original implementation: https://github.com/iShohei220/adopt
    
    Args:
        params (ParamsT): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float or Tensor, optional): Learning rate. Can be a float or 1-element Tensor. 
            Default: 1e-3
        betas (Tuple[float, float], optional): Coefficients (β₁, β₂) for computing running 
            averages of gradient and its square. β₁ controls momentum, β₂ controls variance.
            Default: (0.9, 0.9999)
        eps (float, optional): Term added to denominator to improve numerical stability.
            Default: 1e-6
        clip_lambda (Callable[[int], float], optional): Function that takes the step number 
            and returns the gradient clipping threshold. Common choices:
            - lambda step: step**0.25 (default, gradually increases clipping threshold)
            - lambda step: 1.0 (constant clipping)
            - None (no clipping)
            Default: lambda step: step**0.25
        weight_decay (float, optional): Weight decay coefficient (L2 penalty).
            Default: 0.0
        decouple (bool, optional): If True, uses decoupled weight decay (AdamW-style),
            applying weight decay directly to parameters. If False, adds weight decay
            to gradients (L2 regularization). Default: False
        foreach (bool, optional): If True, uses the faster foreach implementation for
            multi-tensor operations. Default: None (auto-select)
        maximize (bool, optional): If True, maximizes parameters instead of minimizing.
            Useful for reinforcement learning. Default: False
        capturable (bool, optional): If True, the optimizer is safe to capture in a 
            CUDA graph. Requires learning rate as Tensor. Default: False
        differentiable (bool, optional): If True, the optimization step can be 
            differentiated. Useful for meta-learning. Default: False
        fused (bool, optional): If True, uses fused kernel implementation (currently 
            not supported). Default: None
    
    Raises:
        ValueError: If learning rate, epsilon, betas, or weight_decay are invalid.
        RuntimeError: If fused is enabled (not currently supported).
        RuntimeError: If lr is a Tensor with foreach=True and capturable=False.
    
    Example:
        >>> # Basic usage
        >>> optimizer = ADOPT(model.parameters(), lr=0.001)
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()
        
        >>> # With decoupled weight decay
        >>> optimizer = ADOPT(model.parameters(), lr=0.001, weight_decay=0.01, decouple=True)
        
        >>> # Custom gradient clipping
        >>> optimizer = ADOPT(model.parameters(), clip_lambda=lambda step: max(1.0, step**0.5))
    
    Note:
        - For most use cases, the default hyperparameters work well
        - Consider using decouple=True for better generalization (similar to AdamW)
        - The clip_lambda function helps stabilize training in early steps
    """
    
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.9999),
        eps: float = 1e-6,
        clip_lambda: Optional[Callable[[int], float]] = lambda step: step**0.25,
        weight_decay: float = 0.0,
        decouple: bool = False,
        *,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
    ):
        if isinstance(lr, Tensor):
            if foreach and not capturable:
                raise ValueError(
                    "lr as a Tensor is not supported for capturable=False and foreach=True"
                )
            if lr.numel() != 1:
                raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        self.clip_lambda = clip_lambda

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            decouple=decouple,
            maximize=maximize,
            foreach=foreach,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
        )
        super().__init__(params, defaults)

        if fused:
            # TODO: support fused
            raise RuntimeError("`fused` is not currently supported")

            if differentiable:
                raise RuntimeError("`fused` does not support `differentiable`")
            self._step_supports_amp_scaling = True
            # TODO(crcrpar): [low prec params & their higher prec copy]
            # Support AMP with FP16/BF16 model params which would need
            # higher prec copy of params to do update math in higher prec to
            # alleviate the loss of information.
            if foreach:
                raise RuntimeError("`fused` and `foreach` cannot be `True` together.")

    def __setstate__(self, state):
        """Restore optimizer state from a checkpoint.
        
        This method handles backward compatibility when loading optimizer state from
        older versions. It ensures all required fields are present with default values
        and properly converts step counters to tensors if needed.
        
        Key responsibilities:
        1. Set default values for newly added hyperparameters
        2. Convert old-style scalar step counters to tensor format
        3. Place step tensors on appropriate devices based on capturable/fused modes
        
        Args:
            state (dict): Optimizer state dictionary (typically from torch.load()).
        
        Note:
            - This enables loading checkpoints saved with older ADOPT versions
            - Step counters are converted to appropriate device/dtype for compatibility
            - Capturable and fused modes require step tensors on parameter devices
        """
        super().__setstate__(state)
        
        # Set defaults for parameters that may not exist in older checkpoints
        for group in self.param_groups:
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("capturable", False)
            group.setdefault("differentiable", False)
            fused = group.setdefault("fused", None)
            
            # Convert old scalar step counters to tensor format
            for p in group["params"]:
                p_state = self.state.get(p, [])
                if len(p_state) != 0 and not torch.is_tensor(p_state["step"]):
                    step_val = float(p_state["step"])
                    # Place step on parameter device if capturable/fused, else CPU
                    p_state["step"] = (
                        torch.tensor(
                            step_val,
                            dtype=_get_scalar_dtype(is_fused=fused),
                            device=p.device,
                        )
                        if group["capturable"] or group["fused"]
                        else torch.tensor(step_val, dtype=_get_scalar_dtype())
                    )

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
    ):
        """Initialize optimizer state for a parameter group.
        
        This method performs lazy state initialization for parameters that have gradients.
        It sets up the exponential moving averages and step counters needed for the ADOPT
        algorithm. State is only initialized when a parameter receives its first gradient.
        
        The method handles several important tasks:
        1. Identifies parameters with gradients (active parameters)
        2. Validates that gradients are dense (sparse gradients not supported)
        3. Initializes first moment (exp_avg) and second moment (exp_avg_sq) estimates
        4. Sets up step counters with appropriate device placement for performance
        5. Detects complex-valued parameters for special handling
        
        State initialization strategy:
        - exp_avg (m_t): Initialized to zeros, tracks momentum of normalized gradients
        - exp_avg_sq (v_t): Initialized to zeros, tracks variance of raw gradients
        - step: Initialized to 0, placed on CPU for non-capturable/non-fused mode
          to reduce CUDA kernel launch overhead
        
        Args:
            group (dict): Parameter group containing optimization settings and parameters.
            params_with_grad (List[Tensor]): Output list to collect parameters with gradients.
            grads (List[Tensor]): Output list to collect gradient tensors.
            exp_avgs (List[Tensor]): Output list to collect first moment estimates.
            exp_avg_sqs (List[Tensor]): Output list to collect second moment estimates.
            state_steps (List[Tensor]): Output list to collect step counters.
        
        Returns:
            bool: True if any complex parameters are present, False otherwise.
        
        Raises:
            RuntimeError: If sparse gradients are encountered (not supported).
            RuntimeError: If foreach is True with Tensor lr and capturable is False.
            RuntimeError: If step requires_grad in differentiable mode.
        
        Note:
            - Step counters are deliberately placed on CPU when both capturable and
              fused are False to avoid expensive CUDA kernel launches
            - Complex parameters are internally represented as real tensors with an
              extra dimension, requiring special view handling
        """
        has_complex = False
        for p in group["params"]:
            if p.grad is not None:
                has_complex |= torch.is_complex(p)
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError(
                        "ADOPT does not support sparse gradients"
                    )
                grads.append(p.grad)

                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    if group["fused"]:
                        _device_dtype_check_for_fused(p)
                    # note(crcrpar): [special device hosting for step]
                    # Deliberately host `step` on CPU if both capturable and fused are off.
                    # This is because kernel launches are costly on CUDA and XLA.
                    state["step"] = (
                        torch.zeros(
                            (),
                            dtype=_get_scalar_dtype(is_fused=group["fused"]),
                            device=p.device,
                        )
                        if group["capturable"] or group["fused"]
                        else torch.tensor(0.0, dtype=_get_scalar_dtype())
                    )
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])

                if group["differentiable"] and state["step"].requires_grad:
                    raise RuntimeError(
                        "`requires_grad` is not supported for `step` in differentiable mode"
                    )

                # Foreach without capturable does not support a tensor lr
                if (
                    group["foreach"]
                    and torch.is_tensor(group["lr"])
                    and not group["capturable"]
                ):
                    raise RuntimeError(
                        "lr as a Tensor is not supported for capturable=False and foreach=True"
                    )

                state_steps.append(state["step"])
        return has_complex

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step.
        
        This method executes one iteration of the ADOPT optimization algorithm across
        all parameter groups. It orchestrates the following workflow:
        
        1. Optionally evaluates a closure to recompute the loss (useful for algorithms
           like LBFGS or when loss needs multiple evaluations)
        2. For each parameter group:
           - Collects parameters with gradients and their associated state
           - Extracts hyperparameters (betas, learning rate, etc.)
           - Calls the functional adopt() API to perform the actual update
        3. Returns the loss value if a closure was provided
        
        The functional API (adopt()) handles three execution modes:
        - Single-tensor: Updates one parameter at a time (default, JIT-compatible)
        - Multi-tensor (foreach): Batches operations for better performance
        - Fused: Uses fused CUDA kernels (not yet implemented)
        
        Gradient scaling support:
        This method is compatible with automatic mixed precision (AMP) training.
        It can access grad_scale and found_inf attributes for gradient unscaling
        and inf/nan detection when used with GradScaler.
        
        Args:
            closure (Callable, optional): A callable that reevaluates the model and 
                returns the loss. The closure should:
                - Enable gradients (torch.enable_grad())
                - Compute forward pass
                - Compute loss
                - Compute backward pass
                - Return the loss value
                Example: lambda: (loss := model(x), loss.backward(), loss)[-1]
                Default: None
        
        Returns:
            Optional[Tensor]: The loss value returned by the closure, or None if no 
                closure was provided.
        
        Example:
            >>> # Standard usage
            >>> loss = criterion(model(input), target)
            >>> loss.backward()
            >>> optimizer.step()
            
            >>> # With closure (e.g., for line search)
            >>> def closure():
            ...     optimizer.zero_grad()
            ...     output = model(input)
            ...     loss = criterion(output, target)
            ...     loss.backward()
            ...     return loss
            >>> loss = optimizer.step(closure)
        
        Note:
            - Call zero_grad() before computing gradients for the next step
            - CUDA graph capture is checked for safety when capturable=True
            - The method is thread-safe for different parameter groups
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad: List[Tensor] = []
            grads: List[Tensor] = []
            exp_avgs: List[Tensor] = []
            exp_avg_sqs: List[Tensor] = []
            state_steps: List[Tensor] = []
            beta1, beta2 = group["betas"]

            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
            )

            adopt(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                has_complex=has_complex,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                clip_lambda=self.clip_lambda,
                weight_decay=group["weight_decay"],
                decouple=group["decouple"],
                eps=group["eps"],
                maximize=group["maximize"],
                foreach=group["foreach"],
                capturable=group["capturable"],
                differentiable=group["differentiable"],
                fused=group["fused"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
            )

        return loss


def _single_tensor_adopt(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    has_complex: bool,
    beta1: float,
    beta2: float,
    lr: Union[float, Tensor],
    clip_lambda: Optional[Callable[[int], float]],
    weight_decay: float,
    decouple: bool,
    eps: float,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
):
    """Single-tensor implementation of ADOPT optimization algorithm.
    
    This function updates parameters one at a time using the ADOPT algorithm. It's the
    default implementation used when foreach=False, and is compatible with TorchScript
    compilation (torch.jit.script).
    
    Algorithm steps for each parameter:
    1. Apply weight decay (L2 regularization) to gradient if not decoupled
    2. Handle complex parameters by viewing them as real tensors
    3. On first step (step=0): Initialize second moment estimate with g²
    4. On subsequent steps:
       a. Apply decoupled weight decay directly to parameters if enabled
       b. Normalize gradient: normed_grad = g / √(v + ε)
       c. Apply gradient clipping: normed_grad = clip(normed_grad, threshold)
       d. Update first moment (momentum): m = β₁*m + (1-β₁)*normed_grad
       e. Update parameters: θ = θ - lr*m
       f. Update second moment: v = β₂*v + (1-β₂)*g²
    5. Increment step counter
    
    Key differences from Adam:
    - Normalizes gradients BEFORE applying momentum (not after)
    - Uses gradient clipping on normalized gradients for stability
    - First moment tracks normalized gradients, not raw gradients
    
    Args:
        params (List[Tensor]): List of parameters to update.
        grads (List[Tensor]): List of gradients corresponding to parameters.
        exp_avgs (List[Tensor]): List of first moment estimates (momentum).
        exp_avg_sqs (List[Tensor]): List of second moment estimates (variance).
        state_steps (List[Tensor]): List of step counters for each parameter.
        grad_scale (Optional[Tensor]): Gradient scaler for AMP (not used in this impl).
        found_inf (Optional[Tensor]): Infinity detection flag for AMP (not used).
        has_complex (bool): Whether any parameters are complex-valued.
        beta1 (float): Exponential decay rate for first moment (momentum).
        beta2 (float): Exponential decay rate for second moment (variance).
        lr (Union[float, Tensor]): Learning rate.
        clip_lambda (Optional[Callable[[int], float]]): Function mapping step to clip threshold.
        weight_decay (float): Weight decay coefficient (L2 penalty).
        decouple (bool): If True, use decoupled weight decay (AdamW-style).
        eps (float): Small constant for numerical stability.
        maximize (bool): If True, maximize parameters instead of minimize.
        capturable (bool): If True, safe for CUDA graph capture.
        differentiable (bool): If True, optimization step is differentiable.
    
    Note:
        - This implementation is slower than multi-tensor but more flexible
        - Compatible with torch.jit.script for potential speedups
        - First step only updates second moment, not parameters (warmup)
        - Complex parameters are internally handled as real tensors
    """
    assert grad_scale is None and found_inf is None

    if torch.jit.is_scripting():
        # this assert is due to JIT being dumb and not realizing that the ops below
        # have overloads to handle both float and Tensor lrs, so we just assert it's
        # a float since most people using JIT are using floats
        assert isinstance(lr, float)

    for i, param in enumerate(params):
        # Negate gradients if maximizing (e.g., for RL policy gradients)
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        # Verify device compatibility for CUDA graph capture
        # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
        if not torch.compiler.is_compiling() and capturable:
            capturable_supported_devices = _get_capturable_supported_devices()
            assert (
                param.device.type == step_t.device.type
                and param.device.type in capturable_supported_devices
            ), f"If capturable=True, params and state_steps must be on supported devices: {capturable_supported_devices}."

        # Extract step value: keep as tensor for capturable/differentiable, else convert to scalar
        step = step_t if capturable or differentiable else _get_value(step_t)

        # L2 regularization: add weight decay to gradient (traditional approach)
        if weight_decay != 0 and not decouple:
            grad = grad.add(param, alpha=weight_decay)

        # Handle complex parameters by viewing as real tensors with extra dimension
        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            if exp_avg is not None:
                exp_avg = torch.view_as_real(exp_avg)
            if exp_avg_sq is not None:
                exp_avg_sq = torch.view_as_real(exp_avg_sq)
            param = torch.view_as_real(param)

        # First step: Only initialize variance estimate, don't update parameters
        # This warmup step helps stabilize early training
        if step == 0:
            exp_avg_sq.addcmul_(grad, grad.conj())  # v_0 = g_0²
            step_t += 1
            continue

        # Decoupled weight decay (AdamW-style): Apply directly to parameters
        # θ = θ - lr * wd * θ
        if weight_decay != 0 and decouple:
            param.add_(param, alpha=-lr*weight_decay)

        # Normalize gradient by its historical standard deviation
        # denom = √(v + ε) to avoid division by zero
        denom = torch.clamp(exp_avg_sq.sqrt(), eps)
        normed_grad = grad.div(denom)
        
        # Apply gradient clipping for training stability
        # Threshold increases with training steps (default: step^0.25)
        if clip_lambda is not None:
            clip = clip_lambda(step)
            normed_grad.clamp_(-clip, clip)

        # Update first moment with exponential moving average of normalized gradients
        # m_t = β₁ * m_{t-1} + (1 - β₁) * normed_grad
        # Note: lerp(target, weight) = self + weight * (target - self)
        exp_avg.lerp_(normed_grad, 1 - beta1)

        # Update parameters using momentum term
        # θ_t = θ_{t-1} - α * m_t
        param.add_(exp_avg, alpha=-lr)
        
        # Update second moment with exponential moving average of squared gradients
        # v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

        # Increment step counter for next iteration
        step_t += 1


def _multi_tensor_adopt(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    has_complex: bool,
    beta1: float,
    beta2: float,
    lr: Union[float, Tensor],
    clip_lambda: Optional[Callable[[int], float]],
    weight_decay: float,
    decouple: bool,
    eps: float,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
):
    """Multi-tensor (foreach) implementation of ADOPT optimization algorithm.
    
    This function implements the ADOPT algorithm using PyTorch's foreach operations,
    which batch multiple tensor operations together for better performance. This is
    typically 2-3x faster than the single-tensor implementation on GPU.
    
    Performance optimizations:
    1. Groups tensors by device and dtype to minimize kernel launches
    2. Uses torch._foreach_* operations for vectorized updates
    3. Fuses elementwise operations when possible
    4. Handles CPU step counters efficiently to avoid repeated CPU-GPU transfers
    
    Algorithm (same as single-tensor but batched):
    For each device/dtype group:
    1. Handle complex parameters by viewing as real
    2. Optionally negate gradients for maximization
    3. Add L2 weight decay to gradients if not decoupled
    4. On first step: Initialize v with g² for all tensors
    5. On subsequent steps:
       a. Apply decoupled weight decay: θ = θ - lr*wd*θ
       b. Normalize gradients: normed_grad = g / max(√v, ε)
       c. Apply clipping: clip(normed_grad, threshold)
       d. Update momentum: m = β₁*m + (1-β₁)*normed_grad
       e. Update parameters: θ = θ - lr*m
       f. Update variance: v = β₂*v + (1-β₂)*g²
    6. Increment all step counters
    
    Device grouping strategy:
    Parameters are automatically grouped by (device, dtype) to ensure:
    - Operations stay on the same device (no cross-device transfers)
    - Consistent numeric precision within each group
    - Minimal kernel launch overhead
    
    Args:
        params (List[Tensor]): List of parameters to update.
        grads (List[Tensor]): List of gradients corresponding to parameters.
        exp_avgs (List[Tensor]): List of first moment estimates (momentum).
        exp_avg_sqs (List[Tensor]): List of second moment estimates (variance).
        state_steps (List[Tensor]): List of step counters for each parameter.
        grad_scale (Optional[Tensor]): Gradient scaler for AMP (not used in this impl).
        found_inf (Optional[Tensor]): Infinity detection flag for AMP (not used).
        has_complex (bool): Whether any parameters are complex-valued.
        beta1 (float): Exponential decay rate for first moment (momentum).
        beta2 (float): Exponential decay rate for second moment (variance).
        lr (Union[float, Tensor]): Learning rate.
        clip_lambda (Optional[Callable[[int], float]]): Function mapping step to clip threshold.
        weight_decay (float): Weight decay coefficient (L2 penalty).
        decouple (bool): If True, use decoupled weight decay (AdamW-style).
        eps (float): Small constant for numerical stability.
        maximize (bool): If True, maximize parameters instead of minimize.
        capturable (bool): If True, safe for CUDA graph capture.
        differentiable (bool): If True, optimization step is differentiable.
    
    Raises:
        RuntimeError: If lr is a Tensor with capturable=False.
        AssertionError: If differentiable=True (foreach ops don't support autograd).
    
    Note:
        - Significantly faster than single-tensor on GPU (2-3x speedup typical)
        - Not compatible with torch.jit.script
        - Automatically falls back to slower path for CPU step counters
        - Complex parameters require special view handling
    """
    if len(params) == 0:
        return

    if isinstance(lr, Tensor) and not capturable:
        raise RuntimeError(
            "lr as a Tensor is not supported for capturable=False and foreach=True"
        )

    # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
    if not torch.compiler.is_compiling() and capturable:
        capturable_supported_devices = _get_capturable_supported_devices(
            supports_xla=False
        )
        assert all(
            p.device.type == step.device.type
            and p.device.type in capturable_supported_devices
            for p, step in zip(params, state_steps)
        ), f"If capturable=True, params and state_steps must be on supported devices: {capturable_supported_devices}."

    assert grad_scale is None and found_inf is None

    assert not differentiable, "_foreach ops don't support autograd"

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, exp_avgs, exp_avg_sqs, state_steps]  # type: ignore[list-item]
    )
    for (
        device_params_,
        device_grads_,
        device_exp_avgs_,
        device_exp_avg_sqs_,
        device_state_steps_,
    ), _ in grouped_tensors.values():
        device_params = cast(List[Tensor], device_params_)
        device_grads = cast(List[Tensor], device_grads_)
        device_exp_avgs = cast(List[Tensor], device_exp_avgs_)
        device_exp_avg_sqs = cast(List[Tensor], device_exp_avg_sqs_)
        device_state_steps = cast(List[Tensor], device_state_steps_)

        # Handle complex parameters
        if has_complex:
            _view_as_real(
                device_params, device_grads, device_exp_avgs, device_exp_avg_sqs
            )

        if maximize:
            device_grads = torch._foreach_neg(device_grads)  # type: ignore[assignment]

        if weight_decay != 0 and not decouple:
            # Re-use the intermediate memory (device_grads) already allocated for maximize
            if maximize:
                torch._foreach_add_(device_grads, device_params, alpha=weight_decay)
            else:
                device_grads = torch._foreach_add(  # type: ignore[assignment]
                    device_grads, device_params, alpha=weight_decay
                )

        if device_state_steps[0] == 0:
            torch._foreach_addcmul_(device_exp_avg_sqs, device_grads, device_grads)

            # Update steps
            # If steps are on CPU, foreach will fall back to the slow path, which is a for-loop calling t.add(1) over
            # and over. 1 will then be wrapped into a Tensor over and over again, which is slower than if we just
            # wrapped it once now. The alpha is required to assure we go to the right overload.
            if not torch.compiler.is_compiling() and device_state_steps[0].is_cpu:
                torch._foreach_add_(
                    device_state_steps, torch.tensor(1.0, device="cpu"), alpha=1.0
                )
            else:
                torch._foreach_add_(device_state_steps, 1)

            continue

        if weight_decay != 0 and decouple:
            torch._foreach_add_(device_params, device_params, alpha=-lr*weight_decay)

        exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)
        torch._foreach_maximum_(exp_avg_sq_sqrt, eps)

        normed_grad = torch._foreach_div(device_grads, exp_avg_sq_sqrt)
        if clip_lambda is not None:
            clip = clip_lambda(device_state_steps[0])
            torch._foreach_maximum_(normed_grad, -clip)
            torch._foreach_minimum_(normed_grad, clip)

        torch._foreach_lerp_(device_exp_avgs, normed_grad, 1 - beta1)

        torch._foreach_add_(device_params, device_exp_avgs, alpha=-lr)
        torch._foreach_mul_(device_exp_avg_sqs, beta2)
        torch._foreach_addcmul_(
            device_exp_avg_sqs, device_grads, device_grads, value=1 - beta2
        )

        # Update steps
        # If steps are on CPU, foreach will fall back to the slow path, which is a for-loop calling t.add(1) over
        # and over. 1 will then be wrapped into a Tensor over and over again, which is slower than if we just
        # wrapped it once now. The alpha is required to assure we go to the right overload.
        if not torch.compiler.is_compiling() and device_state_steps[0].is_cpu:
            torch._foreach_add_(
                device_state_steps, torch.tensor(1.0, device="cpu"), alpha=1.0
            )
        else:
            torch._foreach_add_(device_state_steps, 1)


@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_adopt)
def adopt(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    foreach: Optional[bool] = None,
    capturable: bool = False,
    differentiable: bool = False,
    fused: Optional[bool] = None,
    grad_scale: Optional[Tensor] = None,
    found_inf: Optional[Tensor] = None,
    has_complex: bool = False,
    *,
    beta1: float,
    beta2: float,
    lr: Union[float, Tensor],
    clip_lambda: Optional[Callable[[int], float]],
    weight_decay: float,
    decouple: bool,
    eps: float,
    maximize: bool,
):
    r"""Functional API that performs ADOPT algorithm computation.
    
    This is the main functional interface for the ADOPT optimization algorithm. It
    dispatches to one of three implementations based on the execution mode:
    
    1. **Single-tensor mode** (default): Updates parameters one at a time
       - Compatible with torch.jit.script
       - More flexible but slower
       - Used when foreach=False or automatically for small models
    
    2. **Multi-tensor (foreach) mode**: Batches operations across tensors
       - 2-3x faster on GPU through vectorization
       - Groups tensors by device/dtype automatically
       - Used when foreach=True
    
    3. **Fused mode**: Uses specialized fused kernels (not yet implemented)
       - Would provide maximum performance
       - Currently raises RuntimeError if enabled
    
    Algorithm overview (ADOPT):
    -------------------------
    ADOPT adapts learning rates per-parameter while using momentum on normalized
    gradients. The key innovation is normalizing gradients before momentum, which
    provides more stable training than standard Adam.
    
    Mathematical formulation:
        # Normalize gradient by its historical variance
        normed_g_t = g_t / √(v_t + ε)
        
        # Optional gradient clipping for stability
        normed_g_t = clip(normed_g_t, threshold(t))
        
        # Momentum on normalized gradients (key difference from Adam)
        m_t = β₁ * m_{t-1} + (1 - β₁) * normed_g_t
        
        # Parameter update
        θ_t = θ_{t-1} - α * m_t
        
        # Update variance estimate
        v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
    
    where:
        - θ: parameters
        - g: gradients
        - m: first moment (momentum of normalized gradients)
        - v: second moment (variance of raw gradients)
        - α: learning rate
        - β₁, β₂: exponential decay rates
        - ε: numerical stability constant
        - clip(): gradient clipping function based on step
    
    Automatic mode selection:
    ------------------------
    When foreach and fused are both None (default), the function automatically
    selects the best implementation based on:
    - Parameter types and devices
    - Whether differentiable mode is enabled
    - Learning rate type (float vs Tensor)
    - Capturable mode requirements
    
    Args:
        params (List[Tensor]): Parameters to optimize.
        grads (List[Tensor]): Gradients for each parameter.
        exp_avgs (List[Tensor]): First moment estimates (momentum).
        exp_avg_sqs (List[Tensor]): Second moment estimates (variance).
        state_steps (List[Tensor]): Step counters (must be singleton tensors).
        foreach (Optional[bool]): Whether to use multi-tensor implementation.
            None: auto-select based on configuration (default).
        capturable (bool): If True, ensure CUDA graph capture safety.
        differentiable (bool): If True, allow gradients through optimization step.
        fused (Optional[bool]): If True, use fused kernels (not implemented).
        grad_scale (Optional[Tensor]): Gradient scaler for AMP training.
        found_inf (Optional[Tensor]): Flag for inf/nan detection in AMP.
        has_complex (bool): Whether any parameters are complex-valued.
        beta1 (float): Exponential decay rate for first moment (momentum).
            Typical range: 0.9-0.95.
        beta2 (float): Exponential decay rate for second moment (variance).
            Typical range: 0.999-0.9999 (higher than Adam).
        lr (Union[float, Tensor]): Learning rate. Can be a scalar Tensor for
            dynamic learning rate with capturable=True.
        clip_lambda (Optional[Callable[[int], float]]): Function that maps step
            number to gradient clipping threshold. None disables clipping.
        weight_decay (float): Weight decay coefficient (L2 penalty).
        decouple (bool): If True, use decoupled weight decay (AdamW-style).
            Recommended for better generalization.
        eps (float): Small constant for numerical stability in normalization.
        maximize (bool): If True, maximize objective instead of minimize.
    
    Raises:
        RuntimeError: If torch.jit.script is used with foreach or fused.
        RuntimeError: If state_steps contains non-tensor elements.
        RuntimeError: If fused=True (not yet implemented).
        RuntimeError: If lr is Tensor with foreach=True and capturable=False.
    
    Example:
        >>> # Typically called by ADOPT optimizer, not directly
        >>> adopt(
        ...     params=[p1, p2],
        ...     grads=[g1, g2],
        ...     exp_avgs=[m1, m2],
        ...     exp_avg_sqs=[v1, v2],
        ...     state_steps=[step1, step2],
        ...     beta1=0.9,
        ...     beta2=0.9999,
        ...     lr=0.001,
        ...     clip_lambda=lambda s: s**0.25,
        ...     weight_decay=0.01,
        ...     decouple=True,
        ...     eps=1e-6,
        ...     maximize=False,
        ... )
    
    Note:
        - For distributed training, this API is compatible with torch/distributed/optim
        - The foreach mode is generally preferred for GPU training
        - Complex parameters are handled transparently by viewing as real
        - First optimization step only initializes variance, doesn't update parameters
    
    See Also:
        - ADOPT class: High-level optimizer interface
        - _single_tensor_adopt: Single-tensor implementation details
        - _multi_tensor_adopt: Multi-tensor implementation details
    """
    # Respect when the user inputs False/True for foreach or fused. We only want to change
    # the default when neither have been user-specified. Note that we default to foreach
    # and pass False to use_fused. This is not a mistake--we want to give the fused impl
    # bake-in time before making it the default, even if it is typically faster.
    if fused is None and foreach is None:
        _, foreach = _default_to_fused_or_foreach(
            params, differentiable, use_fused=False
        )
        # Do not flip on foreach for the unsupported case where lr is a Tensor and capturable=False.
        if foreach and isinstance(lr, Tensor) and not capturable:
            foreach = False
    if fused is None:
        fused = False
    if foreach is None:
        foreach = False

    # this check is slow during compilation, so we skip it
    # if it's strictly needed we can add this check back in dynamo
    if not torch.compiler.is_compiling() and not all(
        isinstance(t, torch.Tensor) for t in state_steps
    ):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of singleton tensors"
        )

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")
    if fused and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with fused optimizers")

    if fused and not torch.jit.is_scripting():
        func = _fused_adopt # noqa: F821
    elif foreach and not torch.jit.is_scripting():
        func = _multi_tensor_adopt
    else:
        func = _single_tensor_adopt

    func(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
        has_complex=has_complex,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        clip_lambda=clip_lambda,
        weight_decay=weight_decay,
        decouple=decouple,
        eps=eps,
        maximize=maximize,
        capturable=capturable,
        differentiable=differentiable,
        grad_scale=grad_scale,
        found_inf=found_inf,
    )