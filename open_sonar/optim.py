import math
import warnings
from functools import partial
from typing import Optional, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

from transformers.trainer_pt_utils import LayerWiseDummyOptimizer, LayerWiseDummyScheduler
from transformers.trainer_utils import SchedulerType
from transformers.utils import logging
from transformers.optimization import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_with_min_lr_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_reduce_on_plateau_schedule,
    get_wsd_schedule,
)

import torch

class MultiLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lambda_factories, last_epoch=-1):
        
        self.schedulers = []
        values = self._get_optimizer_lr(optimizer)
        for idx, factory in enumerate(lambda_factories):
            self.schedulers.append(factory(optimizer))
            values[idx] = self._get_optimizer_lr(optimizer)[idx]
            self._set_optimizer_lr(optimizer, values)
            
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        result = []
        for idx, sched in enumerate(self.schedulers):
            result.append(sched.get_last_lr()[idx])
        return result
    
    @staticmethod
    def _set_optimizer_lr(optimizer, values):
        for param_group, lr in zip(optimizer.param_groups, values):
            param_group['lr'] = lr

    @staticmethod
    def _get_optimizer_lr(optimizer):
        return [group['lr'] for group in optimizer.param_groups]

    def step(self, epoch=None):

        if self.last_epoch != -1:
            values = self._get_optimizer_lr(self.optimizer)
            for idx, sched in enumerate(self.schedulers):
                sched.step()
                values[idx] = self._get_optimizer_lr(self.optimizer)[idx]
                self._set_optimizer_lr(self.optimizer, values)
        super().step()

# torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
      
def get_multi_schedule(
    optimizer: Optimizer, num_warmup_steps: int, timescale: Optional[int] = None, last_epoch: int = -1
):
    """
    Create a schedule with an inverse square-root learning rate, from the initial lr set in the optimizer, after a
    warmup period which increases lr linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        timescale (`int`, *optional*, defaults to `num_warmup_steps`):
            Time scale.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    # Note: this implementation is adapted from
    # https://github.com/google-research/big_vision/blob/f071ce68852d56099437004fd70057597a95f6ef/big_vision/utils.py#L930

    if timescale is None:
        timescale = int(num_warmup_steps) or 10_000

    lr_lambda = partial(get_inverse_sqrt_schedule, num_warmup_steps=num_warmup_steps, timescale=timescale)
    lr_specific = partial(get_inverse_sqrt_schedule, num_warmup_steps=0, timescale=timescale//2)

    return MultiLR(optimizer, [lr_lambda, lr_lambda, lr_specific], last_epoch=last_epoch)


def _get_inverse_sqrt_schedule_lr_lambda(current_step: int, *, num_warmup_steps: int, timescale: Optional[int] = None, epsilon: float = 1e-3):
    if current_step < num_warmup_steps:
        
        ratio = float(current_step) / float(max(1, num_warmup_steps))

        return ratio + (1.0 - ratio) * epsilon
    shift = timescale - num_warmup_steps
    decay = 1.0 / math.sqrt((current_step + shift) / timescale)
    return decay

def get_inverse_sqrt_schedule(
    optimizer: Optimizer, num_warmup_steps: int, timescale: Optional[int] = None, last_epoch: int = -1
):
    """
    Create a schedule with an inverse square-root learning rate, from the initial lr set in the optimizer, after a
    warmup period which increases lr linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        timescale (`int`, *optional*, defaults to `num_warmup_steps`):
            Time scale.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    # Note: this implementation is adapted from
    # https://github.com/google-research/big_vision/blob/f071ce68852d56099437004fd70057597a95f6ef/big_vision/utils.py#L930

    if timescale is None:
        timescale = int(num_warmup_steps) or 10_000

    lr_lambda = partial(_get_inverse_sqrt_schedule_lr_lambda, num_warmup_steps=num_warmup_steps, timescale=timescale)
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)

SchedulerType.MULTI = "multi"


TYPE_TO_SCHEDULER_FUNCTION = {
    SchedulerType.LINEAR: get_linear_schedule_with_warmup,
    SchedulerType.COSINE: get_cosine_schedule_with_warmup,
    SchedulerType.COSINE_WITH_RESTARTS: get_cosine_with_hard_restarts_schedule_with_warmup,
    SchedulerType.POLYNOMIAL: get_polynomial_decay_schedule_with_warmup,
    SchedulerType.CONSTANT: get_constant_schedule,
    SchedulerType.CONSTANT_WITH_WARMUP: get_constant_schedule_with_warmup,
    SchedulerType.INVERSE_SQRT: get_inverse_sqrt_schedule,
    SchedulerType.REDUCE_ON_PLATEAU: get_reduce_on_plateau_schedule,
    SchedulerType.COSINE_WITH_MIN_LR: get_cosine_with_min_lr_schedule_with_warmup,
    SchedulerType.WARMUP_STABLE_DECAY: get_wsd_schedule,
    SchedulerType.MULTI: get_multi_schedule,
}



def get_scheduler(
    name: Union[str, SchedulerType],
    optimizer: Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    scheduler_specific_kwargs: Optional[dict] = None,
):
    """
    Unified API to get any scheduler from its name.

    Args:
        name (`str` or `SchedulerType`):
            The name of the scheduler to use.
        optimizer (`torch.optim.Optimizer`):
            The optimizer that will be used during training.
        num_warmup_steps (`int`, *optional*):
            The number of warmup steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_training_steps (`int``, *optional*):
            The number of training steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        scheduler_specific_kwargs (`dict`, *optional*):
            Extra parameters for schedulers such as cosine with restarts. Mismatched scheduler types and scheduler
            parameters will cause the scheduler function to raise a TypeError.
    """
    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]

    # If a `LayerWiseDummyOptimizer` is passed we extract the optimizer dict and
    # recursively call `get_scheduler` to get the proper schedulers on each parameter
    
    
    
    if optimizer is not None and isinstance(optimizer, LayerWiseDummyOptimizer):
        optimizer_dict = optimizer.optimizer_dict
        scheduler_dict = {}
        

        for param in optimizer_dict.keys():
            scheduler_dict[param] = get_scheduler(
                name,
                optimizer=optimizer_dict[param],
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                scheduler_specific_kwargs=scheduler_specific_kwargs,
            )

        def scheduler_hook(param):
            # Since the optimizer hook has been already attached we only need to
            # attach the scheduler hook, the gradients have been zeroed here
            scheduler_dict[param].step()

        for param in optimizer_dict.keys():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(scheduler_hook)

        return LayerWiseDummyScheduler(optimizer_dict=optimizer_dict, lr=optimizer.defaults["lr"])

    if name == SchedulerType.CONSTANT:
        return schedule_func(optimizer)

    if scheduler_specific_kwargs is None:
        scheduler_specific_kwargs = {}

    if name == SchedulerType.REDUCE_ON_PLATEAU:
        return schedule_func(optimizer, **scheduler_specific_kwargs)

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)

    if name == SchedulerType.INVERSE_SQRT:
        # return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)
        
        for param_group in optimizer.param_groups:
            if "name" in param_group and param_group["name"] == "specific":
                return get_multi_schedule(optimizer, num_warmup_steps=num_warmup_steps)
        
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)

    # wsd scheduler requires either num_training_steps or num_stable_steps
    if name == SchedulerType.WARMUP_STABLE_DECAY:
        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            **scheduler_specific_kwargs,
        )

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")


    return schedule_func(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        **scheduler_specific_kwargs,
    )
