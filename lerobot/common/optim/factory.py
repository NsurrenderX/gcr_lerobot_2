#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.configs.train import TrainPipelineConfig

def scheduler_simple_warpper(optimizer: Optimizer, cfg: TrainPipelineConfig) -> LRScheduler:
    return cfg.scheduler.build(optimizer, cfg.steps) if cfg.scheduler is not None else None

def make_optimizer_and_scheduler(
    cfg: TrainPipelineConfig, policy: PreTrainedPolicy
) -> tuple[Optimizer, LRScheduler | None]:
    """Generates the optimizer and scheduler based on configs.

    Args:
        cfg (TrainPipelineConfig): The training config that contains optimizer and scheduler configs
        policy (PreTrainedPolicy): The policy config from which parameters and presets must be taken from.

    Returns:
        tuple[Optimizer, LRScheduler | None]: The couple (Optimizer, Scheduler). Scheduler can be `None`.
    """
    
    # name, params = policy.get_optim_params() if cfg.use_policy_training_preset else policy.named_parameters()
    bf16_params = []
    fp32_params = []
    bf16_names = []
    fp32_names = []
    for name, param in policy.named_parameters():
        if param.requires_grad:
            if param.dtype == torch.bfloat16:
                bf16_params.append(param)
                bf16_names.append(name)
            elif param.dtype == torch.float32:
                fp32_params.append(param)
                fp32_names.append(name)
    
    # bf16_params = list(filter(lambda p: p.requires_grad and p.dtype == torch.bfloat16, named_params.values()))
    # fp32_params = list(filter(lambda p: p.requires_grad and p.dtype == torch.float32, named_params.values()))
    # bf16_names = list(filter(lambda p: p.requires_grad and p.dtype == torch.bfloat16, named_params.keys()))
    # fp32_names = list(filter(lambda p: p.requires_grad and p.dtype == torch.float32, named_params.keys()))
    params = [{"params": bf16_params, "dtype": torch.bfloat16}, {"params": fp32_params, "dtype": torch.float32}]
    optimizer = cfg.optimizer.build(params)
    lr_scheduler = cfg.scheduler.build(optimizer, cfg.steps) if cfg.scheduler is not None else None
    return optimizer, lr_scheduler, bf16_names, fp32_names
