import logging
import time
import os
import json
from pathlib import Path
from datetime import datetime
from contextlib import nullcontext
from pprint import pformat
from typing import Any
import glob

import deepspeed
from deepspeed import get_accelerator

import torch
from functools import partial
from termcolor import colored
from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.transforms import ImageTransforms
from lerobot.common.datasets.lerobot_dataset import MultiDatasetforDistTraining, extra_collate_fn
from lerobot.common.datasets.sampler import EpisodeAwareSampler, DistEpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.optim.factory import make_optimizer_and_scheduler, scheduler_simple_warpper
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.eval import eval_policy

def init_logger(cfg, rank):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO if rank == 0 else logging.WARN)
    
    if rank == 0:
        formatter = logging.Formatter(
            f'[%(asctime)s] [rank: {rank}] [%(levelname)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        log_path = Path(cfg.log_dir) / f"fsdp_logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def update_policy(
    model_engine: deepspeed.DeepSpeedEngine,
    batch: Any,
    logger
) -> tuple[MetricsTracker, dict]:
    
    batch = {k: v.to(model_engine.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    loss, output_dict = model_engine(batch)
    model_engine.backward(loss)
    grad_norm = model_engine.get_global_grad_norm()
    model_engine.step()

    return loss, grad_norm, output_dict

@parser.wrap()
def train(cfg: TrainPipelineConfig):
    
    cfg.validate()
    
    # world_size = int(os.environ["WORLD_SIZE"], 1)
    # local_rank = int(os.environ["LOCAL_RANK"], 0)
    # world_rank = int(os.environ["RANK"], 0)
    # rank = world_rank
    
    
    with open(cfg.deepspeed, "r") as f:
        ds_cfg = json.load(f)
    ds_cfg['gradient_accumulation_steps'] = cfg.gradient_accumulation_steps
    gradient_accumulation_steps = ds_cfg['gradient_accumulation_steps']
    ds_cfg['train_micro_batch_size_per_gpu'] = cfg.batch_size
    ds_cfg['optimizer']['params']['lr'] = cfg.policy.optimizer_lr
    ds_cfg['optimizer']['params']['betas'] = cfg.policy.optimizer_betas
    ds_cfg['optimizer']['params']['weight_decay'] = cfg.policy.optimizer_weight_decay
    
    deepspeed.init_distributed()
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
        
    logger = init_logger(cfg, rank)
    
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms)
    )
    
    if rank == 0:
        logger.info(pformat(cfg.to_dict()))
        if cfg.wandb.enable and cfg.wandb.project:
            wandb_logger = WandBLogger(cfg)
        else:
            wandb_logger = None
            logger.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))
    else:
        wandb_logger = None
        
    if cfg.seed is not None:
        set_seed(cfg.seed)
    else:
        cfg.seed = 34
        set_seed(cfg.seed)
        
    dataset = MultiDatasetforDistTraining(
        cfg=cfg, 
        image_transforms=image_transforms,
        seed=cfg.seed,
        data_mix=cfg.data_mix,
        vla2root_json="vla2root.json",
    )
    logger.info(f"Dataset: {dataset}")
    
    logger.info("Creating policy...")
    
    policy = make_policy(
        cfg=cfg.policy,
        device="cpu",
        ds_meta=dataset.meta,
        weight_pt_path=cfg.policy.pretrained_path
    )
    
    if rank == 0:
        logger.info(f"Model parameters: {sum(p.numel() for p in policy.parameters())}")
        logger.info(f"Qwen VL visual parameters: {sum(p.numel() for p in policy.model.paligemma_with_expert.qwen25vl.visual.parameters())}")
        logger.info(f"Qwen VL parameters: {sum(p.numel() for p in policy.model.paligemma_with_expert.qwen25vl.parameters())}")
        logger.info(f"kv repre model parameters: {sum(p.numel() for p in policy.model.paligemma_with_expert.kv_repre.parameters())}")
        logger.info(f"AWA Expert parameters: {sum(p.numel() for p in policy.model.paligemma_with_expert.awa_model.parameters())}")
        logger.info(f"Action Expert parameters: {sum(p.numel() for p in policy.model.paligemma_with_expert.qwen_expert.parameters())}")
        logger.info(f"Model trainable parameters: {sum(p.numel() for p in policy.parameters() if p.requires_grad)}")
    
    
    # optimizer, lr_scheduler, bf16_names, fp32_names = make_optimizer_and_scheduler(cfg, policy)
    scheduler_callable = partial(
        scheduler_simple_warpper,
        cfg=cfg
    )
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=cfg.seed,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=8,
        collate_fn=extra_collate_fn,
        pin_memory=True,
    )
    
    # Metrics setup
    dist_step = 50
    
    train_metrics = {
        "loss": AverageMeter("loss", ":.4f"),
        "pos_loss": AverageMeter("pos_loss", ":.4f"),
        "rot_loss": AverageMeter("rot_loss", ":.4f"),
        "grip_loss": AverageMeter("grip_loss", ":.4f"),
        "grad_norm": AverageMeter("grdn", ":.4f"),
        "lr": AverageMeter("lr", ":0.01e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
        "optim_s": AverageMeter("optim_s", ":.3f"),
    }
    
    if rank == 0:
        logger.info(f"Starting Deepspeed training on {world_size} devices")
    
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=policy,
        config=ds_cfg,
        optimizer=None,
        lr_scheduler=scheduler_callable,
    )
    
    step = 0
    if cfg.resume:
        logger.info(f"Resuming training from {cfg.output_dir}")
        ckpt_list = sorted(glob.glob(os.path.join(cfg.output_dir, "step_*")))
        steps = []
        valid_dirs = {}
        for dir_path in ckpt_list:
            if not os.path.isdir(dir_path):
                continue
            dir_name = os.path.basename(dir_path)
            if dir_name.startswith("step_"):
                try:
                    steps.append(int(dir_name.split("_")[-1]))
                    valid_dirs[int(dir_name.split("_")[-1])] = dir_path
                except ValueError:
                    logger.warning(f"Could not parse step number from directory: {dir_path}")
        if len(steps) > 0:
            logger.info(f"Found {len(steps)} checkpoint directories, names are {steps}")
            step = max(steps) 
            resume_tag = f"step_{step}"
        else:
            logger.warning(f"No valid checkpoint directories found in {cfg.output_dir}")
            cfg.resume = False
        
    if cfg.resume:
        logger.info(f"Found checkpoint: {resume_tag}. Attempting to resume...")
        load_path, client_state = model_engine.load_checkpoint(
            cfg.output_dir, 
            tag=resume_tag
        )
        if load_path is None:
            logger.warning(f"DeepSpeed failed to load checkpoint {resume_tag}. Starting from scratch.")
            step = 0
    
    train_tracker = MetricsTracker(
        cfg.batch_size*world_size*cfg.gradient_accumulation_steps,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=int(step/cfg.gradient_accumulation_steps)
    )
         
    model_engine.train()
    
    completed_steps = step + 1
    total_steps = cfg.steps
    
    fwd_bwd_time = 0.0
    dataloading_s = 0.0
    grad_norm_value = 0.0
    loss_value = 0.0
    pos_loss_value = 0.0
    rot_loss_value = 0.0
    grip_loss_value = 0.0
    
    dataloader_iter = cycle(dataloader)
    
    if cfg.resume:
        logger.info(f"Resuming training from step {step}")
        step_to_resume = step % len(dataloader)
        resume_start = time.perf_counter()
        for idx in range(step_to_resume):
            next(dataloader_iter)
            if idx % dist_step == 0:
                resume_time = time.perf_counter() - resume_start
                logger.info(f"Resumed {idx}/{step_to_resume} batches, took {resume_time:.2f} seconds")
        resume_end = time.perf_counter()
        logger.info(f"Resumed training from step {step} in {resume_end - resume_start:.2f} seconds")
    
    for step_idx in range(completed_steps, total_steps):
        start_time = time.perf_counter()
        batch = next(dataloader_iter)
        dataloading_time = time.perf_counter() - start_time
        dataloading_s += dataloading_time
        
        loss, grad_norm, outputs = update_policy(model_engine, batch, logger)
        
        grad_to_record = grad_norm.item() if grad_norm is not None else 0.0
        grad_norm_value += grad_to_record
        loss_value += loss.detach().mean().item()
        pos_loss_value += outputs["pos_loss"].detach().mean().item()
        rot_loss_value += outputs["rot_loss"].detach().mean().item()
        grip_loss_value += outputs["gripper_loss"].detach().mean().item()
        
        step_time = time.perf_counter() - start_time
        fwd_bwd_time += step_time
        
        if model_engine.is_gradient_accumulation_boundary():
            train_tracker.dataloading_s = dataloading_s
            train_tracker.update_s = fwd_bwd_time
            train_tracker.loss = loss_value
            train_tracker.pos_loss = pos_loss_value
            train_tracker.rot_loss = rot_loss_value
            train_tracker.grip_loss = grip_loss_value
            train_tracker.grad_norm = grad_norm_value
            train_tracker.lr = optimizer.param_groups[0]["lr"]
            train_tracker.optim_s = 0.0
            
            train_tracker.step()
            
            fwd_bwd_time = 0.0
            dataloading_s = 0.0
            loss_value = 0.0
            pos_loss_value = 0.0
            rot_loss_value = 0.0
            grip_loss_value = 0.0
            grad_norm_value = 0.0
        
        is_saving_step = (step_idx % cfg.save_freq == 0 or step_idx == cfg.steps) and step_idx > 0
        is_log_step = cfg.log_freq > 0 and step_idx % cfg.log_freq == 0
        
        if cfg.save_checkpoint and is_saving_step:
            
            logger.info(f"Checkpoint policy after step {step_idx}")
            os.makedirs(cfg.output_dir, exist_ok=True)
            global_step = step_idx // gradient_accumulation_steps
            client_state = {
                "step": global_step,
            }
            model_engine.save_checkpoint(save_dir=cfg.output_dir,
                                        tag=f"step_{global_step}",
                                        client_state=client_state)
        
        if is_log_step:
            if rank == 0:
                logger.info(train_tracker)
                if wandb_logger:
                    wandb_log_dict = train_tracker.to_dict()
                    if outputs:
                        wandb_log_dict.update(outputs)
                    wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()
        
        if step_idx % dist_step == 0:
            dist.barrier()
            
if __name__ == "__main__":
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['WANDB_API_KEY'] = '7f1c1acfe477063902c617b0e8ef24d2b76ed447'
    train()