import os
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from xtuner.engine.runner import TrainLoop
from torch.optim import AdamW
from mmengine.config import read_base
from src.models.harmon_dev import HarmonDev

with read_base():
    from ..models.qwen2_5_1_5b_kl16_mar_h import model
    from ..datasets.qwen2_5_1_5b.image2text_text2image import train_dataloader, repeat

#######################################################################
#                          PART 1  Settings                           #
#######################################################################

model_name = os.getenv("MODEL_NAME", "checkpoints/harmon_1.5b.pth")

# Model
model.update(
    type=HarmonDev,
    pretrained_pth=model_name,
    freeze_llm=False,
)

# Scheduler & Optimizer
accumulative_counts = sum(repeat)
dataloader_num_workers = 4
max_iters = 50000
optim_type = AdamW
lr = 1e-5
betas = (0.9, 0.95)
weight_decay = 0.02
max_norm = 1.0  # grad clip
warmup_ratio = 0.01

# Save
save_steps = 5000
save_total_limit = 1

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
train_dataloader = train_dataloader

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    constructor='MAROptimWrapperConstructor',
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='bfloat16')

param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=False,
        begin=0,
        end=warmup_ratio * max_iters),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=False,
        begin=warmup_ratio * max_iters,
        end=max_iters)
]

train_cfg = dict(type=TrainLoop, max_iters=max_iters)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit
    ),
    sampler_seed=dict(type=DistSamplerSeedHook),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

visualizer = None
log_level = 'INFO'
load_from = None
resume = False
randomness = dict(seed=None, deterministic=False)
log_processor = dict(by_epoch=False)
