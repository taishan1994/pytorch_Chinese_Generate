# coding=utf-8
import os
import logging
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
from torch import nn,optim

logger = logging.getLogger(__name__)


def build_optimizer_and_scheduler(args, model, t_total, opt="adam"):
    module = (
        model.module if hasattr(model, "module") else model
    )

    # 差分学习率
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    model_param = list(module.named_parameters())

    params = []

    for name, para in model_param:
      params.append((name, para))

    optimizer_grouped_parameters = [
        # bert other module
        {"params": [p for n, p in params if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.lr},
        {"params": [p for n, p in params if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': args.lr},
    ]

    if opt == "adam":
      optimizer = optim.Adam(model.parameters(), lr=args.lr) # Adam梯度下降
    else:
      optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(args.warmup_proportion * t_total), num_training_steps=t_total
    )

    return optimizer, scheduler

def save_model(args, model, model_name, global_step):
    """保存最好的验证集效果最好那个模型"""
    output_dir = os.path.join(args.output_dir, '{}'.format(model_name, global_step))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # take care of model distributed / parallel training
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )
    logger.info('Saving model checkpoint to {}'.format(output_dir))
    torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'model.pt'))

def save_model_step(args, model, global_step):
    """根据global_step来保存模型"""
    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # take care of model distributed / parallel training
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )
    logger.info('Saving model & optimizer & scheduler checkpoint to {}.format(output_dir)')
    torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'model.pt'))

def load_model_and_parallel(model, gpu_id=None, ckpt_path=None, strict=True):
    """
    加载模型 & 放置到 GPU 中（单卡 / 多卡）
    """

    # set to device to the first cuda
    # device = torch.device("cpu" if gpu_id == '-1' else "cuda:" + gpu_id)

    if ckpt_path is not None:
        logger.info('Load ckpt from {}'.format(ckpt_path))
        model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')), strict=strict)

    # model.to(device)

    # logger.info('Use single gpu in: {}'.format(gpu_id))

    # return model, device
    return model