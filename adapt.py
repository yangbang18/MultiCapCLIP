import os
import argparse
import ruamel.yaml as yaml

from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import numpy as np
import random

import utils
import time
import datetime
import math
import json

from models import build_model
from datasets import create_dataset, create_sampler, create_loader
import configs


def train(model, loader, optimizer, scheduler, epoch):
    model.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    for i, batch in enumerate(metric_logger.log_every(loader, print_freq, header)):
        optimizer.zero_grad()

        loss = model(
            raw_text=batch['caption'], 
            raw_related_text=batch.get('src_caption', batch['caption']),
            clip_text_embs=batch.get('clip_text_embs'),
            related_attn_mask=batch.get('related_attn_mask'),
            lang=batch.get('lang', None),
        )

        loss.backward()
        optimizer.step()
        scheduler.step()

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss=loss.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    stats = {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}
    stats['lr'] = "{:.6f}".format(metric_logger.meters['lr'].global_avg)
    return stats

 
def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    world_size = utils.get_world_size()

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    print("Creating model", flush=True)
    model = build_model(config, mode='adapt')
    model = model.to(device)
    
    print("### Total Params: ", sum(p.numel() for p in model.parameters()))
    print("### Trainable Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    print("Creating dataset", flush=True)
    dataset = create_dataset(config, mode='adapt')

    if args.distributed:
        global_rank = utils.get_rank()            
        samplers = create_sampler([dataset], [True], world_size, global_rank)
    else:
        samplers = [None]

    loader, *_ = create_loader([dataset], samplers, [config['batch_size']], [config['num_workers']], [True])

    print(f"### data {len(dataset)}, batch size, {config['batch_size']} x {world_size}")

    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = utils.create_optimizer(arg_opt, model)
    
    arg_sche = utils.AttrDict(config['schedular'])
    step_per_epoch = math.ceil(len(dataset)/(config['batch_size']*world_size))
    arg_sche['step_per_epoch'] = step_per_epoch
    lr_scheduler = utils.create_scheduler(arg_sche, optimizer)

    checkpointer = utils.Checkpointer(args.output_dir, exclude_prefix='clip.')
    if args.resume:
        start_epoch = checkpointer.resume_latest_states(
            model=model, 
            optimizer=optimizer, 
            lr_scheduler=lr_scheduler, 
            accelerator=None,
            return_type='epoch'
        )
    else:
        start_epoch = 0

    print("Start training", flush=True)
    start_time = time.time()

    for epoch in range(start_epoch, config['schedular']['epochs']):
        stats = train(model, loader, optimizer, lr_scheduler, epoch)
        stats['epoch'] = epoch

        with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
            f.write(json.dumps(stats) + "\n")

        checkpointer.auto_save_checkpoint(
            model, config, epoch, global_step=-1, optimizer=optimizer, scheduler=lr_scheduler, 
            accelerator=None, epoch_flag=True, step_flag=False, only_latest=False)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str), flush=True)
    print('### Time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='coco', choices=['coco', 'msrvtt', 'vatex', 'flickr30k'])
    parser.add_argument('--pickle', action='store_true', help='whether to use the off-the-shelf pickle file that saves text embeddings')

    parser.add_argument('--resume', type=bool, default=True)
    parser.add_argument('--config', type=str, default='configs/adapt.yaml', help='basic configuration')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--output_root', type=str, default='output/adapt')
    parser.add_argument('--folder', type=str, help='the exact folder name you want; otherwise, it will follow the $folder_format')
    parser.add_argument('--folder_format', type=str, default='{clip_arch}_{dataset}_{method}')

    parser.add_argument('--data_path', type=str)
    parser.add_argument('--clip_arch', type=str)
    parser.add_argument('--num_adapt_samples', type=int)
    parser.add_argument('--decoder_config', type=str)
    parser.add_argument('--noise_std', type=float)

    parser.add_argument('--keys_to_override', nargs='+', default=[
        'data_path', 'clip_arch', 'num_adapt_samples', 'decoder_config', 'noise_std',
    ])
    args = parser.parse_args()

    config = configs.create_config(args, mode='adapt')
    print("### Config")
    print(config)

    if args.output_dir is None:
        folder_name = args.folder or utils.get_folder_name(config, args.folder_format)
        args.output_dir = os.path.join(args.output_root, folder_name)
    print('### output_dir:', args.output_dir)
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)
