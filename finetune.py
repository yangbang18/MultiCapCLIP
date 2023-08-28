import argparse
import os
import math
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import utils

from models import build_model
from datasets import create_dataset, create_sampler, create_loader
import configs


def train(model, data_loader, optimizer, epoch, device, scheduler, config):
    model.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.5f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        loss = model(
            raw_text=batch['caption'], 
            raw_related_text=batch.get('src_caption', batch['caption']),
            image=batch.get('image'),
            clip_image_embs=batch.get('clip_image_embs'),
            clip_text_embs=batch.get('clip_text_embs'),
            lang=batch.get('lang'),
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    stats = {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}
    stats['lr'] = "{:.6f}".format(metric_logger.meters['lr'].global_avg)
    return stats


@torch.no_grad()
def evaluation(model, data_loader, device, config):
    # test
    model.eval()

    model_without_ddp = model
    if hasattr(model, 'module'):
        model_without_ddp = model.module

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Caption generation:'
    print_freq = 50
    
    result = []

    for batch in metric_logger.log_every(data_loader, print_freq, header):
        captions = model_without_ddp.generate(
            image=batch.get('image'), 
            clip_image_embs=batch.get('clip_image_embs'),
            lang=config['eval_lang'],
            sample=False, 
            num_beams=config['num_beams'], 
            max_length=config['max_length'],
            min_length=config['min_length'],
        )

        for caption, img_id in zip(captions, batch['image_id']):
            result.append({"image_id": img_id.item(), "caption": caption})

    return result


def main(args, config):
    if args.dataset == 'msrvtt' and config.get('eval_lang', configs.main_lang) == 'zh':
        no_test = True
    else:
        no_test = False

    utils.init_distributed_mode(args)    
    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    print("Creating model")
    if args.from_pretrained:
        checkpoint_path = os.path.join(args.from_pretrained, args.ckpt_name)
        print("### Load pre-trained checkpoint from", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, 'cpu')
        epoch = checkpoint.get('epoch', -1)
        model = build_model(config, mode=args.mode)
        model.load_pretrained_state_dict(checkpoint['model'])
    else:
        epoch = -1
        model = build_model(config, mode=args.mode)

    model = model.to(device)
    print("### Total Params: ", sum(p.numel() for p in model.parameters()))
    print("### Trainable Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    print("Creating fine-tuning dataset")
    train_dataset, val_dataset, test_dataset = create_dataset(config, mode='finetune' if not config.get('data_path') else 'mix', preprocess=model.preprocess)
    datasets = [train_dataset, val_dataset, test_dataset]

    train_dataset_size = len(train_dataset)
    world_size = utils.get_world_size()

    if utils.is_main_process():
        print(f"### data {train_dataset_size}, batch size, {config['batch_size_train']} x {world_size}")

    if args.distributed:
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True, False, False], world_size, global_rank)
    else:
        samplers = [None, None, None]

    train_loader, val_loader, test_loader = create_loader(datasets, samplers,
                                              batch_size=[config['batch_size_train'], config['batch_size_test'], config['batch_size_test']],
                                              num_workers=[4, 4, 4], is_trains=[True, False, False])

    start_time = time.time()

    if args.evaluate:
        print("Start evaluating")
        log_stats = {}
        
        modes = ['val', 'test']
        loaders = [val_loader, test_loader]
        if args.no_val:
            modes, loaders = ['test'], [test_loader]

        for mode, loader in zip(modes, loaders):
            test_result = evaluation(model, loader, device, config)
            test_result_file = utils.collect_result(
                test_result, f'{mode}_eval', local_wdir=args.result_dir,
                save_result=True, remove_duplicate='image_id')

            if not args.no_score:
                if utils.is_main_process():
                    coco_test = utils.coco_caption_eval(config[f'{mode}_gt_file'], test_result_file, config['eval_lang'])
                    # save scores
                    utils.collect_result(coco_test.evalImgs, f'{mode}_eval_scores', local_wdir=args.result_dir, save_result=True, remove_duplicate='image_id')
                    log_stats.update({f'{mode}_{k}': v for k, v in coco_test.eval.items()})
                
        log_stats['epoch'] = epoch

        if args.msg:
            log_stats['msg'] = args.msg # to identify where are the results from
        
        if utils.is_main_process():
            with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats) + "\n")

            print(log_stats, flush=True)

        if utils.is_dist_avail_and_initialized():
            dist.barrier()

    else:
        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = utils.create_optimizer(arg_opt, model)
        arg_sche = utils.AttrDict(config['schedular'])
        arg_sche['step_per_epoch'] = math.ceil(train_dataset_size / (config['batch_size_train'] * world_size))
        lr_scheduler = utils.create_scheduler(arg_sche, optimizer)

        checkpointer = utils.Checkpointer(args.output_dir, exclude_prefix='clip.')

        model_without_ddp = model
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module
         
        max_epoch = config['schedular']['epochs']
        best = 0
        best_epoch = 0
        start_epoch = 0

        if args.resume:
            training_states = checkpointer.resume_latest_states(
                model=model, 
                optimizer=optimizer, 
                lr_scheduler=lr_scheduler, 
                return_type='all',
            )
            if training_states is not None:
                start_epoch = training_states.get('epoch', -1) + 1
                best = training_states.get('best', 0)
                best_epoch = training_states.get('best_epoch', 0)

        for epoch in range(start_epoch, max_epoch):
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            train_stats = train(model, train_loader, optimizer, epoch, device, lr_scheduler, config)

            if utils.is_main_process():
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             'epoch': epoch}

            if epoch >= config['start_eval']:
                val_result = evaluation(model, val_loader, device, config)
                val_result_file = utils.collect_result(
                    val_result, 'val_epoch%d' % epoch, local_wdir=args.result_dir, 
                    save_result=True, remove_duplicate='image_id')

                if not no_test:
                    test_result = evaluation(model, test_loader, device, config)
                    test_result_file = utils.collect_result(
                        test_result, 'test_epoch%d' % epoch, local_wdir=args.result_dir, 
                        save_result=True, remove_duplicate='image_id')

                if utils.is_main_process():
                    coco_val = utils.coco_caption_eval(config['val_gt_file'], val_result_file, config['eval_lang'])
                    if not no_test:
                        coco_test = utils.coco_caption_eval(config['test_gt_file'], test_result_file, config['eval_lang'])

                    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                 **{f'val_{k}': v for k, v in coco_val.eval.items()},
                                 **({f'test_{k}': v for k, v in coco_test.eval.items()} if not no_test else {}),
                                 'epoch': epoch}
                    
                    if coco_val.eval['CIDEr'] > best:
                        save_obj = {
                            'model': model_without_ddp.state_dict(),
                            # 'optimizer': optimizer.state_dict(),
                            # 'lr_scheduler': lr_scheduler.state_dict(),
                            'config': config,
                            # 'epoch': epoch,
                        }
                        torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
                        best = coco_val.eval['CIDEr']
                        best_epoch = epoch

                if utils.is_dist_avail_and_initialized():
                    dist.barrier()

            if utils.is_main_process():
                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")
                
                checkpointer.auto_save_checkpoint(
                    model=model, 
                    config=config, 
                    current_epoch=epoch, 
                    global_step=arg_sche['step_per_epoch'] * (epoch + 1), 
                    optimizer=optimizer, 
                    scheduler=lr_scheduler, 
                    only_latest=True,
                    extra_training_states={'best': best, 'best_epoch': best_epoch}
                )

            if utils.is_dist_avail_and_initialized():
                dist.barrier()

        if utils.is_main_process() and start_epoch < max_epoch:
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write("best epoch: %d\n" % best_epoch)

            os.system(f"cat {args.output_dir}/log.txt")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('### Time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--from_pretrained', type=str)
    parser.add_argument('--ckpt_name', type=str, default='model_state_latest.th')
    parser.add_argument('--dataset', type=str, default='coco', choices=['coco', 'msrvtt', 'vatex', 'flickr30k'])
    parser.add_argument('--pickle', action='store_true')

    parser.add_argument('--config', type=str, default='configs/finetune.yaml')
    parser.add_argument('--fewshot', action='store_true')
    parser.add_argument('--resume', type=bool, default=True, help='automatically resume the latest model states and training states for training')
    
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--output_root', type=str, default='output/finetune')
    parser.add_argument('--folder', type=str)
    parser.add_argument('--folder_format', type=str, default='{clip_arch}_{dataset}_{method}')

    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action='store_false')

    parser.add_argument('--evaluate', action='store_true')

    parser.add_argument('--msg', type=str, default='')

    parser.add_argument('-arch', '--clip_arch', type=str)
    parser.add_argument('--decoder_config', type=str)
    parser.add_argument('--init_with_bert', action='store_true')
    parser.add_argument('-method', '--method', type=str)
    parser.add_argument('--num_tokens', type=int)

    parser.add_argument('-mode', '--mode', type=str, default='finetune')
    parser.add_argument('-dp', '--data_path', type=str)
    parser.add_argument('--num_adapt_samples', type=int, default=-1)
    parser.add_argument('-ipp', '--pickle_path', type=str)
    parser.add_argument('-std', '--noise_std', type=float)

    parser.add_argument('--num_hidden_layers', type=int)
    parser.add_argument('--label_smoothing', type=float)

    parser.add_argument('--train_file', type=str)
    parser.add_argument('--val_file', type=str)
    parser.add_argument('--test_file', type=str)
    parser.add_argument('--subset_fn', type=str)
    
    parser.add_argument('--num_beams', type=int)
    parser.add_argument('--freeze_projector', action='store_true')

    parser.add_argument('--average_pooling', action='store_true')
    parser.add_argument('--no_val', action='store_true')
    parser.add_argument('--no_score', action='store_true')

    parser.add_argument('--keys_to_override', nargs='+', default=[
        'clip_arch', 'decoder_config', 'init_with_bert',
        'data_path', 'num_adapt_samples',
        'pickle_path', 'noise_std', 'num_hidden_layers', 'label_smoothing', 'num_tokens',
        'num_beams', 'freeze_projector', 'average_pooling',
    ])
    args = parser.parse_args()

    config = configs.create_config(args, mode='finetune')
    print("### Config")
    print(config)
    
    if args.output_dir is None:
        folder_name = args.folder or utils.get_folder_name(config, args.folder_format)
        root = args.from_pretrained or args.output_root
        args.output_dir = os.path.join(root, folder_name)
    print('### output_dir:', args.output_dir)

    args.result_dir = os.path.join(args.output_dir, 'result')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)
