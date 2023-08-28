import argparse
import os
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
import glob
import shutil
from models import build_model
from datasets import create_dataset, create_sampler, create_loader
import configs


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
            lang=batch.get('lang'),
            sample=False, 
            num_beams=config['num_beams'], 
            max_length=config['max_length'],
            min_length=config['min_length'],
        )

        for caption, img_id in zip(captions, batch['image_id']):
            result.append({"image_id": img_id.item(), "caption": caption})

    return result


def main(args, config, write=True): 
    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    print("Creating model")
    model = build_model(config, mode='finetune')
    
    checkpoint_path = os.path.join(args.from_pretrained, args.ckpt_name)
    print("### Load pre-trained checkpoint from", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, 'cpu')
    epoch = checkpoint.get('epoch', -1)
    model.load_pretrained_state_dict(checkpoint['model'])

    model = model.to(device)
    print("### Total Params: ", sum(p.numel() for p in model.parameters()))
    print("### Trainable Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    print("Creating fine-tuning dataset")
    train_dataset, val_dataset, test_dataset = create_dataset(config, mode='finetune', preprocess=model.preprocess)
    datasets = [val_dataset, test_dataset]

    world_size = utils.get_world_size()

    if args.distributed:
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [False, False], world_size, global_rank)
    else:
        samplers = [None, None]

    val_loader, test_loader = create_loader(datasets, samplers,
                                            batch_size=[config['batch_size_test'], config['batch_size_test']],
                                            num_workers=[4, 4], is_trains=[False, False])

    start_time = time.time()
    print("### output_dir, ", args.output_dir, flush=True)

    print("Start evaluating")
    log_status = {}
    for mode, loader in zip(['val', 'test'], [val_loader, test_loader]):
        test_result = evaluation(model, loader, device, config)
        test_result_file = utils.collect_result(test_result, f'{mode}_epoch{epoch}', local_wdir=args.result_dir, save_result=True, remove_duplicate='image_id')

        if utils.is_main_process():
            coco_test = utils.coco_caption_eval(config[f'{mode}_gt_file'], test_result_file, config['eval_lang'])
            log_status.update({f'{mode}_{k}': v for k, v in coco_test.eval.items()})
            
    log_status['epoch'] = epoch

    if utils.is_main_process() and write:
        with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
            f.write(json.dumps(log_status) + "\n")

        print(log_status, flush=True)

    if utils.is_dist_avail_and_initialized():
        dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('### Time {}'.format(total_time_str))
    
    return log_status


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--from_pretrained', type=str, required=True)
    parser.add_argument('--ckpt_name', type=str)
    parser.add_argument('--pickle', action='store_true', help='whether to use the off-the-shelf pickle file that saves text embeddings')
    parser.add_argument('--folder', type=str)
    parser.add_argument('--output_dir', type=str)

    parser.add_argument('--config', type=str, default='configs/finetune.yaml')
    parser.add_argument('--resume', type=bool, default=True, help='automatically resume the latest model states and training states for training')

    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action='store_false')

    parser.add_argument('--monitor', type=str, default='val_CIDEr')
    parser.add_argument('--method', type=str)
    parser.add_argument('--dataset', type=str, default='coco', choices=['coco', 'msrvtt', 'vatex', 'flickr30k'])

    parser.add_argument('--average_pooling', action='store_true')
    parser.add_argument('-ipp', '--pickle_path', type=str)
    parser.add_argument('--clip_arch', type=str)
    parser.add_argument('--keys_to_override', nargs='+', default=[
        'average_pooling', 'pickle_path', 'clip_arch',
    ])
    args = parser.parse_args()

    config = configs.create_config(args, mode='finetune')

    if not args.output_dir:
        assert args.folder
        args.output_dir = os.path.join(args.from_pretrained, args.folder)
    args.result_dir = os.path.join(args.output_dir, 'result')
    
    print('### output_dir:', args.output_dir)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    
    print(config)
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    best_ckpt_path = os.path.join(args.from_pretrained, args.ckpt_name or f'{args.dataset}_best.pth')

    # check status
    all_model_checkpoints = glob.glob(os.path.join(args.from_pretrained, "*.th"))
    all_epochs = sorted([
        int(os.path.basename(ckpt).split('.')[0].split('_')[-1]) 
        for ckpt in all_model_checkpoints if 'latest' not in ckpt
    ])
    
    best_score = 0
    best_stats = None
    has_printed_best_epoch = False
    log_path = os.path.join(args.output_dir, "log.txt")
    if os.path.exists(log_path):
        lines = open(log_path, 'r').read().strip().split('\n')
        if 'best epoch' in lines[-1]:
            has_printed_best_epoch = True
            lines = lines[:-1]
        
        exists = set()
        for line in lines:
            if not len(line):
                continue
            status = eval(line)
            if status[args.monitor] > best_score:
                best_score = status[args.monitor]
                best_stats = status

            if status['epoch'] in exists:
                print(f'### It\'s wired, {status["epoch"]} has been evaluated several times')
            else:
                all_epochs.remove(status['epoch'])
            exists.add(status['epoch'])
        
        print("### Remaining epochs:", all_epochs)

    if not os.path.exists(best_ckpt_path):
        utils.init_distributed_mode(args)

        # start evaluation
        for epoch in all_epochs:
            args.ckpt_name = f'model_state_epoch_{epoch}.th'
            status = main(args, config)
            if utils.is_main_process():
                if status['val_CIDEr'] > best_score:
                    best_score = status['val_CIDEr']
                    best_stats = status

        # save sth
        if utils.is_main_process():
            if not has_printed_best_epoch:
                with open(log_path, "a") as f:
                    f.write("best epoch: %d\n" % best_stats['epoch'])
            
            shutil.copy(
                os.path.join(args.from_pretrained, f'model_state_epoch_{best_stats["epoch"]}.th'),
                best_ckpt_path
            )
            print(f"### Best status: {best_stats}")

    print('### The best checkpoint has been saved to', best_ckpt_path)

    epoch = best_stats['epoch']
    json_path = os.path.join(args.result_dir, 'test_epoch%d.json' % epoch)
    if not os.path.exists(json_path):
        args.ckpt_name = f'model_state_epoch_{epoch}.th'
        main(args, config, write=False)
    assert os.path.exists(json_path)

