import json
import os
import time
from collections import defaultdict, deque

import datetime

import numpy as np

import torch
import torch.distributed as dist

from .optim import create_optimizer
from .scheduler import create_scheduler
from .checkpointer import Checkpointer
from .eval import coco_caption_eval, autoencode_eval


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        if np.isnan(value):
            # There occurs gradient overflow in apex.amp
            return
        
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        if len(self.deque) == 0:
            return 0
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        if len(self.deque) == 0:
            return 0
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        if self.count == 0:
            return 0
        return self.total / self.count

    @property
    def max(self):
        if len(self.deque) == 0:
            return 0
        return max(self.deque)

    @property
    def value(self):
        if len(self.deque) == 0:
            return 0
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t", window_size=1, fmt='{value:.4f}'):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.window_size = window_size
        self.fmt = fmt

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.meters:
                from utils import SmoothedValue
                self.add_meter(k, SmoothedValue(window_size=self.window_size, fmt=self.fmt))

            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def global_avg(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.4f}".format(name, meter.global_avg)
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, loader, print_freq, header=None):
        if not header:
            header = ''

        dataset_len = len(loader)
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(dataset_len))) + 'd'

        _msg = [
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            _msg.append('max mem: {memory:.0f}')
        _msg = self.delimiter.join(_msg)
        MB = 1024.0 * 1024.0
        iterable = iter(loader)
        total_train_steps = dataset_len
        
        start_step = 0

        for i in range(start_step, total_train_steps):
            obj = next(iterable)

            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)

            log_msg = header + " " + _msg
            if (i % dataset_len) % print_freq == 0 or i == dataset_len - 1:
                eta_seconds = iter_time.global_avg * (dataset_len - i % dataset_len)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i % dataset_len, dataset_len, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i % dataset_len, dataset_len, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))

            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / dataset_len))


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def read_json(rpath: str):
    result = []
    with open(rpath, 'rt') as f:
        for line in f:
            result.append(json.loads(line.strip()))

    return result


def write_json(result: list, wpath: str):
    with open(wpath, 'wt') as f:
        for res in result:
            f.write(json.dumps(res) + '\n')


def collect_result(result, filename, local_wdir, save_result=False, remove_duplicate='', do_not_collect=False):
    assert isinstance(result, list)
    write_json(result, os.path.join(local_wdir,'%s_rank%d.json' % (filename, get_rank())))
    if is_dist_avail_and_initialized():
        dist.barrier()

    if do_not_collect:
        return None

    result = []
    final_result_file = ''
    if is_main_process():
        # combine results from all processes
        for rank in range(get_world_size()):
            result += read_json(os.path.join(local_wdir, '%s_rank%d.json' % (filename, rank)))

        if remove_duplicate:  # for evaluating captioning tasks
            result_new = []
            id_list = set()
            for res in result:
                if res[remove_duplicate] not in id_list:
                    id_list.add(res[remove_duplicate])
                    result_new.append(res)
            result = result_new

        if save_result:
            final_result_file = os.path.join(local_wdir, '%s.json' % filename)
            json.dump(result, open(final_result_file, 'w'), indent=4)
            print('result file saved to %s' % final_result_file)

    if is_dist_avail_and_initialized():
        dist.barrier()

    return final_result_file if save_result else result


def check_valid(config):
    data_fn_splits = os.path.basename(config['data_path']).split('.')[0].split('_')
    if len(data_fn_splits) > 1:
        clip_arch = data_fn_splits[-1].upper()

        if clip_arch[:3] == 'VIT':
            assert '@' not in clip_arch, f"TODO: {clip_arch}"
            _, model_type, patch_size = clip_arch.split('-')
            clip_arch = f'ViT-{model_type}/{patch_size}'
        else:
            clip_arch = clip_arch.replace('X', 'x')
        
        print(f"### clip_arch in config: {config['clip_arch']}, clip_arch in data_path: {clip_arch}")
        config['clip_arch'] = clip_arch
    

def get_folder_name(config, folder_format='{clip_arch}_{dataset}_{method}'):
    clip_arch = config['clip_arch'].lower().replace('/', '-')
    
    folder_name = folder_format.format(
        clip_arch=clip_arch,
        dataset=config['dataset'],
        method=config['method'],
    )
    return folder_name

