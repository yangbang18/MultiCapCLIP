import torch
import configs
from torch.utils.data import DataLoader

from .adapt_dataset import AdaptDataset
from .finetune_dataset import (
    ImageCaptionDataset,
    VideoCaptionDataset,
    get_uniform_frame_ids,
)


def create_dataset(config, mode='adapt', preprocess=None):
    assert mode in ['adapt', 'finetune', 'mix']

    kwargs_for_AdaptDataset = dict(
        with_related_caption_as_input=config.get('with_related_caption_as_input', False), 
        related_caption_ids_path=config.get('related_caption_ids_path', None),
        related_caption_topk=config.get('related_caption_topk', -1),
        related_caption_prob=config.get('related_caption_prob', 1.0),
        num_related_caption=config.get('num_related_caption', 1),
        auto_generate_related_caption_ids=config.get('auto_generate_related_caption_ids', True),
        src_lang=config.get('src_lang', configs.main_lang),
        train_langs=config.get('train_langs', [configs.main_lang]),
        train_en_prob=config.get('train_en_prob', None),
        hdf5_path=config.get('hdf5_path', None),
        parallel_text_paths=config.get('parallel_text_paths', None),
    )

    if mode == 'adapt':
        return AdaptDataset(config['data_path'], config['num_adapt_samples'], **kwargs_for_AdaptDataset)

    kwargs = dict(
        prompt=config['prompt'], 
        max_words=config['max_tokens'],
        pickle_path=config.get('pickle_path', None),
        i3d=config.get('i3d', False),
    )

    assert preprocess is not None, "Please pass the clip\'s preprocess"
    print('Image Transform:')
    print(preprocess)

    if config['dataset'] in configs.video_datasets:
        DATASET = VideoCaptionDataset
        kwargs['num_frames'] = config['num_frames']
    else:
        DATASET = ImageCaptionDataset

    if mode == 'finetune':
        train_dataset = DATASET(preprocess, config['root'], config['train_file'], **kwargs)
        val_dataset = DATASET(preprocess, config['root'], config['val_file'], **kwargs)
        test_dataset = DATASET(preprocess, config['root'], config['test_file'], **kwargs)
    else:
        train_dataset = AdaptDataset(config['data_path'], config.get('num_adapt_samples', -1), **kwargs_for_AdaptDataset)
        val_dataset = DATASET(preprocess, config['root'], config['val_file'], **kwargs)
        test_dataset = DATASET(preprocess, config['root'], config['test_file'], **kwargs)
    
    print('Train dataset size:', len(train_dataset))
    print('Val dataset size:', len(val_dataset))
    print('Test dataset size:', len(test_dataset))
    return train_dataset, val_dataset, test_dataset


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train in zip(datasets, samplers, batch_size, num_workers, is_trains):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True if len(dataset) >= bs else False
        else:
            shuffle = False
            drop_last = False
        
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=getattr(dataset, 'collate_fn', None),
            drop_last=drop_last,
        )
        loaders.append(loader)

    return loaders
