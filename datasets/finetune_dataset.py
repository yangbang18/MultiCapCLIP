import os
import re
import json
import pickle
import numpy as np
import configs

import torch
from torch.utils.data import Dataset
from PIL import Image


def pre_caption(caption, max_words):
    org = caption
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])

    if not len(caption):
        print(org)
        print('--')
        print(caption)
        print('--')
        raise ValueError("pre_caption yields invalid text")

    return caption


class ImageCaptionDataset(Dataset):
    def __init__(self, transform, image_root, ann_rpath, max_words=30, prompt='', pickle_path=None, **kwargs):
        self.annotation = json.load(open(ann_rpath, 'r'))
        
        if pickle_path is not None:
            data = pickle.load(open(pickle_path, 'rb'))
            self.image2emb = {line['image']: line['image_emb'] for line in data}

        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.prompt = prompt

    def get_item_based_on_id(self, id):
        for i, ann in enumerate(self.annotation):
            image_id = int(ann['image'].split('/')[-1].strip('.jpg').split('_')[-1])
            if image_id == id:
                return self.__getitem__(i)

    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):
        ann = self.annotation[index]
        
        image_id = ann['image'].split('/')[-1].strip('.jpg').split('_')[-1]
        out = {'image_id': int(image_id)}

        if 'caption' in ann: 
            if type(ann['caption']) is str:
                # for the training set, ann['caption'] is a string
                out['caption'] = self.prompt + pre_caption(ann['caption'], self.max_words)
            else:
                # for the validation / testing set, ann['caption'] is a list of string
                out['caption'] = [self.prompt + pre_caption(caption, max_words=10000) for caption in ann['caption']]

        if not hasattr(self, 'image2emb'):
            image_path = os.path.join(self.image_root, ann['image'])
            image = Image.open(image_path).convert('RGB')   
            out['frames'] = [image]
            out['image'] = self.transform(image)
        
        if hasattr(self, 'image2emb'):
            out['clip_image_embs'] = self.image2emb[ann['image']]
        
        lang = ann.get('lang', configs.main_lang) # default to en (English)
        out['lang'] = torch.LongTensor([configs.lang2code[lang]])

        return out
    
    def collate_fn(self, batch):
        # batch contains bsz items, each of which is a dict output from self.__iter__()
        batch_tensors = {}
        for key in batch[0].keys():
            x = [b[key] for b in batch]
            if key in ['caption', 'frames']:
                batch_tensors[key] = x
                continue
            if x[0] is None:
                batch_tensors[key] = None
            elif isinstance(x[0], torch.Tensor):
                batch_tensors[key] = torch.stack(x)
            elif isinstance(x[0], np.ndarray):
                batch_tensors[key] = torch.FloatTensor(np.array(x))
            else:
                batch_tensors[key] = torch.tensor(x, dtype=torch.long)
        
        return batch_tensors


class VideoCaptionDataset(Dataset):
    def __init__(self, transform, video_root, ann_rpath, num_frames=8, max_words=30, prompt='', pickle_path=None, i3d=False):
        self.annotation = json.load(open(ann_rpath, 'r'))
        
        if i3d:
            import h5py
            i3d_path = os.path.join(configs.finetune_root, 'vatex', 'I3D.hdf5')
            db = h5py.File(i3d_path, 'r')

            mapping_path = os.path.join(configs.finetune_root, 'vatex', 'vatex_mapping.txt')
            data = open(mapping_path, 'r').read().strip().split('\n')
            mapping = {line.split(' ')[0]: line.split(' ')[1] for line in data}

            print('### Preparing vatex\'s I3D features ...')
            self.video2embs = {}
            for k in db.keys():
                # e.g., f2uK1SvWf5A_000005_000015 -> video9 via mapping
                video = f'{mapping[k]}.mp4'
                embs = np.asarray(db[k]) # (X, 1024) with a varied X
                ids = get_uniform_frame_ids(embs.shape[0], num_frames) # uniform sampling
                self.video2embs[video] = embs[ids] # (num_frames, 1024)                

        elif pickle_path is not None:
            data = pickle.load(open(pickle_path, 'rb'))
            self.video2embs = {line['image']: line['image_emb'] for line in data}
        
        self.transform = transform
        self.video_root = video_root
        self.num_frames = num_frames
        self.max_words = max_words      
        self.prompt = prompt

    def get_item_based_on_id(self, id):
        for i, ann in enumerate(self.annotation):
            if ann['image_id'] == id:
                return self.__getitem__(i)
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        ann = self.annotation[index]
        
        out = {'image_id': ann['image_id']}

        if 'caption' in ann: 
            if type(ann['caption']) is str:
                # for the training set, ann['caption'] is a string
                out['caption'] = self.prompt + pre_caption(ann['caption'], self.max_words)
            else:
                # for the validation / testing set, ann['caption'] is a list of string
                out['caption'] = [self.prompt + pre_caption(caption, max_words=10000) for caption in ann['caption']]

        if not hasattr(self, 'video2embs'):
            import decord
            video_path = os.path.join(self.video_root, ann['image'])
            reader = decord.VideoReader(video_path)
            frames = reader.get_batch(get_uniform_frame_ids(len(reader), self.num_frames)).asnumpy()
            frames = [Image.fromarray(frame) for frame in frames]
            out['frames'] = frames
            # (num_frames, height, width, 3)
            out['image'] = torch.stack([self.transform(frame) for frame in frames], dim=0)

        if hasattr(self, 'video2embs'):
            out['clip_image_embs'] = self.video2embs[ann['image']]
            #assert out['clip_image_embs'].size(0) == self.num_frames
        
        lang = ann.get('lang', configs.main_lang) # default to en (English)
        out['lang'] = torch.LongTensor([configs.lang2code[lang]])

        return out
    
    def collate_fn(self, batch):
        # batch contains bsz items, each of which is a dict output from self.__iter__()
        batch_tensors = {}
        for key in batch[0].keys():
            x = [b[key] for b in batch]
            if key in ['caption', 'frames']:
                batch_tensors[key] = x
                continue
            if x[0] is None:
                batch_tensors[key] = None
            elif isinstance(x[0], torch.Tensor):
                batch_tensors[key] = torch.stack(x)
            elif isinstance(x[0], np.ndarray):
                batch_tensors[key] = torch.FloatTensor(np.array(x))
            elif type(x[0]) is Image:
                batch_tensors[key] = x
            else:
                batch_tensors[key] = torch.tensor(x, dtype=torch.long)
        
        return batch_tensors


def get_uniform_frame_ids(num_total_frames, num_frames):
    if num_total_frames <= num_frames:
        frame_ids = [_ for _ in range(num_total_frames)]
        frame_ids = frame_ids + [frame_ids[-1]] * (num_frames - num_total_frames)
        return frame_ids

    # there will be num_frames intervals
    ids = np.linspace(0, num_total_frames, num_frames + 1)
    frame_ids = []
    for i in range(num_frames):
        # get the middle frame index of each interval
        frame_ids.append(round((ids[i] + ids[i+1]) / 2))
    return frame_ids
