import os
import sys
REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(REPO)

import torch
import argparse
import pickle
import json
import decord

import clip
from tqdm import tqdm
from PIL import Image

import configs
from datasets.finetune_dataset import get_uniform_frame_ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['coco', 'msrvtt', 'vatex', 'flickr30k'])
    parser.add_argument('--feats_root', type=str, default='./data/feats')
    parser.add_argument('--feats_fn', type=str, help='if not specified, decide `feats_fn` by `feats_fn_format`')
    parser.add_argument('--feats_fn_format', type=str, default='{arch}_image/{dataset}.pkl')
    parser.add_argument('--lang', type=str, default='en', help='load annotations from which language to extract image feats')

    parser.add_argument('--download_root', type=str, default='data/checkpoints')
    parser.add_argument('--arch', type=str, default='ViT-B/16', choices=clip.available_models())
    parser.add_argument('--no_cuda', default=False, action='store_true')
    parser.add_argument('--batch_size', type=int, default=10)
    args = parser.parse_args()

    if args.feats_fn:
        args.save_pickle_path = os.path.join(args.feats_root, args.feats_fn)
    else:
        arch = args.arch.lower().replace('/', '-')
        args.save_pickle_path = os.path.join(
            args.feats_root, 
            args.feats_fn_format.format(arch=arch, dataset=args.dataset)
        )
    print(f'=== Save pickle data to {args.save_pickle_path}')
    if os.path.exists(args.save_pickle_path):
        print('The file exists! Exit now ...')
        sys.exit(0)

    os.makedirs(os.path.dirname(args.save_pickle_path), exist_ok=True)

    key = 'image'
    image_to_fullpath = {}
    for fn in ['train.json', 'val.json', 'test.json']:
        jp = os.path.join(configs.finetune_root, args.dataset, args.lang, fn)
        print(f'=== Load json_data from {jp}')
        data = json.load(open(jp, 'r'))
        for item in data:
            image_to_fullpath[item[key]] = os.path.join(configs.image_video_root[args.dataset], item[key])

    images = list(image_to_fullpath.keys())
    image_paths = [image_to_fullpath[key] for key in images]
    print(f'=== There are {len(image_paths)} {key}s')
    
    if args.no_cuda or not torch.cuda.is_available():
        device = 'cpu'
    else:
        device = 'cuda'

    print(f'=== Load CLIP model from {args.download_root}')
    model, preprocess = clip.load(args.arch, device=device, jit=False, download_root=args.download_root)
    model.float()
    model.eval()
    model.to(device)
    
    steps = len(image_paths) // args.batch_size
    if steps * args.batch_size < len(image_paths):
        steps += 1

    out = []

    for i in tqdm(range(steps)):
        start = i * args.batch_size
        end = start + args.batch_size

        if args.dataset in configs.video_datasets:
            ims = []
            for p in image_paths[start:end]:
                reader = decord.VideoReader(p)
                frames = reader.get_batch(get_uniform_frame_ids(len(reader), configs.num_frames)).asnumpy()
                frames = [preprocess(Image.fromarray(frame)) for frame in frames]
                frames = torch.stack(frames, dim=0)
                ims.append(frames)

            ims = torch.stack(ims, dim=0).to(device)
            B, T, C, H, W = ims.shape
            ims = ims.view(B * T, C, H, W)

            with torch.no_grad():
                image_embs = model.encode_image(ims)
                image_embs = image_embs.view(B, T, -1)
        else:
            ims = []
            for p in image_paths[start:end]:
                im = Image.open(p).convert('RGB')
                im = preprocess(im)
                ims.append(im)
            ims = torch.stack(ims, dim=0).to(device)
            
            with torch.no_grad():
                image_embs = model.encode_image(ims)
            
        for image, image_emb in zip(images[start:end], image_embs):
            line = {f'{key}': image, f'{key}_emb': image_emb.cpu().numpy()}
            out.append(line)

    with open(args.save_pickle_path, 'wb') as wf:
        pickle.dump(out, wf)
