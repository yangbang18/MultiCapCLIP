import os
import sys
REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(REPO)

import torch
import argparse
import pickle
import pandas as pd

import configs

import clip
from tqdm import tqdm


SUPPORTED_EXTENSIONS = ['.txt', '.tsv']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('text_path', type=str)
    parser.add_argument('--feats_root', type=str, default='./data/feats')
    parser.add_argument('--feats_fn', type=str, help='if not specified, decide `feats_fn` by `feats_fn_format`')
    parser.add_argument('--feats_fn_format', type=str, default='{arch}/{text_fn}.pkl')

    parser.add_argument('--download_root', type=str, default='data/checkpoints')
    parser.add_argument('--arch', type=str, default='ViT-B/16', choices=clip.available_models())
    parser.add_argument('--no_cuda', default=False, action='store_true')
    parser.add_argument('--batch_size', type=int, default=512)
    args = parser.parse_args()

    print(f'=== Load text data from {args.text_path}')
    assert os.path.exists(args.text_path)
    assert os.path.isfile(args.text_path)

    if args.feats_fn:
        args.save_pickle_path = os.path.join(args.feats_root, args.feats_fn)
    else:
        text_fn = os.path.basename(args.text_path).split('.')[0]
        arch = args.arch.lower().replace('/', '-')
        args.save_pickle_path = os.path.join(
            args.feats_root, 
            args.feats_fn_format.format(text_fn=text_fn, arch=arch)
        )
    print(f'=== Save pickle data to {args.save_pickle_path}')
    if os.path.exists(args.save_pickle_path):
        print('The file exists! Exit now ...')
        sys.exit(0)
    
    os.makedirs(os.path.dirname(args.save_pickle_path), exist_ok=True)

    if args.no_cuda or not torch.cuda.is_available():
        device = 'cpu'
    else:
        device = 'cuda'

    print(f'=== Load CLIP model from {args.download_root}')
    model, preprocess = clip.load(args.arch, device=device, jit=False, download_root=args.download_root)
    model.float()
    model.eval()
    model.to(device)
    
    file_format = os.path.basename(args.text_path).split('.')[-1]
    assert file_format in ['txt', 'tsv'], file_format

    if file_format == 'txt':
        captions = open(args.text_path, 'r').read().strip().split('\n')
        parallel_data = [{} for _ in range(len(captions))]
    else:
        data = pd.read_csv(args.text_path, sep='\t', lineterminator='\n')
        assert configs.main_lang in data.columns
        captions = []
        parallel_data = []
        for i in range(len(data)):
            captions.append(data.iloc[i][configs.main_lang])
            parallel_data.append(data.iloc[i].to_dict())
    
    print(len(captions))
    
    steps = len(captions) // args.batch_size
    if steps * args.batch_size < len(captions):
        steps += 1

    out = []

    for i in tqdm(range(steps)):
        start = i * args.batch_size
        end = start + args.batch_size

        text_input = clip.tokenize(captions[start:end], truncate=True).to(device)

        with torch.no_grad():
            text_embs = model.encode_text(text_input)
        
        for caption, parallel, text_emb in zip(captions[start:end], parallel_data[start:end], text_embs):
            line = {'caption': caption, 'text_emb': text_emb.cpu().numpy(), **parallel}
            if file_format == 'tsv':
                assert line.pop('caption') == parallel['en']
            out.append(line)

    with open(args.save_pickle_path, 'wb') as wf:
        pickle.dump(out, wf)
