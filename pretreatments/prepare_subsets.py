import os
import sys
REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(REPO)

import argparse
import json
from collections import defaultdict
import random
import configs


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='coco', choices=configs.image_video_root.keys())
parser.add_argument('--lang', type=str, default='en')
parser.add_argument('--ratios', type=float, nargs='+', default=[0.01, 0.1, 1, 10])
parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
parser.add_argument('--save_format', type=str, default='{finetune_root}/{dataset}/{lang}/subsets')
args = parser.parse_args()

key = 'image'
train_file = os.path.join(configs.finetune_root, args.dataset, args.lang, f'train.json')

print('### load json path from', train_file)
data = json.load(open(train_file, 'r'))

id2item = defaultdict(list)
for item in data:
    id2item[item[f'{key}_id']].append(item)

print(f'### there are {len(id2item)} unique {key}s, {len(data)} {key}-caption pairs')

ids = sorted(list(id2item.keys()))

save_path = args.save_format.format(finetune_root=configs.finetune_root, dataset=args.dataset, lang=args.lang)
os.makedirs(save_path, exist_ok=True)

for ratio in args.ratios:
    n_unique_images = int(len(id2item) * ratio / 100)
    if n_unique_images < 1:
        print(f'{ratio} is not applicible')
        continue

    print(f'--- generating a training subset of {ratio}% ({n_unique_images}) unique {key}s')

    for seed in args.seeds:
        json_path = os.path.join(save_path, f'{ratio}%_{seed}.json')
        if os.path.exists(json_path):
            print(json_path, 'exists')
            continue
        
        random.seed(seed)
        
        this_ids = random.sample(ids, n_unique_images)
        this_data = []
        for this_id in this_ids:
            this_data.extend(id2item[this_id])

        print(json_path)
        with open(json_path, 'w') as wf:
            json.dump(this_data, wf)
        
