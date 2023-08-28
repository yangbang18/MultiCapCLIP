import os
import sys
REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(REPO)

import json
from collections import defaultdict, Counter
import argparse
import configs
import pandas as pd
import spacy
import gzip
from tqdm import tqdm


def concept_preparation(train_captions, dataset, source_lang=configs.main_lang, target_lang=None, topk=1000):
    if target_lang is None:
        target_lang = source_lang

    print(f'Parse English captions to get the most frequent {topk} concepts')
    save_path = os.path.join(configs.concepts_root, dataset, target_lang)
    os.makedirs(save_path, exist_ok=True)

    read_path = os.path.join(configs.concepts_root, dataset, source_lang)
    os.makedirs(read_path, exist_ok=True)
    statistics_nouns_path = os.path.join(read_path, f'statistics_nouns.json')
    statistics_verbs_path = os.path.join(read_path, f'statistics_verbs.json')

    if os.path.exists(statistics_nouns_path) and os.path.exists(statistics_verbs_path):
        print(f'Load existing statistics files from {statistics_nouns_path} and {statistics_verbs_path}')
        statistics_nouns = json.load(open(statistics_nouns_path))
        statistics_verbs = json.load(open(statistics_verbs_path))
    else:
        all_noun_chunks = []
        all_verbs = []
        nlp = spacy.load("en_core_web_sm")

        for caption in tqdm(train_captions):
            doc = nlp(caption.lower())
            for nc in doc.noun_chunks:
                all_noun_chunks.append(nc.text)
            
            verbs = [token.text for token in doc if token.pos_ == 'VERB']
            all_verbs.extend(verbs)
    
        statistics_nouns = dict(Counter(all_noun_chunks))
        statistics_verbs = dict(Counter(all_verbs))
        
        print(f'Save statistics files to {statistics_nouns_path} and {statistics_verbs_path}')
        json.dump(statistics_nouns, open(statistics_nouns_path, 'w'))
        json.dump(statistics_verbs, open(statistics_verbs_path, 'w'))

    candidate_nouns = sorted(statistics_nouns.items(), key=lambda x: -x[1])
    candidate_nouns = [item[0] for item in candidate_nouns[:topk]]
    print('-- Noun Phrases')
    print(candidate_nouns[:10])

    candidate_verbs = sorted(statistics_verbs.items(), key=lambda x: -x[1])
    candidate_verbs = [item[0] for item in candidate_verbs[:topk]]
    print('-- Verbs')
    print(candidate_verbs[:10])

    candidate_mix = Counter(statistics_nouns) + Counter(statistics_verbs)
    candidate_mix = sorted(candidate_mix.items(), key=lambda x: -x[1])
    candidate_mix = [item[0] for item in candidate_mix[:topk]]
    print('-- Noun Phrases + Verbs')
    print(candidate_mix[:10])
    
    cp = os.path.join(save_path, 'concepts.txt')
    vp = os.path.join(save_path, 'verbs.txt')
    mp = os.path.join(save_path, 'concepts_with_verbs.txt')
    print(f'Save results to\n\t{cp}\n\t{vp}\n\t{mp}')
    with open(cp, 'w') as f:
        f.write('\n'.join(candidate_nouns))
    with open(vp, 'w') as f:
        f.write('\n'.join(candidate_verbs))
    with open(mp, 'w') as f:
        f.write('\n'.join(candidate_mix))

        
def coco():
    dataset = 'coco'
    data_path = os.path.join(configs.finetune_root, dataset, 'en', 'train.json')
    data = json.load(open(data_path, 'r'))
    captions = [item['caption'].strip().replace('\n', ' ') for item in data]
    print(f'There are {len(captions)} training captions')

    os.makedirs(configs.corpus_root, exist_ok=True)
    with open(os.path.join(configs.corpus_root, f'{dataset}.txt'), 'w') as f:
        f.write('\n'.join(captions))

    concept_preparation(captions, dataset, source_lang='en', target_lang='en')


def flickr30k():
    dataset = 'flickr30k'
    data_path = os.path.join(configs.finetune_root, dataset, 'en', 'train.json')
    data = json.load(open(data_path, 'r'))
    captions = [item['caption'] for item in data]
    print(f'There are {len(captions)} training captions')

    os.makedirs(configs.corpus_root, exist_ok=True)
    with open(os.path.join(configs.corpus_root, f'{dataset}.txt'), 'w') as f:
        f.write('\n'.join(captions))
    
    concept_preparation(captions, dataset, source_lang='en', target_lang='en')

    flickr30k_de()
    flickr30k_fr()


def flickr30k_de():
    print('-' * 20)
    print('Prepare flickr30k_de')

    captions = pd.read_csv(f'{configs.corpus_root}/flickr30k_de.tsv', sep='\t')['en'].to_list()
    # English sentences of `en` and `fr` languages are different
    concept_preparation(captions, 'flickr30k', source_lang='de', target_lang='de')


def flickr30k_fr():
    print('-' * 20)
    print('Prepare flickr30k_fr')
    captions = {'en': [], 'fr': []}
    for lang in ['en', 'fr']:
        captions[lang].extend(open(f'{configs.finetune_root}/flickr30k/en-fr/train.{lang}', 'r').read().strip().split('\n'))
    
    tsv_data = [line for line in zip(captions['en'], captions['fr'])]
    print(f'There are {len(tsv_data)} training English-French pairs')

    df = pd.DataFrame(tsv_data, columns=['en', 'fr'])
    df.to_csv(f'{configs.corpus_root}/flickr30k_fr.tsv', sep='\t', index=False)
    # English sentences of `en` and `fr` languages are different
    concept_preparation(captions['en'], 'flickr30k', source_lang='fr', target_lang='fr')


def msrvtt():
    dataset = 'msrvtt'
    data_path = os.path.join(configs.finetune_root, dataset, 'videodatainfo_2016.json')
    data = json.load(open(data_path, 'r'))

    train_video_ids = []
    for item in data['videos']:
        if item['split'] == 'train':
            train_video_ids.append(item['video_id'])

    train_video_ids = set(train_video_ids)

    captions = []
    for item in data['sentences']:
        vid = item['video_id']
        caption = item['caption']
        if vid in train_video_ids:
            captions.append(caption)

    print(f'There are {len(captions)} training captions')

    os.makedirs(configs.corpus_root, exist_ok=True)
    with open(os.path.join(configs.corpus_root, f'{dataset}.txt'), 'w') as f:
        f.write('\n'.join(captions))

    concept_preparation(captions, dataset, source_lang='en', target_lang='en')
    msrvtt_zh(captions)


def msrvtt_zh(msrvtt_train_captions):
    print('-' * 20)
    print('Prepare msrvtt_zh')

    msrvtt_cn_path = 'data/MSRVTT-CN/msrvtt10kcntrain_google_enc2zh.caption.txt'
    assert os.path.exists(msrvtt_cn_path), msrvtt_cn_path
    data = open(msrvtt_cn_path, 'r').read().strip().split('\n')

    vid2Chinese_captions = defaultdict(list)
    for line in data:
        tag, *caption = line.split(' ')
        vid = int(tag.split('#')[0][5:])
        caption = ' '.join(caption)
        vid2Chinese_captions[vid].append(caption)

    data = json.load(open(f'{configs.finetune_root}/msrvtt/videodatainfo_2016.json', 'r'))

    splits = defaultdict(list)
    for item in data['videos']:
        # 'video1000' -> 1000
        vid = int(item['video_id'][5:])
        splits[item['split']].append(vid)

    for k in splits.keys():
        splits[k] = sorted(splits[k])

    splits['val'] = splits.pop('validate')

    vid2English_captions = defaultdict(list)
    for item in data['sentences']:
        vid = int(item['video_id'][5:])
        caption = item['caption']
        vid2English_captions[vid].append(caption)

    columns = ['en', 'zh']
    assert 'en' in configs.lang2code
    assert 'zh' in configs.lang2code
    tsv_data = []

    for vid in splits['train']:
        for enCap, zhCap in zip(vid2English_captions[vid], vid2Chinese_captions[vid]):
            tsv_data.append([enCap, zhCap])
    
    print(f'There are {len(tsv_data)} training English-Chinese pairs')
    
    df = pd.DataFrame(tsv_data, columns=columns)
    df.to_csv(f'{configs.corpus_root}/msrvtt_zh.tsv', sep='\t', index=False)

    train_captions = [line[0] for line in tsv_data]
    assert set(msrvtt_train_captions) == set(train_captions)
    # we directly copy concept files from `en` 
    # because English sentences of `en` and `zh` languages are identical
    concept_preparation(train_captions, 'msrvtt', source_lang='en', target_lang='zh')


def vatex():
    dataset = 'vatex'
    root = os.path.join(configs.finetune_root, dataset)
    lines = open(os.path.join(root, 'vatex_mapping.txt'), 'r').read().strip().split('\n')
    id2vid = {}
    for line in lines:
        id, vid = line.split(' ')
        id2vid[id] = vid # e.g., Ptf_2VRj-V0_000122_000132 -> video0

    existed_vids = set(open(os.path.join(root, 'vatex_existed_videos.txt'), 'r').read().strip().split('\n'))

    data = json.load(open(os.path.join(root, 'vatex_training_v1.0.json'), 'r'))
    columns = ['en', 'zh']
    assert 'en' in configs.lang2code
    assert 'zh' in configs.lang2code

    tsv_data = []
    failed_count = 0

    for item in data:
        vid = id2vid[item['videoID']]
        if vid not in existed_vids:
            # we do not use the annotations of those unaccessible videos
            failed_count += 1
            continue

        for i in range(10):
            tsv_data.append([item['enCap'][i], item['chCap'][i]])
    
    print(f'annotations of {failed_count} / {len(data)} videos are not used to generate the tsv corpus file')
    captions = [item[0] for item in tsv_data]
    print(f'There are {len(captions)} training captions')

    os.makedirs(configs.corpus_root, exist_ok=True)
    with open(os.path.join(configs.corpus_root, f'{dataset}.txt'), 'w') as f:
        f.write('\n'.join(captions))

    concept_preparation(captions, dataset, source_lang='en', target_lang='en')

    print('-' * 20)
    print('Prepare vatex_zh')
    print(f'There are {len(tsv_data)} training English-Chinese pairs')
    df = pd.DataFrame(tsv_data, columns=columns)
    df.to_csv(f'{configs.corpus_root}/{dataset}_zh.tsv', sep='\t', index=False)
    # we directly copy concept files from `en` 
    # because English sentences of `en` and `zh` languages are identical
    concept_preparation(captions, dataset, source_lang='en', target_lang='zh')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['coco', 'flickr30k', 'msrvtt', 'vatex'])
    args = parser.parse_args()
    
    globals()[args.dataset]()
