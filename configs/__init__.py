import os
import ruamel.yaml as yaml


image_video_root = {
    'coco': 'data/MSCOCO',
    'flickr30k': 'data/Flickr30k',
    'msrvtt': 'data/MSRVTT',
    'vatex': 'data/VATEX',
}

finetune_root = 'data/annotations'
feats_root = 'data/feats'
corpus_root = 'data/corpus'
related_caption_ids_root = 'data/related_caption_ids'
concepts_root = 'data/concepts'
corenlp_root = 'data/stanford-corenlp-4.5.2'

video_datasets = ['msrvtt', 'vatex']
num_frames = 8

main_lang='en'
lang2code = {
    'en': 0,    # English
    'zh': 1,    # Chinese
    'de': 2,    # German
    'fr': 3,    # French
}
# `max_languages` is to affect the type_vocab_size of LMs
# i.e., we use token type embeddings to identify which language to be generated.
# note that the max value in `lang2code` can not exceed `max_languages`
max_languages = 256 


def parse_method(method):
    assert type(method) is str, type(method)
    multilingual = False
    train_langs = [main_lang]
    eval_lang = main_lang
    train_en_prob = None

    if '#' in method:
        multilingual = True
        actual_method, string = method.split('#') # e.g., split "baseline#en,zh-zh" into "baseline" and "en,zh-zh"
        train_langs, eval_lang = string.split('-') # e.g., split "en,zh-zh" into "en,zh" and "zh"
        train_langs = train_langs.split(',')
        
        final_train_langs = []
        for lang in train_langs:
            if 'en' in lang and lang != 'en':
                train_en_prob = float(lang[3:]) # en^0.1 -> 0.1
                final_train_langs.append('en')
            else:
                assert lang in lang2code.keys()
                final_train_langs.append(lang)

        assert eval_lang in lang2code.keys()
    else:
        actual_method = method
        final_train_langs = train_langs
    
    return actual_method, multilingual, final_train_langs, eval_lang, train_en_prob


def get_method_config(key, config={}, yaml_path='configs/methods.yaml', yaml_data=None):
    if not key or key is None:
        return None

    assert yaml_path or yaml_data
    if yaml_data is None:
        yaml_data = yaml.load(open(yaml_path), Loader=yaml.Loader)
    
    assert key in yaml_data.keys(), f"`{key}` can not be found in {yaml_path}"

    specific_data = yaml_data[key]

    if 'inherit_from' in specific_data.keys():
        inherit_from = specific_data.pop('inherit_from')
        if not isinstance(inherit_from, list):
            inherit_from = [inherit_from]

        for new_key in inherit_from:
            config = get_method_config(key=new_key, config=config, yaml_path=yaml_path, yaml_data=yaml_data)

    for k, v in specific_data.items():
        config[k] = v
    
    return config
   

def create_config(args, mode='adapt'):
    assert hasattr(args, 'method')
    assert hasattr(args, 'dataset')

    if getattr(args, 'from_pretrained', None):
        pretrained_config_path = os.path.join(args.from_pretrained, 'config.yaml')
        pretrained_config = yaml.load(open(pretrained_config_path, 'r'), Loader=yaml.Loader)
    else:
        pretrained_config = {}

    assert mode in ['adapt', 'finetune']

    # ==== Basic Config =====
    basic_config_path = getattr(args, 'config', None) or f'configs/{mode}.yaml'
    basic_config = yaml.load(open(basic_config_path, 'r'), Loader=yaml.Loader)
    basic_config['clip_arch'] = pretrained_config.get('clip_arch', basic_config['clip_arch'])
    if getattr(args, 'fewshot', None):
        basic_config['optimizer']['lr'] = basic_config['schedular']['lr'] = 1e-5
        basic_config['schedular']['num_warmup_steps'] = 0
    
    original_method = getattr(args, 'method', None) or pretrained_config['method']
    assert original_method is not None
    method, multilingual, train_langs, eval_lang, train_en_prob = parse_method(original_method)
    
    if eval_lang == 'zh':
        basic_config['max_length'] = 30 # the maximun number of tokens during inference

    if multilingual:
        basic_config['text_model'] = 'data/checkpoints/bert-base-multilingual-cased'
        basic_config['multilingual'] = True
    else:
        basic_config['text_model'] = 'data/checkpoints/bert-base-uncased'

    basic_config['train_langs'] = train_langs
    basic_config['eval_lang'] = eval_lang
    basic_config['train_en_prob'] = train_en_prob

    # ==== Method Config =====
    method_config = get_method_config(key=method)
    method_config['method'] = original_method
    print(method_config)
    
    # ==== Data Config =====
    data_config = {'dataset': args.dataset}
    if args.dataset in video_datasets:
        data_config['num_frames'] = num_frames

    if method_config.get('src_lang', main_lang) != main_lang or method_config.get('multilingual_clip', False):
        setattr(args, 'pickle', False)
    
    if method_config.get('parallel_prefixes', None):
        data_config['parallel_text_paths'] = [
            os.path.join(corpus_root, f'{prefix}_{args.dataset}.txt')
            for prefix in method_config['parallel_prefixes']
        ]

    postfix = f'_{eval_lang}' if eval_lang != main_lang else ''

    if mode == 'adapt':
        clip_arch = getattr(args, 'clip_arch') or method_config.get('clip_arch', None) or basic_config['clip_arch']
        clip_arch = clip_arch.lower().replace('/', '-')

        # where to load the corpus for CLIP-based autoencoding
        flag = False
        if getattr(args, 'pickle', False):
            data_config['data_path'] = getattr(args, 'data_path', None) or os.path.join(feats_root, f'{clip_arch}/{args.dataset}{postfix}.pkl')
            if os.path.exists(data_config['data_path']):
                flag = True

        if not flag:
            if postfix:
                data_config['data_path'] = getattr(args, 'data_path', None) or os.path.join(corpus_root, f'{args.dataset}{postfix}.tsv')
            else:
                data_config['data_path'] = getattr(args, 'data_path', None) or os.path.join(corpus_root, f'{args.dataset}.txt')
            assert os.path.exists(data_config['data_path']), data_config['data_path']
        
        # where to load related caption ids
        if method_config.get('with_related_caption_as_input', False):
            # if this file does not exist, the code will generate and save it
            data_config['related_caption_ids_path'] = os.path.join(related_caption_ids_root, f'{clip_arch}/{args.dataset}{postfix}.txt')
            data_config['auto_generate_related_caption_ids'] = True
    else:
        # the following files are for finetuning or evaluation
        for key, fn in zip(
            ['train_file', 'val_file', 'test_file', 'val_gt_file', 'test_gt_file'],
            ['train.json', 'val.json', 'test.json', 'val_gt.json', 'test_gt.json']):
            data_config[key] = getattr(args, key, None) or os.path.join(finetune_root, args.dataset, eval_lang, fn)
        
        # reduce the size of the training data
        if getattr(args, 'subset_fn', None) is not None:
            data_config['train_file'] = os.path.join(finetune_root, args.dataset, eval_lang, 'subsets', args.subset_fn)

        # speedup training if specified
        if getattr(args, 'pickle', False):
            clip_arch = getattr(args, 'clip_arch', None)  or basic_config['clip_arch']
            clip_arch = clip_arch.lower().replace('/', '-')
            data_config['pickle_path'] = getattr(args, 'pickle_path', None) or os.path.join(feats_root, f'{clip_arch}_image/{args.dataset}.pkl')
            if not os.path.exists(data_config['pickle_path']):
                # avoid bug if the pickle file does not exist
                data_config['pickle_path'] = None

        # where to load images or videos
        data_config['root'] = getattr(args, 'root', None) or image_video_root[args.dataset]
    
    if method_config.get('concepts', False):
        # If we have adapted the model on a specific corpus and its concepts
        # we do not change the concepts related to args.dataset
        data_config['concepts_path'] = pretrained_config.get('concepts_path', None) \
            or os.path.join(concepts_root, args.dataset, eval_lang, method_config.get('concept_fn', 'concepts.txt'))
        assert os.path.exists(data_config['concepts_path']), data_config['concepts_path']
    
    config = {**basic_config, **method_config, **data_config}

    for key in getattr(args, 'keys_to_override', []):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value

    return config
