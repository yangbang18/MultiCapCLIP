# MultiCapCLIP

Data used in our ACL'23 paper:
> **MultiCapCLIP: Auto-Encoding Prompts for Zero-Shot Multilingual Visual Captioning**
> 
> Bang Yang, Fenglin Liu, Xian Wu, Yaowei Wang, Xu Sun, and Yuexian Zou
>
> [ACL Anthology](https://aclanthology.org/2023.acl-long.664/), [arXiv](http://arxiv.org/abs/2308.13218)

## Data
Our data follows the structure shown below:
```
MultiCapCLIP/
    data
    ├── checkpoints                 # off-the-shelf models
    │   ├── ViT-B-16.pt
    │   ├── bert-base-multilingual-cased
    │   │   ├── ...
    │   │   └── vocab.txt
    │   └── bert-base-uncased                       
    │       ├── config.json
    │       ├── pytorch_model.bin
    │       ├── tokenizer.json
    │       ├── tokenizer_config.json
    │       └── vocab.txt
    ├── annotations                 # for evaluations or supervised training (finetuning)
    │   └── $dataset   
    │       └── $lang               # annotations in a specific language
    │           ├── subsets         # for semi-supervised training
    │           │    ├─ 0.1%_0.json # a 0.1% subset of train.json 
    │           │    ├─ 0.1%_1.json # different seed
    │           │    ├─ 0.1%_2.json # totally 3 seeds
    │           │    ├─ ...
    │           │    └─ 10%_2.json  # a 10% subset of train.json 
    │           ├── train.json          
    │           ├── val.json
    │           ├── val_gt.json
    │           ├── test.json     
    │           └── test_gt.json  
    ├── corpus                      # for text-only training
    │   ├── $dataset.txt            # one caption per line
    │   └── $dataset_$lang.tsv      # English-$lang pairs (separated by `\t`) per line
    ├── feats                       # speedup training and inference
    │   ├── vit-b-16                # CLIP's ViT-B/16 English text embeddings
    │   │   └── ...
    │   └── vit-b-16_image          # CLIP's ViT-B/16 image embeddings
    │       └── ...
    ├── concepts                    # for concept prompting
    │   └── ...
    ├── related_caption_ids         # indexes of captions of similar semantics
    │   └── vit-b-16
    │       └── ...
    ├── stanford-corenlp-4.5.2      # segment English, German, and French sentences
    │   └── ...               
    └── ...                         # folders that store raw images and videos
```

**You can download our full data from [Google Drive](https://drive.google.com/drive/folders/1evjzu4aH8DveQzKDIptp0D2BHRX3KEwY?usp=sharing) or [Baidu网盘](https://pan.baidu.com/s/1vnogVEnfX33rIJ3HzQjaEA)(extract code: `huk0`)**


The `data` folder contains the following subfolders:
- The `checkpoints` folder contains pre-trained weights, configs, and vocab files of off-the-shelf models (e.g., CLIP's [ViT-B-16.pt](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt), huggingface's [bert-base-uncased](https://huggingface.co/bert-base-uncased) and [bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased)). We download these files in adavance to avoid network issues and set related configureations in `configs/adapt.yaml` and `configs/finetune.yaml`.
- The `annotations` folder contains many subfolders named with dataset names, where the training, validation, and testing json files for supervised finetuning are included. Note that each json file is a list of dictionaries and each of dictionary looks like, e.g., *{image: path_relative_to_the_root, caption: caption_of_this_image}*. Please refer to [ZeroNLG/data](https://github.com/yangbang18/ZeroNLG/tree/master/data) for more details.
- The `corpus` folder contains `.txt` or `.tsv` files that store the (parallel) corpus for CLIP-based autoencoding or translating. 
- The `feats` folder contains `.pkl` files that store the features of English texts and images/videos.
- The `concepts` folder contains concept files (`.txt`) extracted from the English captions of each dataset and language.
- The `related_caption_ids` folder records indexes of captions of similar semantics for each caption. 
- The `stanford-corenlp-4.5.2` folder has files for segmenting English, German, and French sentences. See [utils/eval.py](/utils/eval.py) for details.
- Other folders that stores the raw images or videos, e.g., `data/MSCOCO/train2014/*.jpg` (see the variable `image_video_root` in [configs](/configs/__init__.py) and the below structure).
    ```
    data
    ├── MSCOCO
    │   ├── train2014
    │   │   └── *.jpg
    │   └── val2014
    │       └── *.jpg
    ├── Flickr30k
    │   └── flickr30k-images   
    │       └── *.jpg
    ├── MSRVTT
    │   └── all_videos   
    │       ├── video0.mp4
    │       ├── ...
    │       └── video9999.mp4
    └── VATEX
        └── all_videos   
            ├── video0.mp4
            ├── ...
            └── video34990.mp4
    ```

 Here are official or shared links to download raw images or videos: 
    <div align="center">
<table border="1" width="100%">
    <tr align="center">
        <th>Datasets</th><th>Official Link</th><th>Shared Link (Others)</th><th>Shared Link (Ours)</th>
    </tr>
    <tr align="center">
        <td>MSCOCO</td><td><a href="https://cocodataset.org/">Link</a></td><td><a href="https://github.com/OFA-Sys/ONE-PEACE/blob/main/datasets.md">Link</a></td><td>N/A</td>
    </tr>
    <tr align="center">
        <td>Flickr30k</td><td><a href="http://shannon.cs.illinois.edu/DenotationGraph/data/index.html">Link</a></td><td><a href="https://github.com/OFA-Sys/ONE-PEACE/blob/main/datasets.md">Link</a></td><td>N/A</td>
    </tr>
    <tr align="center">
        <td>MSRVTT</td><td><a href="http://ms-multimedia-challenge.com/2016/dataset">Link (expired)</a></td><td><a href="https://www.mediafire.com/folder/h14iarbs62e7p/shared">Link</a></td><td>N/A</td>
    </tr>
    <tr align="center">
        <td>VATEX</td><td><a href="https://eric-xw.github.io/vatex-website/download.html">Link</a></td><td>N/A</td><td><a href="https://pkueducn-my.sharepoint.com/:u:/g/personal/2101112290_pkueducn_onmicrosoft_com/EbznKwMvV-1FsxxxRvbiu1cB5aC-NTspM1y5zkyJq6rZSQ?e=IcpHpT">Onedrive</a>, <a href="https://disk.pku.edu.cn:443/link/24892F356463CB7AC6B762ACC7757035">PKU Yun</a> (37.3G)</td>
    </tr>
</table>
</div>

**Note:** 
- After downloading, please reformat the data to follow the above structure. 
- Considering the difficulties to download raw VATEX videos, we share them!
- The official train/val/test splits of VATEX is 25,991: 3,000: 6,000. However, some video clips are no longer available, resulting in the splits 25,006: 2,893: 5,792 (in our case). The same splits are used in our other papers, check them out if you are interested ✨: 

    > [**Concept-Aware Video Captioning: Describing Videos With Effective Prior Information**](https://ieeexplore.ieee.org/abstract/document/10233200/)<br>
    > Accepted by IEEE Transactions on Image Processing | [[Code]](https://github.com/yangbang18/CARE)<br>
    > Bang Yang, Meng Cao and Yuexian Zou

    > [**CLIP Meets Video Captioning: Concept-Aware Representation Learning Does Matter**](https://arxiv.org/abs/2111.15162)<br>
    > Accepted by PRCV 2022 | [[Code]](https://github.com/yangbang18/CLIP-Captioner)<br>
    > Bang Yang, Tong Zhang and Yuexian Zou


## From-Scratch Preparation
### 1. Follow [ZeroNLG/data](https://github.com/yangbang18/ZeroNLG/tree/master/data) to prepare `annotations`.

### 2. Download `MSRVTT-CN` that contains translated Chinese captions from [HuiGuanLab/nrccr](https://github.com/HuiGuanLab/nrccr) 
Note: If the original link is expired, you can download `MSRVTT-CN` from our link (given above).

### 3. Prepare Corpus
Download `en_core_web_sm` for concept extraction. Note that we use the version 3.4.1.
```
wget https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.4.1/en_core_web_sm-3.4.1-py3-none-any.whl

pip install en_core_web_sm-3.4.1-py3-none-any.whl
```
Then run
```
python pretreatments/prepare_corpus.py --dataset coco
python pretreatments/prepare_corpus.py --dataset msrvtt
python pretreatments/prepare_corpus.py --dataset vatex
python pretreatments/prepare_corpus.py --dataset flickr30k
```
**Note**: 
- You should download `data/corpus/flickr30k_de.tsv` from our link before this step.
- We find that the 145K English and German training captions in Multi30K's `task2` are not one-to-one mappings. Therefore, we obtain `data/corpus/flickr30k_de.tsv` by translating German captions into English captions via `Google Translate`. 
- We do not use 29K English-German training pairs in Multi30K's `task1` because its scale is smaller than 145K. For fair comparisons with fully-supervised models trained on 145K image-German pairs, we carry out text-only training on the same scale of texts.
- This step will yield the `concepts` folder except corpora.


### 4. Prepare subsets
Generate subsets of size 0.01% (if applicable), 0.1%, 1%, 10% using three different seeds for semi-supervised training:
```
python pretreatments/prepare_subsets.py --dataset coco
python pretreatments/prepare_subsets.py --dataset msrvtt
```
We highly recommend you to use the **same** subsets (given in the above download links) as ours for fair comparisons.


### 5. Prepare features
Extract English text embeddings in adavnace as follows to avoid extracting them from the frozen CLIP on-the-fly:
```
python pretreatments/extract_text_embs.py data/corpus/coco.txt
python pretreatments/extract_text_embs.py data/corpus/msrvtt.txt
python pretreatments/extract_text_embs.py data/corpus/msrvtt_zh.tsv
python pretreatments/extract_text_embs.py data/corpus/vatex.txt
python pretreatments/extract_text_embs.py data/corpus/vatex_zh.tsv
python pretreatments/extract_text_embs.py data/corpus/flickr30k.txt
python pretreatments/extract_text_embs.py data/corpus/flickr30k_de.tsv
python pretreatments/extract_text_embs.py data/corpus/flickr30k_fr.tsv
```

Extract image embeddings in adavnace as follows to avoid extracting them from the frozen CLIP on-the-fly:
```
python pretreatments/extract_image_embs.py --dataset coco
python pretreatments/extract_image_embs.py --dataset msrvtt
python pretreatments/extract_image_embs.py --dataset vatex
python pretreatments/extract_image_embs.py --dataset flickr30k
```

### 6. The `related_caption_ids` will be generated automatically when running our method.
You can download it from our link to save time. 

### 7. The `stanford-corenlp-4.5.2` will be downloaded automatically during evaluation.
You can download it from our link to avoid network issues. 

## Citation
Please **[★star]** this repo and **[cite]** the following papers if you feel our data useful to your research:

```
@inproceedings{yang-etal-2023-multicapclip,
    title = "{M}ulti{C}ap{CLIP}: Auto-Encoding Prompts for Zero-Shot Multilingual Visual Captioning",
    author = "Yang, Bang and Liu, Fenglin and Wu, Xian and Wang, Yaowei and Sun, Xu and Zou, Yuexian",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    year = "2023",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.664",
    doi = "10.18653/v1/2023.acl-long.664",
    pages = "11908--11922",
}

@article{Yang2023ZeroNLG,
   title={{Z}ero{NLG}: Aligning and Autoencoding Domains for Zero-Shot Multimodal and Multilingual Natural Language Generation},
   author={Yang, Bang and Liu, Fenglin and Zou, Yuexian and Wu, Xian and Wang, Yaowei and Clifton, David A.},
   journal={arXiv preprint arXiv:2303.06458}
   year={2023}
}
```
