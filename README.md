# MultiCapCLIP

PyTroch implementation of our ACL'23 paper:
> **MultiCapCLIP: Auto-Encoding Prompts for Zero-Shot Multilingual Visual Captioning**
> 
> Bang Yang, Fenglin Liu, Xian Wu, Yaowei Wang, Xu Sun, and Yuexian Zou
>
> [ACL Anthology](https://aclanthology.org/2023.acl-long.664/), [arXiv](http://arxiv.org/abs/2308.13218)


## TOC

- [MultiCapCLIP](#multicapclip)
  - [TOC](#toc)
  - [Update Notes](#update-notes)
  - [Environment](#environment)
  - [Data](#data)
  - [Quick Start](#quick-start)
    - [Monolingual Scenario (i.e., English -\> English)](#monolingual-scenario-ie-english---english)
    - [Multilingual Scenario (i.e., English -\> `X`)](#multilingual-scenario-ie-english---x)
    - [Show Results](#show-results)
  - [Reproducibility](#reproducibility)
  - [Citation](#citation)
  - [Acknowledgements](#acknowledgements)


## Update Notes
**[2023-10-22]** Update links for downloading raw images and videos in [README_DATA.md](/README_DATA.md).

**[2023-08-28]** We release the code and data.


## Environment
We run the code based on `Python` 3.8.8, `torch` 1.13.1, and `cuda` 11.7. Please change the version of torch and cuda according to your hardwares.
```
git clone https://github.com/yangbang18/MultiCapCLIP.git
cd MultiCapCLIP

conda create -n zerovc python==3.8.8
conda activate zerovc

# Install a proper version of torch, e.g.:
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117  -f https://download.pytorch.org/whl/cu117/torch_stable.html

pip install -r requirement.txt
```


## Data
Please prepare the data following [README_DATA.md](/README_DATA.md).


## Quick Start
All experiments can be conducted with [scripts/pipe.sh](/scripts/pipe.sh). 
### Monolingual Scenario (i.e., English -> English)
```
train_dataset="msrvtt"
method="baseline"
tasks="adapt adapt_zeroshot"
val_datasets="coco msrvtt"

bash scripts/pipe.sh $train_dataset $method "$tasks" "$val_datasets"
```
In the above command, we make the baseline model auto-encode on MSR-VTT's training captions first (`adapt`), and then evaluate on MS-COCO (out-of-domain) and MSR-VTT (in-donmain) (`adapt_zeroshot`).

**Key Arguments:**
- `train_dataset` supports one of [`coco`, `msrvtt`, `vatex`, `flickr30k`].
- `method` is defined in [configs/methods.yaml](/configs/methods.yaml).
- `tasks` can be combinations of:
  - `finetune`: fully-supervised training, where a model will be trained on 100% training captions.
  - `finetune_fewshot`: fully-supervised training, where a model will be trained on 0.01% (if applicable), 0.1%, 1%, 10% training captions.
  - `adapt`: text-only autoencoding. This task is only responsible for training.
  - `adapt_zeroshot`: evaluate the model that has done `adapt` on a specific dataset.
  - `adapt_fewshot`: use the model that has done `adapt` as a starting point, train the model on 0.01% (if applicable), 0.1%, 1%, 10% training captions. This task is equivalent to semi-supervised learning in the paper.
- `val_datasets` can be combinations of [`coco`, `msrvtt`, `vatex`, `flickr30k`].

### Multilingual Scenario (i.e., English -> `X`)
For the `msrvtt` and `vatex` datasets, `X` denotes Chinese (`zh`). For the `flickr30k` dataset, `X` can be German (`de`) and French (`fr`).
```
train_dataset="flickr30k"
method="baseline#de-de"
tasks="adapt adapt_zeroshot"
val_datasets="flickr30k"

bash scripts/pipe.sh $train_dataset $method "$tasks" "$val_datasets"
```
Different from the monolingual command, we append a postfix `#A-B` to the method:
- `#` activates the multilingual mode, where we use `bert-base-multilingual-cased`'s vocab rather than that of `bert-base-uncased` to embed tokens. See [configs](/configs/__init__.py) for more details.
- `A` denotes which language(s) to be generated during training. For example, when we train model on English-German pairs, we can set `A` to `de` (the model only uses German texts as targets) or `en,de` (the model uses both English and German texts as targets).
- `B` denotes which language to be generated during evaluation. For example, `#A-de` means we require generating German texts during evaluation.

### Show Results
You can run the following command to gather results, where mean metric scores with their standard deviation across a number of runs are reported.
```
python show.py --root output/finetune --csv_path results/ --csv_fn finetune.csv
python show.py --root output/adapt --csv_path results/ --csv_fn adapt.csv
```

## Reproducibility

<details>
<summary>Main</summary>

```
bash scripts/pipe.sh coco baseline "finetune"
bash scripts/pipe.sh msrvtt baseline "finetune"
bash scripts/pipe.sh vatex baseline#zh-zh "finetune"

bash scripts/pipe.sh coco baseline "adapt adapt_zeroshot" "coco msrvtt"
bash scripts/pipe.sh msrvtt baseline "adapt adapt_zeroshot" "coco msrvtt"
bash scripts/pipe.sh msrvtt baseline#zh-zh "adapt adapt_zeroshot" "vatex"
bash scripts/pipe.sh vatex baseline#zh-zh "adapt adapt_zeroshot" "vatex"

bash scripts/pipe.sh coco MultiCapCLIP_001 "adapt adapt_zeroshot" "msrvtt"
bash scripts/pipe.sh coco MultiCapCLIP_01 "adapt adapt_zeroshot" "coco"
bash scripts/pipe.sh msrvtt MultiCapCLIP_001 "adapt adapt_zeroshot" "coco msrvtt"
bash scripts/pipe.sh msrvtt MultiCapCLIP_001#zh-zh "adapt adapt_zeroshot" "vatex"
bash scripts/pipe.sh vatex MultiCapCLIP_001#zh-zh "adapt adapt_zeroshot" "vatex"
```
</details>



<details>
<summary>Semi-Supervised Training</summary>

```
bash scripts/pipe.sh coco baseline "finetune_fewshot"
bash scripts/pipe.sh msrvtt baseline "finetune_fewshot"
bash scripts/pipe.sh msrvtt MultiCapCLIP_001 "adapt adapt_zeroshot adapt_fewshot" "coco msrvtt"
bash scripts/pipe.sh coco MultiCapCLIP_01 "adapt adapt_zeroshot adapt_fewshot" "coco"
bash scripts/pipe.sh coco MultiCapCLIP_001 "adapt adapt_zeroshot adapt_fewshot" "msrvtt"
```
</details>



<details>
<summary>Ablation Study</summary>

```
bash scripts/pipe.sh msrvtt baseline "adapt adapt_zeroshot" "coco"
bash scripts/pipe.sh msrvtt base_CP "adapt adapt_zeroshot" "coco"
bash scripts/pipe.sh msrvtt base_IA "adapt adapt_zeroshot" "coco"
bash scripts/pipe.sh msrvtt base_FA_001 "adapt adapt_zeroshot" "coco"
bash scripts/pipe.sh msrvtt base_IA_FA_001 "adapt adapt_zeroshot" "coco"
bash scripts/pipe.sh msrvtt MultiCapCLIP_001_K4 "adapt adapt_zeroshot" "coco"
bash scripts/pipe.sh msrvtt MultiCapCLIP_001_K8 "adapt adapt_zeroshot" "coco"
bash scripts/pipe.sh msrvtt MultiCapCLIP_001 "adapt adapt_zeroshot" "coco"
bash scripts/pipe.sh msrvtt MultiCapCLIP_001_K32 "adapt adapt_zeroshot" "coco"
bash scripts/pipe.sh msrvtt MultiCapCLIP_001_V "adapt adapt_zeroshot" "coco"
bash scripts/pipe.sh msrvtt MultiCapCLIP_001_NV "adapt adapt_zeroshot" "coco"

bash scripts/pipe.sh coco baseline "adapt adapt_zeroshot" "coco"
bash scripts/pipe.sh coco base_CP "adapt adapt_zeroshot" "coco"
bash scripts/pipe.sh coco base_IA "adapt adapt_zeroshot" "coco"
bash scripts/pipe.sh coco base_FA_01 "adapt adapt_zeroshot" "coco"
bash scripts/pipe.sh coco base_IA_FA_01 "adapt adapt_zeroshot" "coco"
bash scripts/pipe.sh coco MultiCapCLIP_01_K4 "adapt adapt_zeroshot" "coco"
bash scripts/pipe.sh coco MultiCapCLIP_01_K8 "adapt adapt_zeroshot" "coco"
bash scripts/pipe.sh coco MultiCapCLIP_01 "adapt adapt_zeroshot" "coco"
bash scripts/pipe.sh coco MultiCapCLIP_01_K32 "adapt adapt_zeroshot" "coco"
bash scripts/pipe.sh coco MultiCapCLIP_01_V "adapt adapt_zeroshot" "coco"
bash scripts/pipe.sh coco MultiCapCLIP_01_NV "adapt adapt_zeroshot" "coco"
```
</details>



<details>
<summary>Extentions to German and French Languages</summary>

```
bash scripts/pipe.sh flickr30k baseline#de-de "finetune"
bash scripts/pipe.sh flickr30k baseline#de-de "adapt adapt_zeroshot"
bash scripts/pipe.sh flickr30k MultiCapCLIP_001#de-de "adapt adapt_zeroshot"

bash scripts/pipe.sh flickr30k baseline#fr-fr "finetune"
bash scripts/pipe.sh flickr30k baseline#fr-fr "adapt adapt_zeroshot"
bash scripts/pipe.sh flickr30k MultiCapCLIP_001#fr-fr "adapt adapt_zeroshot"
```
</details>

T-SNE Visualization: See [notebooks/zero_shot_tsne.ipynb](/notebooks/zero_shot_tsne.ipynb) for an example.


## Citation
Please **[★star]** this repo and **[cite]** the following papers if you feel our code or data useful to your research:

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

@ARTICLE{Yang2024ZeroNLG,
  author={Yang, Bang and Liu, Fenglin and Zou, Yuexian and Wu, Xian and Wang, Yaowei and Clifton, David A.},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={ZeroNLG: Aligning and Autoencoding Domains for Zero-Shot Multimodal and Multilingual Natural Language Generation}, 
  year={2024},
  volume={46},
  number={8},
  pages={5712-5724},
  doi={10.1109/TPAMI.2024.3371376}
}
```

## Acknowledgements
- Code is inspired by [CapDec](https://github.com/DavidHuji/CapDec) and [X-VLM](https://github.com/zengyan-97/X-VLM).
- Data is derived from [ZeroNLG](https://github.com/yangbang18/ZeroNLG).
