from=$1
dataset=$2
other=$3

from=${from:-output/unsupervised/vit-b-16_coco_n1_p1_0.10}
dataset=${dataset:-coco}
other=${other:-}

if [ $dataset = 'coco' ] || [ $dataset = 'vatex' ];
then
    arr=(0.01% 0.1% 1% 10%)
else
    arr=(0.1% 1% 10%)
fi

ckpt=${dataset}_best.pth

if [ $dataset = 'msrvtt' ] || [ $dataset = 'vatex' ];
then
    key='video'
else
    key='image'
fi

for r in "${arr[@]}"
do
    for n in {0..2}
    do
        subdir=${dataset}_subsets
        name=${r}_${n}

        python3 finetune.py \
        --clip_arch ViT-B/16 \
        --from_pretrained $from \
        --folder $subdir/$name \
        --subset_fn $name.json \
        --ckpt_name $ckpt \
        --dataset $dataset \
        --pickle \
        --fewshot \
        $other
    done
done
