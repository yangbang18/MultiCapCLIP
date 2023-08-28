dataset=$1
method=$2
other=$3

other=${other:-}

if [ $dataset = 'coco' ] || [ $dataset = 'vatex' ];
then
    arr=(0.01% 0.1% 1% 10%)
else
    arr=(0.1% 1% 10%)
fi

for r in "${arr[@]}"
do
    for n in {0..2}
    do
        name=${r}_${n}
        
        python3 finetune.py \
        --clip_arch ViT-B/16 \
        --method ${method} \
        --dataset ${dataset} \
        --pickle \
        --output_dir output/finetune/vit-b-16_${dataset}_${method}/${dataset}_subsets/$name \
        --subset_fn $name.json \
        $other
    done
done
