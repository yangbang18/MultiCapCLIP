folder=$1
device=$2

device=${device:-cuda}

for file in ${folder}/*.th; do
    root=`dirname $file`
    ckpt=`basename $file`
    echo $root
    echo $ckpt
    python3 finetune.py --from_pretrained $root --ckpt_name $ckpt --folder coco_subsets/0% --evaluate --device $device
done
